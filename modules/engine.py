"""SAM Audio: デバイス解決、チェックポイント読み込み、音源分離。"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Protocol

from modules.config import SamAudioConfig
from modules.model_spec import download_spec_files, load_model_spec
from modules.parsing import Anchor
from modules.paths import configure_hf_cache

logger = logging.getLogger("uvicorn.error")


@dataclass
class SeparationOutput:
    """音源分離の結果。target が抜き出した音、residual が残り。"""

    target_wav: bytes
    residual_wav: bytes
    sample_rate: int
    description: str
    model_id: str


class Engine(Protocol):
    """HTTP 層が使うエンジンの契約。"""

    config: SamAudioConfig
    ready: bool

    def load(self) -> None:
        """チェックポイントを読み込む。"""
        ...

    def info(self) -> dict[str, Any]:
        """ロード済みモデル情報を返す。"""
        ...

    def separate(
        self,
        *,
        audio_path: str | None,
        video_path: str | None,
        mask_path: str | None,
        description: str,
        anchors: list[Anchor] | None,
        predict_spans: bool,
        reranking_candidates: int,
    ) -> SeparationOutput:
        """音源分離を実行する。"""
        ...


class SAMAudioEngine:
    """チェックポイントを読み、テキスト指定で音源分離する。"""

    def __init__(self, config: SamAudioConfig) -> None:
        """推論設定を保持し、モデルは load() まで遅延読み込みする。"""
        self.config = config
        self.ready = False
        self._lock = threading.Lock()
        self._model = None
        self._processor = None
        self._device = None
        self._sample_rate = 48_000
        self._spec = None
        self._lite = False
        self._infer_dtype = None

    def load(self) -> None:
        """JSON で指定した重みをデバイスへ載せ、推論可能にする。"""
        if self.config.skip_model_load:
            logger.warning("Skipping SAM Audio model load (--skip-model-load)")
            return
        if self.ready:
            return

        t0 = time.perf_counter()
        configure_hf_cache()

        import torch

        from modules.codec_compat import hide_torchcodec_from_transformers, install_torchcodec_stub
        from modules.rocm_compat import install_xformers_stub, require_expected_backend

        require_expected_backend()
        install_xformers_stub()
        install_torchcodec_stub()

        from huggingface_hub import login
        from safetensors.torch import load_file
        from sam_audio import SAMAudio, SAMAudioProcessor
        from sam_audio.model.config import SAMAudioConfig as CheckpointConfig

        from modules.audio_io import describe_device, patch_torchaudio_io, pick_device
        from modules.paths import MODELS_DIR

        hide_torchcodec_from_transformers()
        patch_torchaudio_io()

        spec = load_model_spec(self.config.model_json, self.config.model_id)
        self._spec = spec
        device = pick_device(self.config.device)
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                logger.info("Visible GPU %s: %s", index, torch.cuda.get_device_name(index))
        logger.info("Using device: %s (requested=%s)", describe_device(device), self.config.device)
        token = self.config.hf_token or os.environ.get("HF_TOKEN") or os.environ.get(
            "HUGGING_FACE_HUB_TOKEN"
        )
        if token:
            login(token=token, add_to_git_credential=False)

        logger.info(
            "Loading SAM Audio id=%s from %s (%s / %s) on %s into %s",
            spec.id,
            spec.repo_id,
            spec.config,
            spec.weights,
            device,
            MODELS_DIR,
        )
        config_path, weights_path = download_spec_files(spec, token=token)
        with config_path.open(encoding="utf-8") as handle:
            config_data = json.load(handle)
        if spec.disable_gated_extras:
            config_data["span_predictor"] = None
            config_data["text_ranker"] = None

        lite = spec.lite
        infer_dtype = _lite_dtype(device) if lite else None
        logger.info(
            "Lite mode: %s (model_id=%s, dtype=%s)",
            "on" if lite else "off",
            spec.id,
            infer_dtype,
        )
        if lite:
            config_data["span_predictor"] = None
            config_data["text_ranker"] = None
            config_data["visual_ranker"] = None
            sam_mod = sys.modules[SAMAudio.__module__]
            sam_mod.PerceptionEncoder = _lite_vision_encoder_type()

        checkpoint_cfg = CheckpointConfig(**config_data)
        processor = SAMAudioProcessor(
            audio_hop_length=checkpoint_cfg.audio_codec.hop_length,
            audio_sampling_rate=checkpoint_cfg.audio_codec.sample_rate,
        )
        model = SAMAudio(checkpoint_cfg)
        state = load_file(str(weights_path), device="cpu")
        missing, unexpected = torch.nn.Module.load_state_dict(model, state, strict=False)
        del state
        if unexpected:
            logger.warning("Unexpected weight keys: %s", unexpected[:8])
        if missing:
            logger.info("Missing optional keys: %s", missing[:8])
        if infer_dtype is not None:
            model = model.eval().to(device=device, dtype=infer_dtype)
        else:
            model = model.eval().to(device)
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("VRAM: %s", _format_vram(device))

        self._processor = processor
        self._model = model
        self._device = device
        self._infer_dtype = infer_dtype
        self._lite = lite
        self._sample_rate = int(processor.audio_sampling_rate)
        self.ready = True
        logger.info(
            "Model load: %ss",
            f"{time.perf_counter() - t0:,.1f}",
        )
        logger.info(
            "SAM Audio ready on %s (sample_rate=%s)",
            describe_device(device),
            self._sample_rate,
        )

    def info(self) -> dict[str, Any]:
        """ロード済みモデルとデバイス情報を返す。"""
        from modules.paths import MODELS_DIR

        spec = self._spec
        return {
            "model_id": spec.id if spec else self.config.model_id,
            "repo_id": spec.repo_id if spec else None,
            "config": spec.config if spec else None,
            "weights": spec.weights if spec else None,
            "model_json": str(spec.source_json if spec else self.config.model_json),
            "device": str(self._device or self.config.device),
            "sample_rate": self._sample_rate,
            "model_loaded": self.ready,
            "lite": self._lite,
            "dtype": str(self._infer_dtype) if self._infer_dtype is not None else None,
            "models_dir": str(MODELS_DIR),
        }

    def separate(
        self,
        *,
        audio_path: str | None,
        video_path: str | None,
        mask_path: str | None,
        description: str,
        anchors: list[Anchor] | None,
        predict_spans: bool,
        reranking_candidates: int,
    ) -> SeparationOutput:
        """入力ミックスから description に合う音を抜き出し、残りと一緒に返す。"""
        if not self.ready or self._model is None or self._processor is None:
            raise RuntimeError("SAM Audio model is not loaded")

        from modules.audio_io import load_waveform, tensor_to_wav_bytes

        media_path = audio_path or video_path
        if not media_path:
            raise ValueError("audio or video input is required")

        kwargs: dict[str, Any] = {
            "audios": [load_waveform(media_path, self._sample_rate)],
            "descriptions": [description],
        }
        if anchors:
            kwargs["anchors"] = [anchors]
        if video_path and mask_path:
            if self._lite:
                raise ValueError("lite mode: visual prompting is disabled")
            kwargs["masked_videos"] = self._processor.mask_videos(
                [video_path],
                [mask_path],
            )
        if self._lite and predict_spans:
            logger.warning("lite mode: predict_spans is ignored")
            predict_spans = False

        spec = self._spec
        model_id = spec.id if spec else "sam-audio"
        import torch

        with self._lock:
            if self._device is not None and self._device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(self._device)
            t0 = time.perf_counter()
            batch = self._processor(**kwargs).to(self._device)
            if self._infer_dtype is not None:
                _cast_batch_floats(batch, self._infer_dtype)
            result = self._model.separate(
                batch,
                predict_spans=predict_spans,
                reranking_candidates=reranking_candidates,
            )
            target = result.target[0]
            residual = result.residual[0]
            if self._device is not None and self._device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = int(round((time.perf_counter() - t0) * 1000))
            logger.info(
                "Inference: %sms  VRAM: %s",
                f"{elapsed_ms:,}",
                _format_vram(self._device, peak=True),
            )
            sample_rate = self._sample_rate

        return SeparationOutput(
            target_wav=tensor_to_wav_bytes(target, sample_rate),
            residual_wav=tensor_to_wav_bytes(residual, sample_rate),
            sample_rate=sample_rate,
            description=description,
            model_id=model_id,
        )


def _lite_dtype(device):
    """Lite の重みは BF16。FP16 は ODE が不安定になり分離精度が落ちる。"""
    import torch

    if device.type != "cuda":
        return None
    return torch.bfloat16


def _format_vram(device, peak: bool = False) -> str:
    """OS の専用 GPU メモリ（HIP/MIOpen/ドライバ込み）と、PyTorch テンソル量。"""
    import torch

    from modules.gpu_mem import query_os_vram

    if device is None or device.type != "cuda" or not torch.cuda.is_available():
        return "n/a"
    tensor_bytes = (
        torch.cuda.max_memory_allocated(device) if peak else torch.cuda.memory_allocated(device)
    )
    reserved_bytes = (
        torch.cuda.max_memory_reserved(device) if peak else torch.cuda.memory_reserved(device)
    )
    tensors = tensor_bytes / (1024**3)
    reserved = reserved_bytes / (1024**3)
    kind = "peak" if peak else "now"
    parts: list[str] = []
    os_vram = query_os_vram(torch.cuda.get_device_name(device))
    if os_vram is not None:
        used, total = os_vram
        parts.append(f"{used:.2f}/{total:.2f} GiB (OS dedicated)")
    parts.append(f"tensors {tensors:.2f} GiB ({kind}) / reserved {reserved:.2f} GiB")
    return "  ".join(parts)


def _cast_batch_floats(batch: Any, dtype) -> None:
    """入力テンソルをモデルと同じ dtype に揃える。"""
    import torch

    for name in ("audios", "masked_video"):
        value = getattr(batch, name, None)
        if torch.is_tensor(value) and value.is_floating_point():
            setattr(batch, name, value.to(dtype=dtype))
        elif isinstance(value, list):
            setattr(
                batch,
                name,
                [
                    item.to(dtype=dtype) if torch.is_tensor(item) and item.is_floating_point() else item
                    for item in value
                ],
            )


def _lite_vision_encoder_type():
    """PE 本体を読まず、dim だけ持つダミー視覚エンコーダ。"""
    import torch

    class LiteVisionEncoder(torch.nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.dim = int(cfg.dim)
            self.batch_size = 0

        def forward(self, videos):
            raise RuntimeError("lite mode: visual prompting is disabled")

    return LiteVisionEncoder
