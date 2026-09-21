"""ROCm では CUDA 用 xFormers を読まず、PyTorch SDPA へ逃がす。"""

from __future__ import annotations

import logging
import os
import sys
import types

logger = logging.getLogger("uvicorn.error")


def is_rocm() -> bool:
    """インストール済み PyTorch が HIP / ROCm ビルドなら True。"""
    try:
        import torch
    except ImportError:
        return False
    return bool(getattr(torch.version, "hip", None))


def require_expected_backend() -> None:
    """run_cuda.bat などは SAM_AUDIO_EXPECT_BACKEND で PyTorch 系統を固定する。"""
    expect = os.environ.get("SAM_AUDIO_EXPECT_BACKEND", "").strip().lower()
    if not expect:
        return
    import torch

    hip = is_rocm()
    name = "none"
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        name = torch.cuda.get_device_name(0)
    if expect == "cuda" and hip:
        raise RuntimeError(
            "NVIDIA CUDA was requested, but this Python environment is ROCm "
            f"(visible GPU: {name}). Run install_cuda.bat then run_cuda.bat "
            "(they use venv-cuda, and keep the 780M ROCm venv)."
        )
    if expect == "rocm" and not hip:
        logger.warning(
            "ROCm was requested, but torch.version.hip is empty (CUDA PyTorch?). GPU=%s",
            name,
        )
    if expect == "cpu" and torch.cuda.is_available():
        logger.info("CPU backend requested; GPU %s is present but will not be used", name)


def install_xformers_stub() -> bool:
    """ROCm のときだけ、本物の xformers より先にスタブを登録する。"""
    if not is_rocm():
        return False
    if "xformers.ops" in sys.modules and getattr(sys.modules["xformers"], "_sam_audio_stub", False):
        return True
    if "xformers" in sys.modules and not getattr(sys.modules["xformers"], "_sam_audio_stub", False):
        logger.warning("xformers is already imported; ROCm stub was not applied")
        return False

    import torch
    import torch.nn.functional as F

    class AttentionBias:
        """xformers.ops.AttentionBias の型だけ合わせる空クラス。"""

        pass

    class _Fmha:
        """メモリ効率アテンションを PyTorch SDPA に置き換える。"""

        @staticmethod
        def memory_efficient_attention(query, key, value, attn_bias=None, **kwargs):
            """query/key/value を SDPA で計算し、xformers と同じ並びで返す。"""
            q = query.transpose(1, 2)
            k = key.transpose(1, 2)
            v = value.transpose(1, 2)
            mask = attn_bias if isinstance(attn_bias, torch.Tensor) else None
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            return out.transpose(1, 2).contiguous()

    xformers = types.ModuleType("xformers")
    xformers._sam_audio_stub = True
    xformers.__version__ = "0.0.0+rocm-stub"

    ops = types.ModuleType("xformers.ops")
    ops.AttentionBias = AttentionBias
    ops.fmha = _Fmha()

    fmha = types.ModuleType("xformers.ops.fmha")
    fmha.memory_efficient_attention = _Fmha.memory_efficient_attention

    profiler = types.ModuleType("xformers.profiler")
    profiler.MemSnapshotsProfiler = object
    profiler.PyTorchProfiler = object

    sys.modules["xformers"] = xformers
    sys.modules["xformers.ops"] = ops
    sys.modules["xformers.ops.fmha"] = fmha
    sys.modules["xformers.profiler"] = profiler
    logger.info("ROCm detected: using PyTorch SDPA instead of xFormers")
    return True
