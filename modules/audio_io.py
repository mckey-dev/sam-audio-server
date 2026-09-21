"""音声ファイルの読み書きと推論デバイスの解決。torchcodec は使わない。"""

from __future__ import annotations

import io
import logging
import shutil
import subprocess
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio

logger = logging.getLogger("uvicorn.error")


def read_audio(path: str) -> tuple[torch.Tensor, int]:
    """音声ファイルを [channels, time] float32 とサンプルレートで読む。"""
    try:
        data, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    except Exception:
        data, sample_rate = _decode_with_ffmpeg(path)
    waveform = torch.from_numpy(np.ascontiguousarray(data.T))
    return waveform, int(sample_rate)


def load_waveform(path: str, target_sr: int) -> torch.Tensor:
    """音声を読み、必要なら target_sr へリサンプルして返す。"""
    waveform, sample_rate = read_audio(path)
    if sample_rate != target_sr:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
    return waveform


def tensor_to_wav_bytes(waveform: Any, sample_rate: int) -> bytes:
    """波形テンソルを PCM 16-bit WAV バイト列に変換する。"""
    wav = waveform.detach().cpu().float()
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    elif wav.ndim == 3:
        wav = wav.squeeze(0)
    data = np.clip(wav.numpy(), -1.0, 1.0)
    if data.ndim == 2:
        data = data.T
    buffer = io.BytesIO()
    sf.write(buffer, data, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def patch_torchaudio_io() -> None:
    """SAM Audio 内部の torchaudio.load/save も soundfile 経由にする。"""
    if getattr(torchaudio, "_sam_audio_io_patched", False):
        return

    def load(uri, *args, **kwargs):
        """torchaudio.load 互換。ファイルまたは file-like から波形を読む。"""
        if hasattr(uri, "read"):
            data, sample_rate = sf.read(uri, always_2d=True, dtype="float32")
            return torch.from_numpy(np.ascontiguousarray(data.T)), int(sample_rate)
        return read_audio(str(uri))

    def save(uri, src, sample_rate, channels_first=True, format=None, **kwargs):
        """torchaudio.save 互換。WAV をパスまたは file-like へ書く。"""
        wav = src.detach().cpu().float()
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        data = np.clip(wav.numpy(), -1.0, 1.0)
        if channels_first and data.ndim == 2:
            data = data.T
        subtype = "PCM_16"
        fmt = (format or "WAV").upper()
        if hasattr(uri, "write"):
            sf.write(uri, data, int(sample_rate), format=fmt, subtype=subtype)
            return
        sf.write(str(uri), data, int(sample_rate), format=fmt, subtype=subtype)

    torchaudio.load = load
    torchaudio.save = save
    torchaudio._sam_audio_io_patched = True
    logger.info("Using soundfile for torchaudio load/save (bypassing torchcodec)")


def pick_device(requested: str):
    """'auto' / 'cpu' / 'cuda' を実際の torch.device に解決する。"""
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    """ターミナル表示用。ROCm でも torch 上は cuda なので GPU 名とバックエンドを添える。"""
    if device.type != "cuda" or not torch.cuda.is_available():
        return str(device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    mem_gb = props.total_memory / (1024**3)
    hip = getattr(torch.version, "hip", None)
    backend = f"ROCm {hip}" if hip else f"CUDA {torch.version.cuda}"
    return f"cuda:{index} ({props.name}, {mem_gb:.1f} GiB, {backend})"


def _decode_with_ffmpeg(path: str) -> tuple[np.ndarray, int]:
    """soundfile で読めない形式を ffmpeg CLI で WAV に変換して読む。"""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            f"Cannot decode audio file {path}. Use wav/flac/ogg, or install ffmpeg and add it to PATH for mp3."
        )
    proc = subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            path,
            "-f",
            "wav",
            "-acodec",
            "pcm_s16le",
            "pipe:1",
        ],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        detail = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed to decode {path}: {detail or 'no output'}")
    data, sample_rate = sf.read(io.BytesIO(proc.stdout), always_2d=True, dtype="float32")
    logger.info("Decoded %s with ffmpeg", path)
    return data, int(sample_rate)
