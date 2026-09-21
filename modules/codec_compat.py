"""torchcodec のネイティブ DLL を読まず、SAM Audio の import を通す。"""

from __future__ import annotations

import logging
import sys
import types
from importlib.machinery import ModuleSpec

logger = logging.getLogger("uvicorn.error")


def install_torchcodec_stub() -> bool:
    """本物の torchcodec より先にスタブを登録する。音声は soundfile で読む。"""
    existing = sys.modules.get("torchcodec")
    if existing is not None and getattr(existing, "_sam_audio_stub", False):
        hide_torchcodec_from_transformers()
        return True
    if existing is not None:
        logger.warning("torchcodec is already imported; stub was not applied")
        return False

    def _fail(*args, **kwargs):
        """本物のデコーダを呼ばれたときに明示的に失敗させる。"""
        raise RuntimeError(
            "torchcodec is disabled. Use wav/flac/ogg audio. Video prompting needs a matching torchcodec/PyTorch pair."
        )

    class _Unavailable:
        """AudioDecoder / VideoDecoder 互換のダミー。実デコードは行わない。"""

        def __init__(self, *args, **kwargs):
            """デコーダ生成は未対応。"""
            _fail()

        @classmethod
        def encode(cls, *args, **kwargs):
            """エンコード要求も同様に拒否する。"""
            _fail()

    torchcodec = _stub_module("torchcodec", is_package=True)
    torchcodec._sam_audio_stub = True
    torchcodec.__version__ = "0.0.0+stub"

    decoders = _stub_module("torchcodec.decoders")
    decoders.AudioDecoder = _Unavailable
    decoders.VideoDecoder = _Unavailable

    encoders = _stub_module("torchcodec.encoders")
    encoders.AudioEncoder = _Unavailable
    encoders.VideoEncoder = _Unavailable

    torchcodec.decoders = decoders
    torchcodec.encoders = encoders
    sys.modules["torchcodec"] = torchcodec
    sys.modules["torchcodec.decoders"] = decoders
    sys.modules["torchcodec.encoders"] = encoders
    hide_torchcodec_from_transformers()
    logger.info("torchcodec stub installed (native DLL will not be loaded)")
    return True


def hide_torchcodec_from_transformers() -> None:
    """transformers の可用性チェックがスタブを本物と誤認しないようにする。"""
    iu = sys.modules.get("transformers.utils.import_utils")
    if iu is None:
        return
    orig = getattr(iu, "_is_package_available", None)
    if orig is None or getattr(orig, "_sam_audio_wrapped", False):
        return

    def wrapped(pkg_name: str, return_version: bool = False):
        """torchcodec だけ未インストール扱いにし、他は元の判定を使う。"""
        if pkg_name == "torchcodec":
            return (False, "N/A") if return_version else (False, None)
        return orig(pkg_name, return_version)

    wrapped._sam_audio_wrapped = True
    iu._is_package_available = wrapped
    available = getattr(iu, "is_torchcodec_available", None)
    if available is not None and hasattr(available, "cache_clear"):
        available.cache_clear()


def _stub_module(name: str, is_package: bool = False) -> types.ModuleType:
    """importlib.find_spec が落ちないよう __spec__ 付きの偽モジュールを作る。"""
    module = types.ModuleType(name)
    module.__spec__ = ModuleSpec(name, loader=None, is_package=is_package)
    parent, _, _ = name.rpartition(".")
    module.__package__ = name if is_package else parent
    return module
