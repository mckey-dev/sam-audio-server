"""HTTP サーバー全体の設定。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from modules.paths import DEFAULT_MODEL_JSON


@dataclass(frozen=True)
class SamAudioConfig:
    """SAM Audio 推論設定。カタログ JSON の ID でモデルを選ぶ。"""

    model_json: Path = field(default_factory=lambda: DEFAULT_MODEL_JSON)
    model_id: str | None = None  # None ならカタログの default
    device: str = "auto"  # auto / cpu / cuda（ROCm も cuda）
    skip_model_load: bool = False
    hf_token: str | None = None


@dataclass(frozen=True)
class ServerConfig:
    """uvicorn / FastAPI が参照するサーバー設定。"""

    host: str = "127.0.0.1"
    port: int = 8765
    cors_origins: tuple[str, ...] = (
        "http://127.0.0.1:8765",
        "http://localhost:8765",
        "http://127.0.0.1:8080",
        "http://localhost:8080",
    )
    max_upload_bytes: int = 100 * 1024 * 1024  # アップロード合計の上限
    sam: SamAudioConfig = field(default_factory=SamAudioConfig)
