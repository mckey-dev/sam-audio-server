"""CLI エントリ: 引数解析と uvicorn による HTTP サーバー起動。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from modules.paths import configure_hf_cache

configure_hf_cache()

import uvicorn

from modules.app import create_app
from modules.config import SamAudioConfig, ServerConfig
from modules.model_spec import load_model_spec
from modules.paths import DEFAULT_MODEL_JSON


def parse_args(argv: list[str] | None = None) -> ServerConfig:
    """コマンドライン引数を解析し、サーバー設定オブジェクトを返す。"""
    parser = argparse.ArgumentParser(description="SAM Audio HTTP server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8765, help="Bind port")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--model",
        default=None,
        help="カタログ上のモデル ID（省略時は models.json の default）",
    )
    parser.add_argument(
        "--model-json",
        default=str(DEFAULT_MODEL_JSON),
        help="モデルカタログ JSON（既定: models/models.json）",
    )
    parser.add_argument(
        "--skip-model-load",
        action="store_true",
        help="Start the HTTP server without loading SAM Audio weights",
    )
    parser.add_argument(
        "--cors-origin",
        action="append",
        dest="cors_origins",
        help="Allowed CORS origin (repeatable). Defaults include localhost test URLs.",
    )
    args = parser.parse_args(argv)

    try:
        spec = load_model_spec(args.model_json, args.model)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    sam = SamAudioConfig(
        model_json=Path(args.model_json),
        model_id=spec.id,
        device=args.device,
        skip_model_load=args.skip_model_load,
    )
    origins = tuple(args.cors_origins) if args.cors_origins else ServerConfig().cors_origins
    return ServerConfig(host=args.host, port=args.port, cors_origins=origins, sam=sam)


def main(argv: list[str] | None = None) -> None:
    """設定を読み込み、FastAPI アプリを uvicorn で待ち受けする。"""
    configure_hf_cache()
    config = parse_args(argv)
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
