"""プロジェクト内のディレクトリパスと Hugging Face キャッシュの設定。"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # リポジトリルート
MODELS_DIR = ROOT / "models"  # 重みと HF キャッシュ
TEST_DIR = ROOT / "test"  # ブラウザテスト UI
DEFAULT_MODEL_JSON = MODELS_DIR / "models.json"


def configure_hf_cache() -> Path:
    """モデル保存先を <project>/models に固定し、関連環境変数を設定する。"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    hub = MODELS_DIR / "hub"
    transformers_dir = MODELS_DIR / "transformers"
    hub.mkdir(parents=True, exist_ok=True)
    transformers_dir.mkdir(parents=True, exist_ok=True)
    models = str(MODELS_DIR.resolve())
    hub_path = str(hub.resolve())
    os.environ["HF_HOME"] = models
    os.environ["HUGGINGFACE_HUB_CACHE"] = hub_path
    os.environ["HF_HUB_CACHE"] = hub_path
    os.environ["TRANSFORMERS_CACHE"] = str(transformers_dir.resolve())
    os.environ["HF_DATASETS_CACHE"] = str((MODELS_DIR / "datasets").resolve())
    return MODELS_DIR
