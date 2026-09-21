"""モデルカタログ JSON の読み込みと Hugging Face からのファイル取得。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from modules.paths import DEFAULT_MODEL_JSON, MODELS_DIR, ROOT


@dataclass(frozen=True)
class ModelSpec:
    """カタログ上の 1 モデル。Hugging Face の config / 重みと Lite 可否。"""

    id: str
    repo_id: str
    config: str
    weights: str
    disable_gated_extras: bool
    source_json: Path
    lite: bool


def resolve_model_json(path: str | Path | None) -> Path:
    """カタログ JSON のパスを絶対パスにする。"""
    if path is None or str(path).strip() == "":
        return DEFAULT_MODEL_JSON
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    return resolved.resolve()


def load_model_spec(path: str | Path | None = None, model_id: str | None = None) -> ModelSpec:
    """カタログ JSON から ID で 1 件を選び ModelSpec にする。"""
    json_path = resolve_model_json(path)
    if not json_path.is_file():
        raise FileNotFoundError(f"モデルカタログ JSON が見つかりません: {json_path}")
    with json_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    models = data.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError(f"{json_path} には models オブジェクトが必要です")
    requested = (model_id or "").strip() or str(data.get("default") or "")
    if not requested:
        raise ValueError(f"{json_path} に default が無く、--model も指定されていません")
    entry = models.get(requested)
    if not isinstance(entry, dict):
        known = ", ".join(sorted(models))
        raise ValueError(f"未知のモデル ID: {requested}（利用可能: {known}）")
    config = entry.get("config")
    weights = entry.get("weights")
    if not config or not weights:
        raise ValueError(f"{json_path} の {requested} には config と weights が必要です")
    if "lite" not in entry:
        raise ValueError(f"{json_path} の {requested} には lite が必要です")
    return ModelSpec(
        id=requested,
        repo_id=str(entry.get("repo_id") or "aoiandroid/sam-audio-models"),
        config=str(config),
        weights=str(weights),
        disable_gated_extras=bool(entry.get("disable_gated_extras", True)),
        source_json=json_path,
        lite=bool(entry["lite"]),
    )


def download_spec_files(spec: ModelSpec, token: str | None = None) -> tuple[Path, Path]:
    """config と重みを models/ へ取得し、ローカルパスを返す。"""
    from huggingface_hub import hf_hub_download

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config_path = Path(
        hf_hub_download(
            repo_id=spec.repo_id,
            filename=spec.config,
            local_dir=str(MODELS_DIR),
            token=token,
        )
    )
    weights_path = Path(
        hf_hub_download(
            repo_id=spec.repo_id,
            filename=spec.weights,
            local_dir=str(MODELS_DIR),
            token=token,
        )
    )
    return config_path, weights_path
