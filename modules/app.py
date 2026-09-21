"""FastAPI アプリケーション: HTTP API とブラウザテスト UI の配信。"""

from __future__ import annotations

import logging
import shutil
import tempfile
import zipfile
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from modules.config import ServerConfig
from modules.engine import Engine, SAMAudioEngine, SeparationOutput
from modules.parsing import parse_anchors, parse_bool
from modules.paths import TEST_DIR, configure_hf_cache

logger = logging.getLogger("uvicorn.error")

OutputMode = Literal["zip", "target", "residual", "json"]


def create_app(config: ServerConfig, engine: Engine | None = None) -> FastAPI:
    """サーバー設定から FastAPI アプリを組み立てる。"""
    engine = engine or SAMAudioEngine(config.sam)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """起動時にモデルを読み、終了時に ready を落とす。"""
        configure_hf_cache()
        app.state.config = config
        app.state.engine = engine
        engine.load()
        yield
        engine.ready = False

    app = FastAPI(title="sam-audio-server", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if TEST_DIR.is_dir():
        app.mount("/test", StaticFiles(directory=str(TEST_DIR), html=True), name="test")

    @app.get("/")
    async def root() -> dict[str, str]:
        """サービス名と主要エンドポイントへの案内を返す。"""
        return {
            "service": "sam-audio-server",
            "test_ui": "/test/",
            "health": "/health",
        }

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """稼働状態とモデル読み込み済みかどうかを返す。"""
        loaded = bool(getattr(engine, "ready", False))
        return {"status": "ok", "model_loaded": loaded}

    @app.get("/v1/info")
    async def info() -> dict[str, Any]:
        """ロード済みモデル、デバイス、サンプルレートを返す。"""
        return engine.info()

    @app.post("/v1/separate")
    async def separate(
        audio: Annotated[UploadFile | None, File()] = None,
        video: Annotated[UploadFile | None, File()] = None,
        mask: Annotated[UploadFile | None, File()] = None,
        description: Annotated[str, Form()] = "",
        anchors: Annotated[str | None, Form()] = None,
        predict_spans: Annotated[str, Form()] = "false",
        reranking_candidates: Annotated[int, Form()] = 1,
        output: Annotated[OutputMode, Form()] = "zip",
    ) -> Response:
        """アップロードされた音声を SAM Audio で分離する。"""
        if audio is None and video is None:
            raise HTTPException(status_code=400, detail="audio or video file is required")
        if reranking_candidates < 1:
            raise HTTPException(status_code=400, detail="reranking_candidates must be >= 1")
        if not engine.ready:
            raise HTTPException(status_code=503, detail="SAM Audio model is not loaded")

        total = 0
        for upload in (audio, video, mask):
            if upload is not None and upload.size:
                total += int(upload.size)
        if total > config.max_upload_bytes:
            raise HTTPException(status_code=413, detail="File too large")

        try:
            parsed_anchors = parse_anchors(anchors)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        with tempfile.TemporaryDirectory(prefix="sam-audio-") as tmp:
            tmp_dir = Path(tmp)
            audio_path = await _save_upload(audio, tmp_dir, "input_audio")
            video_path = await _save_upload(video, tmp_dir, "input_video")
            mask_path = await _save_upload(mask, tmp_dir, "input_mask")
            prompt = description.strip()
            source = audio.filename if audio is not None and audio.filename else (
                video.filename if video is not None else None
            )
            logger.info("Separation request: %s (%s)", prompt or "(no description)", source or "upload")
            try:
                result = engine.separate(
                    audio_path=audio_path,
                    video_path=video_path,
                    mask_path=mask_path,
                    description=description.strip(),
                    anchors=parsed_anchors,
                    predict_spans=parse_bool(predict_spans),
                    reranking_candidates=reranking_candidates,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                logger.exception("Separation failed")
                raise HTTPException(status_code=500, detail=str(exc)) from exc

        return _build_response(result, output)

    return app


async def _save_upload(upload: UploadFile | None, tmp_dir: Path, stem: str) -> str | None:
    """アップロードを一時ディレクトリへ保存し、パスを返す。未指定なら None。"""
    if upload is None or not upload.filename:
        return None
    suffix = Path(upload.filename).suffix or ".bin"
    dest = tmp_dir / f"{stem}{suffix}"
    with dest.open("wb") as handle:
        shutil.copyfileobj(upload.file, handle)
    return str(dest)


def _build_response(result: SeparationOutput, output: OutputMode) -> Response:
    """分離結果を zip / 単体 WAV / JSON のいずれかで返す。"""
    if output == "target":
        return Response(
            content=result.target_wav,
            media_type="audio/wav",
            headers={"Content-Disposition": 'attachment; filename="target.wav"'},
        )
    if output == "residual":
        return Response(
            content=result.residual_wav,
            media_type="audio/wav",
            headers={"Content-Disposition": 'attachment; filename="residual.wav"'},
        )
    if output == "json":
        import base64

        return JSONResponse(
            {
                "model_id": result.model_id,
                "description": result.description,
                "sample_rate": result.sample_rate,
                "target_wav_b64": base64.b64encode(result.target_wav).decode("ascii"),
                "residual_wav_b64": base64.b64encode(result.residual_wav).decode("ascii"),
            }
        )

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("target.wav", result.target_wav)
        archive.writestr("residual.wav", result.residual_wav)
        archive.writestr(
            "meta.txt",
            f"model_id={result.model_id}\nsample_rate={result.sample_rate}\ndescription={result.description}\n",
        )
    return Response(
        content=buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="sam-audio-output.zip"'},
    )
