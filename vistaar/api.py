"""HTTP API. Upload a document, ask questions, get cited answers."""
from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from .config import Settings
from .pipeline import RAGPipeline, RetrievalConfig

logger = logging.getLogger("vistaar")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
limiter = Limiter(key_func=get_remote_address, default_limits=[])


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    namespace: str = Field(default="default", max_length=128)


class ConfigRequest(BaseModel):
    namespace: str = Field(default="default", max_length=128)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    pipeline = RAGPipeline(settings)

    app = FastAPI(title="Vistaar", version="0.2.0")
    app.state.limiter = limiter
    app.state.settings = settings
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.state.pipeline = pipeline

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request):
        return templates.TemplateResponse(
            request,
            "index.html",
            {"reader": settings.reader, "model": settings.reader_model if settings.reader == "local" else settings.model},
        )

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "reader": settings.reader,
            "namespaces": pipeline.store.namespaces(),
            "settings": settings.as_safe_dict(),
        }

    @app.post("/upload")
    @limiter.limit("10/minute")
    async def upload(
        request: Request,
        file: UploadFile = File(...),
        namespace: str = Form("default"),
    ):
        started = time.monotonic()
        content = await file.read()

        if len(content) > settings.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds {settings.max_upload_bytes // (1024*1024)} MB limit.",
            )
        if not content:
            raise HTTPException(status_code=400, detail="Empty file.")

        try:
            stats = pipeline.index_document(
                file.filename or "upload.txt", content, namespace=namespace
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # unexpected
            logger.exception("indexing failed")
            raise HTTPException(status_code=500, detail="Failed to process document.") from exc

        logger.info(
            "indexed doc=%s chunks=%s ns=%s in %.2fs",
            stats["doc_id"], stats["chunks"], namespace, time.monotonic() - started,
        )
        return {"message": "Document indexed.", **stats}

    @app.post("/ask")
    @limiter.limit("30/minute")
    def ask(request: Request, req: QuestionRequest):
        answer = pipeline.answer(req.question, namespace=req.namespace)
        payload = answer.to_dict()
        payload["message"] = "ok"
        return JSONResponse(payload)

    @app.delete("/namespace/{namespace}")
    def clear(namespace: str):
        pipeline.store.reset_namespace(namespace)
        return {"message": f"Cleared {namespace}"}

    return app


def new_namespace() -> str:
    return f"session-{uuid.uuid4().hex[:8]}"
