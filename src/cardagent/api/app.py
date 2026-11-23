"""REST API.  Run: uvicorn cardagent.api.app:app --host 0.0.0.0 --port 8080 --workers N"""
from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .. import __version__
from ..agent import CardAgent
from ..observability.telemetry import audit_record, setup_logging
from ..schemas import ValidationResult
from ..settings import get_settings
from ..store.base import StoreUnavailable
from . import admin_cards, console
from .deps import STATE, agent, require_admin

log = logging.getLogger("cardagent.api")


class _IdempotencyCache:
    """Same request_id AND same image bytes -> same result (capture systems retry on timeouts).
    Keyed on both, so reusing a request_id with a different photo is processed fresh instead of
    replaying the earlier card's decision. Size 0 disables caching.
    Single-process cache; use Redis when running several replicas."""

    def __init__(self, size: int):
        self.size, self.data, self.lock = size, OrderedDict(), Lock()

    def get(self, k):
        with self.lock:
            v = self.data.get(k)
            if v is not None:
                self.data.move_to_end(k)
            return v

    def put(self, k, v):
        if self.size <= 0:
            return
        with self.lock:
            self.data[k] = v
            if len(self.data) > self.size:
                self.data.popitem(last=False)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    s = get_settings()
    setup_logging(s.logging.level, s.logging.json_logs)
    STATE["agent"] = CardAgent(s)
    STATE["cache"] = _IdempotencyCache(s.api.idempotency_cache_size)
    log.info("card agent ready", extra={"extra_json": {"registry": STATE["agent"].state.version,
                                                        "vector_store": STATE["agent"].store.backend}})
    yield


app = FastAPI(title="Card Validation Agent", version=__version__, lifespan=lifespan)
app.include_router(admin_cards.router)
app.include_router(console.router)

if get_settings().console.cors_origins:                 # opt-in: only when the console runs on another origin
    app.add_middleware(CORSMiddleware, allow_origins=get_settings().console.cors_origins,
                       allow_methods=["GET", "POST", "PUT", "DELETE"],
                       allow_headers=["Content-Type", "X-Admin-Token"])


@app.exception_handler(StoreUnavailable)
async def _store_unavailable(_req: Request, exc: StoreUnavailable):
    log.error("vector store unavailable: %s", exc)
    return JSONResponse(status_code=503, content={"detail": "vector store unavailable, retry later"},
                        headers={"Retry-After": "5"})


async def run_validation(image: UploadFile, request_id: str | None, a: CardAgent) -> dict:
    """Shared by /v1/cards/validate and the console API (/console/v1/cards/validate)."""
    s = get_settings()
    if image.content_type not in s.api.allowed_content_types:
        raise HTTPException(415, f"unsupported content type {image.content_type}")
    data = await image.read(s.api.max_image_bytes + 1)
    if len(data) > s.api.max_image_bytes:
        raise HTTPException(413, "image too large")
    cache = STATE["cache"]
    key = (request_id, hashlib.sha256(data).hexdigest()) if request_id else None
    if key and (hit := cache.get(key)) is not None:
        return hit
    result = await run_in_threadpool(a.process, data, request_id)   # CPU/GPU work off the event loop
    if key:
        cache.put(key, result)
    log.info("decision", extra={"extra_json": audit_record(result, s.logging.mask_pii)})
    return result


STATE["run_validation"] = run_validation


@app.post("/v1/cards/validate", response_model=ValidationResult)
async def validate(image: UploadFile = File(...), request_id: str | None = Form(None),
                   a: CardAgent = Depends(agent)):
    return await run_validation(image, request_id, a)


@app.get("/v1/registry")
def registry(a: CardAgent = Depends(agent)):
    a.refresh_if_changed()
    return {**a.state.summary(), "vector_store": a.store.backend}


@app.post("/v1/admin/registry/reload", dependencies=[Depends(require_admin)])
async def reload_registry(a: CardAgent = Depends(agent)):
    try:
        return await run_in_threadpool(a.reload)
    except Exception as e:
        raise HTTPException(422, f"reload rejected, previous version kept: {e}")


@app.get("/health/live")
def live():
    return {"status": "ok"}


@app.get("/health/ready")
def ready():
    a = STATE.get("agent")
    if a is None:
        raise HTTPException(503, "loading")
    if not a.store.healthy():
        raise HTTPException(503, f"vector store ({a.store.backend}) unreachable")
    return {"status": "ready", "registry": a.state.version, "references": a.state.ref_count,
            "vector_store": a.store.backend, "embedder": a.embedder.name, "ocr": a.ocr.name,
            "vlm": bool(a.vlm and a.vlm.available)}


@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
