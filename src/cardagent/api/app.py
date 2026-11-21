"""REST API.  Run: uvicorn cardagent.api.app:app --host 0.0.0.0 --port 8080 --workers N"""
from __future__ import annotations

import hashlib
import hmac
import logging
from collections import OrderedDict
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .. import __version__
from ..agent import CardAgent
from ..observability.telemetry import audit_record, setup_logging
from ..schemas import ValidationResult
from ..settings import get_settings

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


STATE: dict = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    s = get_settings()
    setup_logging(s.logging.level, s.logging.json_logs)
    STATE["agent"] = CardAgent(s)
    STATE["cache"] = _IdempotencyCache(s.api.idempotency_cache_size)
    log.info("card agent ready", extra={"extra_json": {"registry": STATE["agent"].state.snapshot.version}})
    yield


app = FastAPI(title="Card Validation Agent", version=__version__, lifespan=lifespan)


def agent() -> CardAgent:
    a = STATE.get("agent")
    if a is None:
        raise HTTPException(503, "not ready")
    return a


def require_admin(x_admin_token: str = Header(default="")):
    token = get_settings().api.admin_token
    if not token:
        raise HTTPException(403, "admin endpoints disabled (set CARDAGENT_API__ADMIN_TOKEN)")
    if not hmac.compare_digest(x_admin_token, token):
        raise HTTPException(401, "invalid admin token")


@app.post("/v1/cards/validate", response_model=ValidationResult)
async def validate(image: UploadFile = File(...), request_id: str | None = Form(None),
                   a: CardAgent = Depends(agent)):
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


@app.get("/v1/registry")
def registry(a: CardAgent = Depends(agent)):
    return a.state.snapshot.summary()


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
    return {"status": "ready", "registry": a.state.snapshot.version, "references": len(a.state.index.entries),
            "embedder": a.embedder.name, "ocr": a.ocr.name, "vlm": bool(a.vlm and a.vlm.available)}


@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
