"""Card Validation Agent - REST API (one unified /v1 API; OpenAPI at /docs).

Run: uvicorn cardagent.api.app:app --host 0.0.0.0 --port 8080
Groups: validation (+ live stream), cards & reference images, registry, runs & review queue,
statistics, settings, evaluation, health & metrics. Endpoint reference: docs/API.md.
"""
from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from threading import Lock

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .. import __version__
from ..agent import CardAgent
from ..observability.telemetry import setup_logging
from ..settings import get_settings
from ..store.base import StoreUnavailable
from .deps import STATE
from .routes import cards, evaluations, registry, runs, settings, validate

log = logging.getLogger("cardagent.api")
PRUNE_EVERY_S = 3600


class _IdempotencyCache:
    """Same request_id AND same image bytes -> same result (capture systems retry on timeouts).
    Keyed on both, so reusing a request_id with a different photo is processed fresh. Size 0 disables
    it. Per process; put a shared cache (e.g. Redis) in front when running several replicas."""

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


async def _prune_loop(a: CardAgent):
    while True:
        await asyncio.sleep(PRUNE_EVERY_S)
        try:
            await run_in_threadpool(a.runs.prune)
        except Exception:                                            # noqa: BLE001 - retried next hour
            log.exception("run pruning failed")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    s = get_settings()
    setup_logging(s.logging.level, s.logging.json_logs)
    a = CardAgent(s)
    STATE.update(agent=a, cache=_IdempotencyCache(s.api.idempotency_cache_size),
                 executor=ThreadPoolExecutor(max_workers=1, thread_name_prefix="evaluation"))
    pruner = asyncio.create_task(_prune_loop(a)) if s.runs.enabled and s.runs.retention_days > 0 else None
    log.info("card agent ready", extra={"extra_json": {"registry": a.state.version, "families": len(a.state.families),
                                                        "references": a.state.ref_count, "embedder": a.embedder.name}})
    yield
    if pruner:
        pruner.cancel()
        with suppress(asyncio.CancelledError):
            await pruner
    STATE["executor"].shutdown(wait=False, cancel_futures=True)
    STATE.clear()


app = FastAPI(title="Card Validation Agent", version=__version__, lifespan=lifespan,
              description="Self-hosted validation of GCC discount and membership cards. "
                          "Admin endpoints need the X-Admin-Token header.")
for r in (validate, cards, registry, runs, settings, evaluations):
    app.include_router(r.router)

if get_settings().api.cors_origins:                   # opt-in: only when a browser app runs on another origin
    app.add_middleware(CORSMiddleware, allow_origins=get_settings().api.cors_origins,
                       allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
                       allow_headers=["Content-Type", "X-Admin-Token"])


@app.exception_handler(StoreUnavailable)
async def _store_unavailable(_req: Request, exc: StoreUnavailable):
    log.error("qdrant unavailable: %s", exc)
    return JSONResponse(status_code=503, content={"detail": "Qdrant unavailable, retry later"},
                        headers={"Retry-After": "5"})


@app.get("/health/live", tags=["health"])
def live():
    return {"status": "ok"}


@app.get("/health/ready", tags=["health"])
def ready():
    a = STATE.get("agent")
    if a is None:
        raise HTTPException(503, "loading")
    if not a.store.healthy():
        raise HTTPException(503, "Qdrant unreachable")
    return {"status": "ready", "version": __version__, "registry": a.state.version,
            "families": len(a.state.families), "references": a.state.ref_count,
            "embedder": a.embedder.name, "ocr": a.ocr.name, "vlm": bool(a.vlm and a.vlm.available),
            "runs_recorded": a.s.runs.enabled, "settings_overrides": bool(a.overrides)}


@app.get("/metrics", tags=["health"])
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
