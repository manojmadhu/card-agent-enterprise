"""Validation: one photo in -> one decision out, plain or streamed step by step."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from ...agent import CardAgent
from ...observability.telemetry import audit_record
from ...pipeline.trace import STEPS, StepTrace
from ...schemas import ValidationResult
from ...settings import get_settings
from ..deps import STATE, agent, read_image

log = logging.getLogger("cardagent.api")
router = APIRouter(prefix="/v1", tags=["validation"])


def _record(a: CardAgent, result: dict, image: bytes, source: str, store: bool) -> dict:
    """Store the run (history + review queue) and log the PII-masked decision."""
    s = get_settings()
    if store and s.runs.enabled:
        result["run_id"] = a.runs.add(result, image, source)
    log.info("decision", extra={"extra_json": {**audit_record(result, s.logging.mask_pii), "run_id": result.get("run_id"),
                                               "source": source}})
    return result


@router.post("/cards/validate", response_model=ValidationResult)
async def validate(image: UploadFile = File(..., description="card photo (jpeg/png/webp)"),
                   request_id: str | None = Form(None, description="your id; a retry with the same id AND image "
                                                                   "returns the first answer"),
                   store: bool = Query(True, description="record in the run history (runs.enabled must be on)"),
                   a: CardAgent = Depends(agent)):
    data = await read_image(image)
    cache = STATE["cache"]
    key = (request_id, hashlib.sha256(data).hexdigest()) if request_id else None
    if key and (hit := cache.get(key)) is not None:
        return hit
    result = await run_in_threadpool(a.process, data, request_id)
    result = await run_in_threadpool(_record, a, result, data, "api", store)
    if key:
        cache.put(key, result)
    return result


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"


@router.post("/cards/validate/stream")
async def validate_stream(image: UploadFile = File(...), request_id: str | None = Form(None),
                          mode: Literal["delta", "snapshot"] = "delta", store: bool = True,
                          a: CardAgent = Depends(agent)):
    """Server-Sent Events: the run step by step (StepState objects, see GET /v1/pipeline/steps).

    mode=delta (default): `snapshot` (all steps pending) -> `step` (one StepState each time a step
      starts or finishes) and `image` ({normalisedUrl}, once) -> `result` (full snapshot + result).
    mode=snapshot: every event is a full `{steps, normalisedUrl?, result?}` snapshot.
    Failures after the stream has started arrive as an `error` event. Never served from the cache."""
    data = await read_image(image)
    rid = request_id or f"req_{uuid.uuid4().hex[:8]}"
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    trace = StepTrace(a.s, a.state.families, a.state.ref_count)

    def on_stage(event, step, payload=None):                      # called from the worker thread
        loop.call_soon_threadsafe(queue.put_nowait, ("event", event, step, payload))

    async def worker():
        try:
            result = await run_in_threadpool(a.process, data, rid, None, on_stage)
            await queue.put(("result", await run_in_threadpool(_record, a, result, data, "stream", store)))
        except Exception as e:                                    # noqa: BLE001 - reported to the client
            log.exception("stream validation failed")
            await queue.put(("error", e))

    task = asyncio.create_task(worker())

    async def events():
        yield _sse("snapshot", trace.snapshot())
        sent_image = False
        while True:
            item = await queue.get()
            if item[0] == "event":
                changed = trace.on_event(*item[1:])
                if mode == "snapshot":
                    yield _sse("snapshot", trace.snapshot())
                    continue
                if trace.normalised_url and not sent_image:
                    sent_image = True
                    yield _sse("image", {"normalisedUrl": trace.normalised_url})
                for st in changed:
                    yield _sse("step", st)
            elif item[0] == "result":
                trace.finish(item[1])
                yield _sse("result", trace.snapshot(item[1]))
                break
            else:
                yield _sse("error", {"detail": f"validation failed: {type(item[1]).__name__}"})
                break
        await task

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/pipeline/steps")
def pipeline_steps():
    """The 11 steps of the stream (id, title, stage number)."""
    return {"steps": STEPS}
