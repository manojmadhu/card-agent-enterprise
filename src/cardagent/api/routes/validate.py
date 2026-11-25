"""Validation: one card in (one photo, or one photo per side) -> one decision out, plain or streamed.

Upload fields (multipart), up to 2 photos per request:
  image          one photo, side detected automatically (the original API, unchanged)
  front, back    a photo of that side (the label is trusted unless the photo clearly shows the other side)
  images         1-2 photos, sides detected automatically
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from ...agent import MAX_PHOTOS, CardAgent, Photo
from ...observability.telemetry import audit_record
from ...pipeline.trace import STEPS, StepTrace
from ...schemas import ValidationResult
from ...settings import get_settings
from ..deps import STATE, agent, read_image

log = logging.getLogger("cardagent.api")
router = APIRouter(prefix="/v1", tags=["validation"])


async def read_photos(image: UploadFile | None, front: UploadFile | None, back: UploadFile | None,
                      images: list[UploadFile] | None) -> list[Photo]:
    """Upload fields -> photos in processing order: front, back, image, images."""
    parts = [(f, "front", "front") for f in [front] if f] + [(f, "back", "back") for f in [back] if f] \
        + [(f, None, "image") for f in [image] if f] + [(f, None, "images") for f in (images or []) if f]
    if not parts:
        raise HTTPException(422, "send a card photo: field 'image', or 'front' and/or 'back', or 'images'")
    if len(parts) > MAX_PHOTOS:
        raise HTTPException(422, f"at most {MAX_PHOTOS} photos per request (one card, front and back)")
    return [Photo(await read_image(f), declared, upload) for f, declared, upload in parts]


def _cache_key(request_id: str | None, photos: list[Photo]):
    if not request_id:
        return None
    h = hashlib.sha256()
    for ph in photos:
        h.update(f"{ph.upload}:{ph.declared}:".encode())
        h.update(hashlib.sha256(ph.data).digest())
    return request_id, h.hexdigest()


def _record(a: CardAgent, result: dict, photos: list[Photo], source: str, store: bool) -> dict:
    """Store the run (history + review queue) and log the PII-masked decision."""
    s = get_settings()
    if store and s.runs.enabled:
        result["run_id"] = a.runs.add(result, photos, source)
    log.info("decision", extra={"extra_json": {**audit_record(result, s.logging.mask_pii), "run_id": result.get("run_id"),
                                               "source": source}})
    return result


IMAGE_DOC = "card photo, side detected automatically (jpeg/png/webp)"
FRONT_DOC = "photo of the card's front"
BACK_DOC = "photo of the card's back"
IMAGES_DOC = "1-2 card photos, sides detected automatically"


@router.post("/cards/validate", response_model=ValidationResult)
async def validate(image: UploadFile | None = File(None, description=IMAGE_DOC),
                   front: UploadFile | None = File(None, description=FRONT_DOC),
                   back: UploadFile | None = File(None, description=BACK_DOC),
                   images: list[UploadFile] | None = File(None, description=IMAGES_DOC),
                   request_id: str | None = Form(None, description="your id; a retry with the same id AND the same "
                                                                   "photos returns the first answer"),
                   store: bool = Query(True, description="record in the run history (runs.enabled must be on)"),
                   a: CardAgent = Depends(agent)):
    """Validate one card from one photo, or from a photo of each side (at most 2 photos)."""
    photos = await read_photos(image, front, back, images)
    cache = STATE["cache"]
    key = _cache_key(request_id, photos)
    if key and (hit := cache.get(key)) is not None:
        return hit
    result = await run_in_threadpool(lambda: a.process(request_id=request_id, photos=photos))
    result = await run_in_threadpool(_record, a, result, photos, "api", store)
    if key:
        cache.put(key, result)
    return result


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"


@router.post("/cards/validate/stream")
async def validate_stream(image: UploadFile | None = File(None, description=IMAGE_DOC),
                          front: UploadFile | None = File(None, description=FRONT_DOC),
                          back: UploadFile | None = File(None, description=BACK_DOC),
                          images: list[UploadFile] | None = File(None, description=IMAGES_DOC),
                          request_id: str | None = Form(None),
                          mode: Literal["delta", "snapshot"] = "delta", store: bool = True,
                          a: CardAgent = Depends(agent)):
    """Server-Sent Events: the run step by step (StepState objects, see GET /v1/pipeline/steps).

    mode=delta (default): `snapshot` (all steps pending) -> `step` (one StepState each time a step
      starts or finishes) and `image` ({normalisedUrl}, once) -> `result` (full snapshot + result).
    mode=snapshot: every event is a full `{steps, normalisedUrl?, result?}` snapshot.
    Two photos: also `photo_image` ({photo, label, normalisedUrl}) once per photo, and snapshots carry
      `photos` ([{photo, label, side, normalisedUrl}]); per-photo steps are merged (docs/API.md §1).
    Failures after the stream has started arrive as an `error` event. Never served from the cache."""
    photos = await read_photos(image, front, back, images)
    rid = request_id or f"req_{uuid.uuid4().hex[:8]}"
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    trace = StepTrace(a.s, a.state.families, a.state.ref_count,
                      photos=[ph.declared or f"photo {i + 1}" for i, ph in enumerate(photos)])

    def on_stage(event, step, payload=None, photo=None):          # called from the worker thread
        loop.call_soon_threadsafe(queue.put_nowait, ("event", event, step, payload, photo))

    async def worker():
        try:
            result = await run_in_threadpool(lambda: a.process(request_id=rid, trace=on_stage, photos=photos))
            await queue.put(("result", await run_in_threadpool(_record, a, result, photos, "stream", store)))
        except Exception as e:                                    # noqa: BLE001 - reported to the client
            log.exception("stream validation failed")
            await queue.put(("error", e))

    task = asyncio.create_task(worker())

    async def events():
        yield _sse("snapshot", trace.snapshot())
        sent_image = False
        sent_photo = [False] * len(photos)
        while True:
            item = await queue.get()
            if item[0] == "event":
                _, event, step, payload, photo = item
                changed = trace.on_event(event, step, payload, photo=photo)
                if mode == "snapshot":
                    yield _sse("snapshot", trace.snapshot())
                    continue
                if trace.normalised_url and not sent_image:
                    sent_image = True
                    yield _sse("image", {"normalisedUrl": trace.normalised_url})
                if trace.multi:
                    for i, url in enumerate(trace.photo_urls):
                        if url and not sent_photo[i]:
                            sent_photo[i] = True
                            yield _sse("photo_image", {"photo": i, "label": trace.labels[i], "normalisedUrl": url})
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
