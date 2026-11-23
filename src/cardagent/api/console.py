"""Card Agent Console API (card-agent-web), mounted at /console/v1.

Point the console's "API base URL" (Settings -> Engine -> card-agent API) at
    http://<host>:8080/console/v1

Routes the console calls today (exact paths and formats of its src/engine/api.ts):
  POST   /cards/validate                       multipart image + request_id -> MVP result envelope
  POST   /cards/validate/stream                the same run as Server-Sent Events, step by step (StepState)
  GET    /pipeline/steps                       the 11 step ids / titles / stages
  GET    /registry                             {families: [...console YAML as JSON...], version, ...}
  POST   /registry/reload                      reload the card registry
  PUT    /registry/families/{id}               create / replace a family from console YAML

Additional routes for the console's family editor and reference-card pages:
  GET    /registry/families/{id}               the family as console YAML
  DELETE /registry/families/{id}               delete the family and its reference images
  GET    /registry/families/{id}/references    reference images per variant
  POST   /registry/families/{id}/variants/{variant}/references   enroll / upload images (multipart "images")
  DELETE /registry/families/{id}/references/{ref_id}            remove one reference image

Writes need X-Admin-Token unless console.require_admin_token=false. Storage follows
vector_store.backend: with qdrant, writes go through the card store (incremental, shared by
replicas, same registry-ownership rules as /v1/admin/cards); with faiss, they edit registry/
files and reload. The /v1 API is unchanged.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import yaml
from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from ..agent import CardAgent
from ..console.schema import (ConsoleFamily, console_id, resolve_enterprise_id, to_console, to_console_result,
                              to_enterprise)
from ..console.stream import STEPS, ConsoleTrace
from ..observability.telemetry import audit_record
from ..registry.models import CardFamily
from ..registry.store import IMAGE_EXT
from ..settings import get_settings
from ..vision import normalise as normalise_mod
from .deps import STATE, agent, require_admin

log = logging.getLogger("cardagent.console")
MAX_IMAGES_PER_REQUEST = 20


def _enabled():
    if not get_settings().console.enabled:
        raise HTTPException(404, "console API disabled (console.enabled=false)")


def _write_auth(x_admin_token: str = Header(default="")):
    if get_settings().console.require_admin_token:
        require_admin(x_admin_token)


router = APIRouter(prefix="/console/v1", tags=["console (card-agent-web)"], dependencies=[Depends(_enabled)])


def _audit(action: str, **kw):
    log.info("console %s", action, extra={"extra_json": {"event": "console_admin", "action": action, **kw}})


def _ids(a: CardAgent) -> list[str]:
    return list(a.state.families)


def _resolve(a: CardAgent, cid: str) -> str:
    return resolve_enterprise_id(cid, _ids(a), get_settings().console.default_country)


def _ops(a: CardAgent):
    return _QdrantOps(a) if a.store.supports_admin else _FileOps(a)


# ================================================================================== validate
@router.post("/cards/validate")
async def validate(image: UploadFile = File(...), request_id: str | None = Form(None),
                   a: CardAgent = Depends(agent)):
    result = await STATE["run_validation"](image, request_id, a)
    return to_console_result(result, a.state.families)


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"


@router.post("/cards/validate/stream")
async def validate_stream(image: UploadFile = File(...), request_id: str | None = Form(None),
                          mode: Literal["delta", "snapshot"] = "delta", a: CardAgent = Depends(agent)):
    """Server-Sent Events: the run step by step, in the console browser engine's StepState format.

    mode=delta (default): `snapshot` (all steps pending) -> `step` (one StepState, each time a step
      starts or finishes) and `image` ({normalisedUrl}, once) -> `result` (full snapshot + result).
    mode=snapshot: every event is a full `{steps, normalisedUrl?, result?}` snapshot, exactly what
      the browser engine's generator yields (simplest to consume, more bytes).
    Errors after the stream has started arrive as an `error` event. Not cached: every call runs."""
    data = await STATE["read_upload"](image)
    rid = request_id or f"req_{uuid.uuid4().hex[:8]}"
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    families, s = a.state.families, get_settings()
    trace = ConsoleTrace(s, families, a.state.ref_count)

    def on_stage(event, step, payload=None):                      # called from the worker thread
        loop.call_soon_threadsafe(queue.put_nowait, ("event", event, step, payload))

    async def worker():
        try:
            await queue.put(("result", await run_in_threadpool(a.process, data, rid, None, on_stage)))
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
                result = item[1]
                console_result = to_console_result(result, families)
                trace.finish(result, console_result)
                log.info("decision", extra={"extra_json": {**audit_record(result, s.logging.mask_pii), "stream": True}})
                yield _sse("result", trace.snapshot(console_result))
                break
            else:
                yield _sse("error", {"detail": f"validation failed: {type(item[1]).__name__}"})
                break
        await task

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/pipeline/steps")
def pipeline_steps():
    """The 11 step ids/titles/stages the trace uses (same as the console's browser engine)."""
    return {"steps": STEPS}


# ================================================================================== registry
@router.get("/registry")
def registry(a: CardAgent = Depends(agent)):
    a.refresh_if_changed()
    st, ids = a.state, _ids(a)
    fams = []
    for f in sorted(st.families.values(), key=lambda x: x.id):
        fams.append({**to_console(f, console_id(f.id, ids)), "enterprise_id": f.id, "country": f.country,
                     "reference_images": {v.id: st.ref_counts.get((f.id, v.id), 0) for v in f.variants}})
    return {"version": st.version, "registry_version": st.version, "vector_store": a.store.backend,
            "reference_images": st.ref_count, "families": fams}


@router.post("/registry/reload", dependencies=[Depends(_write_auth)])
async def reload(a: CardAgent = Depends(agent)):
    try:
        summary = await run_in_threadpool(a.reload)
    except Exception as e:
        raise HTTPException(422, f"reload rejected, previous version kept: {e}")
    _audit("reload", version=summary.get("version"))
    return {"version": summary.get("version"), "families": summary.get("families"),
            "reference_images": summary.get("reference_images"), "vector_store": a.store.backend}


# ================================================================================== families
@router.get("/registry/families/{family}")
def get_family(family: str, a: CardAgent = Depends(agent)):
    a.refresh_if_changed()
    fam = a.state.families.get(_resolve(a, family))
    if fam is None:
        raise HTTPException(404, f"family {family} not found")
    doc = to_console(fam, console_id(fam.id, _ids(a)))
    header = f"# {fam.id} ({fam.country}) — served by card-agent-enterprise registry {a.state.version}\n"
    return Response(header + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True),
                    media_type="application/yaml")


async def _parse_console_family(request: Request) -> ConsoleFamily:
    raw = await request.body()
    ctype = request.headers.get("content-type", "")
    try:
        data = json.loads(raw) if "json" in ctype else yaml.safe_load(raw)
    except (json.JSONDecodeError, yaml.YAMLError) as e:
        raise HTTPException(400, f"body is not valid {'JSON' if 'json' in ctype else 'YAML'}: {e}")
    if not isinstance(data, dict):
        raise HTTPException(400, "body must be a card family mapping")
    try:
        return ConsoleFamily.model_validate(data)
    except ValidationError as e:
        raise HTTPException(422, json.loads(e.json(include_url=False)))


@router.put("/registry/families/{family}", dependencies=[Depends(_write_auth)])
async def put_family(family: str, request: Request, response: Response, force: bool = False,
                     a: CardAgent = Depends(agent)):
    """Body: console YAML (application/yaml) or the same as JSON. The `family` key must match the URL."""
    cf = await _parse_console_family(request)
    if cf.family != family:
        raise HTTPException(422, f"family '{cf.family}' in the body does not match URL '{family}'")
    eid = _resolve(a, family)
    existing = a.state.families.get(eid)
    try:
        fam = to_enterprise(cf, eid, existing)
    except ValidationError as e:
        raise HTTPException(422, json.loads(e.json(include_url=False)))
    out = await run_in_threadpool(_ops(a).put, fam, force)
    response.status_code = 201 if out["created"] else 200
    _audit("put_family", family=eid, created=out["created"], forced=force, backend=a.store.backend)
    return {"family": console_id(eid, _ids(a)), "enterprise_id": eid, **out,
            "registry_version": a.state.version, "vector_store": a.store.backend,
            "note": "weights are console-only; the service uses its global scoring.family_weights"}


@router.delete("/registry/families/{family}", dependencies=[Depends(_write_auth)])
async def delete_family(family: str, force: bool = False, a: CardAgent = Depends(agent)):
    eid = _resolve(a, family)
    if eid not in a.state.families:
        raise HTTPException(404, f"family {family} not found")
    out = await run_in_threadpool(_ops(a).delete, eid, force)
    _audit("delete_family", family=eid, forced=force, backend=a.store.backend)
    return {"family": family, "enterprise_id": eid, **out, "registry_version": a.state.version}


# ================================================================================== references
def _family_or_404(a: CardAgent, family: str) -> CardFamily:
    fam = a.state.families.get(_resolve(a, family))
    if fam is None:
        raise HTTPException(404, f"family {family} not found")
    return fam


@router.get("/registry/families/{family}/references")
def list_references(family: str, a: CardAgent = Depends(agent)):
    a.refresh_if_changed()
    fam = _family_or_404(a, family)
    refs = _ops(a).list_refs(fam.id)
    return {"family": family, "enterprise_id": fam.id,
            "variants": [{"id": v.id, "name": v.name, "references": [r for r in refs if r["variant"] == v.id]}
                         for v in fam.variants]}


@router.post("/registry/families/{family}/variants/{variant}/references", status_code=201,
             dependencies=[Depends(_write_auth)])
async def add_references(family: str, variant: str, response: Response,
                         images: list[UploadFile] = File(..., description="1..20 images of this variant"),
                         detect_card: bool = True, a: CardAgent = Depends(agent)):
    """Enroll reference images: flat scans, or photos (the card is found and straightened when
    detect_card=true). Re-sending the same image is a no-op (status "exists")."""
    s = get_settings()
    fam = _family_or_404(a, family)
    if fam.variant(variant) is None:
        raise HTTPException(404, f"variant {variant} not in family {family} "
                                 f"(variants: {', '.join(v.id for v in fam.variants)})")
    if not 1 <= len(images) <= MAX_IMAGES_PER_REQUEST:
        raise HTTPException(422, f"send 1..{MAX_IMAGES_PER_REQUEST} images per request")
    payloads = []
    for f in images:
        if f.content_type not in s.api.allowed_content_types:
            raise HTTPException(415, f"{f.filename}: unsupported content type {f.content_type}")
        data = await f.read(s.api.max_image_bytes + 1)
        if len(data) > s.api.max_image_bytes:
            raise HTTPException(413, f"{f.filename}: image too large")
        payloads.append((f.filename or "upload", data))
    results = await run_in_threadpool(_ops(a).add_refs, fam, variant, payloads, detect_card)
    created = sum(r["status"] == "created" for r in results)
    if not created:
        response.status_code = 422 if all(r["status"] == "rejected" for r in results) else 200
    _audit("add_references", family=fam.id, variant=variant, created=created, total=len(results))
    return {"family": family, "enterprise_id": fam.id, "variant": variant, "created": created, "results": results,
            "reference_images_in_variant": a.state.ref_counts.get((fam.id, variant), 0),
            "registry_version": a.state.version}


@router.delete("/registry/families/{family}/references/{ref_id}", dependencies=[Depends(_write_auth)])
async def delete_reference(family: str, ref_id: str, force: bool = False, a: CardAgent = Depends(agent)):
    fam = _family_or_404(a, family)
    out = await run_in_threadpool(_ops(a).delete_ref, fam.id, ref_id, force)
    _audit("delete_reference", family=fam.id, ref_id=ref_id, forced=force)
    return {"family": family, "enterprise_id": fam.id, **out,
            "reference_images_in_variant": a.state.ref_counts.get((fam.id, out["variant"]), 0),
            "registry_version": a.state.version}


# ================================================================================== storage backends
class _QdrantOps:
    """Qdrant backend: incremental writes through the card store (same rules as /v1/admin/cards)."""

    def __init__(self, a: CardAgent):
        self.a, self.store = a, a.store

    def _owned(self, doc: dict | None, force: bool, action: str):
        if doc and doc.get("source") == "registry" and not force:
            raise HTTPException(409, f"family {doc['family_id']} is managed by the file registry "
                                     f"(registry/cards in git); {action} it there and run a sync, or retry "
                                     f"with ?force=true to hand it over to the API")

    def put(self, fam: CardFamily, force: bool) -> dict:
        self._owned(self.store.get_family(fam.id), force, "change")
        out = self.store.put_family(fam, "api")
        self.a.refresh_if_changed(force=True)
        return {"created": out["created"], "changed": out["changed"], "references_deleted": out["references_deleted"],
                "keypoints_refreshed": out["keypoints_refreshed"], "source": "api"}

    def delete(self, eid: str, force: bool) -> dict:
        doc = self.store.get_family(eid)
        self._owned(doc, force, "delete")
        out = self.store.delete_family(eid) or {}
        self.a.refresh_if_changed(force=True)
        return {"references_deleted": out.get("references_deleted", 0),
                **({"note": "registry-managed: the next sync re-creates it unless its YAML and images are "
                            "removed from registry/"} if (doc or {}).get("source") == "registry" else {})}

    def list_refs(self, eid: str) -> list[dict]:
        return [{"ref_id": r["ref_id"], "variant": r["variant"], "filename": r.get("filename"),
                 "sha256": r.get("sha256"), "source": r.get("source"), "added_at": r.get("added_at")}
                for r in self.store.list_refs(eid)]

    def add_refs(self, fam: CardFamily, variant: str, payloads, detect_card: bool) -> list[dict]:
        results = []
        for name, data in payloads:
            try:
                r = self.store.add_reference(fam, variant, data, name, "api", None, detect_card, False)
                results.append({"filename": name, **r})
            except ValueError as e:
                results.append({"filename": name, "status": "rejected", "error": str(e)})
        if any(r["status"] == "created" for r in results):
            self.store.bump_revision()
            self.a.refresh_if_changed(force=True)
        return results

    def delete_ref(self, eid: str, ref_id: str, force: bool) -> dict:
        refs = [r for r in self.store.list_refs(eid) if r["ref_id"] == ref_id]
        if not refs:
            raise HTTPException(404, f"reference {ref_id} not found in family {eid}")
        if refs[0].get("source") == "registry" and not force:
            raise HTTPException(409, "reference image comes from registry/refs; remove the file there and run "
                                     "a sync, or pass ?force=true (it returns on the next sync)")
        out = self.store.delete_reference(eid, ref_id)
        self.a.refresh_if_changed(force=True)
        return {"ref_id": ref_id, "variant": out["variant"]}


class _FileOps:
    """FAISS backend: the file registry is the source of truth. Writes edit registry/ atomically
    (replaced or deleted files go to registry/.deleted/ for recovery), then reload; a definition
    the service cannot load is rolled back."""

    def __init__(self, a: CardAgent):
        self.a = a
        self.root = a.s.path(a.s.registry_dir)

    # ---- helpers
    def _yaml_path(self, eid: str) -> Path:
        for p in sorted((self.root / "cards").rglob("*.yaml")):
            try:
                if (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("id") == eid:
                    return p
            except yaml.YAMLError:
                continue
        cc, short = eid.split(".", 1)
        return self.root / "cards" / cc / f"{short}.yaml"

    def _trash(self, p: Path) -> Path:
        dst = self.root / ".deleted" / time.strftime("%Y%m%d-%H%M%S") / p.relative_to(self.root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(dst))
        return dst

    def _reload_or_rollback(self, undo) -> None:
        try:
            self.a.reload()
        except Exception as e:
            undo()
            self.a.reload()
            raise HTTPException(422, f"registry rejected the change, previous version restored: {e}")

    def _ref_id(self, p: Path) -> str:
        return hashlib.sha256(p.relative_to(self.root).as_posix().encode()).hexdigest()[:16]

    # ---- families
    def put(self, fam: CardFamily, force: bool) -> dict:
        path = self._yaml_path(fam.id)
        created = not path.exists()
        backup = None if created else self._trash_copy(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".yaml.tmp")
        body = "# managed from the Card Agent Console\n" + yaml.safe_dump(
            fam.model_dump(mode="json", exclude_none=True), sort_keys=False, allow_unicode=True)
        tmp.write_text(body, encoding="utf-8")
        os.replace(tmp, path)

        def undo():
            if backup:
                shutil.copy2(backup, path)
            else:
                path.unlink(missing_ok=True)

        self._reload_or_rollback(undo)
        return {"created": created, "changed": True, "source": "registry", "file": path.relative_to(self.root).as_posix()}

    def _trash_copy(self, p: Path) -> Path:
        dst = self.root / ".deleted" / time.strftime("%Y%m%d-%H%M%S") / p.relative_to(self.root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        return dst

    def delete(self, eid: str, force: bool) -> dict:
        path = self._yaml_path(eid)
        n_refs = len(self.list_refs(eid))
        moved = []
        if path.exists():
            moved.append((self._trash(path), path))
        refs_dir = self.root / "refs" / eid
        if refs_dir.exists():
            moved.append((self._trash(refs_dir), refs_dir))

        def undo():
            for dst, src in moved:
                shutil.move(str(dst), str(src))

        self._reload_or_rollback(undo)
        return {"references_deleted": n_refs, "recoverable_from": "registry/.deleted/"}

    # ---- references
    def list_refs(self, eid: str) -> list[dict]:
        out = []
        base = self.root / "refs" / eid
        for p in sorted(base.glob("*/*")) if base.exists() else []:
            if p.suffix.lower() in IMAGE_EXT:
                out.append({"ref_id": self._ref_id(p), "variant": p.parent.name, "filename": p.name,
                            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "source": "registry",
                            "added_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(p.stat().st_mtime))})
        return out

    def add_refs(self, fam: CardFamily, variant: str, payloads, detect_card: bool) -> list[dict]:
        vdir = self.root / "refs" / fam.id / variant
        vdir.mkdir(parents=True, exist_ok=True)
        results, written = [], []
        for name, data in payloads:
            sha = hashlib.sha256(data).hexdigest()
            dst = vdir / f"api_{sha[:16]}.png"
            if dst.exists():
                results.append({"filename": name, "ref_id": self._ref_id(dst), "status": "exists", "sha256": sha})
                continue
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                results.append({"filename": name, "status": "rejected", "error": f"{name}: not a decodable image"})
                continue
            detected = None
            if detect_card:
                img, info = normalise_mod.normalise(img)
                detected = info["card_detected"]
            cv2.imwrite(str(dst), img)                        # stored as the straightened card
            written.append(dst)
            results.append({"filename": name, "ref_id": self._ref_id(dst), "status": "created", "sha256": sha,
                            "card_detected": detected})
        if written:
            self._reload_or_rollback(lambda: [p.unlink(missing_ok=True) for p in written])
        return results

    def delete_ref(self, eid: str, ref_id: str, force: bool) -> dict:
        base = self.root / "refs" / eid
        hit = next((p for p in base.glob("*/*") if self._ref_id(p) == ref_id), None) if base.exists() else None
        if hit is None:
            raise HTTPException(404, f"reference {ref_id} not found in family {eid}")
        variant, dst = hit.parent.name, self._trash(hit)
        self._reload_or_rollback(lambda: shutil.move(str(dst), str(hit)))
        return {"ref_id": ref_id, "variant": variant}
