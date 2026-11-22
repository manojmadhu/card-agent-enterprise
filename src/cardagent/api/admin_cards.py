"""Card management API (admin token required; Qdrant backend).

  GET    /v1/admin/cards                                              list card families
  GET    /v1/admin/cards/{family_id}                                  definition + reference images
  PUT    /v1/admin/cards/{family_id}                                  create / replace a definition (JSON or YAML body)
  DELETE /v1/admin/cards/{family_id}                                  remove the family and all its references
  POST   /v1/admin/cards/{family_id}/variants/{variant_id}/references upload reference images (multipart "images")
  DELETE /v1/admin/cards/{family_id}/references/{ref_id}              remove one reference image
  POST   /v1/admin/cards/sync                                         incremental sync from the file registry

Every change touches only the affected card's points in Qdrant (no re-index), takes effect on this
replica immediately and on every other replica within vector_store.qdrant.refresh_seconds.

Ownership: families from registry/ (source=registry) are managed by git + sync. Changing them here
returns 409 unless ?force=true; a forced PUT transfers ownership to the API, so later syncs leave
the family alone. A forced DELETE of a registry family lasts until the next sync unless its YAML
is removed as well.
"""
from __future__ import annotations

import json
import logging

import yaml
from fastapi import APIRouter, Depends, File, HTTPException, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError

from ..agent import CardAgent
from ..registry.models import CardFamily
from ..settings import get_settings
from .deps import agent, require_admin

log = logging.getLogger("cardagent.admin")
router = APIRouter(prefix="/v1/admin/cards", tags=["card management"], dependencies=[Depends(require_admin)])

MAX_IMAGES_PER_REQUEST = 20


def _store(a: CardAgent):
    if not a.store.supports_admin:
        raise HTTPException(501, f"card management needs vector_store.backend=qdrant "
                                 f"(current: {a.store.backend}); edit registry/ and run tools/build_index.py instead")
    return a.store


def _audit(action: str, **kw):
    log.info("admin %s", action, extra={"extra_json": {"event": "card_admin", "action": action, **kw}})


def _owned_check(doc: dict | None, force: bool, action: str):
    if doc and doc.get("source") == "registry" and not force:
        raise HTTPException(409, f"family {doc['family_id']} is managed by the file registry (registry/cards); "
                                 f"{action} it there and run a sync, or pass ?force=true")


def _family_view(store, doc: dict, refs: list[dict]) -> dict:
    fam = CardFamily.model_validate(doc["definition"])
    per_variant = {v.id: [] for v in fam.variants}
    for r in refs:
        per_variant.setdefault(r["variant"], []).append(
            {k: r.get(k) for k in ("ref_id", "filename", "sha256", "source", "added_at")})
    return {"family": fam.id, "source": doc.get("source"), "created_at": doc.get("created_at"),
            "updated_at": doc.get("updated_at"), "definition": doc["definition"],
            "variants": [{"id": v.id, "name": v.name, "reference_images": per_variant.get(v.id, [])}
                         for v in fam.variants],
            "variants_without_references": [v.id for v in fam.variants if not per_variant.get(v.id)]}


async def _parse_definition(request: Request) -> CardFamily:
    raw = await request.body()
    ctype = request.headers.get("content-type", "")
    try:
        data = yaml.safe_load(raw) if ("yaml" in ctype or "yml" in ctype) else json.loads(raw)
    except (json.JSONDecodeError, yaml.YAMLError) as e:
        raise HTTPException(400, f"body is not valid {'YAML' if 'yaml' in ctype else 'JSON'}: {e}")
    try:
        return CardFamily.model_validate(data)
    except ValidationError as e:
        raise HTTPException(422, json.loads(e.json(include_url=False)))


# ------------------------------------------------------------------------------ families
@router.get("")
def list_cards(a: CardAgent = Depends(agent)):
    store = _store(a)
    a.refresh_if_changed(force=True)
    st = a.state
    docs = {r.payload["family_id"]: r.payload
            for r in store._scroll(store.fams_col, store._eq("kind", "family"), ["family_id", "source", "updated_at"])}
    items = [{"family": f.id, "name": f.name, "country": f.country, "issuer": f.issuer,
              "source": docs.get(f.id, {}).get("source"), "updated_at": docs.get(f.id, {}).get("updated_at"),
              "variants": [{"id": v.id, "reference_images": st.ref_counts.get((f.id, v.id), 0)} for v in f.variants]}
             for f in sorted(st.families.values(), key=lambda x: x.id)]
    return {"registry_version": st.version, "families": len(items), "reference_images": st.ref_count, "items": items}


@router.get("/{family_id}")
def get_card(family_id: str, a: CardAgent = Depends(agent)):
    store = _store(a)
    doc = store.get_family(family_id)
    if doc is None:
        raise HTTPException(404, f"family {family_id} not found")
    return _family_view(store, doc, store.list_refs(family_id))


@router.put("/{family_id}")
async def put_card(family_id: str, request: Request, response: Response, force: bool = False,
                   a: CardAgent = Depends(agent)):
    """Create or replace a card family definition. Body: the same schema as registry/cards/*.yaml,
    as JSON (application/json) or YAML (application/yaml). Removing a variant deletes its reference
    images; changing field ROIs re-extracts this family's keypoints. Other cards are not touched."""
    store = _store(a)
    fam = await _parse_definition(request)
    if fam.id != family_id:
        raise HTTPException(422, f"definition id '{fam.id}' does not match URL '{family_id}'")
    _owned_check(store.get_family(family_id), force, "change")
    out = await run_in_threadpool(store.put_family, fam, "api")
    a.refresh_if_changed(force=True)
    response.status_code = 201 if out["created"] else 200
    _audit("put_family", family=family_id, created=out["created"], forced=force)
    doc = store.get_family(family_id)
    return {**out, "registry_version": a.state.version, **_family_view(store, doc, store.list_refs(family_id))}


@router.delete("/{family_id}")
async def delete_card(family_id: str, force: bool = False, a: CardAgent = Depends(agent)):
    store = _store(a)
    doc = store.get_family(family_id)
    if doc is None:
        raise HTTPException(404, f"family {family_id} not found")
    _owned_check(doc, force, "delete")
    out = await run_in_threadpool(store.delete_family, family_id)
    a.refresh_if_changed(force=True)
    _audit("delete_family", family=family_id, references=out["references_deleted"], forced=force)
    note = ("registry-managed: it will be re-created by the next sync unless "
            "its YAML and reference images are removed from registry/") if doc.get("source") == "registry" else None
    return {**out, "registry_version": a.state.version, **({"note": note} if note else {})}


# ------------------------------------------------------------------------------ reference images
@router.post("/{family_id}/variants/{variant_id}/references", status_code=201)
async def add_references(family_id: str, variant_id: str, response: Response,
                         images: list[UploadFile] = File(..., description="1..20 reference images of this variant"),
                         detect_card: bool = True, a: CardAgent = Depends(agent)):
    """Upload reference images (a flat scan, or a photo: the card is detected and straightened when
    detect_card=true). Each image is embedded and upserted on its own; re-uploading the same image
    is a no-op (status "exists")."""
    store = _store(a)
    s = get_settings()
    doc = store.get_family(family_id)
    if doc is None:
        raise HTTPException(404, f"family {family_id} not found")
    fam = CardFamily.model_validate(doc["definition"])
    if fam.variant(variant_id) is None:
        raise HTTPException(404, f"variant {variant_id} not in family {family_id} "
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

    results = []
    for name, data in payloads:
        try:
            r = await run_in_threadpool(store.add_reference, fam, variant_id, data, name, "api",
                                        None, detect_card, False)
            results.append({"filename": name, **r})
        except ValueError as e:
            results.append({"filename": name, "status": "rejected", "error": str(e)})
    created = [r for r in results if r["status"] == "created"]
    if created:
        store.bump_revision()
        a.refresh_if_changed(force=True)
    if not created:
        response.status_code = 422 if all(r["status"] == "rejected" for r in results) else 200
    _audit("add_references", family=family_id, variant=variant_id, created=len(created), total=len(results))
    return {"family": family_id, "variant": variant_id, "created": len(created), "results": results,
            "reference_images_in_variant": a.state.ref_counts.get((family_id, variant_id), 0),
            "registry_version": a.state.version}


@router.delete("/{family_id}/references/{ref_id}")
async def delete_reference(family_id: str, ref_id: str, force: bool = False, a: CardAgent = Depends(agent)):
    store = _store(a)
    refs = [r for r in store.list_refs(family_id) if r["ref_id"] == ref_id]
    if not refs:
        raise HTTPException(404, f"reference {ref_id} not found in family {family_id}")
    if refs[0].get("source") == "registry" and not force:
        raise HTTPException(409, "reference image comes from registry/refs; remove the file there and run a sync, "
                                 "or pass ?force=true (it will return on the next sync)")
    out = await run_in_threadpool(store.delete_reference, family_id, ref_id)
    a.refresh_if_changed(force=True)
    _audit("delete_reference", family=family_id, ref_id=ref_id, forced=force)
    return {**out, "reference_images_in_variant": a.state.ref_counts.get((family_id, out["variant"]), 0),
            "registry_version": a.state.version}


# ------------------------------------------------------------------------------ registry sync
@router.post("/sync")
async def sync(prune: bool = True, a: CardAgent = Depends(agent)):
    """Incremental sync from the file registry (same as tools/build_index.py): only new, changed or
    removed cards are processed."""
    store = _store(a)
    from ..store.sync import sync_registry
    try:
        out = await run_in_threadpool(sync_registry, store, a.s.path(a.s.registry_dir), prune)
    except (ValueError, ValidationError) as e:
        raise HTTPException(422, f"registry invalid, nothing changed: {e}")
    a.refresh_if_changed(force=True)
    _audit("sync", **{k: v for k, v in out.items() if isinstance(v, (int, bool))})
    return out
