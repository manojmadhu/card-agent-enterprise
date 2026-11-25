"""Card families and their reference images.

Every change touches only the affected card's points in Qdrant (no re-index) and takes effect on
this replica immediately and on every other replica within qdrant.refresh_seconds.

Card sides: a definition lists `sides` (front, back) and every field names the `side` it is printed
on. Reference images are uploaded per variant AND side (?side=back); the front is the default, so
one-sided cards work exactly as before.

Ownership: families imported from registry/ (source=registry) are managed by git + POST
/v1/registry/sync. Changing them here returns 409 unless ?force=true; a forced change hands the
family over to the API, so later syncs leave it alone.
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
import yaml

from ...agent import CardAgent
from ...registry.models import CardFamily
from ..deps import agent, audit, image_media_type, parse_body, read_image, require_admin

router = APIRouter(prefix="/v1/cards", tags=["cards"])
MAX_IMAGES_PER_REQUEST = 20


def _owned_check(doc: dict | None, force: bool, action: str):
    if doc and doc.get("source") == "registry" and not force:
        raise HTTPException(409, f"family {doc['family_id']} is managed by the file registry (registry/cards); "
                                 f"{action} it there and run POST /v1/registry/sync, or pass ?force=true")


def _doc_or_404(a: CardAgent, family_id: str) -> dict:
    doc = a.store.get_family(family_id)
    if doc is None:
        raise HTTPException(404, f"family {family_id} not found")
    return doc


def _variant_rows(a: CardAgent, fam: CardFamily) -> list[dict]:
    counts, sides = a.state.ref_counts, a.state.side_counts
    return [{"id": v.id, "name": v.name, "reference_images": counts.get((fam.id, v.id), 0),
             "reference_images_by_side": {s: sides.get((fam.id, v.id, s), 0) for s in fam.side_ids}}
            for v in fam.variants]


def _view(a: CardAgent, doc: dict) -> dict:
    fam = CardFamily.model_validate(doc["definition"])
    counts, sides = a.state.ref_counts, a.state.side_counts
    return {"family": fam.id, "name": fam.name, "country": fam.country, "issuer": fam.issuer,
            "source": doc.get("source"), "created_at": doc.get("created_at"), "updated_at": doc.get("updated_at"),
            "definition": doc["definition"],
            "sides": [{"id": s, "required": s in fam.required_sides(), "fields": [f.name for f in fam.fields_on(s)]}
                      for s in fam.side_ids],
            "variants": _variant_rows(a, fam),
            "variants_without_references": [v.id for v in fam.variants if not counts.get((fam.id, v.id))],
            "missing_references": [{"variant": v.id, "side": s} for v in fam.variants for s in fam.side_ids
                                   if not sides.get((fam.id, v.id, s))]}


# ------------------------------------------------------------------------------ families
@router.get("")
def list_cards(a: CardAgent = Depends(agent)):
    a.refresh_if_changed(force=True)
    st = a.state
    docs = {r.payload["family_id"]: r.payload for r in
            a.store._scroll(a.store.fams_col, a.store._eq("kind", "family"), ["family_id", "source", "updated_at"])}
    items = [{"family": f.id, "name": f.name, "country": f.country, "issuer": f.issuer,
              "source": docs.get(f.id, {}).get("source"), "updated_at": docs.get(f.id, {}).get("updated_at"),
              "sides": f.side_ids, "variants": _variant_rows(a, f)}
             for f in sorted(st.families.values(), key=lambda x: x.id)]
    return {"registry_version": st.version, "families": len(items), "reference_images": st.ref_count, "items": items}


@router.get("/{family_id}")
def get_card(family_id: str, format: str = "json", a: CardAgent = Depends(agent)):
    """The family definition and reference counts. ?format=yaml returns the definition as YAML."""
    a.refresh_if_changed(force=True)
    doc = _doc_or_404(a, family_id)
    if format == "yaml":
        return Response(yaml.safe_dump(doc["definition"], sort_keys=False, allow_unicode=True),
                        media_type="application/yaml")
    return _view(a, doc)


@router.put("/{family_id}", dependencies=[Depends(require_admin)])
async def put_card(family_id: str, request: Request, response: Response, force: bool = False,
                   a: CardAgent = Depends(agent)):
    """Create or replace a family. Body: a card definition as JSON, or YAML with
    Content-Type: application/yaml (schema: docs/API.md, examples/cards/). Removing a variant deletes its
    reference images; changing field ROIs re-extracts this family's keypoints."""
    fam = await parse_body(request, CardFamily)
    if fam.id != family_id:
        raise HTTPException(422, f"definition id '{fam.id}' does not match URL '{family_id}'")
    _owned_check(a.store.get_family(family_id), force, "change")
    out = await run_in_threadpool(a.store.put_family, fam, "api")
    a.refresh_if_changed(force=True)
    response.status_code = 201 if out["created"] else 200
    audit("put_card", family=family_id, created=out["created"], forced=force)
    return {**out, "registry_version": a.state.version, **_view(a, a.store.get_family(family_id))}


@router.delete("/{family_id}", dependencies=[Depends(require_admin)])
async def delete_card(family_id: str, force: bool = False, a: CardAgent = Depends(agent)):
    doc = _doc_or_404(a, family_id)
    _owned_check(doc, force, "delete")
    out = await run_in_threadpool(a.store.delete_family, family_id)
    a.refresh_if_changed(force=True)
    audit("delete_card", family=family_id, references=out["references_deleted"], forced=force)
    note = ("registry-managed: the next sync re-creates it unless its YAML and images are removed from "
            "registry/") if doc.get("source") == "registry" else None
    return {**out, "registry_version": a.state.version, **({"note": note} if note else {})}


# ------------------------------------------------------------------------------ reference images
@router.get("/{family_id}/references", dependencies=[Depends(require_admin)])
def list_references(family_id: str, a: CardAgent = Depends(agent)):
    doc = _doc_or_404(a, family_id)
    fam = CardFamily.model_validate(doc["definition"])
    refs = a.store.list_refs(family_id)
    keep = ("ref_id", "side", "filename", "sha256", "source", "added_at")
    return {"family": family_id, "sides": fam.side_ids,
            "variants": [{"id": v.id, "name": v.name,
                          "references": [{k: r.get(k) for k in keep} for r in refs if r["variant"] == v.id]}
                         for v in fam.variants]}


@router.get("/{family_id}/references/{ref_id}/image", dependencies=[Depends(require_admin)])
def reference_image(family_id: str, ref_id: str, a: CardAgent = Depends(agent)):
    ref = next((r for r in a.store.list_refs(family_id) if r["ref_id"] == ref_id), None)
    if ref is None or not ref.get("image_path"):
        raise HTTPException(404, f"reference {ref_id} not found in family {family_id}")
    p = a.store._abs(ref["image_path"])
    if not p.exists():
        raise HTTPException(404, "reference image file is missing on disk")
    with open(p, "rb") as fh:
        media = image_media_type(fh.read(16))
    return FileResponse(p, media_type=media)


@router.post("/{family_id}/variants/{variant_id}/references", status_code=201,
             dependencies=[Depends(require_admin)])
async def add_references(family_id: str, variant_id: str, response: Response,
                         images: list[UploadFile] = File(..., description="1..20 reference images of this variant"),
                         detect_card: bool = True,
                         side: Literal["front", "back"] = Query("front", description="the card side the images show"),
                         a: CardAgent = Depends(agent)):
    """Upload reference images of one side of a variant: flat scans, or photos (the card is found and
    straightened when detect_card=true). Each image is embedded and upserted on its own; re-uploading the
    same image is a no-op (status "exists")."""
    fam = CardFamily.model_validate(_doc_or_404(a, family_id)["definition"])
    if fam.variant(variant_id) is None:
        raise HTTPException(404, f"variant {variant_id} not in family {family_id} "
                                 f"(variants: {', '.join(v.id for v in fam.variants)})")
    if side not in fam.side_ids:
        raise HTTPException(422, f"{family_id} has no {side} side (sides: {', '.join(fam.side_ids)}); "
                                 f"add it to the definition's `sides` first")
    if not 1 <= len(images) <= MAX_IMAGES_PER_REQUEST:
        raise HTTPException(422, f"send 1..{MAX_IMAGES_PER_REQUEST} images per request")
    payloads = [(f.filename or "upload", await read_image(f)) for f in images]
    results = []
    for name, data in payloads:
        try:
            r = await run_in_threadpool(a.store.add_reference, fam, variant_id, data, name, "api", None, detect_card,
                                        False, side)
            results.append({"filename": name, **r})
        except ValueError as e:
            results.append({"filename": name, "status": "rejected", "error": str(e)})
    created = [r for r in results if r["status"] == "created"]
    if created:
        a.store.bump_revision()
        a.refresh_if_changed(force=True)
    else:
        response.status_code = 422 if all(r["status"] == "rejected" for r in results) else 200
    audit("add_references", family=family_id, variant=variant_id, created=len(created), total=len(results))
    return {"family": family_id, "variant": variant_id, "side": side, "created": len(created), "results": results,
            "reference_images_in_variant": a.state.ref_counts.get((family_id, variant_id), 0),
            "reference_images_in_side": a.state.side_counts.get((family_id, variant_id, side), 0),
            "registry_version": a.state.version}


@router.delete("/{family_id}/references/{ref_id}", dependencies=[Depends(require_admin)])
async def delete_reference(family_id: str, ref_id: str, force: bool = False, a: CardAgent = Depends(agent)):
    ref = next((r for r in a.store.list_refs(family_id) if r["ref_id"] == ref_id), None)
    if ref is None:
        raise HTTPException(404, f"reference {ref_id} not found in family {family_id}")
    if ref.get("source") == "registry" and not force:
        raise HTTPException(409, "reference image comes from registry/refs; remove the file there and run "
                                 "POST /v1/registry/sync, or pass ?force=true (it returns on the next sync)")
    out = await run_in_threadpool(a.store.delete_reference, family_id, ref_id)
    a.refresh_if_changed(force=True)
    audit("delete_reference", family=family_id, ref_id=ref_id, forced=force)
    return {**out, "reference_images_in_variant": a.state.ref_counts.get((family_id, out["variant"]), 0),
            "registry_version": a.state.version}
