"""Run history, the human review queue and operating statistics (admin).

Every validation is stored as a run (payload-only point in <prefix>_runs, photo on disk under
runs.store_dir) unless runs.enabled=false or the request passed ?store=false. REVIEW decisions enter
the queue with review_status=pending; a reviewer settles them with POST /v1/runs/{id}/review and can
enroll the photo as a new reference image so the same design is recognised next time.
"""
from __future__ import annotations

import time
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ...agent import CardAgent
from ...registry.models import CardFamily
from ...store.runs import compute_stats, parse_time
from ..deps import agent, audit, image_media_type, require_admin

router = APIRouter(prefix="/v1", tags=["runs & review"], dependencies=[Depends(require_admin)])
Status = Literal["ACCEPT", "REVIEW", "REJECT"]
ReviewStatus = Literal["none", "pending", "accepted", "rejected"]


def _time(value: str | None, name: str) -> float | None:
    try:
        return parse_time(value)
    except ValueError:
        raise HTTPException(422, f"{name}: use an ISO-8601 time like 2026-11-24 or 2026-11-24T08:00:00Z")


def _run_or_404(a: CardAgent, run_id: str) -> dict:
    run = a.runs.get(run_id)
    if run is None:
        raise HTTPException(404, f"run {run_id} not found")
    return run


@router.get("/runs")
def list_runs(status: Status | None = None, family: str | None = None, review_status: ReviewStatus | None = None,
              since: str | None = None, until: str | None = None,
              before: float | None = Query(None, description="paging cursor: next_before from the previous page"),
              limit: int = Query(50, ge=1, le=200), a: CardAgent = Depends(agent)):
    """Newest first, without the full result (GET /v1/runs/{id} has it)."""
    items, cursor = a.runs.list(status, family, review_status, _time(since, "since"), _time(until, "until"),
                                before, limit)
    return {"items": items, "next_before": cursor}


@router.get("/review")
def review_queue(family: str | None = None, before: float | None = None, limit: int = Query(50, ge=1, le=200),
                 a: CardAgent = Depends(agent)):
    """Runs waiting for a human decision (status REVIEW, not reviewed yet)."""
    items, cursor = a.runs.list(None, family, "pending", None, None, before, limit)
    return {"pending": a.runs.count(review_status="pending", family=family), "items": items, "next_before": cursor}


@router.get("/runs/{run_id}")
def get_run(run_id: str, a: CardAgent = Depends(agent)):
    return _run_or_404(a, run_id)


@router.get("/runs/{run_id}/image")
def run_image(run_id: str, a: CardAgent = Depends(agent)):
    p = a.runs.image_file(_run_or_404(a, run_id))
    if p is None:
        raise HTTPException(404, "no photo stored for this run (runs.store_images=false or pruned)")
    with open(p, "rb") as fh:
        return FileResponse(p, media_type=image_media_type(fh.read(16)))


@router.delete("/runs/{run_id}")
def delete_run(run_id: str, a: CardAgent = Depends(agent)):
    if not a.runs.delete(run_id):
        raise HTTPException(404, f"run {run_id} not found")
    audit("delete_run", run_id=run_id)
    return {"deleted": run_id}


class ReviewIn(BaseModel):
    decision: Literal["accept", "reject"]
    family: str | None = Field(None, description="the correct family (default: what the agent identified)")
    variant: str | None = Field(None, description="the correct variant (default: what the agent identified)")
    fields: dict[str, str] = Field(default_factory=dict, description="corrected field values, e.g. {\"card_number\": \"...\"}")
    note: str | None = None
    reviewer: str | None = None
    enroll: bool = Field(False, description="accept only: add this photo as a reference image of family/variant")


@router.post("/runs/{run_id}/review")
async def review_run(run_id: str, body: ReviewIn, a: CardAgent = Depends(agent)):
    """Settle a run (normally one from the review queue; any run can be reviewed, e.g. a wrong ACCEPT).
    The verdict and corrections are kept with the run and feed evaluation sets built from runs."""
    run = _run_or_404(a, run_id)
    if body.decision == "reject" and body.enroll:
        raise HTTPException(422, "enroll is only possible with decision=accept")
    family = body.family or run.get("family")
    variant = body.variant or run.get("variant")
    fam = None
    if body.decision == "accept":
        if not family or not variant:
            raise HTTPException(422, "the agent did not identify this card: send family and variant")
        doc = a.store.get_family(family)
        if doc is None:
            raise HTTPException(422, f"unknown family {family}")
        fam = CardFamily.model_validate(doc["definition"])
        if fam.variant(variant) is None:
            raise HTTPException(422, f"variant {variant} not in family {family}")
        unknown = set(body.fields) - {f.name for f in fam.fields}
        if unknown:
            raise HTTPException(422, f"fields not defined for {family}: {', '.join(sorted(unknown))}")
    review = {"decision": body.decision, "family": family if body.decision == "accept" else None,
              "variant": variant if body.decision == "accept" else None, "fields": body.fields,
              "note": body.note, "reviewer": body.reviewer, "reviewed_at": time.time(), "enrolled": None}
    if body.enroll:
        p = a.runs.image_file(run)
        if p is None:
            raise HTTPException(422, "this run has no stored photo to enroll")
        try:
            r = await run_in_threadpool(a.store.add_reference, fam, variant, p.read_bytes(), f"run-{run_id}.img", "api")
        except ValueError as e:
            raise HTTPException(422, f"photo could not be enrolled: {e}")
        a.refresh_if_changed(force=True)
        review["enrolled"] = {"ref_id": r["ref_id"], "status": r["status"]}
    out = a.runs.set_review(run_id, review)
    audit("review_run", run_id=run_id, decision=body.decision, family=family, variant=variant,
          enrolled=bool(review["enrolled"]))
    out.pop("result", None)
    return out


@router.post("/runs/prune")
def prune_runs(older_than_days: int | None = Query(None, ge=1, description="default: runs.retention_days"),
               a: CardAgent = Depends(agent)):
    """Delete runs (and their photos) older than the retention period. Also runs hourly in the background."""
    n = a.runs.prune(older_than_days)
    audit("prune_runs", deleted=n)
    return {"deleted": n, "older_than_days": older_than_days or a.s.runs.retention_days}


@router.get("/stats")
async def stats(since: str | None = None, until: str | None = None, family: str | None = None,
                a: CardAgent = Depends(agent)):
    """Decision mix, automation and review rates, reviewer agreement, latency, top reasons, per family
    and per day, computed from the stored runs (default: the last 7 days)."""
    t_since = _time(since, "since") or time.time() - 7 * 86400
    t_until = _time(until, "until")
    fields = ["created_at", "status", "reasons", "family", "variant", "processing_ms", "review_status", "review"]
    runs = await run_in_threadpool(a.runs.scan, t_since, t_until, family, fields, a.s.runs.stats_max_runs)
    out = compute_stats(runs, t_since, t_until)
    out["truncated"] = len(runs) >= a.s.runs.stats_max_runs
    return out
