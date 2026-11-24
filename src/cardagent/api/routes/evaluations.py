"""Evaluation: measure accuracy on labelled photos before and after a change (admin).

1. Create an evaluation set: upload a zip (manifest.json + photos, format in docs/API.md), or build
   one from reviewed runs (the reviewers' verdicts are the ground truth).
2. POST /v1/evaluations runs the whole set through the pipeline in the background, with the live
   settings or with what-if settings ({"settings": {"scoring": {"family_min": 0.6}}}) that are not
   applied to live traffic. Poll GET /v1/evaluations/{id} until status is done.
3. The report has status agreement, ACCEPT precision, wrong accepts, field accuracy, latency, family
   confusion and a pass/fail gate (evaluation.min_accept_precision / min_status_agreement).
"""
from __future__ import annotations

import logging
from datetime import date

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from ...agent import CardAgent
from ...evaluation import ManifestError, evaluate, load_manifest
from ...store.evaluations import SUMMARY_KEYS
from ...store.runs import parse_time
from ...store.runtime_settings import SettingsError, apply_overrides, merge
from ..deps import STATE, agent, audit, require_admin

log = logging.getLogger("cardagent.api")
router = APIRouter(prefix="/v1", tags=["evaluation"], dependencies=[Depends(require_admin)])


# ------------------------------------------------------------------------------ sets
@router.post("/evaluation-sets", status_code=201)
async def upload_set(file: UploadFile = File(..., description="zip with manifest.json and the photos it lists"),
                     name: str = Form(""), a: CardAgent = Depends(agent)):
    data = await file.read(a.s.evaluation.max_set_bytes + 1)
    if len(data) > a.s.evaluation.max_set_bytes:
        raise HTTPException(413, f"zip larger than {a.s.evaluation.max_set_bytes} bytes")
    try:
        out = await run_in_threadpool(a.evaluations.create_set_from_zip, name, data, a.s.evaluation.max_set_bytes)
    except (ManifestError, ValueError) as e:
        raise HTTPException(422, f"evaluation set rejected: {e}")
    except Exception as e:                                           # BadZipFile and friends
        raise HTTPException(422, f"not a readable zip file: {e}")
    audit("create_evaluation_set", set_id=out["set_id"], cases=out["cases"], origin="upload")
    return out


class FromRunsIn(BaseModel):
    name: str = ""
    since: str | None = Field(None, description="ISO time; default: all reviewed runs still stored")
    until: str | None = None
    family: str | None = None


@router.post("/evaluation-sets/from-runs", status_code=201)
async def set_from_runs(body: FromRunsIn, a: CardAgent = Depends(agent)):
    """Build a set from runs a reviewer accepted or rejected (needs runs.store_images=true)."""
    try:
        since, until = parse_time(body.since), parse_time(body.until)
    except ValueError:
        raise HTTPException(422, "since/until: use ISO-8601 like 2026-11-01")
    runs = []
    for rs in ("accepted", "rejected"):
        flt = dict(review_status=rs, since=since, until=until, family=body.family)
        items, cursor = a.runs.list(**flt, limit=200)
        while True:
            runs += items
            if cursor is None:
                break
            items, cursor = a.runs.list(**flt, before=cursor, limit=200)
    full = [a.runs.get(r["run_id"]) for r in runs]
    try:
        out = await run_in_threadpool(a.evaluations.create_set_from_runs, body.name, [r for r in full if r], a.runs)
    except ValueError as e:
        raise HTTPException(422, str(e))
    audit("create_evaluation_set", set_id=out["set_id"], cases=out["cases"], origin="reviewed_runs")
    return out


@router.get("/evaluation-sets")
def list_sets(a: CardAgent = Depends(agent)):
    return {"items": a.evaluations.list_sets()}


@router.get("/evaluation-sets/{set_id}")
def get_set(set_id: str, a: CardAgent = Depends(agent)):
    s = a.evaluations.get_set(set_id)
    if s is None:
        raise HTTPException(404, f"evaluation set {set_id} not found")
    return s


@router.delete("/evaluation-sets/{set_id}")
def delete_set(set_id: str, a: CardAgent = Depends(agent)):
    if not a.evaluations.delete_set(set_id):
        raise HTTPException(404, f"evaluation set {set_id} not found")
    audit("delete_evaluation_set", set_id=set_id)
    return {"deleted": set_id}


# ------------------------------------------------------------------------------ evaluations
class EvaluationIn(BaseModel):
    set_id: str
    settings: dict = Field(default_factory=dict, description="what-if overrides on top of the live ones "
                                                             "(same shape as PATCH /v1/settings; null removes)")
    today: date | None = Field(None, description="the date expiry rules compare against (default: today)")


def _run_evaluation(a: CardAgent, eid: str, set_doc: dict, settings, today: date | None):
    ev = a.evaluations
    try:
        ev.update_evaluation(eid, status="running")
        set_dir = ev.set_dir(set_doc["set_id"])
        manifest = set_dir / set_doc["manifest"]
        cases = load_manifest(manifest)
        last = [0]

        def progress(done, total):
            if done == total or done - last[0] >= 10:
                last[0] = done
                ev.update_evaluation(eid, progress={"done": done, "total": total})

        report = evaluate(a, cases, manifest.parent, today, settings,
                          (a.s.evaluation.min_accept_precision, a.s.evaluation.min_status_agreement), progress)
        ev.save_report(eid, report)
        ev.update_evaluation(eid, status="done", summary={k: report[k] for k in SUMMARY_KEYS},
                             registry_version=report["registry_version"], embedder=report["embedder"])
    except Exception as e:                                           # noqa: BLE001 - stored on the evaluation
        log.exception("evaluation %s failed", eid)
        ev.update_evaluation(eid, status="failed", error=f"{type(e).__name__}: {e}")


@router.post("/evaluations", status_code=202)
def start_evaluation(body: EvaluationIn, a: CardAgent = Depends(agent)):
    set_doc = a.evaluations.get_set(body.set_id)
    if set_doc is None:
        raise HTTPException(404, f"evaluation set {body.set_id} not found")
    a.refresh_if_changed(force=True)
    try:
        overrides = merge(a.overrides, body.settings)
        settings = apply_overrides(a.base_settings, overrides) if body.settings else None
    except SettingsError as e:
        raise HTTPException(422, str(e))
    doc = a.evaluations.create_evaluation(body.set_id, body.settings, body.today.isoformat() if body.today else None)
    STATE["executor"].submit(_run_evaluation, a, doc["evaluation_id"], set_doc, settings, body.today)
    audit("start_evaluation", evaluation_id=doc["evaluation_id"], set_id=body.set_id, what_if=bool(body.settings))
    return doc


@router.get("/evaluations")
def list_evaluations(set_id: str | None = None, a: CardAgent = Depends(agent)):
    return {"items": a.evaluations.list_evaluations(set_id)}


@router.get("/evaluations/{evaluation_id}")
def get_evaluation(evaluation_id: str, rows: bool = True, a: CardAgent = Depends(agent)):
    """Status, progress and (when done) the full report. rows=false leaves out the per-photo rows."""
    doc = a.evaluations.get_evaluation(evaluation_id)
    if doc is None:
        raise HTTPException(404, f"evaluation {evaluation_id} not found")
    report = a.evaluations.load_report(evaluation_id) if doc["status"] == "done" else None
    if report is not None and not rows:
        report.pop("rows", None)
    return {**doc, "report": report}


@router.delete("/evaluations/{evaluation_id}")
def delete_evaluation(evaluation_id: str, a: CardAgent = Depends(agent)):
    doc = a.evaluations.get_evaluation(evaluation_id)
    if doc is None:
        raise HTTPException(404, f"evaluation {evaluation_id} not found")
    if doc["status"] in ("queued", "running"):
        raise HTTPException(409, "evaluation is still running")
    a.evaluations.delete_evaluation(evaluation_id)
    audit("delete_evaluation", evaluation_id=evaluation_id)
    return {"deleted": evaluation_id}
