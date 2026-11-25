"""Evaluation sets and evaluation runs (Qdrant collection <prefix>_evaluations, payload-only).

  kind=set         a labelled set: manifest.json + photos, kept on disk in evaluation.store_dir/sets/<id>/
  kind=evaluation  one evaluation of a set (optionally with what-if settings); the full report is kept
                   on disk in evaluation.store_dir/reports/<id>.json, the summary in the payload
Sets come from an uploaded zip or from reviewed runs (the review decisions are the ground truth).
"""
from __future__ import annotations

import io
import json
import shutil
import time
import uuid
import warnings
import zipfile
from pathlib import Path

from ..evaluation import load_manifest
from .qdrant_store import _guard
from .runs import _iso

SUMMARY_KEYS = ["cases", "status_agreement", "automation_rate", "accept_precision", "wrong_accepts",
                "field_accuracy", "latency_ms", "gate"]


class EvaluationStore:
    def __init__(self, card_store, store_dir: str, root: Path):
        self.client, self.m = card_store.client, card_store.m
        self.col = f"{card_store.cfg.collection_prefix}_evaluations"
        d = Path(store_dir)
        self.dir = d if d.is_absolute() else root / d
        self.ensure()

    @_guard
    def ensure(self):
        if not self.client.collection_exists(self.col):
            self.client.create_collection(self.col, vectors_config={})
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.client.create_payload_index(self.col, "kind", self.m.PayloadSchemaType.KEYWORD)
                self.client.create_payload_index(self.col, "set_id", self.m.PayloadSchemaType.KEYWORD)

    def set_dir(self, set_id: str) -> Path:
        return self.dir / "sets" / set_id

    # ------------------------------------------------------------------ sets
    @_guard
    def create_set_from_zip(self, name: str, data: bytes, max_bytes: int) -> dict:
        set_id = uuid.uuid4().hex[:16]
        dst = self.set_dir(set_id)
        dst.mkdir(parents=True)
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                if sum(i.file_size for i in z.infolist()) > max_bytes:
                    raise ValueError(f"zip expands to more than {max_bytes} bytes")
                root = dst.resolve()
                for info in z.infolist():
                    target = (dst / info.filename).resolve()
                    if root != target and root not in target.parents:       # zip-slip protection
                        raise ValueError(f"unsafe path in zip: {info.filename}")
                z.extractall(dst)
            manifests = sorted(dst.rglob("manifest.json"), key=lambda p: len(p.parts))
            if not manifests:
                raise ValueError("zip has no manifest.json")
            cases = load_manifest(manifests[0])
        except Exception:
            shutil.rmtree(dst, ignore_errors=True)
            raise
        return self._save_set(set_id, name, manifests[0].relative_to(dst).as_posix(), cases, "upload")

    @_guard
    def create_set_from_runs(self, name: str, runs: list[dict], run_store) -> dict:
        """Ground truth from human reviews: accepted -> the reviewed family/variant, expected ACCEPT (with
        corrected fields); rejected -> expected REJECT. Runs without a stored photo are skipped."""
        set_id = uuid.uuid4().hex[:16]
        dst = self.set_dir(set_id)
        (dst / "photos").mkdir(parents=True)
        cases, skipped = [], 0
        for r in runs:
            photos = run_store.photo_files(r)
            rev = r.get("review") or {}
            if not photos or not rev:
                skipped += 1
                continue
            sides = {f["photo"]: f.get("side") for f in photos}                  # side the agent used
            sides.update({e["photo"]: e["side"] for e in rev.get("enrolled_photos") or []})   # reviewer's sides
            names = []
            for f in photos:
                fn = f"photos/{r['run_id']}.{f['photo']}.img" if len(photos) > 1 else f"photos/{r['run_id']}.img"
                shutil.copy2(f["file"], dst / fn)
                names.append((sides.get(f["photo"]), fn))
            if len(names) == 1:
                where = {"file": names[0][1]}
            elif all(s for s, _ in names) and len({s for s, _ in names}) == len(names):
                where = {"files": {s: fn for s, fn in names}}
            else:
                where = {"files": [fn for _, fn in names]}
            card = (r.get("result") or {}).get("card") or {}
            if rev["decision"] == "accept":
                fields = {k: v.get("value") for k, v in ((r.get("result") or {}).get("fields") or {}).items()
                          if v.get("value") is not None}
                fields.update(rev.get("fields") or {})
                cases.append({**where, "family": rev.get("family") or card.get("family"),
                              "variant": rev.get("variant") or card.get("variant"),
                              "expected_status": "ACCEPT", "fields": fields, "run_id": r["run_id"]})
            else:
                cases.append({**where, "family": None, "variant": None, "expected_status": "REJECT",
                              "fields": {}, "run_id": r["run_id"]})
        if not cases:
            shutil.rmtree(dst, ignore_errors=True)
            raise ValueError("no reviewed runs with a stored photo in that range")
        (dst / "manifest.json").write_text(json.dumps(cases, indent=1), encoding="utf-8")
        out = self._save_set(set_id, name, "manifest.json", cases, "reviewed_runs")
        return {**out, "runs_skipped": skipped}

    def _save_set(self, set_id, name, manifest, cases, origin) -> dict:
        now = time.time()
        payload = {"kind": "set", "set_id": set_id, "name": name or set_id, "manifest": manifest, "origin": origin,
                   "cases": len(cases), "created_at": now, "created_iso": _iso(now),
                   "expected": {s: sum(c["expected_status"] == s for c in cases) for s in ("ACCEPT", "REVIEW", "REJECT")},
                   "families": sorted({c["family"] for c in cases if c.get("family")})}
        self.client.upsert(self.col, [self.m.PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"set:{set_id}")),
                                                         vector={}, payload=payload)])
        return payload

    @_guard
    def list_sets(self) -> list[dict]:
        return sorted(self._scroll("set"), key=lambda x: -x["created_at"])

    @_guard
    def get_set(self, set_id: str) -> dict | None:
        return next((s for s in self._scroll("set", set_id=set_id)), None)

    @_guard
    def delete_set(self, set_id: str) -> bool:
        s = self.get_set(set_id)
        if s is None:
            return False
        self.client.delete(self.col, points_selector=self.m.PointIdsList(
            points=[str(uuid.uuid5(uuid.NAMESPACE_URL, f"set:{set_id}"))]))
        shutil.rmtree(self.set_dir(set_id), ignore_errors=True)
        return True

    # ------------------------------------------------------------------ evaluations
    @_guard
    def create_evaluation(self, set_id: str, overrides: dict, today: str | None) -> dict:
        eid = uuid.uuid4().hex[:16]
        now = time.time()
        payload = {"kind": "evaluation", "evaluation_id": eid, "set_id": set_id, "status": "queued",
                   "settings_overrides": overrides, "today": today, "created_at": now, "created_iso": _iso(now),
                   "progress": {"done": 0, "total": None}, "summary": None, "error": None}
        self.client.upsert(self.col, [self.m.PointStruct(id=self._eid(eid), vector={}, payload=payload)])
        return payload

    @_guard
    def update_evaluation(self, eid: str, **fields):
        self.client.set_payload(self.col, fields, points=[self._eid(eid)])

    @_guard
    def get_evaluation(self, eid: str) -> dict | None:
        r = self.client.retrieve(self.col, [self._eid(eid)], with_payload=True)
        return dict(r[0].payload) if r and r[0].payload.get("kind") == "evaluation" else None

    @_guard
    def list_evaluations(self, set_id: str | None = None) -> list[dict]:
        return sorted(self._scroll("evaluation", set_id=set_id), key=lambda x: -x["created_at"])

    def save_report(self, eid: str, report: dict):
        p = self.dir / "reports" / f"{eid}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")

    def load_report(self, eid: str) -> dict | None:
        p = self.dir / "reports" / f"{eid}.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    @_guard
    def delete_evaluation(self, eid: str) -> bool:
        if self.get_evaluation(eid) is None:
            return False
        self.client.delete(self.col, points_selector=self.m.PointIdsList(points=[self._eid(eid)]))
        (self.dir / "reports" / f"{eid}.json").unlink(missing_ok=True)
        return True

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _eid(eid: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"evaluation:{eid}"))

    def _scroll(self, kind: str, set_id: str | None = None) -> list[dict]:
        m = self.m
        must = [m.FieldCondition(key="kind", match=m.MatchValue(value=kind))]
        if set_id:
            must.append(m.FieldCondition(key="set_id", match=m.MatchValue(value=set_id)))
        out, offset = [], None
        while True:
            pts, offset = self.client.scroll(self.col, scroll_filter=m.Filter(must=must), limit=500, offset=offset,
                                             with_payload=True)
            out += [dict(p.payload) for p in pts]
            if offset is None:
                return out
