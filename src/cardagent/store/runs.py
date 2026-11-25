"""Run history and review queue (Qdrant collection <prefix>_runs, payload-only points).

One point per validation. The payload keeps the full result JSON plus indexed fields for
filtering (status, family, review_status) and time (created_at, epoch seconds). The uploaded
photos (one, or one per card side) are kept on disk (runs.store_dir) when runs.store_images is on,
so a REVIEW case can be inspected and, once confirmed, enrolled as reference images.
Payload `images`: [{photo, side, upload, path}]; `image_path` is the main photo (front when present).

Runs contain personal data (names, card and Emirates ID numbers): the API serves them only with
the admin token, and runs older than runs.retention_days are deleted with their photos.
"""
from __future__ import annotations

import logging
import time
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path

from ..settings import RunsCfg
from .qdrant_store import _guard

log = logging.getLogger(__name__)
LIST_FIELDS = ["run_id", "request_id", "created_at", "created_iso", "status", "reasons", "family", "variant",
               "family_confidence", "variant_confidence", "processing_ms", "source", "review_status",
               "has_image", "photos", "registry_version"]


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds")


def parse_time(value: str | None) -> float | None:
    """ISO-8601 date/time (2026-11-24 or 2026-11-24T08:00:00+04:00) -> epoch seconds."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError as e:
        raise ValueError(f"invalid date/time '{value}': use ISO-8601, e.g. 2026-11-24 or 2026-11-24T08:00:00+04:00") from e
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


class RunStore:
    def __init__(self, card_store, cfg: RunsCfg, root: Path):
        self.cs, self.cfg = card_store, cfg
        self.client, self.m = card_store.client, card_store.m
        self.col = f"{card_store.cfg.collection_prefix}_runs"
        d = Path(cfg.store_dir)
        self.dir = d if d.is_absolute() else root / d
        self.root = root
        self.ensure()

    @_guard
    def ensure(self):
        if not self.client.collection_exists(self.col):
            self.client.create_collection(self.col, vectors_config={})
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for f in ("status", "family", "review_status", "request_id"):
                    self.client.create_payload_index(self.col, f, self.m.PayloadSchemaType.KEYWORD)
                self.client.create_payload_index(self.col, "created_at", self.m.PayloadSchemaType.FLOAT)

    # ------------------------------------------------------------------ write
    @_guard
    def add(self, result: dict, photos, source: str) -> str:
        """photos: the uploaded photos (agent.Photo list), or raw bytes of a single photo."""
        run_id = uuid.uuid4().hex
        now = time.time()
        if isinstance(photos, (bytes, bytearray)) or photos is None:
            from ..agent import Photo
            photos = [] if photos is None else [Photo(bytes(photos), None, "image")]
        info = {p["photo"]: p for p in result.get("photos") or []}
        images = []
        if self.cfg.store_images:
            day = datetime.fromtimestamp(now, timezone.utc)
            for i, ph in enumerate(photos):
                if not ph.data:
                    continue
                name = f"{run_id}.img" if i == 0 else f"{run_id}.{i}.img"
                p = self.dir / f"{day:%Y}" / f"{day:%m}" / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(ph.data)
                images.append({"photo": i, "side": (info.get(i) or {}).get("side"), "upload": ph.upload,
                               "path": p.relative_to(self.dir).as_posix()})
        main = next((im for im in images if im["side"] == "front"), images[0] if images else None)
        image_path = main["path"] if main else None
        card = result.get("card") or {}
        payload = {
            "kind": "run", "run_id": run_id, "request_id": result.get("request_id"),
            "created_at": now, "created_iso": _iso(now), "source": source,
            "status": result["status"], "reasons": result.get("reasons", []),
            "family": card.get("family"), "variant": card.get("variant"),
            "family_confidence": card.get("family_confidence"), "variant_confidence": card.get("variant_confidence"),
            "processing_ms": result.get("processing_ms"),
            "registry_version": (result.get("model_versions") or {}).get("registry"),
            "review_status": "pending" if result["status"] == "REVIEW" else "none", "review": None,
            "has_image": image_path is not None, "image_path": image_path, "images": images,
            "photos": len(photos), "result": result,
        }
        self.client.upsert(self.col, [self.m.PointStruct(id=run_id, vector={}, payload=payload)])
        return run_id

    @_guard
    def set_review(self, run_id: str, review: dict) -> dict | None:
        run = self.get(run_id)
        if run is None:
            return None
        status = "accepted" if review["decision"] == "accept" else "rejected"
        self.client.set_payload(self.col, {"review_status": status, "review": review}, points=[run_id])
        return {**run, "review_status": status, "review": review}

    @_guard
    def delete(self, run_id: str) -> bool:
        run = self.get(run_id)
        if run is None:
            return False
        for rel in self._paths(run):
            self._unlink(rel)
        self.client.delete(self.col, points_selector=self.m.PointIdsList(points=[run_id]))
        return True

    @_guard
    def prune(self, retention_days: int | None = None) -> int:
        days = self.cfg.retention_days if retention_days is None else retention_days
        if days <= 0:
            return 0
        cutoff = time.time() - days * 86400
        flt = self._filter(until=cutoff)
        n = 0
        while True:
            pts, _ = self.client.scroll(self.col, scroll_filter=flt, limit=500, with_payload=["image_path", "images"])
            if not pts:
                break
            for p in pts:
                for rel in self._paths(p.payload):
                    self._unlink(rel)
            self.client.delete(self.col, points_selector=self.m.PointIdsList(points=[p.id for p in pts]))
            n += len(pts)
        if n:
            log.info("pruned %d runs older than %d days", n, days)
        return n

    # ------------------------------------------------------------------ read
    @_guard
    def get(self, run_id: str) -> dict | None:
        try:
            r = self.client.retrieve(self.col, [run_id], with_payload=True)
        except Exception:                                        # malformed id
            return None
        return dict(r[0].payload) if r else None

    def image_file(self, run: dict, side: str | None = None, photo: int | None = None) -> Path | None:
        """The main photo, or the photo used as `side`, or photo number `photo` (0-based)."""
        rel = run.get("image_path")
        if (side is not None or photo is not None) and run.get("images"):
            im = next((x for x in run["images"]
                       if (side is None or x.get("side") == side) and (photo is None or x.get("photo") == photo)), None)
            rel = im["path"] if im else None
        elif side not in (None, "front") or photo not in (None, 0):
            rel = None                          # runs stored before card sides existed: one (front) photo
        if not rel:
            return None
        p = (self.dir / rel).resolve()
        return p if self.dir.resolve() in p.parents and p.exists() else None

    def photo_files(self, run: dict) -> list[dict]:
        """[{photo, side, upload, file}] of every stored photo of a run (runs from before card sides
        existed: the single photo)."""
        ims = run.get("images") or ([{"photo": 0, "side": None, "upload": "image", "path": run["image_path"]}]
                                    if run.get("image_path") else [])
        out = []
        for im in ims:
            p = (self.dir / im["path"]).resolve()
            if self.dir.resolve() in p.parents and p.exists():
                out.append({**{k: v for k, v in im.items() if k != "path"}, "file": p})
        return out

    @_guard
    def list(self, status=None, family=None, review_status=None, since=None, until=None, before=None,
             limit: int = 50) -> tuple[list[dict], float | None]:
        """Newest first. Pagination: pass the returned cursor as `before` for the next page."""
        upper, strict = (before, True) if before is not None else (until, False)
        flt = self._filter(status, family, review_status, since, upper, strict_until=strict)
        pts, _ = self.client.scroll(self.col, scroll_filter=flt, limit=limit + 1, with_payload=LIST_FIELDS,
                                    order_by=self.m.OrderBy(key="created_at", direction=self.m.Direction.DESC))
        items = [dict(p.payload) for p in pts[:limit]]
        cursor = items[-1]["created_at"] if len(pts) > limit else None
        return items, cursor

    @_guard
    def count(self, **kw) -> int:
        return self.client.count(self.col, count_filter=self._filter(**kw), exact=True).count

    @_guard
    def scan(self, since=None, until=None, family=None, fields=None, max_runs: int = 50000) -> list[dict]:
        out, offset = [], None
        flt = self._filter(family=family, since=since, until=until)
        while len(out) < max_runs:
            pts, offset = self.client.scroll(self.col, scroll_filter=flt, limit=min(1000, max_runs - len(out)),
                                             offset=offset, with_payload=fields or LIST_FIELDS)
            out += [dict(p.payload) for p in pts]
            if offset is None:
                break
        return out

    # ------------------------------------------------------------------ helpers
    def _filter(self, status=None, family=None, review_status=None, since=None, until=None, strict_until=False):
        m, must = self.m, []
        for key, val in (("status", status), ("family", family), ("review_status", review_status)):
            if val:
                must.append(m.FieldCondition(key=key, match=m.MatchValue(value=val)))
        if since is not None or until is not None:
            rng = {"gte": since} if since is not None else {}
            if until is not None:
                rng["lt" if strict_until else "lte"] = until
            must.append(m.FieldCondition(key="created_at", range=m.Range(**rng)))
        return m.Filter(must=must) if must else None

    @staticmethod
    def _paths(run: dict) -> list[str]:
        return list(dict.fromkeys([x["path"] for x in run.get("images") or []] + [run.get("image_path")]))

    def _unlink(self, rel: str | None):
        if not rel:
            return
        p = (self.dir / rel).resolve()
        if self.dir.resolve() in p.parents and p.exists():
            p.unlink()


def compute_stats(runs: list[dict], since: float | None, until: float | None) -> dict:
    """Dashboard numbers from a list of run payloads (see RunStore.scan)."""
    n = len(runs)
    by_status = {s: 0 for s in ("ACCEPT", "REVIEW", "REJECT")}
    reasons: dict[str, int] = {}
    fams: dict[str, dict] = {}
    days: dict[str, dict] = {}
    lat = sorted(r["processing_ms"] for r in runs if r.get("processing_ms") is not None)
    review = {"pending": 0, "accepted": 0, "rejected": 0}
    for r in runs:
        st = r["status"]
        by_status[st] = by_status.get(st, 0) + 1
        for code in r.get("reasons") or []:
            k = code.split(":")[0]
            reasons[k] = reasons.get(k, 0) + 1
        f = fams.setdefault(r.get("family") or "unidentified", {"total": 0, "ACCEPT": 0, "REVIEW": 0, "REJECT": 0})
        f["total"] += 1
        f[st] = f.get(st, 0) + 1
        day = _iso(r["created_at"])[:10]
        d = days.setdefault(day, {"date": day, "total": 0, "ACCEPT": 0,
                                                    "REVIEW": 0, "REJECT": 0})
        d["total"] += 1
        d[st] += 1
        if r.get("review_status") in review:
            review[r["review_status"]] += 1
    reviewed = review["accepted"] + review["rejected"]
    pct = lambda q: lat[min(len(lat) - 1, int(len(lat) * q))] if lat else None  # noqa: E731
    return {
        "since": _iso(since) if since else None, "until": _iso(until) if until else None, "runs": n,
        "by_status": by_status,
        "auto_accept_rate": round(by_status["ACCEPT"] / n, 4) if n else None,
        "review_rate": round(by_status["REVIEW"] / n, 4) if n else None,
        "review_queue": review,
        "review_agreement": round(review["accepted"] / reviewed, 4) if reviewed else None,
        "latency_ms": {"p50": pct(0.5), "p95": pct(0.95), "max": lat[-1] if lat else None},
        "top_reasons": sorted(({"reason": k, "count": v} for k, v in reasons.items()), key=lambda x: -x["count"])[:15],
        "by_family": sorted(({"family": k, **v} for k, v in fams.items()), key=lambda x: -x["total"]),
        "per_day": sorted(days.values(), key=lambda x: x["date"]),
    }
