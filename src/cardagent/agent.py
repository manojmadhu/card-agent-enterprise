"""CardAgent: orchestrates all stages and owns hot-reloadable engine state.

Engine state is a StoreSnapshot (card definitions + keyword index + a read view of the reference
vectors) taken from Qdrant. A reload builds a new snapshot off to the side and swaps a single
reference: requests in flight keep the state they started with, and a failed reload leaves the
current state untouched. Every replica checks the store revision every qdrant.refresh_seconds and
reloads when cards or runtime settings were changed elsewhere.

Settings: `base_settings` come from config/settings.yaml + environment; `s` (effective) adds the
threshold overrides stored in Qdrant (PATCH /v1/settings). process(settings=...) runs one request
with other settings (what-if evaluations) without touching the live ones.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from datetime import date
from typing import Callable

import cv2
import numpy as np

from . import __version__
from .embed.embedders import Embedder, build_embedder
from .observability import telemetry as tm
from .ocr.engines import OcrEngine, build_ocr
from .pipeline import extract, identify, rules
from .schemas import ValidationResult
from .settings import PROJECT_ROOT, Settings
from .store.base import StoreSnapshot
from .store.evaluations import EvaluationStore
from .store.qdrant_store import QdrantCardStore
from .store.runs import RunStore
from .store.runtime_settings import RuntimeSettingsStore, apply_overrides
from .vision import normalise, quality
from .vision.features import body_colour_rgb, extract_keypoints
from .vlm.client import VlmClient

log = logging.getLogger(__name__)


def _no_trace(*_a, **_k):
    return None


class CardAgent:
    def __init__(self, settings: Settings):
        self.base_settings = settings
        self.s = settings
        self.embedder: Embedder = build_embedder(settings.embedder, PROJECT_ROOT)
        self.ocr: OcrEngine = build_ocr(settings.ocr, PROJECT_ROOT)
        self.vlm = VlmClient(settings.vlm) if settings.vlm.enabled else None
        self.store = QdrantCardStore(settings.qdrant, self.embedder, settings.retrieval.orb_features, PROJECT_ROOT)
        self.runtime_settings = RuntimeSettingsStore(self.store)
        self.runs = RunStore(self.store, settings.runs, PROJECT_ROOT)
        self.evaluations = EvaluationStore(self.store, settings.evaluation.store_dir, PROJECT_ROOT)
        if settings.qdrant.sync_on_startup:
            from .store.sync import sync_registry
            log.info("registry sync on startup: %s", sync_registry(self.store, settings.path(settings.registry_dir)))
        self._reload_lock = threading.Lock()
        self._last_check = time.monotonic()
        self._apply_runtime_settings()
        self.state: StoreSnapshot = self.store.snapshot()
        tm.REGISTRY_INFO.set(self.state.ref_count)

    def _apply_runtime_settings(self) -> None:
        overrides = self.runtime_settings.get()
        try:
            self.s = apply_overrides(self.base_settings, overrides)
        except ValueError as e:                                  # stored overrides no longer valid
            log.error("ignoring invalid runtime settings in Qdrant: %s", e)
            self.s = self.base_settings
        self.overrides = overrides

    # ------------------------------------------------------------------ registry
    def reload(self, rebuild: bool = False) -> dict:
        with self._reload_lock:
            try:
                new = self.store.snapshot(rebuild=rebuild)
            except Exception:
                tm.RELOADS.labels("failed").inc()
                log.exception("registry reload failed; keeping version %s", self.state.version)
                raise
            old = self.state.version
            self._apply_runtime_settings()
            self.state = new                                   # atomic reference swap
            self._last_check = time.monotonic()
            tm.RELOADS.labels("ok").inc()
            tm.REGISTRY_INFO.set(new.ref_count)
            if old != new.version:
                log.info("registry reloaded %s -> %s", old, new.version)
            return new.summary()

    def refresh_if_changed(self, force: bool = False) -> None:
        """Pick up card or settings changes made by any replica (cheap revision check)."""
        if not force and time.monotonic() - self._last_check < self.s.qdrant.refresh_seconds:
            return
        self._last_check = time.monotonic()
        if self.store.current_version() != self.state.version:
            self.reload()

    # ------------------------------------------------------------------ main entry
    def process(self, image_bytes: bytes, request_id: str | None = None, today: date | None = None,
                trace: Callable[..., None] | None = None, settings: Settings | None = None) -> dict:
        """trace (optional): called as trace("start", step) / trace("done", step, data) for each stage,
        for the live step-by-step stream. settings (optional): use these instead of the live settings
        for this call only (what-if evaluations). Without them, behaviour is unchanged."""
        emit = trace or _no_trace
        self.refresh_if_changed()
        s = settings or self.s
        st = self.state                                        # pin state for this request
        t0 = time.perf_counter()
        timings: dict[str, int] = {}

        def lap(name, t):
            dt = time.perf_counter() - t
            timings[name] = int(dt * 1000)
            tm.STAGE.labels(name).observe(dt)
            return time.perf_counter()

        out = {"request_id": request_id or str(uuid.uuid4()), "status": "REJECT", "reasons": [],
               "model_versions": {"pipeline": __version__, "registry": st.version,
                                  "embedder": self.embedder.name, "ocr": self.ocr.name,
                                  **({"vlm": s.vlm.model} if self.vlm else {})}}

        emit("start", "ingest")
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        emit("done", "ingest", {"request_id": out["request_id"], "bytes": len(image_bytes),
                                "width": None if img is None else img.shape[1],
                                "height": None if img is None else img.shape[0]})
        if img is None:
            return self._finish(out, "REJECT", ["INVALID_IMAGE"], t0, timings)

        emit("start", "quality")
        t = time.perf_counter()
        q = quality.assess(img, s.quality)
        out["quality"] = q
        t = lap("quality", t)
        emit("done", "quality", {"quality": q, "ms": timings["quality"]})
        if not q["passed"]:
            return self._finish(out, "REJECT", ["RETAKE_REQUIRED", *q["issues"]], t0, timings)

        emit("start", "normalise")
        card, ninfo = normalise.normalise(img)
        t = lap("normalise", t)
        emit("done", "normalise", {"info": ninfo, "card": card, "ms": timings["normalise"]})
        emit("start", "ocr")
        card, full, rotation = self._ocr_orientation(card, st.keywords)
        q.update(card_detected=ninfo["card_detected"], detector=ninfo["detector"], rotation=rotation)
        t = lap("ocr_full", t)
        emit("done", "ocr", {"ocr": full, "rotation": rotation, "card": card, "ms": timings["ocr_full"]})

        emit("start", "embed")
        qvec = self.embedder.embed(card)
        qkp = extract_keypoints(card, s.retrieval.orb_features)
        qcol = body_colour_rgb(card)
        t = lap("features", t)
        emit("done", "embed", {"vector": qvec, "keypoints": len(qkp.pts), "body_rgb": qcol,
                               "ms": timings["features"]})

        emit("start", "search")
        itrace = {} if trace else None
        ident = identify.identify(st.refs, st.keywords, st.families, full, qvec, qkp, qcol,
                                  s.retrieval, s.scoring, s.embedder.similarity_floor, itrace)
        t = lap("identify", t)
        emit("done", "identify", {"ident": ident, "trace": itrace, "ref_count": st.ref_count})
        out["candidates"] = {"family": ident.family_candidates}
        if not ident.family_decided:
            sc = s.scoring
            reason = "UNKNOWN_CARD_TYPE" if ident.family_score < sc.family_min else "AMBIGUOUS_CARD_TYPE"
            status = "REJECT" if ident.family_score < sc.reject_below else "REVIEW"
            return self._finish(out, status, [reason], t0, timings)

        fam = st.families[ident.family_id]
        v = fam.variant(ident.variant_id)
        out["candidates"]["variant"] = ident.variant_candidates
        out["card"] = {"family": fam.id, "family_name": fam.name, "country": fam.country, "issuer": fam.issuer,
                       "family_confidence": ident.family_score, "variant": v.id if v else None,
                       "variant_name": v.name if v else None, "variant_confidence": ident.variant_score}

        H = ident.best_match.H if (ident.best_match and ident.best_match.inliers >= s.retrieval.min_inliers) else None
        if H is not None and ninfo["card_detected"]:
            shift = extract.corner_shift(H)
            q["align_shift"] = round(shift, 3)
            if shift > s.retrieval.max_align_shift:
                H = None                                       # implausible correction: keep the rectified card
        emit("start", "fields")
        aligned, did_align = extract.align(card, H)
        q["aligned_to_reference"] = did_align
        fields = extract.extract_fields(fam, aligned, full, self.ocr, self.vlm)
        t = lap("extract", t)
        emit("done", "fields", {"family": fam, "fields": fields, "aligned": did_align,
                                "align_shift": q.get("align_shift"), "ms": timings["extract"]})
        emit("start", "rules")
        checks = rules.run_rules(fam, v.id if v else None, fields, today)
        lap("rules", t)
        emit("done", "rules", {"checks": checks, "ms": timings["rules"]})

        status, reasons = self._decide(v, fields, checks, s)
        for name, f in fields.items():
            if f["source"] == "vlm":
                tm.VLM_CALLS.inc()
        out["fields"] = {}
        for name, f in fields.items():
            fr = {k: f[k] for k in ("value", "confidence", "source", "valid_format")}
            spec = next(s for s in fam.fields if s.name == name)
            if spec.normalise.startswith("date") and f["valid_format"]:
                d = rules.parse_expiry(f["value"])
                fr["valid_until"] = d.isoformat() if d else None
            out["fields"][name] = fr
        out["validation"] = checks
        tm.FAMILY.labels(fam.id, v.id if v else "unknown").inc()
        return self._finish(out, status, reasons, t0, timings)

    # ------------------------------------------------------------------ helpers
    def _ocr_orientation(self, card, keywords):
        best = None
        for rot in (0, 180):
            img = card if rot == 0 else cv2.rotate(card, cv2.ROTATE_180)
            o = self.ocr.read_full(img)
            kw = max(keywords.keyword_families(o["tokens"], o["text"]).values(), default=0.0)
            score = kw + 0.1 * o["mean_conf"]
            if best is None or score > best[0]:
                best = (score, img, o, rot)
            if kw >= 0.99:
                break
        return best[1], best[2], best[3]

    def _decide(self, v, fields, checks, s: Settings) -> tuple[str, list[str]]:
        reasons = []
        if v is None:
            reasons.append("AMBIGUOUS_VARIANT")
        for name, f in fields.items():
            if f["required"] and not f["valid_format"]:
                reasons.append(f"FIELD_UNREADABLE:{name}")
            elif f["valid_format"] and (f["confidence"] < s.scoring.field_min_confidence or f["source"] == "vlm"):
                reasons.append(f"LOW_CONFIDENCE:{name}")
        expired = any(c["rule"] == "not_expired" and c["result"] == "fail" and (c["detail"] or "").startswith("CARD_EXPIRED")
                      for c in checks)
        reasons += [f"RULE_FAILED:{c['rule']}:{c['field']}" for c in checks if c["result"] == "fail"]
        if expired:
            return "REJECT", ["CARD_EXPIRED", *reasons]
        return ("REVIEW", reasons) if reasons else ("ACCEPT", [])

    def _finish(self, out, status, reasons, t0, timings) -> dict:
        out["status"], out["reasons"] = status, reasons
        out["timings_ms"] = timings
        out["processing_ms"] = int((time.perf_counter() - t0) * 1000)
        result = ValidationResult.model_validate(out).model_dump()   # enforce the output contract
        tm.REQUESTS.labels(status).inc()
        tm.LATENCY.observe(out["processing_ms"] / 1000)
        for r in reasons:
            tm.REASONS.labels(r.split(":")[0]).inc()
        return result
