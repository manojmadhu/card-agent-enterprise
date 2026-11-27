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
from dataclasses import dataclass
from dataclasses import field as dc_field
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


MAX_PHOTOS = 2                       # a card has two sides


def _no_trace(*_a, **_k):
    return None


@dataclass
class Photo:
    """One uploaded photo. declared: the side named by the client (front/back), None = detect it."""
    data: bytes | None
    declared: str | None = None
    upload: str = "image"


@dataclass
class _Work:
    index: int
    photo: Photo
    status: str = "ok"                   # ok | invalid | retake | not_used
    quality: dict | None = None
    ninfo: dict | None = None
    card: object = None
    full: dict | None = None
    rotation: int = 0
    qvec: object = None
    qkp: object = None
    qcol: tuple = ()
    ev: object = None
    side: str | None = None
    side_corrected: bool = False
    ref_id: str | None = None
    inliers: int | None = None
    fields: list = dc_field(default_factory=list)


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
    def process(self, image_bytes: bytes | None = None, request_id: str | None = None, today: date | None = None,
                trace: Callable[..., None] | None = None, settings: Settings | None = None,
                photos: list[Photo] | None = None) -> dict:
        """Validate one card from one photo (image_bytes) or from up to two photos of its sides (photos).

        trace (optional): called as trace("start", step, photo=i) / trace("done", step, data, photo=i)
        for each stage, for the live step-by-step stream. settings (optional): use these instead of the
        live settings for this call only (what-if evaluations)."""
        emit = trace or _no_trace
        if photos is None:
            photos = [Photo(image_bytes, None, "image")]
        if not 1 <= len(photos) <= MAX_PHOTOS:
            raise ValueError(f"send 1 to {MAX_PHOTOS} photos")
        self.refresh_if_changed()
        s = settings or self.s
        st = self.state                                        # pin state for this request
        t0 = time.perf_counter()
        timings: dict[str, int] = {}

        def lap(name, t):
            dt = time.perf_counter() - t
            timings[name] = timings.get(name, 0) + int(dt * 1000)
            tm.STAGE.labels(name).observe(dt)
            return time.perf_counter()

        out = {"request_id": request_id or str(uuid.uuid4()), "status": "REJECT", "reasons": [],
               "model_versions": {"pipeline": __version__, "registry": st.version,
                                  "embedder": self.embedder.name, "ocr": self.ocr.name,
                                  **({"vlm": s.vlm.model} if self.vlm else {})}}

        # ---- stages 0-4a for every photo: decode, quality gate, rectify, OCR, features
        work = [self._prepare(i, ph, out["request_id"], s, st, emit, lap) for i, ph in enumerate(photos)]
        usable = [w for w in work if w.status == "ok"]
        out["quality"] = next((w.quality for w in work if w.quality), None)
        if not usable:
            retake = [w for w in work if w.status == "retake"]
            reasons = (["RETAKE_REQUIRED", *dict.fromkeys(i for w in retake for i in w.quality["issues"])]
                       if retake else []) + (["INVALID_IMAGE"] if any(w.status == "invalid" for w in work) else [])
            out["photos"] = [self._photo_info(w) for w in work]
            return self._finish(out, "REJECT", reasons, t0, timings)

        # ---- stage 4b-5: identification over all usable photos (pooled text, per-photo visuals)
        pooled = usable[0].full if len(usable) == 1 else {
            "tokens": [tok for w in usable for tok in w.full["tokens"]],
            "text": " ".join(w.full["text"] for w in usable),
            "mean_conf": float(np.mean([w.full["mean_conf"] for w in usable]))}
        emit("start", "search")
        t = time.perf_counter()
        for w in usable:
            w.ev = identify.gather(st.refs, st.keywords, st.families, pooled, w.qvec, w.qkp, w.qcol, s.retrieval)
        per_photo = [identify.score_families(w.ev, st.families, s.scoring, s.retrieval, s.embedder.similarity_floor)
                     for w in usable]
        fam_cands = identify.combine([c for c, _, _ in per_photo])
        decided, margin = identify._decide(fam_cands, s.scoring.family_min, s.scoring.family_margin)
        if len(usable) == 1:
            fam_cands = [{k: v for k, v in c.items() if k != "photo"} for c in fam_cands]
        mismatch = identify.sides_mismatch([w.ev for w in usable], s.retrieval, s.scoring.sides_mismatch_geometry) \
            if len(usable) > 1 else None
        ident = identify.Identification(decided and not mismatch, fam_cands[0]["id"] if decided and not mismatch else None,
                                        fam_cands[0]["score"] if fam_cands else 0.0, margin, fam_cands[:5],
                                        refs_verified=sum(len(w.ev.geo) for w in usable))
        lead = max(range(len(usable)), key=lambda i: per_photo[i][0][0]["score"] if per_photo[i][0] else -1)
        itrace = None
        if trace:
            _, w_lead, tm_lead = per_photo[lead]
            itrace = identify.trace_info(usable[lead].ev, st.families, w_lead, tm_lead, margin)
            if len(usable) > 1:
                itrace["photos"] = [{"photo": w.index, "top": (per_photo[k][0][:1] or [None])[0],
                                     "verified": len(w.ev.geo)} for k, w in enumerate(usable)]
                itrace["sides_mismatch"] = mismatch
        out["candidates"] = {"family": ident.family_candidates}
        fam = st.families.get(ident.family_id) if ident.family_decided else None
        v = None
        if fam is not None:
            tv = time.perf_counter()
            var_cands = identify.combine([identify.score_variants(w.ev, fam, s.scoring, s.retrieval,
                                                                  s.embedder.similarity_floor) for w in usable])
            if len(usable) == 1:
                var_cands = [{k: x for k, x in c.items() if k != "photo"} for c in var_cands]
            vdec, vmargin = identify._decide(var_cands, s.scoring.variant_min, s.scoring.variant_margin)
            ident.variant_decided, ident.variant_candidates = vdec, var_cands
            ident.variant_id = var_cands[0]["id"] if vdec else None
            ident.variant_score = var_cands[0]["score"] if var_cands else 0.0
            v = fam.variant(ident.variant_id)
            if itrace is not None:
                itrace["variant_margin"] = vmargin
                itrace["number_seen"] = usable[0].ev.digits[0] if usable[0].ev.digits else None
                itrace["ms"]["variant"] = int((time.perf_counter() - tv) * 1000)
        lap("identify", t)
        emit("done", "identify", {"ident": ident, "trace": itrace, "ref_count": st.ref_count})

        if fam is None:
            out["photos"] = [self._photo_info(w) for w in work]
            if mismatch:
                return self._finish(out, "REVIEW", ["SIDES_MISMATCH"], t0, timings)
            sc = s.scoring
            reason = "UNKNOWN_CARD_TYPE" if ident.family_score < sc.family_min else "AMBIGUOUS_CARD_TYPE"
            status = "REJECT" if ident.family_score < sc.reject_below else "REVIEW"
            return self._finish(out, status, [reason], t0, timings)

        out["candidates"]["variant"] = ident.variant_candidates
        out["card"] = {"family": fam.id, "family_name": fam.name, "country": fam.country, "issuer": fam.issuer,
                       "family_confidence": ident.family_score, "variant": v.id if v else None,
                       "variant_name": v.name if v else None, "variant_confidence": ident.variant_score}

        # ---- which side each photo shows
        evs = [w.ev if w.status == "ok" else None for w in work]
        assigned, side_detail = identify.assign_sides(evs, [w.photo.declared for w in work], fam, s.retrieval,
                                                      s.embedder.similarity_floor)
        for w, side in zip(work, assigned):
            w.side = side
            w.side_corrected = bool(side and w.photo.declared and w.photo.declared != side)
            if w.status == "ok" and side is None:
                w.status = "not_used"
        covered = {w.side for w in work if w.side}
        missing = [sd for sd in fam.required_sides() if sd not in covered]

        # ---- stage 6: align each used photo to a reference of its side, read that side's fields
        emit("start", "fields")
        t = time.perf_counter()
        fields: dict[str, dict] = {}
        for w in work:
            if not w.side:
                continue
            ref_id, gm = identify.best_alignment(w.ev, fam.id, ident.variant_id, w.side)
            H = gm.H if (gm is not None and gm.inliers >= s.retrieval.min_inliers) else None
            w.ref_id, w.inliers = ref_id, (gm.inliers if gm is not None else None)
            if H is not None and w.ninfo["card_detected"]:
                shift = extract.corner_shift(H)
                w.quality["align_shift"] = round(shift, 3)
                if shift > s.retrieval.max_align_shift:
                    H = None                                   # implausible correction: keep the rectified card
            aligned, did_align = extract.align(w.card, H)
            w.quality["aligned_to_reference"] = did_align
            got = extract.extract_fields(fam, aligned, w.full, self.ocr, self.vlm,
                                         side=None if len(fam.sides) == 1 else w.side)
            w.fields = list(got)
            fields.update(got)
        for f in fam.fields:
            if f.name not in fields:
                fields[f.name] = extract.not_captured(f)
        fields = {f.name: fields[f.name] for f in fam.fields}           # definition order
        lap("extract", t)
        main = self._main_photo(work)
        out["quality"] = main.quality
        emit("done", "fields", {"family": fam, "fields": fields, "aligned": main.quality.get("aligned_to_reference"),
                                "align_shift": main.quality.get("align_shift"), "ms": timings["extract"],
                                "photos": [{"photo": w.index, "side": w.side, "fields": w.fields,
                                            "aligned": w.quality.get("aligned_to_reference") if w.quality else None}
                                           for w in work], "missing_sides": missing, "side_detail": side_detail})

        # ---- stage 7: rules (fields of a side nobody photographed are skipped, not failed)
        emit("start", "rules")
        t = time.perf_counter()
        checks = rules.run_rules(fam, v.id if v else None, fields, today)
        for c in checks:
            f = fields.get(c["field"])
            if f and not f.get("captured", True):
                c.update(result="skip", detail=f"SIDE_NOT_CAPTURED:{f['side']}")
        lap("rules", t)
        emit("done", "rules", {"checks": checks, "ms": timings["rules"]})

        # ---- stage 8: decision
        status, reasons = self._decide(v, fields, checks, s)
        if missing:
            side_reasons = []
            for sd in missing:
                # a photo that could have been this side but was unusable -> retake it; else ask for it
                bad = [w for w in work if w.status in ("retake", "invalid") and w.photo.declared in (None, sd)]
                if bad and bad[0].status == "retake":
                    side_reasons += ["RETAKE_REQUIRED", *bad[0].quality["issues"]]
                elif bad:
                    side_reasons.append("INVALID_IMAGE")
                side_reasons.append(f"SIDE_REQUIRED:{sd}")
            status, reasons = "REJECT", [*dict.fromkeys(side_reasons), *reasons]
        for name, f in fields.items():
            if f["source"] == "vlm":
                tm.VLM_CALLS.inc()
        out["fields"] = {}
        for name, f in fields.items():
            fr = {k: f[k] for k in ("value", "confidence", "source", "valid_format", "side")}
            spec = next(x for x in fam.fields if x.name == name)
            if spec.normalise.startswith("date") and f["valid_format"]:
                d = rules.parse_expiry(f["value"])
                fr["valid_until"] = d.isoformat() if d else None
            out["fields"][name] = fr
        out["validation"] = checks
        out["photos"] = [self._photo_info(w) for w in work]
        out["missing_sides"] = missing
        tm.FAMILY.labels(fam.id, v.id if v else "unknown").inc()
        return self._finish(out, status, reasons, t0, timings)

    def _prepare(self, i: int, ph: "Photo", request_id: str, s: Settings, st: StoreSnapshot, emit, lap) -> "_Work":
        """Stages 0-4a for one photo."""
        w = _Work(i, ph)

        def stage(name, t0):                           # record the stage, return its duration in ms
            ms = int((time.perf_counter() - t0) * 1000)
            lap(name, t0)
            return ms

        emit("start", "ingest", photo=i)
        img = cv2.imdecode(np.frombuffer(ph.data, np.uint8), cv2.IMREAD_COLOR) if ph.data else None
        emit("done", "ingest", {"request_id": request_id, "bytes": len(ph.data or b""),
                                "width": None if img is None else img.shape[1],
                                "height": None if img is None else img.shape[0]}, photo=i)
        if img is None:
            w.status = "invalid"
            return w
        emit("start", "quality", photo=i)
        t = time.perf_counter()
        w.quality = quality.assess(img, s.quality, s.ocr.engine)
        ms = stage("quality", t)
        emit("done", "quality", {"quality": w.quality, "ms": ms}, photo=i)
        if not w.quality["passed"]:
            w.status = "retake"
            return w
        emit("start", "normalise", photo=i)
        t = time.perf_counter()
        card, w.ninfo = normalise.normalise(img)
        ms = stage("normalise", t)
        emit("done", "normalise", {"info": w.ninfo, "card": card, "ms": ms}, photo=i)
        emit("start", "ocr", photo=i)
        t = time.perf_counter()
        w.card, w.full, w.rotation = self._ocr_orientation(card, st.keywords)
        w.quality.update(card_detected=w.ninfo["card_detected"], detector=w.ninfo["detector"], rotation=w.rotation)
        ms = stage("ocr_full", t)
        emit("done", "ocr", {"ocr": w.full, "rotation": w.rotation, "card": w.card, "ms": ms}, photo=i)
        emit("start", "embed", photo=i)
        t = time.perf_counter()
        w.qvec = self.embedder.embed(w.card)
        w.qkp = extract_keypoints(w.card, s.retrieval.orb_features)
        w.qcol = body_colour_rgb(w.card)
        ms = stage("features", t)
        emit("done", "embed", {"vector": w.qvec, "keypoints": len(w.qkp.pts), "body_rgb": w.qcol,
                               "ms": ms}, photo=i)
        return w

    @staticmethod
    def _main_photo(work: list["_Work"]) -> "_Work":
        used = [w for w in work if w.side]
        return next((w for w in used if w.side == "front"), used[0] if used else work[0])

    @staticmethod
    def _photo_info(w: "_Work") -> dict:
        return {"photo": w.index, "upload": w.photo.upload, "declared": w.photo.declared,
                "side": w.side, "status": "used" if w.side else ("not_used" if w.status == "ok" else w.status),
                "side_corrected": w.side_corrected, "quality": w.quality, "reference_id": w.ref_id,
                "inliers": w.inliers, "fields": w.fields}

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
            if not f.get("captured", True):            # side not photographed: SIDE_REQUIRED covers it
                continue
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
