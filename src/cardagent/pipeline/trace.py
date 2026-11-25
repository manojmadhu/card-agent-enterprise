"""Live step trace for POST /v1/cards/validate/stream.

Snapshots are `{steps: StepState[], normalisedUrl?, result?}` where StepState is
`{id, status: pending|running|pass|warn|fail|skip, summary, ms, detail}` over the 11 STEPS below
(the step format the web console renders, so a server run shows as a live step timeline).
StepTrace turns the agent's stage events (agent.process(trace=...)) into StepState objects with a
`detail` per step: quality checks, OCR tokens, embedding, shortlist + verified references, family
and variant candidates with their signals, extracted fields, rule results, decision.

Two photos (both sides of a card): the per-photo steps (ingest, quality, normalise, ocr, embed) are
tracked per photo and merged into the same step: status = the worst finished photo (running while
any photo is still in it), summary "front: … · back: …", and detail = the first photo's detail plus
`photos: [{photo, label, status, summary, detail}]`. Snapshots then also carry
`photos: [{photo, label, side, normalisedUrl}]`. A single photo produces exactly the 2.0 trace.
"""
from __future__ import annotations

import base64

import cv2
import numpy as np

from ..settings import Settings

STEPS = [
    {"id": "ingest", "title": "Ingest", "stage": "0"},
    {"id": "quality", "title": "Quality gate", "stage": "1"},
    {"id": "normalise", "title": "Normalise", "stage": "2"},
    {"id": "ocr", "title": "Full-card OCR", "stage": "3"},
    {"id": "embed", "title": "Embed card", "stage": "4a"},
    {"id": "search", "title": "Reference search", "stage": "4b"},
    {"id": "family", "title": "Family classification", "stage": "4c"},
    {"id": "variant", "title": "Variant classification", "stage": "5"},
    {"id": "fields", "title": "Field extraction", "stage": "6"},
    {"id": "rules", "title": "Validation rules", "stage": "7"},
    {"id": "decision", "title": "Decision", "stage": "8"},
]
_IDS = [s["id"] for s in STEPS]
PER_PHOTO = ("ingest", "quality", "normalise", "ocr", "embed")
_RANK = {"pending": 0, "skip": 1, "pass": 2, "warn": 3, "fail": 4}
MAX_VECTOR_VALUES = 768          # the console only draws the vector; DINOv2-small is 768-d


def _hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(int(c) for c in rgb)


def card_data_url(card_bgr: np.ndarray, width: int = 380) -> str:
    h = int(card_bgr.shape[0] * width / card_bgr.shape[1])
    ok, buf = cv2.imencode(".jpg", cv2.resize(card_bgr, (width, h), interpolation=cv2.INTER_AREA),
                           [cv2.IMWRITE_JPEG_QUALITY, 75])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


class StepTrace:
    """Accumulates StepState objects; `on_event` returns the steps that changed (to send)."""

    def __init__(self, settings: Settings, families: dict, ref_total: int, source: str = "api upload",
                 photos: list[str] | None = None):
        """photos: one label per uploaded photo (declared side or "photo 1"); default one photo."""
        self.s, self.families, self.ref_total, self.source = settings, families, ref_total, source
        self.steps = {sid: _blank(sid) for sid in _IDS}
        self.normalised_url: str | None = None
        self.family_id: str | None = None
        self.labels = list(photos or ["photo 1"])
        self.multi = len(self.labels) > 1
        self.photo_steps = [{sid: _blank(sid) for sid in _IDS} for _ in self.labels]
        self.photo_urls: list[str | None] = [None] * len(self.labels)
        self.photo_sides: list[str | None] = [None] * len(self.labels)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def cid(eid):
        return eid

    def snapshot(self, result: dict | None = None) -> dict:
        snap = {"steps": [dict(self.steps[i]) for i in _IDS]}
        if self.normalised_url:
            snap["normalisedUrl"] = self.normalised_url
        if self.multi:
            snap["photos"] = self.photos_view()
        if result is not None:
            snap["result"] = result
        return snap

    def _set(self, sid, status, summary, detail, ms=0):
        self.steps[sid].update(status=status, summary=summary, detail=detail, ms=max(1, int(ms)) if ms else 1)
        return self.steps[sid]

    def _skip_from(self, first: str, summary: str) -> list[dict]:
        out, on = [], False
        for sid in _IDS:
            on = on or sid == first
            if on and sid != "decision" and self.steps[sid]["status"] in ("pending", "running"):
                self.steps[sid].update(status="skip", summary=summary, ms=0)
                out.append(self.steps[sid])
        return out

    def photos_view(self) -> list[dict]:
        return [{"photo": i, "label": self.labels[i], "side": self.photo_sides[i], "normalisedUrl": self.photo_urls[i]}
                for i in range(len(self.labels))]

    # ------------------------------------------------------------------ events
    def on_event(self, event: str, step: str, data: dict | None = None, photo: int | None = None) -> list[dict]:
        if self.multi and photo is not None and step in PER_PHOTO:
            return self._photo_event(event, step, data, photo)
        if event == "start":
            if step in self.steps:
                self.steps[step]["status"] = "running"
                return [self.steps[step]]
            return []
        return getattr(self, f"_done_{step}")(data or {})

    def _photo_event(self, event, step, data, i) -> list[dict]:
        """Run the normal step logic on photo i's own steps, then merge the photos into the shared step."""
        shared_steps, shared_url = self.steps, self.normalised_url
        self.steps, self.normalised_url = self.photo_steps[i], self.photo_urls[i]
        try:
            if event == "start":
                self.steps[step]["status"] = "running"
                touched = [step]
            else:
                touched = [st["id"] for st in getattr(self, f"_done_{step}")(data or {})]
            self.photo_urls[i] = self.normalised_url
        finally:
            self.steps, self.normalised_url = shared_steps, shared_url
        if self.normalised_url is None and self.photo_urls[0]:
            self.normalised_url = self.photo_urls[0]
        return [self._merge(sid) for sid in dict.fromkeys(touched) if sid in PER_PHOTO]

    def _merge(self, sid: str) -> dict:
        per = [ps[sid] for ps in self.photo_steps]
        if any(p["status"] == "running" for p in per) or (
                any(p["status"] == "pending" for p in per) and any(p["status"] != "pending" for p in per)):
            status = "running"
        else:
            status = max((p["status"] for p in per), key=lambda s: _RANK.get(s, 0))
        first = next((p["detail"] for p in per if p["detail"] is not None), None)
        detail = {**(first or {}), "photos": [{"photo": i, "label": self.labels[i], "status": p["status"],
                                                "summary": p["summary"], "detail": p["detail"]}
                                               for i, p in enumerate(per)]}
        summary = " · ".join(f"{self.labels[i]}: {p['summary']}" for i, p in enumerate(per) if p["summary"])
        self.steps[sid].update(status=status, summary=summary, detail=detail, ms=sum(p["ms"] for p in per))
        return self.steps[sid]

    def _done_ingest(self, d):
        if d.get("width") is None:
            return [self._set("ingest", "fail", "not a decodable image", {"request_id": d["request_id"],
                                                                          "bytes": d["bytes"], "source": self.source})]
        return [self._set("ingest", "pass", f"{d['width']}×{d['height']} · {self.source}",
                          {"request_id": d["request_id"], "width": d["width"], "height": d["height"],
                           "source": self.source})]

    def _done_quality(self, d):
        q, c = d["quality"], self.s.quality
        checks = [
            {"name": "sharpness (Laplacian var.)", "value": q["blur"], "op": "≥", "limit": c.min_blur,
             "ok": q["blur"] >= c.min_blur, "code": "IMAGE_BLURRY"},
            {"name": "brightness", "value": q["brightness"], "op": "≥", "limit": c.brightness_min,
             "ok": q["brightness"] >= c.brightness_min, "code": "BAD_EXPOSURE"},
            {"name": "brightness", "value": q["brightness"], "op": "≤", "limit": c.brightness_max,
             "ok": q["brightness"] <= c.brightness_max, "code": "BAD_EXPOSURE"},
            {"name": "glare ratio", "value": q["glare_ratio"], "op": "≤", "limit": c.max_glare,
             "ok": q["glare_ratio"] <= c.max_glare, "code": "GLARE"},
            {"name": "resolution (short side px)", "value": min(q["width"], q["height"]), "op": "≥",
             "limit": c.min_side, "ok": min(q["width"], q["height"]) >= c.min_side, "code": "LOW_RESOLUTION"},
        ]
        metrics = {"blur": q["blur"], "brightness": q["brightness"], "glare_ratio": q["glare_ratio"]}
        if not q["passed"]:
            changed = [self._set("quality", "fail", f"Failed: {', '.join(q['issues'])} → RETAKE_REQUIRED",
                                 {"metrics": metrics, "checks": checks}, d["ms"])]
            return changed + self._skip_from("normalise", "skipped — quality gate failed")
        return [self._set("quality", "pass", f"blur {q['blur']} · brightness {q['brightness']} · "
                                             f"glare {q['glare_ratio'] * 100:.1f}%", {"metrics": metrics, "checks": checks}, d["ms"])]

    def _done_normalise(self, d):
        info = d["info"]
        self.normalised_url = card_data_url(d["card"])
        detected = info["card_detected"]
        return [self._set("normalise", "pass" if detected else "warn",
                          "warped to 1012×638" + ("" if detected else " · card edges not found, full frame used"),
                          {"size": [1012, 638], "orientation": 0, "tried": [{"deg": 0}],
                           "card_detected": detected, "detector": info["detector"]}, d["ms"])]

    def _done_ocr(self, d):
        o, rot = d["ocr"], d["rotation"]
        changed = []
        if rot:                                              # orientation is settled during OCR
            self.normalised_url = card_data_url(d["card"])
            n = self.steps["normalise"]
            n["detail"] = {**(n["detail"] or {}), "orientation": rot, "tried": [{"deg": 0}, {"deg": 180}]}
            n["summary"] = f"{n['summary']} · rotation {rot}°"
            changed.append(n)
        tokens = [{"text": t, "conf": round(o["mean_conf"], 3)} for t in o["tokens"]]
        changed.append(self._set("ocr", "pass" if tokens else "warn",
                                 f"{len(tokens)} tokens · mean conf {o['mean_conf']:.2f}" if tokens
                                 else "no text read — text signal will be 0",
                                 {"engine": self.s.ocr.engine, "tokens": tokens, "mean_conf": round(o["mean_conf"], 3),
                                  "rotation": rot}, d["ms"]))
        return changed

    def _done_embed(self, d):
        v = d["vector"]
        return [self._set("embed", "pass", f"{len(v)}-d vector · body colour {_hex(d['body_rgb'])} · "
                                           f"{d['keypoints']} keypoints",
                          {"embedder": self.s.embedder.kind, "dim": len(v),
                           "vector": [round(float(x), 4) for x in v[:MAX_VECTOR_VALUES]],
                           "vector_truncated": len(v) > MAX_VECTOR_VALUES, "body_rgb": list(d["body_rgb"]),
                           "keypoints": d["keypoints"]}, d["ms"])]

    def _done_identify(self, d):
        ident, tr = d["ident"], d["trace"] or {}
        ms = tr.get("ms", {})
        sc, floor = self.s.scoring, self.s.embedder.similarity_floor
        changed = []

        # ---- search (stage 1 shortlist + stage 2 geometric verification)
        top = [{**h, "family": self.cid(h["family"]), "source": "index"}
               for h in tr.get("shortlist", [])]
        best = top[0]["sim"] if top else 0.0
        matched = best >= floor
        verified = [{**r, "family": self.cid(r["family"])} for r in tr.get("verified", [])]
        summary = (f"best match {top[0]['family']}/{top[0]['variant']} · sim {best:.3f} · "
                   f"{len(tr.get('candidate_families', []))} families shortlisted · {len(verified)} refs verified"
                   if top else "no reference cards enrolled")
        changed.append(self._set("search", "pass" if matched else "warn",
                                 summary if matched or not top else
                                 f"no reference above floor {floor} (best {best:.3f}) — not matching",
                                 {"scanned": d["ref_count"], "total": d["ref_count"], "top": top, "best": best,
                                  "floor": floor, "matched": matched, "all": top,
                                  "keyword_families": {self.cid(k): v for k, v in tr.get("keyword_hits", {}).items()},
                                  "candidate_families": [self.cid(f) for f in tr.get("candidate_families", [])],
                                  "verified": verified, "refs_scored": tr.get("refs_scored", 0)},
                                 ms.get("search", 1)))

        # ---- family
        w = tr.get("weights", sc.family_weights)
        cands = []
        for c in ident.family_candidates:
            fam = self.families.get(c["id"])
            sg = c["signals"]
            cands.append({"family": self.cid(c["id"]), "name": fam.name if fam else c["id"], "text": sg["text"],
                          "keyword_hit": None, "negative_hit": None, "visual": sg["embed"],
                          "best_ref_sim": round(tr.get("family_best_sim", {}).get(c["id"], 0.0), 3),
                          "colour": sg["colour"], "geometry": sg["geometry"],
                          "weights": {"text": w.get("text", 0), "visual": w.get("embed", 0),
                                      "colour": w.get("colour", 0), "geometry": w.get("geometry", 0)},
                          "score": c["score"]})
        t1 = cands[0] if cands else None
        t2 = cands[1] if len(cands) > 1 else None
        margin = ident.family_margin
        if ident.family_decided:
            outcome, status = "decided", "pass"
            fsum = f"{t1['name']} · score {t1['score']:.2f} · margin {margin:.2f}"
            self.family_id = ident.family_id
        elif t1 and ident.family_score >= sc.family_min:
            outcome, status = "ambiguous", "warn"
            fsum = (f"{t1['name']} {t1['score']:.2f} vs {t2['name']} {t2['score']:.2f} — "
                    f"margin {margin:.2f} < δ {sc.family_margin}") if t2 else f"{t1['name']} — margin < δ"
        else:
            outcome, status = "unknown", "fail"
            fsum = f"top-1 {t1['name'] + ' ' + format(t1['score'], '.2f') if t1 else '—'} < τ {sc.family_min} → unknown card"
        changed.append(self._set("family", status, fsum,
                                 {"candidates": cands, "tau": sc.family_min, "delta": sc.family_margin,
                                  "top1": t1["score"] if t1 else 0.0, "margin": margin, "outcome": outcome,
                                  "text_missing": tr.get("text_missing", False)}, ms.get("family", 1)))
        if not ident.family_decided:
            reason = "card family ambiguous" if outcome == "ambiguous" else "card family not recognised"
            return changed + self._skip_from("variant", f"skipped — {reason}")

        # ---- variant
        fam = self.families[ident.family_id]
        vc = []
        for c in ident.variant_candidates:
            v, sg = fam.variant(c["id"]), c["signals"]
            vc.append({"variant": c["id"], "name": v.name if v else c["id"], "colour": sg["colour"],
                       "keyword": sg["keyword"], "keyword_hit": None, "prefix": sg["prefix"],
                       "geometry": sg["geometry"], "embed": sg["embed"],
                       "rgb": list(v.body_color_rgb) if v else None, "score": c["score"]})
        vm = tr.get("variant_margin", 0.0)
        vdetail = {"candidates": vc, "tau": sc.variant_min, "delta": sc.variant_margin, "margin": vm,
                   "number_seen": tr.get("number_seen"), "body_rgb": None}
        if ident.variant_decided:
            changed.append(self._set("variant", "pass", f"{vc[0]['name']} · score {vc[0]['score']:.2f} · margin {vm:.2f}",
                                     vdetail, ms.get("variant", 1)))
        else:
            changed.append(self._set("variant", "warn", f"{vc[0]['name']} {vc[0]['score']:.2f} — below τ or "
                                                        f"margin {vm:.2f} < δ", vdetail, ms.get("variant", 1)))
        return changed

    def _done_fields(self, d):
        fam, fields = d["family"], d["fields"]
        rows = []
        for spec in fam.fields:
            f = fields.get(spec.name, {})
            iso = None
            if spec.normalise.startswith("date") and f.get("valid_format"):
                from ..pipeline.rules import parse_expiry
                p = parse_expiry(f.get("value"))
                iso = p.isoformat() if p else None
            rows.append({"name": spec.name, "side": spec.side, "required": spec.required, "roi": list(spec.roi),
                         "regex": spec.regex,
                         "normalise": spec.normalise, "value": f.get("value"), "confidence": f.get("confidence", 0),
                         "source": {"ocr_roi": "ocr"}.get(f.get("source"), f.get("source")), "raw": f.get("raw"),
                         "iso": iso, "ok": bool(f.get("valid_format"))})
        missing = [r["name"] for r in rows if r["required"] and not r["ok"]]
        if self.multi:
            for ph in d.get("photos", []):
                self.photo_sides[ph["photo"]] = ph["side"]
                if ph["side"]:
                    self.labels[ph["photo"]] = ph["side"]
        extra = {} if not self.multi else {"photos": d.get("photos", []), "missing_sides": d.get("missing_sides", []),
                                           "side_assignment": d.get("side_detail", [])}
        vlm = "enabled" if self.s.vlm.enabled else "disabled"
        summary = (f"{len(rows) - len(missing)}/{len(rows)} extracted · missing {', '.join(missing)}"
                   + ("" if self.s.vlm.enabled else " (VLM fallback off)")) if missing else \
            f"{sum(r['ok'] for r in rows)}/{len(rows)} fields extracted"
        return [self._set("fields", "warn" if missing else "pass", summary,
                          {"fields": rows, "vlm": vlm, "aligned_to_reference": d["aligned"],
                           "align_shift": d.get("align_shift"), **extra}, d["ms"])]

    def _done_rules(self, d):
        checks = d["checks"]
        crit = [c for c in checks if c["result"] == "fail" and c["critical"]]
        soft = [c for c in checks if c["result"] == "fail" and not c["critical"]]
        npass = sum(c["result"] == "pass" for c in checks)
        return [self._set("rules", "fail" if crit else "warn" if soft else "pass",
                          f"{npass} pass · {len(crit) + len(soft)} fail · {len(checks) - npass - len(crit) - len(soft)} skip",
                          {"results": checks}, d["ms"])]

    # ------------------------------------------------------------------ end
    def finish(self, result: dict) -> list[dict]:
        """Close the trace from the final result (also covers early exits)."""
        changed = []
        reasons = result["reasons"]
        if self.multi:
            for ph in result.get("photos") or []:
                self.photo_sides[ph["photo"]] = ph["side"]
                if ph["side"]:
                    self.labels[ph["photo"]] = ph["side"]
            for ps in self.photo_steps:                        # close per-photo steps of unusable photos
                for sid in PER_PHOTO:
                    if ps[sid]["status"] in ("pending", "running"):
                        ps[sid].update(status="skip", summary="skipped", ms=0)
            for sid in PER_PHOTO:
                changed.append(self._merge(sid))
        if ("RETAKE_REQUIRED" in reasons or "INVALID_IMAGE" in reasons) and not result.get("card"):
            changed += self._skip_from("normalise", "skipped — quality gate failed")
        changed += self._skip_from("ingest", "skipped")
        status = result["status"]
        conf = (result.get("card") or {}).get("variant_confidence") or (result.get("card") or {}).get("family_confidence") or 0.0
        changed.append(self._set("decision", {"ACCEPT": "pass", "REVIEW": "warn"}.get(status, "fail"),
                                 f"{status}{' · ' + ', '.join(reasons) if reasons else ''}",
                                 {"status": status, "reasons": reasons, "confidence": conf}, 1))
        return changed


def _blank(sid: str) -> dict:
    return {"id": sid, "status": "pending", "summary": "", "ms": 0, "detail": None}


__all__ = ["STEPS", "PER_PHOTO", "StepTrace"]
