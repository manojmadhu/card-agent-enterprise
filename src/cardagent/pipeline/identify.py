"""Stages 4-5: two-stage identification.

Stage 1 (recall, cheap):   vector search over ALL reference images  +  OCR keyword inverted index
                           -> candidate families (shortlist)
Stage 2 (precision):       similarity to every reference of the candidate families (filtered vector
                           query), ORB + RANSAC geometric verification on a capped set of them,
                           fused with text, embedding and colour scores -> family, then variant.
Decisions need score >= min AND margin over runner-up >= margin.

Works against the Qdrant card store through RefView; the cost per request
depends on the shortlisted families, never on the total number of cards.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import numpy as np

from ..registry.models import CardFamily
from ..settings import RetrievalCfg, ScoringCfg
from ..store.base import RefView
from ..store.keywords import KeywordIndex
from ..vision.features import GeoMatch, Keypoints, colour_similarity, verify


@dataclass
class Identification:
    family_decided: bool
    family_id: str | None
    family_score: float
    family_margin: float
    family_candidates: list[dict]
    variant_decided: bool = False
    variant_id: str | None = None
    variant_score: float = 0.0
    variant_candidates: list[dict] = field(default_factory=list)
    best_ref: str | None = None              # ref_id of the reference used for alignment
    best_match: GeoMatch | None = None       # its homography (query -> reference)
    refs_verified: int = 0


def _kw_score(tokens: list[str], text: str, keywords: list[str]) -> float:
    best = 0.0
    for kw in keywords:
        if re.search(rf"(?<!\w){re.escape(kw)}(?!\w)", text):
            return 1.0
        for t in tokens:
            if abs(len(t) - len(kw)) <= 2 and len(kw) >= 4:
                best = max(best, SequenceMatcher(None, t, kw).ratio() if
                           SequenceMatcher(None, t, kw).ratio() >= 0.8 else 0.0)
    return best


def _norm_sim(sim: float, floor: float) -> float:
    return float(np.clip((sim - floor) / max(1e-6, 1 - floor), 0, 1))


def _decide(cands: list[dict], min_score: float, margin: float) -> tuple[bool, float]:
    if not cands:
        return False, 0.0
    m = cands[0]["score"] - (cands[1]["score"] if len(cands) > 1 else 0.0)
    return cands[0]["score"] >= min_score and m >= margin, round(m, 3)


def identify(refs_view: RefView, keywords: KeywordIndex, families: dict[str, CardFamily], ocr: dict,
             qvec: np.ndarray, qkp: Keypoints, qcolour, rcfg: RetrievalCfg, scfg: ScoringCfg,
             sim_floor: float, trace: dict | None = None) -> Identification:
    """trace (optional): filled with the intermediate evidence (shortlist, keyword hits, verified
    references, per-sub-stage timings) for step-by-step views such as the validation stream."""
    t0 = time.perf_counter()
    tokens, text = ocr["tokens"], ocr["text"]
    nn = refs_view.search(qvec, rcfg.shortlist_refs)                # stage-1 vector shortlist
    kw_hits = keywords.keyword_families(tokens, text)               # stage-1 keyword shortlist
    negatives = keywords.negative_hits(text)
    cand_fams = {h.family_id for h in nn} | set(kw_hits)
    cand_fams = {f for f in cand_fams if f in families}

    # cosine to every reference of the candidate families (one filtered vector query)
    hits = refs_view.family_refs(qvec, cand_fams, rcfg.max_candidate_refs)
    all_sims = {h.ref_id: h.score for h in hits}
    ref_family = {h.ref_id: h.family_id for h in hits}
    ref_variant = {h.ref_id: h.variant_id for h in hits}

    # ---- choose which references to verify geometrically (bounded cost) ----
    by_variant: dict[tuple[str, str], list[str]] = {}
    for h in hits:
        by_variant.setdefault((h.family_id, h.variant_id), []).append(h.ref_id)
    to_verify: list[str] = []
    for refs in by_variant.values():
        to_verify += sorted(refs, key=lambda i: -all_sims[i])[:rcfg.refs_per_variant_verify]
    to_verify = sorted(to_verify, key=lambda i: -(all_sims[i] + (1.0 if ref_family[i] in kw_hits else 0)))
    to_verify = to_verify[:rcfg.max_verify_refs]
    kps = refs_view.keypoints(to_verify)
    geo: dict[str, GeoMatch] = {i: verify(qkp, kps[i], rcfg.ratio_test) for i in to_verify if i in kps}
    t_search = time.perf_counter()

    def geo_score(inl: int) -> float:
        # inliers at or below the baseline are what two cards sharing a template/font produce
        return float(np.clip((inl - rcfg.inliers_baseline) / max(1, rcfg.inliers_full_score - rcfg.inliers_baseline), 0, 1))

    # ---- family scoring ----
    # No programme keyword readable at all (glare on the logo, unsupported script...) means the
    # text signal is MISSING, not negative: re-weight over visual signals but demand strong geometry.
    text_missing = not kw_hits
    w = dict(scfg.family_weights)
    if text_missing:
        rest = sum(v for k, v in w.items() if k != "text")
        w = {k: (0.0 if k == "text" else v / rest) for k, v in w.items()}
    fam_cands = []
    for fid in cand_fams:
        fam = families[fid]
        refs = [i for (f, _), rs in by_variant.items() if f == fid for i in rs]
        s_text = kw_hits.get(fid, 0.0) * (0.2 if fid in negatives else 1.0)
        s_emb = _norm_sim(max((all_sims[i] for i in refs), default=0.0), sim_floor)
        s_geo = geo_score(max((geo[i].inliers for i in refs if i in geo), default=0))
        s_col = max(colour_similarity(qcolour, v.body_color_rgb) for v in fam.variants)
        score = w["text"] * s_text + w["embed"] * s_emb + w["geometry"] * s_geo + w["colour"] * s_col
        if text_missing and s_geo < scfg.geometry_required_without_text:
            score = min(score, scfg.family_min - 0.01)      # cannot be accepted on looks alone
        fam_cands.append({"id": fid, "score": round(score, 3),
                          "signals": {"text": round(s_text, 3), "embed": round(s_emb, 3),
                                      "geometry": round(s_geo, 3), "colour": round(s_col, 3)}})
    fam_cands.sort(key=lambda c: -c["score"])
    decided, margin = _decide(fam_cands, scfg.family_min, scfg.family_margin)
    t_family = time.perf_counter()
    if trace is not None:
        best_sim: dict[str, float] = {}
        for h in hits:
            best_sim[h.family_id] = max(best_sim.get(h.family_id, 0.0), h.score)
        trace.update(
            shortlist=[{"ref_id": h.ref_id, "family": h.family_id, "variant": h.variant_id,
                        "sim": round(h.score, 4)} for h in nn],
            keyword_hits={k: round(v, 3) for k, v in kw_hits.items()}, negative_hits=sorted(negatives),
            candidate_families=sorted(cand_fams), refs_scored=len(hits), family_best_sim=best_sim,
            verified=[{"ref_id": i, "family": ref_family[i], "variant": ref_variant[i], "inliers": g.inliers,
                       "good_matches": g.good_matches} for i, g in sorted(geo.items(), key=lambda x: -x[1].inliers)],
            weights=w, text_missing=text_missing, margin=margin,
            ms={"search": int((t_search - t0) * 1000), "family": int((t_family - t_search) * 1000)})
    res = Identification(decided, fam_cands[0]["id"] if decided else None,
                         fam_cands[0]["score"] if fam_cands else 0.0, margin, fam_cands[:5],
                         refs_verified=len(geo))
    if not decided:
        return res

    # ---- variant scoring (only inside the chosen family) ----
    fam = families[res.family_id]
    wv = scfg.variant_weights
    digits = re.findall(r"\d{6,}", text.replace(" ", ""))
    var_cands = []
    for v in fam.variants:
        refs = by_variant.get((fam.id, v.id), [])
        s_col = colour_similarity(qcolour, v.body_color_rgb, scale=22.0)
        s_kw = _kw_score(tokens, text, v.keywords) if v.keywords else 0.0
        s_geo = geo_score(max((geo[i].inliers for i in refs if i in geo), default=0))
        s_emb = _norm_sim(max((all_sims[i] for i in refs), default=0.0), sim_floor)
        s_pre = 1.0 if v.number_prefix and any(d.startswith(v.number_prefix) for d in digits) else 0.0
        score = (wv["colour"] * s_col + wv["keyword"] * s_kw + wv["geometry"] * s_geo
                 + wv["embed"] * s_emb + wv["prefix"] * s_pre)
        var_cands.append({"id": v.id, "score": round(score, 3),
                          "signals": {"colour": round(s_col, 3), "keyword": round(s_kw, 3),
                                      "geometry": round(s_geo, 3), "embed": round(s_emb, 3), "prefix": s_pre}})
    var_cands.sort(key=lambda c: -c["score"])
    vdec, vmargin = _decide(var_cands, scfg.variant_min, scfg.variant_margin)
    if trace is not None:
        trace["variant_margin"] = vmargin
        trace["number_seen"] = digits[0] if digits else None
        trace["ms"]["variant"] = int((time.perf_counter() - t_family) * 1000)
    res.variant_decided, res.variant_candidates = vdec, var_cands
    res.variant_id = var_cands[0]["id"] if vdec else None
    res.variant_score = var_cands[0]["score"]

    # best reference for alignment: most inliers, preferring the chosen variant
    pool = [i for i in geo if ref_family[i] == fam.id
            and (res.variant_id is None or ref_variant[i] == res.variant_id)]
    if pool:
        b = max(pool, key=lambda i: geo[i].inliers)
        res.best_ref, res.best_match = b, geo[b]
    return res
