"""Stages 4-5: two-stage identification.

Stage 1 (recall, cheap):   vector search over ALL reference images  +  OCR keyword inverted index
                           -> candidate families (shortlist)
Stage 2 (precision):       ORB + RANSAC geometric verification on a capped set of shortlisted refs,
                           fused with text, embedding and colour scores -> family, then variant.
Decisions need score >= min AND margin over runner-up >= margin.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import numpy as np

from ..index.reference_index import ReferenceIndex
from ..registry.models import CardFamily
from ..settings import RetrievalCfg, ScoringCfg
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
    best_ref: int | None = None              # index into ReferenceIndex.entries
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


def identify(idx: ReferenceIndex, families: dict[str, CardFamily], ocr: dict, qvec: np.ndarray,
             qkp: Keypoints, qcolour, rcfg: RetrievalCfg, scfg: ScoringCfg, sim_floor: float) -> Identification:
    tokens, text = ocr["tokens"], ocr["text"]
    all_sims = idx.vectors @ qvec                                   # cosine to every reference (cheap)
    nn = idx.search(qvec, rcfg.shortlist_refs)                      # stage-1 vector shortlist
    kw_hits = idx.keyword_families(tokens, text)                    # stage-1 keyword shortlist
    negatives = idx.negative_hits(text)
    cand_fams = {idx.entries[i].family_id for i, _ in nn} | set(kw_hits)
    cand_fams = {f for f in cand_fams if f in families}

    # ---- choose which references to verify geometrically (bounded cost) ----
    by_variant: dict[tuple[str, str], list[int]] = {}
    for i, e in enumerate(idx.entries):
        if e.family_id in cand_fams:
            by_variant.setdefault((e.family_id, e.variant_id), []).append(i)
    to_verify: list[int] = []
    for refs in by_variant.values():
        to_verify += sorted(refs, key=lambda i: -all_sims[i])[:rcfg.refs_per_variant_verify]
    to_verify = sorted(to_verify, key=lambda i: -(all_sims[i] + (1.0 if idx.entries[i].family_id in kw_hits else 0)))
    to_verify = to_verify[:rcfg.max_verify_refs]
    geo: dict[int, GeoMatch] = {i: verify(qkp, idx.entries[i].kp, rcfg.ratio_test) for i in to_verify}

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
    vdec, _ = _decide(var_cands, scfg.variant_min, scfg.variant_margin)
    res.variant_decided, res.variant_candidates = vdec, var_cands
    res.variant_id = var_cands[0]["id"] if vdec else None
    res.variant_score = var_cands[0]["score"]

    # best reference for alignment: most inliers, preferring the chosen variant
    pool = [i for i in geo if idx.entries[i].family_id == fam.id
            and (res.variant_id is None or idx.entries[i].variant_id == res.variant_id)]
    if pool:
        b = max(pool, key=lambda i: geo[i].inliers)
        res.best_ref, res.best_match = b, geo[b]
    return res
