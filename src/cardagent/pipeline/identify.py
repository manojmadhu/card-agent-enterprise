"""Stages 4-5: two-stage identification, for one or two photos (the sides of one card).

Stage 1 (recall, cheap):   vector search over ALL reference images  +  OCR keyword inverted index
                           -> candidate families (shortlist)
Stage 2 (precision):       similarity to every reference of the candidate families (filtered vector
                           query), ORB + RANSAC geometric verification on a capped set of them,
                           fused with text, embedding and colour scores -> family, then variant.
Decisions need score >= min AND margin over runner-up >= margin.

Card sides: every reference image shows one side (front/back). Each photo is compared with the
references of BOTH sides, so it does not matter which side a photo shows or how it was labelled.
With two photos, the OCR text of both is pooled (a programme name on the front and a card number on
the back are evidence about the same card), each photo collects its own visual evidence, and the
family/variant scores are the best over the photos. assign_sides() then decides which side each
photo shows, and sides_mismatch() catches two photos of different cards.

Works against the Qdrant card store through RefView; the cost per request depends on the
shortlisted families, never on the total number of cards.
"""
from __future__ import annotations

import itertools
import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import numpy as np

from ..registry.models import CardFamily
from ..settings import RetrievalCfg, ScoringCfg
from ..store.base import RefHit, RefView
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


@dataclass
class Evidence:
    """What one photo tells about every shortlisted card (filled by gather)."""
    tokens: list[str]
    text: str
    digits: list[str]
    kw_hits: dict[str, float]
    negatives: set[str]
    nn: list[RefHit]
    cand_fams: set[str]
    hits: dict[str, RefHit]                  # ref_id -> hit (cosine similarity, family, variant, side)
    by_variant: dict[tuple[str, str], list[str]]
    geo: dict[str, GeoMatch]
    qcolour: tuple
    ms_search: int = 0


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


def geo_score(inl: int, rcfg: RetrievalCfg) -> float:
    """Inliers at or below the baseline are what two cards sharing a template/font produce."""
    return float(np.clip((inl - rcfg.inliers_baseline) / max(1, rcfg.inliers_full_score - rcfg.inliers_baseline), 0, 1))


# ------------------------------------------------------------------------------ evidence per photo
def gather(refs_view: RefView, keywords: KeywordIndex, families: dict[str, CardFamily], ocr: dict,
           qvec: np.ndarray, qkp: Keypoints, qcolour, rcfg: RetrievalCfg) -> Evidence:
    t0 = time.perf_counter()
    tokens, text = ocr["tokens"], ocr["text"]
    nn = refs_view.search(qvec, rcfg.shortlist_refs)                # stage-1 vector shortlist
    kw_hits = keywords.keyword_families(tokens, text)               # stage-1 keyword shortlist
    negatives = keywords.negative_hits(text)
    cand_fams = {h.family_id for h in nn} | set(kw_hits)
    cand_fams = {f for f in cand_fams if f in families}

    # cosine to every reference of the candidate families (one filtered vector query)
    hits = {h.ref_id: h for h in refs_view.family_refs(qvec, cand_fams, rcfg.max_candidate_refs)}

    # ---- choose which references to verify geometrically (bounded cost) ----
    by_variant: dict[tuple[str, str], list[str]] = {}
    by_side: dict[tuple[str, str, str], list[str]] = {}
    for h in hits.values():
        by_variant.setdefault((h.family_id, h.variant_id), []).append(h.ref_id)
        by_side.setdefault((h.family_id, h.variant_id, h.side), []).append(h.ref_id)
    to_verify: list[str] = []
    for refs in by_side.values():                                   # best refs of every side of every variant
        to_verify += sorted(refs, key=lambda i: -hits[i].score)[:rcfg.refs_per_variant_verify]
    to_verify = sorted(to_verify, key=lambda i: -(hits[i].score + (1.0 if hits[i].family_id in kw_hits else 0)))
    to_verify = to_verify[:rcfg.max_verify_refs]
    kps = refs_view.keypoints(to_verify)
    geo = {i: verify(qkp, kps[i], rcfg.ratio_test) for i in to_verify if i in kps}
    digits = re.findall(r"\d{6,}", text.replace(" ", ""))
    return Evidence(tokens, text, digits, kw_hits, negatives, nn, cand_fams, hits, by_variant, geo,
                    tuple(qcolour), int((time.perf_counter() - t0) * 1000))


def _back_colours(ev: Evidence, family_id: str) -> list:
    """Measured colours of non-front references (the definition's body colour describes the front)."""
    return [h.colour for h in ev.hits.values() if h.family_id == family_id and h.side != "front" and h.colour]


def score_families(ev: Evidence, families: dict[str, CardFamily], scfg: ScoringCfg, rcfg: RetrievalCfg,
                   sim_floor: float) -> tuple[list[dict], dict, bool]:
    # No programme keyword readable at all (glare on the logo, unsupported script...) means the
    # text signal is MISSING, not negative: re-weight over visual signals but demand strong geometry.
    text_missing = not ev.kw_hits
    w = dict(scfg.family_weights)
    if text_missing:
        rest = sum(v for k, v in w.items() if k != "text")
        w = {k: (0.0 if k == "text" else v / rest) for k, v in w.items()}
    cands = []
    for fid in ev.cand_fams:
        fam = families[fid]
        refs = [i for (f, _), rs in ev.by_variant.items() if f == fid for i in rs]
        s_text = ev.kw_hits.get(fid, 0.0) * (0.2 if fid in ev.negatives else 1.0)
        s_emb = _norm_sim(max((ev.hits[i].score for i in refs), default=0.0), sim_floor)
        s_geo = geo_score(max((ev.geo[i].inliers for i in refs if i in ev.geo), default=0), rcfg)
        s_col = max([colour_similarity(ev.qcolour, v.body_color_rgb) for v in fam.variants]
                    + [colour_similarity(ev.qcolour, c) for c in _back_colours(ev, fid)])
        score = w["text"] * s_text + w["embed"] * s_emb + w["geometry"] * s_geo + w["colour"] * s_col
        if text_missing and s_geo < scfg.geometry_required_without_text:
            score = min(score, scfg.family_min - 0.01)      # cannot be accepted on looks alone
        cands.append({"id": fid, "score": round(score, 3),
                      "signals": {"text": round(s_text, 3), "embed": round(s_emb, 3),
                                  "geometry": round(s_geo, 3), "colour": round(s_col, 3)}})
    cands.sort(key=lambda c: -c["score"])
    return cands, w, text_missing


def photo_side(ev: Evidence, family_id: str, rcfg: RetrievalCfg) -> str | None:
    """Side of the family's reference this photo matches best by geometry (None = no reliable match)."""
    pool = [i for i in ev.geo if ev.hits[i].family_id == family_id and ev.geo[i].inliers >= rcfg.min_inliers]
    return ev.hits[max(pool, key=lambda i: ev.geo[i].inliers)].side if pool else None


def score_variants(ev: Evidence, fam: CardFamily, scfg: ScoringCfg, rcfg: RetrievalCfg, sim_floor: float) -> list[dict]:
    wv = dict(scfg.variant_weights)
    # The variants' body colours describe the FRONT. A photo of another side (usually the same for every
    # tier) carries no colour evidence: re-weight over the other signals, as for missing text.
    no_colour = photo_side(ev, fam.id, rcfg) not in (None, "front")
    if no_colour:
        rest = sum(x for k, x in wv.items() if k != "colour")
        wv = {k: (0.0 if k == "colour" else x / rest) for k, x in wv.items()}
    cands = []
    for v in fam.variants:
        refs = ev.by_variant.get((fam.id, v.id), [])
        s_col = 0.0 if no_colour else colour_similarity(ev.qcolour, v.body_color_rgb, scale=22.0)
        s_kw = _kw_score(ev.tokens, ev.text, v.keywords) if v.keywords else 0.0
        s_geo = geo_score(max((ev.geo[i].inliers for i in refs if i in ev.geo), default=0), rcfg)
        s_emb = _norm_sim(max((ev.hits[i].score for i in refs), default=0.0), sim_floor)
        s_pre = 1.0 if v.number_prefix and any(d.startswith(v.number_prefix) for d in ev.digits) else 0.0
        score = (wv["colour"] * s_col + wv["keyword"] * s_kw + wv["geometry"] * s_geo
                 + wv["embed"] * s_emb + wv["prefix"] * s_pre)
        cands.append({"id": v.id, "score": round(score, 3),
                      "signals": {"colour": round(s_col, 3), "keyword": round(s_kw, 3),
                                  "geometry": round(s_geo, 3), "embed": round(s_emb, 3), "prefix": s_pre}})
    cands.sort(key=lambda c: -c["score"])
    return cands


def best_alignment(ev: Evidence, family_id: str, variant_id: str | None, side: str | None = None):
    """Reference with the most inliers for aligning this photo: same family (and side), preferring the
    chosen variant. Returns (ref_id, GeoMatch) or (None, None)."""
    pool = [i for i in ev.geo if ev.hits[i].family_id == family_id and (side is None or ev.hits[i].side == side)]
    preferred = [i for i in pool if variant_id is None or ev.hits[i].variant_id == variant_id]
    pool = preferred or (pool if variant_id is None else [])
    if not pool:
        return None, None
    b = max(pool, key=lambda i: ev.geo[i].inliers)
    return b, ev.geo[b]


def trace_info(ev: Evidence, families: dict[str, CardFamily], w: dict, text_missing: bool, margin: float) -> dict:
    best_sim: dict[str, float] = {}
    for h in ev.hits.values():
        best_sim[h.family_id] = max(best_sim.get(h.family_id, 0.0), h.score)
    return dict(
        shortlist=[{"ref_id": h.ref_id, "family": h.family_id, "variant": h.variant_id, "side": h.side,
                    "sim": round(h.score, 4)} for h in ev.nn],
        keyword_hits={k: round(v, 3) for k, v in ev.kw_hits.items()}, negative_hits=sorted(ev.negatives),
        candidate_families=sorted(ev.cand_fams), refs_scored=len(ev.hits), family_best_sim=best_sim,
        verified=[{"ref_id": i, "family": ev.hits[i].family_id, "variant": ev.hits[i].variant_id,
                   "side": ev.hits[i].side, "inliers": g.inliers, "good_matches": g.good_matches}
                  for i, g in sorted(ev.geo.items(), key=lambda x: -x[1].inliers)],
        weights=w, text_missing=text_missing, margin=margin, ms={"search": ev.ms_search})


# ------------------------------------------------------------------------------ single photo (classic API)
def identify(refs_view: RefView, keywords: KeywordIndex, families: dict[str, CardFamily], ocr: dict,
             qvec: np.ndarray, qkp: Keypoints, qcolour, rcfg: RetrievalCfg, scfg: ScoringCfg,
             sim_floor: float, trace: dict | None = None) -> Identification:
    """One photo. trace (optional): filled with the intermediate evidence (shortlist, keyword hits,
    verified references, per-sub-stage timings) for step-by-step views such as the validation stream."""
    ev = gather(refs_view, keywords, families, ocr, qvec, qkp, qcolour, rcfg)
    t = time.perf_counter()
    cands, w, text_missing = score_families(ev, families, scfg, rcfg, sim_floor)
    decided, margin = _decide(cands, scfg.family_min, scfg.family_margin)
    if trace is not None:
        trace.update(trace_info(ev, families, w, text_missing, margin))
        trace["ms"]["family"] = int((time.perf_counter() - t) * 1000)
    res = Identification(decided, cands[0]["id"] if decided else None, cands[0]["score"] if cands else 0.0,
                         margin, cands[:5], refs_verified=len(ev.geo))
    if not decided:
        return res
    fam = families[res.family_id]
    t = time.perf_counter()
    vc = score_variants(ev, fam, scfg, rcfg, sim_floor)
    vdec, vmargin = _decide(vc, scfg.variant_min, scfg.variant_margin)
    if trace is not None:
        trace["variant_margin"] = vmargin
        trace["number_seen"] = ev.digits[0] if ev.digits else None
        trace["ms"]["variant"] = int((time.perf_counter() - t) * 1000)
    res.variant_decided, res.variant_candidates = vdec, vc
    res.variant_id = vc[0]["id"] if vdec else None
    res.variant_score = vc[0]["score"]
    res.best_ref, res.best_match = best_alignment(ev, fam.id, res.variant_id)
    return res


# ------------------------------------------------------------------------------ several photos
def combine(per_photo: list[list[dict]]) -> list[dict]:
    """Best score per candidate over the photos (each photo is evidence about the same card)."""
    best: dict[str, dict] = {}
    for i, cands in enumerate(per_photo):
        for c in cands:
            if c["id"] not in best or c["score"] > best[c["id"]]["score"]:
                best[c["id"]] = {**c, "photo": i}
    return sorted(best.values(), key=lambda c: -c["score"])


def visual_family(ev: Evidence, rcfg: RetrievalCfg) -> tuple[str | None, float]:
    """The family this photo matches best by geometry alone (no text): (family_id, geometry score)."""
    if not ev.geo:
        return None, 0.0
    i = max(ev.geo, key=lambda r: ev.geo[r].inliers)
    return ev.hits[i].family_id, geo_score(ev.geo[i].inliers, rcfg)


def sides_mismatch(evs: list[Evidence], rcfg: RetrievalCfg, threshold: float) -> list[str] | None:
    """Two photos that each match a DIFFERENT card family strongly are not two sides of one card."""
    strong = [f for f, g in (visual_family(e, rcfg) for e in evs) if f and g >= threshold]
    return sorted(set(strong)) if len(set(strong)) > 1 else None


def side_affinity(ev: Evidence, family_id: str, side: str, rcfg: RetrievalCfg, sim_floor: float) -> float:
    """How strongly a photo looks like this side of this card (geometry first, embedding second)."""
    refs = [h for h in ev.hits.values() if h.family_id == family_id and h.side == side]
    if not refs:
        return 0.0
    inl = max((ev.geo[h.ref_id].inliers for h in refs if h.ref_id in ev.geo), default=0)
    g = float(np.clip((inl - rcfg.min_inliers) / max(1, rcfg.inliers_full_score - rcfg.min_inliers), 0, 1))
    return g + 0.3 * _norm_sim(max(h.score for h in refs), sim_floor)


PRIOR_DECLARED, PRIOR_UNDECLARED, PRIOR_OTHER = 0.5, 0.25, -0.5


def assign_sides(evs: list[Evidence | None], declared: list[str | None], fam: CardFamily, rcfg: RetrievalCfg,
                 sim_floor: float) -> tuple[list[str | None], list[dict]]:
    """Which side each usable photo shows (None = not used: unusable, duplicate or extra photo).

    value(photo, side) = affinity to that side - best affinity to any other side + prior, where the
    prior trusts a label (+0.5), slightly favours using an unlabelled photo (+0.25) and resists using a
    photo labelled as the other side (-0.5). Only pairs with a positive value can be used; the
    one-to-one assignment with the highest total wins (ties keep upload order, front first). So a
    label is kept unless the photo's geometry clearly shows the other side (swapped uploads are
    corrected), and a second photo of the same side is left unused instead of being read as the other."""
    sides = fam.side_ids
    usable = [i for i, e in enumerate(evs) if e is not None]
    aff = {(i, s): side_affinity(evs[i], fam.id, s, rcfg, sim_floor) for i in usable for s in sides}

    def value(i, s):
        other = max((aff[(i, o)] for o in sides if o != s), default=0.0)
        prior = PRIOR_UNDECLARED if declared[i] is None else PRIOR_DECLARED if declared[i] == s else PRIOR_OTHER
        return aff[(i, s)] - other + prior

    vals = {(i, s): value(i, s) for i in usable for s in sides}
    best, best_total = {}, 0.0
    for k in range(1, min(len(usable), len(sides)) + 1):
        for chosen in itertools.permutations(sides, k):
            for photos in itertools.permutations(usable, k):
                pairs = list(zip(photos, chosen))
                if any(vals[p] <= 0 for p in pairs):
                    continue
                total = sum(vals[p] for p in pairs)
                total -= 1e-6 * sum(abs(usable.index(i) - sides.index(s)) for i, s in pairs)   # order tie-break
                if total > best_total + 1e-9:
                    best, best_total = dict(pairs), total
    out = [best.get(i) for i in range(len(evs))]
    details = [{"photo": i, "affinity": {s: round(aff[(i, s)], 3) for s in sides},
                "value": {s: round(vals[(i, s)], 3) for s in sides}} for i in usable]
    return out, details
