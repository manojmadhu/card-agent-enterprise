"""Evaluation: run the agent over a labelled set and compute the numbers that decide whether a card,
threshold or model change may ship. Used by POST /v1/evaluations and tools/evaluate.py.

Manifest (manifest.json, a JSON list), paths relative to the manifest:
  {"file": "photos/IMG_0012.jpg", "family": "ae.fazaa", "variant": "gold",
   "expected_status": "ACCEPT", "fields": {"card_number": "2210457783"}}
Two-sided cards: "files" instead of "file", either by side or as a list (sides detected):
  {"files": {"front": "photos/0012_f.jpg", "back": "photos/0012_b.jpg"}, ...}
  {"files": ["photos/0012_1.jpg", "photos/0012_2.jpg"], ...}
family/variant null = a card that must not be identified (other programme, non-card).
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Callable

STATUSES = {"ACCEPT", "REVIEW", "REJECT"}


class ManifestError(ValueError):
    pass


def load_manifest(path: Path) -> list[dict]:
    try:
        cases = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ManifestError(f"cannot read {path.name}: {e}") from e
    if not isinstance(cases, list) or not cases:
        raise ManifestError("manifest.json must be a non-empty JSON list")
    base = path.parent.resolve()
    for i, c in enumerate(cases):
        if not isinstance(c, dict) or ("file" not in c and "files" not in c) or "expected_status" not in c:
            raise ManifestError(f"case {i}: needs 'file' (or 'files') and 'expected_status'")
        if c["expected_status"] not in STATUSES:
            raise ManifestError(f"case {i}: expected_status must be one of {sorted(STATUSES)}")
        c["photos"] = _case_photos(c, i)
        for _, rel in c["photos"]:
            f = (base / rel).resolve()
            if base not in f.parents or not f.is_file():
                raise ManifestError(f"case {i}: image '{rel}' not found next to manifest.json")
        c.setdefault("file", c["photos"][0][1])
        c.setdefault("family", None)
        c.setdefault("variant", None)
        c.setdefault("fields", {})
    return cases


def _case_photos(c: dict, i: int) -> list[tuple[str | None, str]]:
    """[(declared side or None, relative path)] of a manifest case."""
    if "files" not in c:
        return [(None, c["file"])]
    fs = c["files"]
    if isinstance(fs, dict) and fs and set(fs) <= {"front", "back"}:
        return [(s, fs[s]) for s in ("front", "back") if s in fs]
    if isinstance(fs, list) and 1 <= len(fs) <= 2 and all(isinstance(x, str) for x in fs):
        return [(None, x) for x in fs]
    raise ManifestError(f"case {i}: 'files' must be {{\"front\": path, \"back\": path}} or a list of 1-2 paths")


def evaluate(agent, cases: list[dict], base_dir: Path, today: date | None = None, settings=None,
             gate: tuple[float, float] = (0.99, 0.90), progress: Callable[[int, int], None] | None = None) -> dict:
    """settings: optional Settings for a what-if run (the live settings are not changed)."""
    rows, conf_family, conf_variant = [], defaultdict(Counter), defaultdict(Counter)
    accepted = accepted_correct = status_ok = 0
    field_tot, field_ok = Counter(), Counter()
    latencies = []
    for i, c in enumerate(cases):
        from .agent import Photo
        photos = [Photo((base_dir / rel).read_bytes(), side, side or ("image" if len(c["photos"]) == 1 else "images"))
                  for side, rel in c.get("photos") or [(None, c["file"])]]
        r = agent.process(request_id=f"eval:{c['file']}", today=today, settings=settings, photos=photos)
        card = r["card"] or {}
        pf, pv = card.get("family"), card.get("variant")
        retake = "RETAKE_REQUIRED" in r["reasons"] or any(x.startswith("SIDE_REQUIRED") for x in r["reasons"])
        conf_family[str(c["family"])][str(pf)] += 1
        if c["family"] and not retake:
            conf_variant[f"{c['family']}/{c['variant']}"][f"{pf}/{pv}"] += 1
        correct_id = pf == c["family"] and pv == c["variant"]
        if r["status"] == "ACCEPT":
            accepted += 1
            fields_ok = all((r["fields"].get(k) or {}).get("value") == v for k, v in c["fields"].items() if k in r["fields"])
            accepted_correct += int(correct_id and fields_ok)
        status_ok += int(r["status"] == c["expected_status"])
        for k, v in c["fields"].items():
            if k in r["fields"]:
                field_tot[k] += 1
                field_ok[k] += int(r["fields"][k]["value"] == v)
        latencies.append(r["processing_ms"])
        rows.append({"file": c["file"], **({"files": c["files"]} if "files" in c else {}),
                     "expected": c["expected_status"], "got": r["status"],
                     "expected_family": c["family"], "expected_variant": c["variant"], "family": pf, "variant": pv,
                     "id_correct": correct_id if c["family"] and not retake else None,
                     "reasons": r["reasons"], "ms": r["processing_ms"], "timings": r["timings_ms"]})
        if progress:
            progress(i + 1, len(cases))

    n = len(cases)
    lat = sorted(latencies)
    precision = round(accepted_correct / accepted, 4) if accepted else None
    agreement = round(status_ok / n, 4)
    min_precision, min_agreement = gate
    return {
        "cases": n,
        "status_agreement": agreement,
        "automation_rate": round(accepted / n, 4),
        "accept_precision": precision,
        "wrong_accepts": accepted - accepted_correct,
        "field_accuracy": {k: round(field_ok[k] / field_tot[k], 4) for k in field_tot},
        "latency_ms": {"p50": lat[n // 2], "p95": lat[min(n - 1, int(n * .95))], "max": lat[-1]},
        "family_confusion": {k: dict(v) for k, v in conf_family.items()},
        "identification_errors": {k: dict(v) for k, v in conf_variant.items() if set(v) != {k}},
        "gate": {"min_accept_precision": min_precision, "min_status_agreement": min_agreement,
                 "passed": (precision is None or precision >= min_precision) and agreement >= min_agreement},
        "registry_version": agent.state.version, "embedder": agent.embedder.name,
        "rows": rows,
    }
