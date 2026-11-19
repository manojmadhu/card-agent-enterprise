"""Evaluation gate: run the agent over a labelled manifest and report the numbers that decide
whether a registry/model/threshold change may ship. Exits non-zero when the gate fails (use in CI).

Manifest format (JSON list): {"file", "family", "variant", "expected_status", "fields": {...}}

  python tools/evaluate.py --manifest data/samples/manifest.json --today 2026-11-18
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cardagent.agent import CardAgent  # noqa: E402
from cardagent.settings import get_settings  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data/samples/manifest.json"))
    ap.add_argument("--today", default=None, help="fix 'today' for reproducible expiry checks (YYYY-MM-DD)")
    ap.add_argument("--min-accept-precision", type=float, default=0.99)
    ap.add_argument("--min-status-agreement", type=float, default=0.90)
    ap.add_argument("--out", default=str(ROOT / "data/eval_report.json"))
    a = ap.parse_args()

    today = date.fromisoformat(a.today) if a.today else None
    manifest_path = Path(a.manifest)
    cases = json.loads(manifest_path.read_text())
    agent = CardAgent(get_settings())

    rows, conf_family, conf_variant = [], defaultdict(Counter), defaultdict(Counter)
    accepted = accepted_correct = status_ok = 0
    field_tot, field_ok = Counter(), Counter()
    latencies = []
    for c in cases:
        r = agent.process((manifest_path.parent / c["file"]).read_bytes(), c["file"], today)
        card = r["card"] or {}
        pf, pv = card.get("family"), card.get("variant")
        conf_family[str(c["family"])][str(pf)] += 1
        if c["family"] and "RETAKE_REQUIRED" not in r["reasons"]:
            conf_variant[f"{c['family']}/{c['variant']}"][f"{pf}/{pv}"] += 1
        correct_id = pf == c["family"] and pv == c["variant"]
        if r["status"] == "ACCEPT":
            accepted += 1
            fields_ok = all((r["fields"].get(k) or {}).get("value") == v for k, v in c.get("fields", {}).items()
                            if k in r["fields"])
            accepted_correct += int(correct_id and fields_ok)
        status_ok += int(r["status"] == c["expected_status"])
        for k, v in c.get("fields", {}).items():
            if k in r["fields"]:
                field_tot[k] += 1
                field_ok[k] += int(r["fields"][k]["value"] == v)
        latencies.append(r["processing_ms"])
        rows.append({"file": c["file"], "expected": c["expected_status"], "got": r["status"],
                     "family": pf, "variant": pv, "id_correct": (correct_id if c["family"] and "RETAKE_REQUIRED" not in r["reasons"] else None),
                     "reasons": r["reasons"], "ms": r["processing_ms"], "timings": r["timings_ms"]})

    n = len(cases)
    lat = sorted(latencies)
    report = {
        "cases": n,
        "status_agreement": round(status_ok / n, 4),
        "automation_rate": round(accepted / n, 4),
        "accept_precision": round(accepted_correct / accepted, 4) if accepted else None,
        "field_accuracy": {k: round(field_ok[k] / field_tot[k], 4) for k in field_tot},
        "latency_ms": {"p50": lat[n // 2], "p95": lat[min(n - 1, int(n * .95))], "max": lat[-1]},
        "family_confusion": {k: dict(v) for k, v in conf_family.items()},
        "variant_errors": {k: dict(v) for k, v in conf_variant.items() if set(v) != {k}},
        "registry_version": agent.state.snapshot.version, "embedder": agent.embedder.name,
        "rows": rows,
    }
    Path(a.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"{'file':38} {'expected':8} {'got':8} {'family':12} {'variant':10} {'ms':>5}  reasons")
    for r in rows:
        mark = "OK " if r["expected"] == r["got"] and r["id_correct"] in (True, None) else "XX "
        print(f"{mark}{r['file'][:35]:35} {r['expected']:8} {r['got']:8} {str(r['family']):12} "
              f"{str(r['variant']):10} {r['ms']:5}  {', '.join(r['reasons'])[:60]}")
    print(f"\nstatus agreement {report['status_agreement']:.1%} | automation {report['automation_rate']:.1%} | "
          f"ACCEPT precision {report['accept_precision']} | p50 {report['latency_ms']['p50']} ms "
          f"p95 {report['latency_ms']['p95']} ms")
    print("field accuracy:", report["field_accuracy"])
    if report["variant_errors"]:
        print("identification errors:", report["variant_errors"])

    gate_ok = (report["accept_precision"] is None or report["accept_precision"] >= a.min_accept_precision) \
        and report["status_agreement"] >= a.min_status_agreement
    print("GATE:", "PASS" if gate_ok else "FAIL")
    sys.exit(0 if gate_ok else 1)


if __name__ == "__main__":
    main()
