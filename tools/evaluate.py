"""Evaluation gate from the command line (CI): run a labelled set through the pipeline and fail when
ACCEPT precision or status agreement is below the gate. The same report is available through the API
(POST /v1/evaluation-sets + POST /v1/evaluations), which also supports what-if settings.

Manifest (JSON list, photos next to it):
  [{"file": "photos/001.jpg", "family": "ae.fazaa", "variant": "gold", "expected_status": "ACCEPT",
    "fields": {"card_number": "1234567890"}}, ...]
  family/variant null + expected_status REJECT for photos that are not a known card.

  python tools/evaluate.py --manifest data/eval/manifest.json --today 2026-11-24
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cardagent.agent import CardAgent  # noqa: E402
from cardagent.evaluation import evaluate, load_manifest  # noqa: E402
from cardagent.settings import get_settings  # noqa: E402


def main():
    s = get_settings()
    ap = argparse.ArgumentParser(description="evaluation gate")
    ap.add_argument("--manifest", required=True, help="manifest.json of the labelled set")
    ap.add_argument("--today", default=None, help="fix 'today' for reproducible expiry checks (YYYY-MM-DD)")
    ap.add_argument("--min-accept-precision", type=float, default=s.evaluation.min_accept_precision)
    ap.add_argument("--min-status-agreement", type=float, default=s.evaluation.min_status_agreement)
    ap.add_argument("--out", default=str(ROOT / "data/eval_report.json"))
    a = ap.parse_args()

    manifest = Path(a.manifest)
    cases = load_manifest(manifest)
    agent = CardAgent(s)
    report = evaluate(agent, cases, manifest.parent, date.fromisoformat(a.today) if a.today else None,
                      gate=(a.min_accept_precision, a.min_status_agreement))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2, ensure_ascii=False))
    print(f"\nfull report: {a.out}")
    print("GATE PASSED" if report["gate"]["passed"] else "GATE FAILED")
    sys.exit(0 if report["gate"]["passed"] else 1)


if __name__ == "__main__":
    main()
