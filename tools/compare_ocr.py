"""Compare OCR engines on YOUR labelled photos: same cards, same settings, only the OCR changes.

Runs the full pipeline once per engine over an evaluation manifest (the same format as
tools/evaluate.py and POST /v1/evaluation-sets) against the configured Qdrant, then prints decision
agreement, ACCEPT precision, automation rate, accuracy per field and latency side by side.

  python tools/compare_ocr.py --manifest data/eval/manifest.json --today 2026-11-27
  python tools/compare_ocr.py --manifest ... --engines paddle tesseract --out data/ocr_compare.json

Both engines must be installed (Tesseract binary for "tesseract"; the Arabic model for Arabic text
with "paddle": python tools/download_ocr_models.py).
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

KEYS = ("cases", "status_agreement", "accept_precision", "wrong_accepts", "automation_rate")


def run(engine: str, cases, base, today, gate):
    s = get_settings().model_copy(deep=True)
    s.ocr.engine = engine
    s.qdrant.sync_on_startup = False
    agent = CardAgent(s)
    try:
        return evaluate(agent, cases, base, today, gate=gate)
    finally:
        try:
            agent.store.client.close()
        except Exception:  # noqa: BLE001
            pass


def main():
    s = get_settings()
    ap = argparse.ArgumentParser(description="compare OCR engines on a labelled set")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--engines", nargs="+", default=["paddle", "tesseract"], choices=["paddle", "tesseract"])
    ap.add_argument("--today", default=None, help="YYYY-MM-DD for reproducible expiry checks")
    ap.add_argument("--out", default=str(ROOT / "data/ocr_compare.json"))
    a = ap.parse_args()
    manifest = Path(a.manifest)
    today = date.fromisoformat(a.today) if a.today else None
    gate = (s.evaluation.min_accept_precision, s.evaluation.min_status_agreement)
    reports = {}
    for eng in a.engines:
        print(f"running {eng} ...", flush=True)
        reports[eng] = run(eng, load_manifest(manifest), manifest.parent, today, gate)

    fields = sorted({f for r in reports.values() for f in r["field_accuracy"]})
    w = max(22, *(len(f) + 9 for f in fields))
    print("\n" + "".ljust(w) + "".join(e.rjust(14) for e in a.engines))
    for k in KEYS:
        print(k.ljust(w) + "".join(str(reports[e][k]).rjust(14) for e in a.engines))
    for f in fields:
        print(f"field {f}".ljust(w) + "".join(str(reports[e]["field_accuracy"].get(f, "-")).rjust(14) for e in a.engines))
    for k in ("p50", "p95"):
        print(f"latency {k} ms".ljust(w) + "".join(str(reports[e]["latency_ms"][k]).rjust(14) for e in a.engines))
    print("gate passed".ljust(w) + "".join(str(reports[e]["gate"]["passed"]).rjust(14) for e in a.engines))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nfull reports: {a.out}")


if __name__ == "__main__":
    main()
