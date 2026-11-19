"""Validate the registry and (re)build the reference index ahead of deployment (CI step).

  python tools/build_index.py            # validate + build (fails on any invalid YAML)
  python tools/build_index.py --force    # rebuild even if a cached index exists
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cardagent.embed.embedders import build_embedder  # noqa: E402
from cardagent.index.reference_index import build_or_load  # noqa: E402
from cardagent.registry.store import load_registry  # noqa: E402
from cardagent.settings import PROJECT_ROOT, get_settings  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    s = get_settings()
    snap = load_registry(s.path(s.registry_dir))
    missing = [f"{f.id}/{v.id}" for f in snap.families.values() for v in f.variants
               if not any(r.family_id == f.id and r.variant_id == v.id for r in snap.refs)]
    if missing:
        print("ERROR: variants without reference images:", ", ".join(missing))
        sys.exit(2)
    idx = build_or_load(snap, build_embedder(s.embedder, PROJECT_ROOT), s.path(s.index_dir),
                        s.retrieval.orb_features, force=a.force)
    print(json.dumps({"registry_version": snap.version, "families": len(snap.families),
                      "references": len(idx.entries), "embedder": idx.embedder_name}, indent=2))


if __name__ == "__main__":
    main()
