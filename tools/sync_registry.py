"""Validate the file registry and import it into Qdrant (CI / deploy step).

Git-managed cards live in registry/cards/<country>/<family>.yaml with reference images in
registry/refs/<family_id>/<variant_id>/*.png|jpg. Cards can also be managed entirely through the API
(PUT /v1/cards/{id} + reference uploads); then this tool is not needed. The sync is incremental:
only new, changed or removed files are processed, and API-owned cards are never touched.

  python tools/sync_registry.py                  # validate, then sync (deletes cards removed from registry/)
  python tools/sync_registry.py --validate-only  # schema + missing-reference check, no Qdrant needed
  python tools/sync_registry.py --no-prune       # keep cards whose files were removed
  python tools/sync_registry.py --reembed        # after changing the embedder: re-embed every reference
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--validate-only", action="store_true", help="validate registry/ only, do not touch Qdrant")
    ap.add_argument("--no-prune", action="store_true", help="do not delete cards removed from registry/")
    ap.add_argument("--reembed", action="store_true", help="re-embed every reference (the embedder changed)")
    ap.add_argument("--allow-missing-refs", action="store_true",
                    help="do not fail when a variant has no reference image in registry/refs")
    a = ap.parse_args()

    from cardagent.registry.store import load_registry
    from cardagent.settings import PROJECT_ROOT, get_settings

    s = get_settings()
    reg = s.path(s.registry_dir)
    snap = load_registry(reg)                                  # raises on any invalid YAML
    missing = [f"{f.id}/{v.id}" for f in snap.families.values() for v in f.variants
               if not any(r.family_id == f.id and r.variant_id == v.id for r in snap.refs)]
    if missing and not a.allow_missing_refs:
        print("ERROR: variants without reference images (add them to registry/refs or upload them "
              "through the API and pass --allow-missing-refs):", ", ".join(missing))
        sys.exit(2)
    if a.validate_only:
        print(json.dumps({"registry": "valid", "registry_version": snap.version, "families": len(snap.families),
                          "references": len(snap.refs), "variants_without_references": missing}, indent=2))
        return

    from cardagent.embed.embedders import build_embedder
    from cardagent.store.qdrant_store import QdrantCardStore
    from cardagent.store.sync import sync_registry

    q = s.qdrant
    try:
        store = QdrantCardStore(q, build_embedder(s.embedder, PROJECT_ROOT), s.retrieval.orb_features, PROJECT_ROOT,
                                allow_embedder_change=a.reembed)
    except RuntimeError as e:                                  # embedder changed since the last sync
        print(f"ERROR: {e}")
        sys.exit(3)
    out = sync_registry(store, reg, prune=not a.no_prune, reembed=a.reembed)
    print(json.dumps({"qdrant": q.location or q.path or q.url, **out}, indent=2))


if __name__ == "__main__":
    main()
