"""Validate the registry and bring the vector store up to date with it (CI / deploy step).

  python tools/build_index.py              # validate YAML, then:
                                           #   qdrant: incremental sync (only new/changed/removed cards)
                                           #   faiss:  build the file index if missing
  python tools/build_index.py --force      # faiss: rebuild even if a cached index exists
  python tools/build_index.py --reembed    # qdrant: new embedder -> recreate vectors for every reference
  python tools/build_index.py --no-prune   # qdrant: keep cards whose YAML/images were removed from registry/
  python tools/build_index.py --backend faiss   # override vector_store.backend from config
  python tools/build_index.py --validate-only   # schema + missing-reference check only (docker build / CI)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="faiss: rebuild the index from scratch")
    ap.add_argument("--reembed", action="store_true", help="qdrant: re-embed every reference (embedder changed)")
    ap.add_argument("--no-prune", action="store_true", help="qdrant: do not delete cards removed from registry/")
    ap.add_argument("--backend", choices=["faiss", "qdrant"], help="override vector_store.backend")
    ap.add_argument("--validate-only", action="store_true", help="validate the registry, touch no index")
    a = ap.parse_args()
    if a.backend:
        os.environ["CARDAGENT_VECTOR_STORE__BACKEND"] = a.backend

    from cardagent.embed.embedders import build_embedder
    from cardagent.registry.store import load_registry
    from cardagent.settings import PROJECT_ROOT, get_settings
    from cardagent.store.factory import build_store

    s = get_settings()
    reg = s.path(s.registry_dir)
    snap = load_registry(reg)                                  # raises on any invalid YAML
    missing = [f"{f.id}/{v.id}" for f in snap.families.values() for v in f.variants
               if not any(r.family_id == f.id and r.variant_id == v.id for r in snap.refs)]
    if missing:
        print("ERROR: variants without reference images:", ", ".join(missing))
        sys.exit(2)
    if a.validate_only:
        print(json.dumps({"registry": "valid", "registry_version": snap.version, "families": len(snap.families),
                          "references": len(snap.refs)}, indent=2))
        return

    embedder = build_embedder(s.embedder, PROJECT_ROOT)
    if s.vector_store.backend == "qdrant":
        from cardagent.store.qdrant_store import QdrantCardStore
        from cardagent.store.sync import sync_registry
        q = s.vector_store.qdrant
        try:
            store = QdrantCardStore(q, embedder, s.retrieval.orb_features, PROJECT_ROOT,
                                    allow_embedder_change=a.reembed)
        except RuntimeError as e:                              # embedder changed since the last sync
            print(f"ERROR: {e}")
            sys.exit(3)
        out = sync_registry(store, reg, prune=not a.no_prune, reembed=a.reembed)
        print(json.dumps({"vector_store": "qdrant", "url": q.location or q.path or q.url, **out}, indent=2))
    else:
        st = build_store(s, embedder)
        snapshot = st.snapshot(rebuild=a.force)
        print(json.dumps({"vector_store": "faiss", "registry_version": snapshot.version,
                          "families": len(snapshot.families), "references": snapshot.ref_count,
                          "embedder": embedder.name}, indent=2))


if __name__ == "__main__":
    main()
