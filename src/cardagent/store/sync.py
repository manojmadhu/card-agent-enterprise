"""Incremental sync: make Qdrant match the file registry (registry/cards/*.yaml + registry/refs/).

Only differences are applied:
  * new/changed YAML      -> family definition upserted (ROI change re-extracts that family's keypoints)
  * new reference image   -> embedded and upserted          (unchanged images are never re-embedded)
  * removed image / YAML  -> its points deleted              (prune; registry-managed data only)
Ownership: every family has a source. Cards created through the admin API (source=api), or taken
over with ?force=true, are never modified or pruned by a registry sync - even if a YAML file with
the same id still exists (it is reported under "skipped_api_owned").
"""
from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

import cv2

from ..registry.models import CardFamily
from ..registry.store import load_registry
from .base import prepare_reference
from .qdrant_store import QdrantCardStore, ref_point_id

log = logging.getLogger(__name__)


def sync_registry(store: QdrantCardStore, registry_root: Path, prune: bool = True, reembed: bool = False) -> dict:
    t0 = time.time()
    snap = load_registry(registry_root)                       # validates every YAML first; raises on error
    st = {"families_created": 0, "families_updated": 0, "families_deleted": 0, "references_added": 0,
          "references_deleted": 0, "references_unchanged": 0, "keypoints_refreshed": 0, "api_references_reembedded": 0}

    reingest = []
    if reembed:
        reingest = [r for r in store.reset_refs() if r.get("source") == "api"]

    existing = {rec.payload["family_id"]: rec.payload
                for rec in store._scroll(store.fams_col, store._eq("kind", "family"), ["family_id", "source"])}
    api_owned = sorted(f for f in snap.families if existing.get(f, {}).get("source") == "api")
    managed = {fid: fam for fid, fam in snap.families.items() if fid not in api_owned}

    # ---- family definitions
    for fam in managed.values():
        r = store.put_family(fam, "registry", bump=False)
        st["families_created"] += r["created"]
        st["families_updated"] += r["changed"] and not r["created"]
        st["references_deleted"] += r["references_deleted"]
        st["keypoints_refreshed"] += r["keypoints_refreshed"]

    # ---- reference images: diff by content hash
    desired: dict[str, tuple[CardFamily, str, Path, str, str]] = {}
    for ref in snap.refs:
        if ref.family_id not in managed:
            continue
        sha = hashlib.sha256(ref.path.read_bytes()).hexdigest()
        desired[ref_point_id(ref.family_id, ref.variant_id, sha, ref.side)] = (managed[ref.family_id], ref.variant_id,
                                                                              ref.path, sha, ref.side)
    current = {r["ref_id"]: r for r in store.list_refs() if r.get("source") == "registry"}
    to_add = [pid for pid in desired if pid not in current]
    to_delete = [pid for pid, r in current.items() if pid not in desired and r["family"] in managed]
    st["references_unchanged"] = len(desired) - len(to_add)

    batch = []
    for pid in to_add:
        fam, vid, path, sha, side = desired[pid]
        img = cv2.imread(str(path))
        if img is None:
            log.warning("unreadable reference image %s (skipped)", path)
            continue
        prep = prepare_reference(img, fam, store.embedder, store.n_features, side)
        batch.append((pid, fam, vid, sha, path.name, path, "registry", prep, side))
        if len(batch) >= 64:
            store.add_prepared(batch)
            st["references_added"] += len(batch)
            batch = []
    if batch:
        store.add_prepared(batch)
        st["references_added"] += len(batch)
    st["references_deleted"] += store.delete_refs_by_ids(to_delete)

    # ---- families removed from the registry
    if prune:
        for fid, p in existing.items():
            if p.get("source") == "registry" and fid not in snap.families:
                r = store.delete_family(fid, bump=False)
                st["families_deleted"] += 1
                st["references_deleted"] += r["references_deleted"] if r else 0

    # ---- after --reembed: API-uploaded references are re-embedded from their stored copies
    for r in reingest:
        doc = store.get_family(r["family"])
        path = store._abs(r["image_path"])
        if doc is None or not path.exists():
            log.warning("cannot re-embed API reference %s (family or stored image missing)", r["ref_id"])
            continue
        fam = CardFamily.model_validate(doc["definition"])
        side = r.get("side") or "front"
        if side not in fam.side_ids:
            continue
        store.add_reference(fam, r["variant"], path.read_bytes(), r["filename"], "api", image_path=path, bump=False,
                            side=side)
        st["api_references_reembedded"] += 1

    changed = any(v for k, v in st.items() if k != "references_unchanged")
    version = store.bump_revision() if changed or reembed else store.current_version()
    counts = store.ref_counts()
    return {**st, "changed": bool(changed), "skipped_api_owned": api_owned,
            "registry_version": version, "families": len(snap.families),
            "references_total": sum(counts.values()), "embedder": store.embedder.name,
            "seconds": round(time.time() - t0, 1)}
