"""FAISS backend: the original file-based index, rebuilt from the on-disk registry when it changes.
Kept for development and single-process use; card management through the API needs Qdrant."""
from __future__ import annotations

import numpy as np

from ..index.reference_index import ReferenceIndex, build_or_load
from ..registry.store import load_registry
from ..vision.features import Keypoints
from .base import RefHit, StoreSnapshot
from .keywords import KeywordIndex


class FaissRefView:
    """Identification read side over one immutable ReferenceIndex (ref_id = row number)."""

    def __init__(self, idx: ReferenceIndex):
        self.idx = idx

    def _hit(self, i: int, s: float) -> RefHit:
        e = self.idx.entries[i]
        return RefHit(str(i), e.family_id, e.variant_id, float(s))

    def search(self, qvec: np.ndarray, k: int) -> list[RefHit]:
        return [self._hit(i, s) for i, s in self.idx.search(qvec, k)]

    def family_refs(self, qvec: np.ndarray, family_ids: set[str], limit: int) -> list[RefHit]:
        rows = [i for i, e in enumerate(self.idx.entries) if e.family_id in family_ids]
        if not rows:
            return []
        sims = self.idx.vectors[rows] @ qvec
        hits = [self._hit(i, s) for i, s in zip(rows, sims)]
        return hits if len(hits) <= limit else sorted(hits, key=lambda h: -h.score)[:limit]

    def keypoints(self, ref_ids: list[str]) -> dict[str, Keypoints]:
        return {r: self.idx.entries[int(r)].kp for r in ref_ids}


class FaissCardStore:
    backend = "faiss"
    supports_admin = False

    def __init__(self, settings, embedder):
        self.s, self.embedder = settings, embedder
        self._version = None

    def snapshot(self, rebuild: bool = False) -> StoreSnapshot:
        snap = load_registry(self.s.path(self.s.registry_dir))
        idx = build_or_load(snap, self.embedder, self.s.path(self.s.index_dir),
                            self.s.retrieval.orb_features, force=rebuild)
        if self.s.strict_index_version and idx.registry_version != snap.version:
            raise RuntimeError("index/registry version mismatch")
        counts: dict[tuple[str, str], int] = {}
        for e in idx.entries:
            counts[(e.family_id, e.variant_id)] = counts.get((e.family_id, e.variant_id), 0) + 1
        self._version = snap.version
        return StoreSnapshot(snap.version, snap.families, KeywordIndex(snap.families), FaissRefView(idx), counts)

    def current_version(self) -> str:
        return self._version or "unloaded"      # file registry changes are picked up by an explicit reload

    def healthy(self) -> bool:
        return True
