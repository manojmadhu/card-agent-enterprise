"""FAISS backend index = everything precomputed from the registry's reference images:
  * FAISS inner-product index over L2-normalised embeddings (cosine similarity)
  * ORB keypoints/descriptors per reference (for RANSAC verification + alignment)
  * body colour per reference

Persisted under <index_dir>/<registry_version>__<embedder>/ so restarts are instant and
an index can never be used with the wrong registry version. (The keyword index lives in
store/keywords.py and is shared by both backends.)
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import faiss
import numpy as np

from ..embed.embedders import Embedder
from ..registry.store import RegistrySnapshot
from ..vision.features import Keypoints, body_colour_rgb, design_mask, extract_keypoints
from ..vision.geometry import CARD_H, CARD_W

log = logging.getLogger(__name__)


@dataclass
class RefEntry:
    family_id: str
    variant_id: str
    path: str
    colour: tuple[int, int, int]
    kp: Keypoints


@dataclass
class ReferenceIndex:
    registry_version: str
    embedder_name: str
    entries: list[RefEntry]
    faiss_index: faiss.Index
    vectors: np.ndarray = None        # (N, D) copy for exact re-scoring of shortlisted families

    def search(self, vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        k = min(k, len(self.entries))
        if k == 0:
            return []
        sims, ids = self.faiss_index.search(vec.reshape(1, -1).astype(np.float32), k)
        return [(int(i), float(s)) for i, s in zip(ids[0], sims[0]) if i >= 0]


INDEX_FORMAT = "f2"   # bump when indexing logic changes so stale caches are never reused


def _dir(index_root: Path, version: str, embedder: Embedder) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", embedder.name)
    return index_root / f"{version}__{safe}__{INDEX_FORMAT}"


def build_or_load(snapshot: RegistrySnapshot, embedder: Embedder, index_root: Path,
                  n_features: int = 1000, force: bool = False) -> ReferenceIndex:
    d = _dir(index_root, snapshot.version, embedder)
    if not force and (d / "meta.json").exists():
        try:
            return _load(d, snapshot, embedder)
        except Exception as e:  # corrupt cache -> rebuild
            log.warning("index cache unreadable (%s); rebuilding", e)
    return _build(d, snapshot, embedder, n_features)


def _build(d: Path, snapshot, embedder, n_features) -> ReferenceIndex:
    t0 = time.time()
    entries, vecs = [], []
    for r in snapshot.refs:
        img = cv2.imread(str(r.path))
        if img is None:
            log.warning("unreadable reference image %s (skipped)", r.path)
            continue
        img = cv2.resize(img, (CARD_W, CARD_H), interpolation=cv2.INTER_AREA)
        vecs.append(embedder.embed(img))
        mask = design_mask([f.roi for f in snapshot.families[r.family_id].fields], CARD_W, CARD_H)
        entries.append(RefEntry(r.family_id, r.variant_id, str(r.path), body_colour_rgb(img),
                                extract_keypoints(img, n_features, mask)))
    dim = len(vecs[0]) if vecs else embedder.dim
    index = faiss.IndexFlatIP(dim)   # exact search; switch to IndexHNSWFlat beyond ~100k refs
    if vecs:
        index.add(np.stack(vecs).astype(np.float32))

    d.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(d / "vectors.faiss"))
    np.savez_compressed(d / "keypoints.npz",
                        **{f"p{i}": e.kp.pts for i, e in enumerate(entries)},
                        **{f"d{i}": (e.kp.des if e.kp.des is not None else np.zeros((0, 32), np.uint8))
                           for i, e in enumerate(entries)})
    (d / "meta.json").write_text(json.dumps({
        "registry_version": snapshot.version, "embedder": embedder.name, "dim": dim,
        "entries": [{"family": e.family_id, "variant": e.variant_id, "path": e.path, "colour": e.colour}
                    for e in entries]}, indent=1), encoding="utf-8")
    log.info("built reference index: %d refs, dim=%d in %.1fs -> %s", len(entries), dim, time.time() - t0, d)
    vmat = np.stack(vecs).astype(np.float32) if vecs else np.zeros((0, dim), np.float32)
    return ReferenceIndex(snapshot.version, embedder.name, entries, index, vmat)


def _load(d: Path, snapshot, embedder) -> ReferenceIndex:
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    if meta["registry_version"] != snapshot.version or meta["embedder"] != embedder.name:
        raise ValueError("index metadata does not match registry/embedder")
    kp = np.load(d / "keypoints.npz")
    entries = []
    for i, m in enumerate(meta["entries"]):
        des = kp[f"d{i}"]
        entries.append(RefEntry(m["family"], m["variant"], m["path"], tuple(m["colour"]),
                                Keypoints(kp[f"p{i}"], des if len(des) else None)))
    index = faiss.read_index(str(d / "vectors.faiss"))
    vmat = index.reconstruct_n(0, index.ntotal) if index.ntotal else np.zeros((0, index.d), np.float32)
    log.info("loaded reference index from %s (%d refs)", d, len(entries))
    return ReferenceIndex(snapshot.version, embedder.name, entries, index, vmat)
