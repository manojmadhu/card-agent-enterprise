"""Qdrant-backed card store.

Collections (prefix configurable, default "cardagent"):
  <prefix>_refs      one point per reference image
                     vector  = image embedding (cosine)
                     payload = family, variant, variant_key, country, sha256, filename, image_path,
                               colour, source, added_at, orb (base64 ORB keypoints, stored on disk)
  <prefix>_families  no vectors: one point per card family (payload = validated definition)
                     + one "meta" point: revision, embedder name, vector dim

Point ids are content addressed (uuid5 of family/variant/sha256), so re-adding the same image is a
no-op and adding/removing a card never touches any other card's vectors. Every write bumps the
meta revision; API replicas poll it and reload family definitions when it changes.
"""
from __future__ import annotations

import functools
import logging
import threading
import uuid
import warnings
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from ..registry.models import CardFamily
from ..settings import QdrantCfg
from ..vision import normalise as normalise_mod
from ..vision.features import Keypoints
from .base import (AdminUnsupported, PreparedRef, RefHit, StoreSnapshot, StoreUnavailable,  # noqa: F401
                   definition_hash, keypoints_from_b64, keypoints_to_b64, prepare_reference,
                   reference_keypoints, rois_signature)
from .keywords import KeywordIndex

log = logging.getLogger(__name__)

_NS = uuid.UUID("6f1c3f7e-2b7a-4c55-9d0e-6a1f2c9b8e01")
META_ID = str(uuid.uuid5(_NS, "__meta__"))
REF_PAYLOAD = ["family", "variant", "variant_key", "country", "sha256", "filename", "image_path",
               "colour", "source", "added_at"]
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp"}


def family_point_id(family_id: str) -> str:
    return str(uuid.uuid5(_NS, f"family:{family_id}"))


def ref_point_id(family_id: str, variant_id: str, sha256: str) -> str:
    return str(uuid.uuid5(_NS, f"ref:{family_id}/{variant_id}/{sha256}"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _guard(fn):
    """Connectivity problems -> StoreUnavailable (HTTP 503); anything else propagates."""
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
        try:
            return fn(*a, **kw)
        except ResponseHandlingException as e:
            raise StoreUnavailable(f"qdrant unreachable: {e}") from e
        except UnexpectedResponse as e:
            if e.status_code is None or e.status_code >= 500:
                raise StoreUnavailable(f"qdrant error {e.status_code}: {e}") from e
            raise
        except (ConnectionError, TimeoutError) as e:
            raise StoreUnavailable(f"qdrant unreachable: {e}") from e
    return wrapper


class _LRU:
    def __init__(self, size: int):
        self.size, self.data, self.lock = size, OrderedDict(), threading.Lock()

    def get_many(self, keys):
        with self.lock:
            out = {}
            for k in keys:
                if k in self.data:
                    self.data.move_to_end(k)
                    out[k] = self.data[k]
            return out

    def put(self, k, v):
        with self.lock:
            self.data[k] = v
            self.data.move_to_end(k)
            while len(self.data) > self.size:
                self.data.popitem(last=False)

    def drop(self, keys):
        with self.lock:
            for k in keys:
                self.data.pop(k, None)


def make_client(cfg: QdrantCfg, root: Path):
    from qdrant_client import QdrantClient
    if cfg.location:
        return QdrantClient(location=cfg.location)
    if cfg.path:
        p = Path(cfg.path)
        return QdrantClient(path=str(p if p.is_absolute() else root / p))
    return QdrantClient(url=cfg.url, api_key=cfg.api_key or None, prefer_grpc=cfg.prefer_grpc,
                        timeout=int(cfg.timeout_s))


class QdrantCardStore:
    backend = "qdrant"
    supports_admin = True

    def __init__(self, cfg: QdrantCfg, embedder, n_features: int, root: Path, client=None,
                 allow_embedder_change: bool = False):
        from qdrant_client import models as m
        self.m = m
        self.cfg, self.embedder, self.n_features, self.root = cfg, embedder, n_features, root
        self.client = client or make_client(cfg, root)
        self.refs_col = f"{cfg.collection_prefix}_refs"
        self.fams_col = f"{cfg.collection_prefix}_families"
        self.blob_dir = Path(cfg.reference_store_dir)
        if not self.blob_dir.is_absolute():
            self.blob_dir = root / self.blob_dir
        self._kp = _LRU(cfg.keypoint_cache_size)
        self._write_lock = threading.RLock()
        self.ensure_collections(allow_embedder_change)

    # ================================================================ setup
    @_guard
    def ensure_collections(self, allow_embedder_change: bool = False):
        m, c = self.m, self.client
        if not c.collection_exists(self.fams_col):
            c.create_collection(self.fams_col, vectors_config={})
            self._index(self.fams_col, ["kind", "family_id", "source"])
        meta = self._meta()
        if meta and (meta.get("embedder") != self.embedder.name or meta.get("dim") != self.embedder.dim):
            if not allow_embedder_change:
                raise RuntimeError(
                    f"Qdrant collection '{self.refs_col}' was built with embedder {meta.get('embedder')} "
                    f"(dim {meta.get('dim')}) but the service uses {self.embedder.name} (dim {self.embedder.dim}). "
                    "Run: python tools/build_index.py --reembed")
        if not c.collection_exists(self.refs_col):
            c.create_collection(self.refs_col,
                                vectors_config=m.VectorParams(size=self.embedder.dim, distance=m.Distance.COSINE),
                                on_disk_payload=True)       # ORB blobs stay on disk, not in RAM
            self._index(self.refs_col, ["family", "variant_key", "source", "sha256"])
        if not meta:
            self._write_meta(revision=uuid.uuid4().hex[:12])

    def _index(self, col, fields):
        with warnings.catch_warnings():               # local/embedded mode ignores payload indexes
            warnings.simplefilter("ignore")
            for f in fields:
                self.client.create_payload_index(col, f, self.m.PayloadSchemaType.KEYWORD)

    def _meta(self) -> dict | None:
        r = self.client.retrieve(self.fams_col, [META_ID], with_payload=True)
        return r[0].payload if r else None

    def _write_meta(self, **changes):
        meta = self._meta() or {"kind": "meta"}
        meta.update(changes, embedder=self.embedder.name, dim=self.embedder.dim, updated_at=_now())
        self.client.upsert(self.fams_col, [self.m.PointStruct(id=META_ID, vector={}, payload=meta)])

    @_guard
    def bump_revision(self) -> str:
        rev = uuid.uuid4().hex[:12]
        self._write_meta(revision=rev)
        return rev

    # ================================================================ read side
    @_guard
    def current_version(self) -> str:
        meta = self._meta()
        return meta["revision"] if meta else "empty"

    def healthy(self) -> bool:
        try:
            self.current_version()
            return True
        except Exception:
            return False

    @_guard
    def snapshot(self, rebuild: bool = False) -> StoreSnapshot:
        version = self.current_version()
        families: dict[str, CardFamily] = {}
        for rec in self._scroll(self.fams_col, self._eq("kind", "family"), True):
            try:
                fam = CardFamily.model_validate(rec.payload["definition"])
                families[fam.id] = fam
            except Exception as e:
                log.error("skipping invalid family definition %s: %s", rec.payload.get("family_id"), e)
        counts = self.ref_counts()
        return StoreSnapshot(version, families, KeywordIndex(families), self, counts)

    @_guard
    def ref_counts(self) -> dict[tuple[str, str], int]:
        total = self.client.count(self.refs_col, exact=True).count
        if total == 0:
            return {}
        res = self.client.facet(self.refs_col, "variant_key", limit=max(1000, total), exact=True)
        out = {}
        for h in res.hits:
            fam, _, var = str(h.value).partition("/")
            out[(fam, var)] = h.count
        return out

    @_guard
    def search(self, qvec: np.ndarray, k: int) -> list[RefHit]:
        r = self.client.query_points(self.refs_col, query=qvec.astype(np.float32).tolist(), limit=k,
                                     with_payload=["family", "variant"])
        return [RefHit(str(p.id), p.payload["family"], p.payload["variant"], float(p.score)) for p in r.points]

    @_guard
    def family_refs(self, qvec: np.ndarray, family_ids: set[str], limit: int) -> list[RefHit]:
        if not family_ids:
            return []
        flt = self.m.Filter(must=[self.m.FieldCondition(key="family", match=self.m.MatchAny(any=sorted(family_ids)))])
        r = self.client.query_points(self.refs_col, query=qvec.astype(np.float32).tolist(), query_filter=flt,
                                     limit=limit, with_payload=["family", "variant"])
        return [RefHit(str(p.id), p.payload["family"], p.payload["variant"], float(p.score)) for p in r.points]

    @_guard
    def keypoints(self, ref_ids: list[str]) -> dict[str, Keypoints]:
        have = self._kp.get_many(ref_ids)
        missing = [i for i in ref_ids if i not in have]
        if missing:
            for rec in self.client.retrieve(self.refs_col, missing, with_payload=["orb"]):
                kp = keypoints_from_b64(rec.payload["orb"])
                self._kp.put(str(rec.id), kp)
                have[str(rec.id)] = kp
        return have

    # ================================================================ admin: families
    @_guard
    def get_family(self, family_id: str) -> dict | None:
        r = self.client.retrieve(self.fams_col, [family_point_id(family_id)], with_payload=True)
        return r[0].payload if r else None

    @_guard
    def list_refs(self, family_id: str | None = None) -> list[dict]:
        flt = self._eq("family", family_id) if family_id else None
        return [{"ref_id": str(r.id), **r.payload} for r in self._scroll(self.refs_col, flt, REF_PAYLOAD)]

    @_guard
    def put_family(self, fam: CardFamily, source: str, bump: bool = True) -> dict:
        """Create or replace a family definition. Removing a variant deletes its references; changing
        field ROIs re-extracts that family's keypoints (from the stored images) - nothing else is touched."""
        with self._write_lock:
            old = self.get_family(fam.id)
            old_fam = CardFamily.model_validate(old["definition"]) if old else None
            new_hash = definition_hash(fam)
            out = {"family": fam.id, "created": old is None, "changed": old is None or old.get("def_hash") != new_hash,
                   "references_deleted": 0, "keypoints_refreshed": 0}
            if not out["changed"] and old.get("source") == source:
                return out
            self.client.upsert(self.fams_col, [self.m.PointStruct(id=family_point_id(fam.id), vector={}, payload={
                "kind": "family", "family_id": fam.id, "country": fam.country, "source": source,
                "definition": fam.model_dump(mode="json"), "def_hash": new_hash, "rois": rois_signature(fam),
                "created_at": (old or {}).get("created_at", _now()), "updated_at": _now()})])
            if old_fam:
                removed = {v.id for v in old_fam.variants} - {v.id for v in fam.variants}
                for vid in removed:
                    out["references_deleted"] += self._delete_refs(self._eq("variant_key", f"{fam.id}/{vid}"))
                if old.get("rois") != rois_signature(fam):
                    out["keypoints_refreshed"] = self._refresh_keypoints(fam)
                if old_fam.country != fam.country:
                    self.client.set_payload(self.refs_col, {"country": fam.country},
                                            points=self.m.FilterSelector(filter=self._eq("family", fam.id)))
            if bump:
                self.bump_revision()
            return out

    @_guard
    def delete_family(self, family_id: str, bump: bool = True) -> dict | None:
        with self._write_lock:
            old = self.get_family(family_id)
            if old is None:
                return None
            refs = self.list_refs(family_id)
            n = self._delete_refs(self._eq("family", family_id))
            self.client.delete(self.fams_col, points_selector=self.m.PointIdsList(points=[family_point_id(family_id)]))
            files = self._delete_blobs(refs)
            if bump:
                self.bump_revision()
            return {"family": family_id, "source": old.get("source"), "references_deleted": n,
                    "stored_images_deleted": files}

    # ================================================================ admin: references
    @_guard
    def add_reference(self, fam: CardFamily, variant_id: str, data: bytes, filename: str, source: str,
                      image_path: Path | None = None, detect_card: bool = True, bump: bool = True) -> dict:
        """Embed one reference image and upsert it. Idempotent: the same image bytes for the same
        variant map to the same point id. Uploaded images are kept in reference_store_dir so the
        family can be re-processed later (new embedder, changed ROIs) without asking for them again."""
        if fam.variant(variant_id) is None:
            raise KeyError(f"variant '{variant_id}' not in family {fam.id}")
        sha = _sha256(data)
        pid = ref_point_id(fam.id, variant_id, sha)
        with self._write_lock:
            if self.client.retrieve(self.refs_col, [pid], with_payload=False):
                return {"ref_id": pid, "status": "exists", "sha256": sha}
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"{filename}: not a decodable image")
            detected = None
            if detect_card:
                img, info = normalise_mod.normalise(img)
                detected = info["card_detected"]
            if image_path is None:
                ext = Path(filename).suffix.lower() if Path(filename).suffix.lower() in IMAGE_EXT else ".png"
                image_path = self.blob_dir / fam.id / variant_id / f"{sha[:24]}{ext}"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.write_bytes(data)
            prep = prepare_reference(img, fam, self.embedder, self.n_features)
            self.client.upsert(self.refs_col, [self._ref_point(pid, fam, variant_id, sha, filename,
                                                               image_path, source, prep)])
            if bump:
                self.bump_revision()
            return {"ref_id": pid, "status": "created", "sha256": sha, "card_detected": detected}

    @_guard
    def add_prepared(self, items: list[tuple[str, CardFamily, str, str, str, Path, str, PreparedRef]]):
        """Batch upsert of already-prepared references (used by registry sync)."""
        pts = [self._ref_point(pid, fam, vid, sha, fn, path, src, prep) for pid, fam, vid, sha, fn, path, src, prep in items]
        for i in range(0, len(pts), 64):
            self.client.upsert(self.refs_col, pts[i:i + 64])

    @_guard
    def delete_reference(self, family_id: str, ref_id: str, bump: bool = True) -> dict | None:
        with self._write_lock:
            r = self.client.retrieve(self.refs_col, [ref_id], with_payload=REF_PAYLOAD)
            if not r or r[0].payload.get("family") != family_id:
                return None
            ref = {"ref_id": ref_id, **r[0].payload}
            self.client.delete(self.refs_col, points_selector=self.m.PointIdsList(points=[ref_id]))
            self._kp.drop([ref_id])
            files = self._delete_blobs([ref])
            if bump:
                self.bump_revision()
            return {"ref_id": ref_id, "family": family_id, "variant": ref["variant"], "stored_images_deleted": files}

    @_guard
    def delete_refs_by_ids(self, ids: list[str]) -> int:
        if ids:
            self.client.delete(self.refs_col, points_selector=self.m.PointIdsList(points=ids))
            self._kp.drop(ids)
        return len(ids)

    @_guard
    def reset_refs(self) -> list[dict]:
        """Drop and recreate the reference collection for a new embedder. Returns the previous reference
        records (without vectors) so the caller can re-ingest them from their stored images."""
        with self._write_lock:
            old = self.list_refs() if self.client.collection_exists(self.refs_col) else []
            if self.client.collection_exists(self.refs_col):
                self.client.delete_collection(self.refs_col)
            self._kp = _LRU(self.cfg.keypoint_cache_size)
            self._write_meta(revision=uuid.uuid4().hex[:12])      # records the new embedder + dim
            self.ensure_collections()
            return old

    # ================================================================ helpers
    def _ref_point(self, pid, fam: CardFamily, vid, sha, filename, image_path: Path, source, prep: PreparedRef):
        return self.m.PointStruct(id=pid, vector=prep.vector.astype(np.float32).tolist(), payload={
            "family": fam.id, "variant": vid, "variant_key": f"{fam.id}/{vid}", "country": fam.country,
            "sha256": sha, "filename": filename, "image_path": self._rel(image_path), "colour": list(prep.colour),
            "source": source, "added_at": _now(), "orb": keypoints_to_b64(prep.keypoints)})

    def _rel(self, p: Path) -> str:
        p = Path(p).resolve()
        try:
            return p.relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return p.as_posix()

    def _abs(self, s: str) -> Path:
        p = Path(s)
        return p if p.is_absolute() else self.root / p

    def _eq(self, key, value):
        return self.m.Filter(must=[self.m.FieldCondition(key=key, match=self.m.MatchValue(value=value))])

    def _scroll(self, col, flt, with_payload):
        offset = None
        while True:
            recs, offset = self.client.scroll(col, scroll_filter=flt, limit=256, offset=offset,
                                              with_payload=with_payload, with_vectors=False)
            yield from recs
            if offset is None:
                break

    def _delete_refs(self, flt) -> int:
        ids = [str(r.id) for r in self._scroll(self.refs_col, flt, False)]
        return self.delete_refs_by_ids(ids)

    def _delete_blobs(self, refs: list[dict]) -> int:
        """Remove stored upload copies (never files from the git-managed registry)."""
        n = 0
        blob_root = self.blob_dir.resolve()
        for r in refs:
            if r.get("source") != "api" or not r.get("image_path"):
                continue
            p = self._abs(r["image_path"]).resolve()
            if blob_root in p.parents and p.exists():
                p.unlink()
                n += 1
                for d in p.parents:                    # prune now-empty variant/family folders
                    if d == blob_root or blob_root not in d.parents or any(d.iterdir()):
                        break
                    d.rmdir()
        return n

    def _refresh_keypoints(self, fam: CardFamily) -> int:
        n = 0
        for ref in self.list_refs(fam.id):
            p = self._abs(ref["image_path"])
            img = cv2.imread(str(p))
            if img is None:
                log.warning("cannot re-extract keypoints for %s: image %s missing", ref["ref_id"], p)
                continue
            if ref.get("source") == "api":
                img, _ = normalise_mod.normalise(img)
            kp = reference_keypoints(img, fam, self.n_features)
            self.client.set_payload(self.refs_col, {"orb": keypoints_to_b64(kp)}, points=[ref["ref_id"]])
            self._kp.drop([ref["ref_id"]])
            n += 1
        return n


def _sha256(data: bytes) -> str:
    import hashlib
    return hashlib.sha256(data).hexdigest()
