"""Shared types for the card store (snapshot, reference hits, reference preparation)."""
from __future__ import annotations

import base64
import hashlib
import io
import json
from dataclasses import dataclass, field
from typing import Protocol

import cv2
import numpy as np

from ..registry.models import CardFamily
from ..vision.features import Keypoints, body_colour_rgb, design_mask, extract_keypoints
from ..vision.geometry import CARD_H, CARD_W
from .keywords import KeywordIndex


class StoreUnavailable(RuntimeError):
    """The vector store could not be reached; the API maps this to HTTP 503."""


@dataclass(frozen=True)
class RefHit:
    ref_id: str
    family_id: str
    variant_id: str
    score: float          # cosine similarity to the query embedding
    side: str = "front"   # card side the reference shows (references stored before sides existed: front)
    colour: tuple | None = None   # body colour measured on the reference image


class RefView(Protocol):
    """Read side used by identification; pinned for the duration of one request."""

    def search(self, qvec: np.ndarray, k: int) -> list[RefHit]: ...

    def family_refs(self, qvec: np.ndarray, family_ids: set[str], limit: int) -> list[RefHit]: ...

    def keypoints(self, ref_ids: list[str]) -> dict[str, Keypoints]: ...


@dataclass
class StoreSnapshot:
    version: str
    families: dict[str, CardFamily]
    keywords: KeywordIndex
    refs: RefView
    ref_counts: dict[tuple[str, str], int] = field(default_factory=dict)
    side_counts: dict[tuple[str, str, str], int] = field(default_factory=dict)   # (family, variant, side)

    @property
    def ref_count(self) -> int:
        return sum(self.ref_counts.values())

    def summary(self) -> dict:
        return {
            "version": self.version,
            "families": len(self.families),
            "variants": sum(len(f.variants) for f in self.families.values()),
            "reference_images": self.ref_count,
            "items": [{"family": f.id, "country": f.country, "name": f.name,
                       "sides": f.side_ids,
                       "variants": [{"id": v.id, "name": v.name,
                                     "reference_images": self.ref_counts.get((f.id, v.id), 0),
                                     "reference_images_by_side": {s: self.side_counts.get((f.id, v.id, s), 0)
                                                                  for s in f.side_ids}}
                                    for v in f.variants]}
                      for f in sorted(self.families.values(), key=lambda x: x.id)],
        }


# ---------------------------------------------------------------- reference preparation
@dataclass
class PreparedRef:
    vector: np.ndarray
    keypoints: Keypoints
    colour: tuple[int, int, int]


def _side_mask(fam: CardFamily, side: str):
    return design_mask([f.roi for f in fam.fields_on(side)], CARD_W, CARD_H)


def prepare_reference(card_bgr: np.ndarray, fam: CardFamily, embedder, n_features: int,
                      side: str = "front") -> PreparedRef:
    """Exactly what the index stores per reference image. Personal-data regions (the ROIs of the
    fields printed on this side) are masked out of the keypoints so verification matches the card
    design, not names."""
    img = cv2.resize(card_bgr, (CARD_W, CARD_H), interpolation=cv2.INTER_AREA)
    return PreparedRef(embedder.embed(img), extract_keypoints(img, n_features, _side_mask(fam, side)),
                       body_colour_rgb(img))


def reference_keypoints(card_bgr: np.ndarray, fam: CardFamily, n_features: int, side: str = "front") -> Keypoints:
    img = cv2.resize(card_bgr, (CARD_W, CARD_H), interpolation=cv2.INTER_AREA)
    return extract_keypoints(img, n_features, _side_mask(fam, side))


def keypoints_to_b64(kp: Keypoints) -> str:
    buf = io.BytesIO()
    des = kp.des if kp.des is not None else np.zeros((0, 32), np.uint8)
    np.savez_compressed(buf, pts=kp.pts.astype(np.float32), des=des.astype(np.uint8))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def keypoints_from_b64(s: str) -> Keypoints:
    with np.load(io.BytesIO(base64.b64decode(s))) as z:
        pts, des = z["pts"], z["des"]
    return Keypoints(pts, des if len(des) else None)


def definition_hash(fam: CardFamily) -> str:
    return hashlib.sha256(json.dumps(fam.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()[:16]


def rois_signature(fam: CardFamily) -> str:
    """Keypoints depend only on the field ROIs per side (masked regions); re-extract when this changes.
    Front-side ROIs hash as before sides existed, so existing families keep their signature."""
    items = [list(f.roi) if f.side == "front" else [f.side, *f.roi] for f in fam.fields]
    legacy = all(f.side == "front" for f in fam.fields)
    items = sorted(items) if legacy else sorted(items, key=json.dumps)
    return hashlib.sha256(json.dumps(items).encode()).hexdigest()[:16]
