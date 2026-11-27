"""Public output contract. Export JSON Schema with: python -m cardagent.schemas > result.schema.json"""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel

SCHEMA_VERSION = "2.1"          # 2.1: two-sided cards (photos, missing_sides, field side); additive to 2.0
Status = Literal["ACCEPT", "REVIEW", "REJECT"]
Side = Literal["front", "back"]


class CardInfo(BaseModel):
    family: str
    family_name: str
    country: str
    issuer: str
    family_confidence: float
    variant: str | None
    variant_name: str | None
    variant_confidence: float


class FieldResult(BaseModel):
    value: str | None
    confidence: float
    source: Literal["ocr_roi", "ocr_fulltext", "vlm", "none"]
    valid_format: bool
    valid_until: str | None = None    # ISO date, expiry fields only
    side: Side = "front"               # card side the field is printed on


class RuleResult(BaseModel):
    rule: str
    field: str
    critical: bool
    result: Literal["pass", "fail", "skip"]
    detail: str | None = None


class Candidate(BaseModel):
    id: str
    score: float
    signals: dict[str, float]


class QualityInfo(BaseModel):
    width: int
    height: int
    sharpness: float | None = None     # contrast-normalised sharpness (the gate's measure)
    min_sharpness: float | None = None # threshold it was compared with
    blur: float                        # raw Laplacian variance (legacy measure)
    brightness: float
    glare_ratio: float
    passed: bool
    issues: list[str]
    card_detected: bool | None = None
    detector: str | None = None
    rotation: int | None = None
    aligned_to_reference: bool | None = None


class PhotoInfo(BaseModel):
    """One uploaded photo and what it was used for."""
    photo: int                         # 0-based position in the request
    upload: str                        # form field it came in: image | front | back | images
    declared: Side | None = None       # side named by the upload field (front/back), else null
    side: Side | None = None           # side this photo was used as; null = not used
    status: Literal["used", "not_used", "retake", "invalid"]
    side_corrected: bool = False       # the photo shows the other side than declared (uploads swapped)
    quality: QualityInfo | None = None
    reference_id: str | None = None    # reference image it was aligned to
    inliers: int | None = None         # RANSAC inliers against that reference
    fields: list[str] = []             # fields read from this photo


class ValidationResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    request_id: str
    run_id: str | None = None          # id in the run history (GET /v1/runs/{run_id}); null when runs are disabled
    status: Status
    reasons: list[str]
    card: CardInfo | None = None
    fields: dict[str, FieldResult] = {}
    validation: list[RuleResult] = []
    quality: QualityInfo | None = None   # of the main photo (front when present); every photo: `photos`
    photos: list[PhotoInfo] = []
    missing_sides: list[Side] = []       # sides the card needs but no usable photo shows
    candidates: dict[str, list[Candidate]] = {}
    model_versions: dict[str, str] = {}
    timings_ms: dict[str, int] = {}
    processing_ms: int = 0


if __name__ == "__main__":
    print(json.dumps(ValidationResult.model_json_schema(), indent=2))
