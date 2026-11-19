"""Public output contract. Export JSON Schema with: python -m cardagent.schemas > result.schema.json"""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel

SCHEMA_VERSION = "2.0"
Status = Literal["ACCEPT", "REVIEW", "REJECT"]


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
    blur: float
    brightness: float
    glare_ratio: float
    passed: bool
    issues: list[str]
    card_detected: bool | None = None
    detector: str | None = None
    rotation: int | None = None
    aligned_to_reference: bool | None = None


class ValidationResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    request_id: str
    status: Status
    reasons: list[str]
    card: CardInfo | None = None
    fields: dict[str, FieldResult] = {}
    validation: list[RuleResult] = []
    quality: QualityInfo | None = None
    candidates: dict[str, list[Candidate]] = {}
    model_versions: dict[str, str] = {}
    timings_ms: dict[str, int] = {}
    processing_ms: int = 0


if __name__ == "__main__":
    print(json.dumps(ValidationResult.model_json_schema(), indent=2))
