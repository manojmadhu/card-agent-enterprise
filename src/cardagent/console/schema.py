"""Compatibility layer for the Card Agent Console (card-agent-web).

The console was built against the card-agent MVP, so it speaks the MVP formats:

* card families as MVP YAML  (family: fazaa, family_name, weights: {text, visual, colour}, ...)
  -- mirrored here by ConsoleFamily, field for field with the console's zod schema
* validation results as the MVP envelope (flat `candidates`, reason codes UNKNOWN_CARD /
  AMBIGUOUS_FAMILY, `quality` always present)

This module converts between those formats and the enterprise ones (registry.models.CardFamily,
schemas.ValidationResult). It holds no state and does no I/O, so it is unit-testable on its own.
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from ..registry.models import CardFamily

# ---------------------------------------------------------------------------------- console schema
ConsoleNormalise = Literal["upper", "digits", "date_mm_yy", "emirates_id", "none"]
ConsoleRule = Literal["regex", "not_expired", "date_order", "luhn", "emirates_id_format",
                      "prefix_matches_variant", "lookup"]
DEFAULT_WEIGHTS = {"text": 0.55, "visual": 0.30, "colour": 0.15}


class ConsoleVariant(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9_-]+$")
    name: str = Field(min_length=1)
    body_color_rgb: tuple[int, int, int]
    keywords: list[str] = []
    number_prefix: str = Field("", pattern=r"^\d*$")

    @field_validator("body_color_rgb")
    @classmethod
    def _rgb(cls, v):
        if any(not 0 <= c <= 255 for c in v):
            raise ValueError("body_color_rgb values must be 0..255")
        return v


class ConsoleField(BaseModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    roi: tuple[float, float, float, float]
    regex: str = Field(min_length=1)
    required: bool = False
    normalise: ConsoleNormalise = "none"
    anchor: str | None = None


class ConsoleRuleSpec(BaseModel):
    type: ConsoleRule
    field: str
    critical: bool = False
    other_field: str | None = None


class ConsoleWeights(BaseModel):
    text: float = Field(ge=0, le=1)
    visual: float = Field(ge=0, le=1)
    colour: float = Field(ge=0, le=1)


class ConsoleFamily(BaseModel):
    """The console's family schema (src/engine/schema.ts in card-agent-web)."""
    family: str = Field(pattern=r"^[a-z0-9_-]+$")
    family_name: str = Field(min_length=1)
    issuer: str = ""
    schema_version: int = 1
    keywords: list[str] = Field(min_length=1)
    negative_keywords: list[str] = []
    weights: ConsoleWeights = ConsoleWeights(**DEFAULT_WEIGHTS)
    variants: list[ConsoleVariant] = Field(min_length=1)
    fields: list[ConsoleField] = []
    rules: list[ConsoleRuleSpec] = []

    @model_validator(mode="after")
    def _check(self):
        w = self.weights
        if abs(w.text + w.visual + w.colour - 1) > 0.011:
            raise ValueError(f"weights must sum to 1 (got {w.text + w.visual + w.colour:.2f})")
        return self


# ---------------------------------------------------------------------------------- id mapping
def console_id(enterprise_id: str, all_ids) -> str:
    """ae.fazaa -> 'fazaa' when that short name is unique, else 'ae-fazaa' (console ids allow a-z0-9_-)."""
    short = enterprise_id.split(".", 1)[1]
    clash = sum(1 for i in all_ids if i.split(".", 1)[1] == short) > 1
    return enterprise_id.replace(".", "-") if clash else short


def resolve_enterprise_id(cid: str, all_ids, default_country: str) -> str:
    """Inverse of console_id; an unknown id becomes a new family in default_country."""
    all_ids = list(all_ids)
    for eid in all_ids:
        if console_id(eid, all_ids) == cid:
            return eid
    m = re.fullmatch(r"([a-z]{2})-([a-z0-9_-]+)", cid)
    if m:
        return f"{m.group(1)}.{m.group(2).replace('-', '_')}"
    return f"{default_country.lower()}.{cid.replace('-', '_')}"


# ---------------------------------------------------------------------------------- family conversion
def to_enterprise(cf: ConsoleFamily, enterprise_id: str, existing: CardFamily | None = None) -> CardFamily:
    """Console -> enterprise. Attributes the console does not know about (country, a field's script,
    extra rule params) are kept from the existing definition. Console `weights` are not stored: the
    enterprise service uses the global scoring.family_weights for every family."""
    old_fields = {f.name: f for f in existing.fields} if existing else {}
    old_rules = {(r.type, r.field): r for r in existing.rules} if existing else {}
    fields = []
    for f in cf.fields:
        prev = old_fields.get(f.name)
        fields.append({"name": f.name, "roi": list(f.roi), "regex": f.regex, "required": f.required,
                       "normalise": f.normalise, "anchor": f.anchor or None,
                       **({"script": prev.script} if prev else {})})
    rules = []
    for r in cf.rules:
        params = dict(old_rules[(r.type, r.field)].params) if (r.type, r.field) in old_rules else {}
        if r.other_field:
            params["other_field"] = r.other_field
        rules.append({"type": r.type, "field": r.field, "critical": r.critical, "params": params})
    return CardFamily.model_validate({
        "id": enterprise_id,
        "country": existing.country if existing else enterprise_id.split(".")[0].upper(),
        "issuer": cf.issuer or cf.family_name,
        "name": cf.family_name,
        "schema_version": cf.schema_version,
        "keywords": cf.keywords,
        "negative_keywords": cf.negative_keywords,
        "variants": [{"id": v.id, "name": v.name, "body_color_rgb": list(v.body_color_rgb),
                      "keywords": v.keywords, "number_prefix": v.number_prefix or None} for v in cf.variants],
        "fields": fields,
        "rules": rules,
    })


def to_console(fam: CardFamily, cid: str) -> dict:
    """Enterprise -> console dict (YAML-ready, in the console's key order)."""
    return {
        "family": cid,
        "family_name": fam.name,
        "issuer": fam.issuer,
        "schema_version": fam.schema_version,
        "keywords": list(fam.keywords),
        "negative_keywords": list(fam.negative_keywords),
        "weights": dict(DEFAULT_WEIGHTS),
        "variants": [{"id": v.id, "name": v.name, "body_color_rgb": list(v.body_color_rgb),
                      "keywords": list(v.keywords), "number_prefix": v.number_prefix or ""} for v in fam.variants],
        "fields": [{"name": f.name, "roi": list(f.roi), "regex": f.regex, "required": f.required,
                    "normalise": f.normalise, **({"anchor": f.anchor} if f.anchor else {})} for f in fam.fields],
        "rules": [{"type": r.type, "field": r.field, "critical": r.critical,
                   **({"other_field": r.params["other_field"]} if r.params.get("other_field") else {})}
                  for r in fam.rules],
    }


# ---------------------------------------------------------------------------------- result translation
REASON_MAP = {"UNKNOWN_CARD_TYPE": "UNKNOWN_CARD", "AMBIGUOUS_CARD_TYPE": "AMBIGUOUS_FAMILY"}
_EMPTY_QUALITY = {"width": 0, "height": 0, "blur": 0.0, "brightness": 0.0, "glare_ratio": 0.0,
                  "passed": False, "issues": []}


def to_console_result(result: dict, families: dict[str, CardFamily]) -> dict:
    """Enterprise ValidationResult dict -> the MVP envelope the console renders.

    Guarantees the console relies on: `quality` is always an object; `candidates` is a flat list of
    {family, name, score}; UNKNOWN_CARD / AMBIGUOUS_FAMILY / RETAKE_REQUIRED reason codes; and `card`
    is present whenever the decision went past family identification (for an ambiguous family it
    is the top candidate, marked provisional)."""
    ids = list(families)
    cid = lambda eid: console_id(eid, ids) if eid in families else eid  # noqa: E731
    reasons = [REASON_MAP.get(r, r) for r in result.get("reasons", [])]
    if "INVALID_IMAGE" in reasons and "RETAKE_REQUIRED" not in reasons:
        reasons.insert(0, "RETAKE_REQUIRED")          # unreadable upload = retake, as the console expects

    fam_cands = (result.get("candidates") or {}).get("family", [])
    var_cands = (result.get("candidates") or {}).get("variant", [])
    candidates = [{"family": cid(c["id"]), "name": families[c["id"]].name if c["id"] in families else c["id"],
                   "score": c["score"], "signals": c.get("signals", {})} for c in fam_cands]

    card = result.get("card")
    if card:
        card = {**card, "family": cid(card["family"]), "enterprise_family": card["family"], "provisional": False}
    elif "AMBIGUOUS_FAMILY" in reasons and fam_cands:
        top = fam_cands[0]
        fam = families.get(top["id"])
        card = {"family": cid(top["id"]), "enterprise_family": top["id"],
                "family_name": fam.name if fam else top["id"], "country": fam.country if fam else None,
                "issuer": fam.issuer if fam else None, "family_confidence": top["score"],
                "variant": None, "variant_name": None, "variant_confidence": 0.0, "provisional": True}

    return {
        "schema_version": result.get("schema_version"),
        "request_id": result["request_id"],
        "status": result["status"],
        "reasons": reasons,
        "card": card,
        "fields": result.get("fields") or {},
        "validation": result.get("validation") or [],
        "quality": result.get("quality") or dict(_EMPTY_QUALITY, issues=["INVALID_IMAGE"]),
        "candidates": candidates,
        "variant_candidates": var_cands,
        "model_versions": result.get("model_versions", {}),
        "timings_ms": result.get("timings_ms", {}),
        "processing_ms": result.get("processing_ms", 0),
    }
