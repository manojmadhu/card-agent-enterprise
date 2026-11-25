"""Validated schema for card-family YAML files. A bad file fails loudly at load time,
so a broken registry can never be hot-reloaded into production."""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

NormaliseKind = Literal["digits", "upper", "alnum", "date_mm_yy", "date_dd_mm_yyyy", "emirates_id", "none"]
RuleKind = Literal["regex", "not_expired", "emirates_id_format", "luhn", "prefix_matches_variant",
                   "length", "lookup", "date_order"]   # date_order: field <= params.other_field
SideId = Literal["front", "back"]
SIDE_ORDER = ("front", "back")


class FieldSpec(BaseModel):
    name: str
    roi: tuple[float, float, float, float] = Field(description="x0,y0,x1,y1 as fractions of the reference card")
    regex: str
    required: bool = True
    normalise: NormaliseKind = "none"
    anchor: str | None = None          # printed label to strip, e.g. "EXP"
    script: Literal["latin", "arabic"] = "latin"
    side: SideId = "front"             # the side the field is printed on; roi is on that side's reference

    @field_validator("regex")
    @classmethod
    def _compiles(cls, v: str) -> str:
        try:
            re.compile(v)
        except re.error as e:                  # surface as a validation error (422), not a crash
            raise ValueError(f"regex does not compile: {e}") from e
        return v

    @field_validator("roi")
    @classmethod
    def _roi_ok(cls, v):
        x0, y0, x1, y1 = v
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError(f"invalid roi {v}")
        return v


class RuleSpec(BaseModel):
    type: RuleKind
    field: str
    critical: bool = False
    params: dict = {}


class Variant(BaseModel):
    id: str
    name: str
    body_color_rgb: tuple[int, int, int]
    keywords: list[str] = []
    number_prefix: str | None = None

    @field_validator("keywords")
    @classmethod
    def _upper(cls, v):
        return [k.upper() for k in v]


class SideSpec(BaseModel):
    """One physical side of the card. `required`: a photo of this side must be submitted. Default
    (null): required when a required field is printed on it, or when it is the card's only side."""
    id: SideId
    required: bool | None = None


class CardFamily(BaseModel):
    id: str = Field(description="globally unique, e.g. ae.fazaa")
    country: str = Field(description="ISO-3166 alpha-2, e.g. AE, SA, QA")
    issuer: str
    name: str
    keywords: list[str]
    negative_keywords: list[str] = []
    variants: list[Variant]
    fields: list[FieldSpec]
    rules: list[RuleSpec] = []
    sides: list[SideSpec] = Field(default_factory=lambda: [SideSpec(id="front")],
                                  description="card sides; default front only")
    schema_version: int = 1

    @field_validator("keywords", "negative_keywords")
    @classmethod
    def _upper(cls, v):
        return [k.upper() for k in v]

    @model_validator(mode="after")
    def _consistency(self):
        if not re.fullmatch(r"[a-z]{2}\.[a-z0-9_]+", self.id):
            raise ValueError(f"family id '{self.id}' must look like 'ae.fazaa'")
        ids = [v.id for v in self.variants]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{self.id}: duplicate variant ids")
        names = {f.name for f in self.fields}
        for r in self.rules:
            if r.field not in names:
                raise ValueError(f"{self.id}: rule {r.type} refers to unknown field '{r.field}'")
            if r.type == "date_order" and r.params.get("other_field") not in names:
                raise ValueError(f"{self.id}: date_order rule needs params.other_field naming a field")
        sides = [s.id for s in self.sides]
        if not sides or len(sides) != len(set(sides)):
            raise ValueError(f"{self.id}: sides must list front and/or back once each")
        for f in self.fields:
            if f.side not in sides:
                raise ValueError(f"{self.id}: field '{f.name}' is on side '{f.side}', which is not in sides {sides}")
        for s in self.sides:
            if s.required is False and any(f.required and f.side == s.id for f in self.fields):
                raise ValueError(f"{self.id}: side '{s.id}' has required fields, so it cannot be required: false")
        self.sides = sorted(self.sides, key=lambda s: SIDE_ORDER.index(s.id))
        return self

    @property
    def side_ids(self) -> list[str]:
        return [s.id for s in self.sides]

    def required_sides(self) -> list[str]:
        out = []
        for s in self.sides:
            req = s.required
            if req is None:
                req = len(self.sides) == 1 or any(f.required and f.side == s.id for f in self.fields)
            if req:
                out.append(s.id)
        return out

    def fields_on(self, side: str) -> list[FieldSpec]:
        return [f for f in self.fields if f.side == side]

    def variant(self, vid: str | None) -> Variant | None:
        return next((v for v in self.variants if v.id == vid), None)
