"""Runtime settings: decision thresholds changed through the API (PATCH /v1/settings).

Only TUNABLE keys can be changed at runtime: they affect decisions, not what is stored in Qdrant.
Anything that changes stored data or loaded models (embedder kind, ORB feature count, OCR engine,
paths, credentials) stays in config/settings.yaml / environment variables and needs a restart.

Overrides are kept in one point of the families collection and bump the store revision, so every
replica applies them within qdrant.refresh_seconds. Effective settings = file/env settings + overrides.
"""
from __future__ import annotations

import uuid

from ..settings import EmbedderCfg, QualityCfg, RetrievalCfg, ScoringCfg, Settings

_NS = uuid.UUID("6f1c3f7e-2b7a-4c55-9d0e-6a1f2c9b8e01")
SETTINGS_ID = str(uuid.uuid5(_NS, "__settings__"))

SECTION_MODELS = {"scoring": ScoringCfg, "quality": QualityCfg, "retrieval": RetrievalCfg, "embedder": EmbedderCfg}
TUNABLE: dict[str, set[str]] = {
    "scoring": set(ScoringCfg.model_fields),
    "quality": set(QualityCfg.model_fields),
    "retrieval": set(RetrievalCfg.model_fields) - {"orb_features"},   # stored keypoints depend on it
    "embedder": {"similarity_floor"},
}


class SettingsError(ValueError):
    pass


def validate_overrides(overrides: dict) -> dict:
    """Return the cleaned overrides; raise SettingsError naming every key that is not allowed."""
    if not isinstance(overrides, dict):
        raise SettingsError("settings must be an object like {\"scoring\": {\"family_min\": 0.55}}")
    bad = []
    for section, values in overrides.items():
        if section not in TUNABLE:
            bad.append(section)
            continue
        if not isinstance(values, dict):
            raise SettingsError(f"'{section}' must be an object")
        bad += [f"{section}.{k}" for k in values if k not in TUNABLE[section]]
    if bad:
        raise SettingsError(f"not changeable at runtime: {', '.join(sorted(bad))} "
                            f"(tunable: {', '.join(f'{s}.*' if s != 'retrieval' and s != 'embedder' else s for s in TUNABLE)}; "
                            "see GET /v1/settings → tunable)")
    return {s: dict(v) for s, v in overrides.items() if v}


def merge(a: dict, b: dict) -> dict:
    """Merge patch b into overrides a (JSON merge-patch style): a value of null removes that override,
    and a whole section set to null removes every override in it."""
    out = {s: dict(v) for s, v in a.items()}
    for s, v in (b or {}).items():
        if v is None:
            out.pop(s, None)
            continue
        if not isinstance(v, dict):
            raise SettingsError(f"'{s}' must be an object")
        sec = out.setdefault(s, {})
        for k, val in v.items():
            if val is None:
                sec.pop(k, None)
            else:
                sec[k] = val
    return {s: v for s, v in out.items() if v}


def apply_overrides(base: Settings, overrides: dict) -> Settings:
    """A new Settings object with the overrides applied (each section fully re-validated)."""
    overrides = validate_overrides(overrides)
    update = {}
    for section, values in overrides.items():
        model = SECTION_MODELS[section]
        current = getattr(base, section).model_dump()
        try:
            update[section] = model.model_validate({**current, **values})
        except Exception as e:                                     # pydantic ValidationError
            raise SettingsError(f"invalid value in '{section}': {e}") from e
    _check_weights(update.get("scoring"))
    return base.model_copy(update=update)


def _check_weights(scoring: ScoringCfg | None):
    if scoring is None:
        return
    need = {"family_weights": {"text", "embed", "geometry", "colour"},
            "variant_weights": {"colour", "keyword", "geometry", "embed", "prefix"}}
    for name, keys in need.items():
        w = getattr(scoring, name)
        if set(w) != keys:
            raise SettingsError(f"scoring.{name} needs exactly the keys {sorted(keys)}")
        if abs(sum(w.values()) - 1) > 0.011 or any(v < 0 for v in w.values()):
            raise SettingsError(f"scoring.{name} must be non-negative and sum to 1 (got {sum(w.values()):.2f})")


def tunable_view(s: Settings) -> dict:
    return {section: {k: getattr(getattr(s, section), k) for k in sorted(keys)} for section, keys in TUNABLE.items()}


class RuntimeSettingsStore:
    def __init__(self, card_store):
        self.cs = card_store                      # QdrantCardStore: shares client, collection and revision

    def get(self) -> dict:
        r = self.cs.client.retrieve(self.cs.fams_col, [SETTINGS_ID], with_payload=True)
        return (r[0].payload.get("overrides") or {}) if r else {}

    def put(self, overrides: dict, updated_by: str = "api") -> dict:
        from .qdrant_store import _now
        overrides = validate_overrides(overrides)
        with self.cs._write_lock:
            self.cs.client.upsert(self.cs.fams_col, [self.cs.m.PointStruct(
                id=SETTINGS_ID, vector={},
                payload={"kind": "settings", "overrides": overrides, "updated_at": _now(), "updated_by": updated_by})])
            self.cs.bump_revision()
        return overrides
