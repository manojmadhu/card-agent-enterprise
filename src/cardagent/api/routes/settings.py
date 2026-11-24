"""Runtime settings: decision thresholds and weights, changed without a restart.

Effective settings = config/settings.yaml + CARDAGENT_* environment + overrides stored in Qdrant.
Overrides apply to every replica within qdrant.refresh_seconds. Only the `tunable` keys can be
overridden; the rest (models, OCR, storage, credentials) is shown under `fixed` and needs a restart.
Try a change first with POST /v1/evaluations {"settings": {...}} (what-if) before applying it here.
"""
from __future__ import annotations

from fastapi import APIRouter, Body, Depends, HTTPException

from ...agent import CardAgent
from ...store.runtime_settings import TUNABLE, SettingsError, apply_overrides, merge, tunable_view
from ..deps import agent, audit, require_admin

router = APIRouter(prefix="/v1/settings", tags=["settings"])


def _fixed(s) -> dict:
    return {"embedder": {"kind": s.embedder.kind, "onnx_path": s.embedder.onnx_path if s.embedder.kind != "classic" else None},
            "ocr": {"engine": s.ocr.engine, "tesseract_lang": s.ocr.tesseract_lang},
            "retrieval": {"orb_features": s.retrieval.orb_features},
            "qdrant": {"url": s.qdrant.url if not s.qdrant.location else None, "location": s.qdrant.location,
                       "collection_prefix": s.qdrant.collection_prefix, "refresh_seconds": s.qdrant.refresh_seconds},
            "vlm": {"enabled": s.vlm.enabled, "model": s.vlm.model},
            "runs": s.runs.model_dump(), "evaluation": s.evaluation.model_dump()}


def _view(a: CardAgent) -> dict:
    return {"effective": tunable_view(a.s), "defaults": tunable_view(a.base_settings), "overrides": a.overrides,
            "tunable": {k: sorted(v) for k, v in TUNABLE.items()}, "fixed": _fixed(a.s)}


@router.get("")
def get_settings_view(a: CardAgent = Depends(agent)):
    """effective = defaults + overrides. Only `tunable` keys can be changed through PATCH."""
    a.refresh_if_changed(force=True)
    return _view(a)


@router.patch("", dependencies=[Depends(require_admin)])
def patch_settings(patch: dict = Body(..., examples=[{"scoring": {"family_min": 0.55, "family_margin": None}}]),
                   a: CardAgent = Depends(agent)):
    """Merge-patch the overrides: {"scoring": {"family_min": 0.55}} sets a value, null removes an override
    (back to the default), {"scoring": null} removes a whole section. The result is validated as a whole
    (ranges, weights summing to 1) before it is stored; an invalid patch changes nothing."""
    a.refresh_if_changed(force=True)
    try:
        new = merge(a.overrides, patch)
        apply_overrides(a.base_settings, new)
        a.runtime_settings.put(new)
    except SettingsError as e:
        raise HTTPException(422, str(e))
    a.reload()
    audit("patch_settings", patch=patch)
    return _view(a)


@router.delete("", dependencies=[Depends(require_admin)])
def reset_settings(section: str | None = None, a: CardAgent = Depends(agent)):
    """Remove all overrides, or only one section's (?section=scoring)."""
    a.refresh_if_changed(force=True)
    if section and section not in TUNABLE:
        raise HTTPException(422, f"unknown section {section} (sections: {', '.join(TUNABLE)})")
    new = {k: v for k, v in a.overrides.items() if section and k != section}
    a.runtime_settings.put(new)
    a.reload()
    audit("reset_settings", section=section or "all")
    return _view(a)
