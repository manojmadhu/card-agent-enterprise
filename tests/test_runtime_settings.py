"""Runtime settings overrides: what may change, merge-patch semantics, whole-section validation."""
import pytest

from cardagent.settings import Settings
from cardagent.store.runtime_settings import SettingsError, apply_overrides, merge, tunable_view, validate_overrides


def test_only_tunable_keys_are_accepted():
    assert validate_overrides({"scoring": {"family_min": 0.6}}) == {"scoring": {"family_min": 0.6}}
    for bad in ({"retrieval": {"orb_features": 500}}, {"embedder": {"kind": "classic"}}, {"qdrant": {"url": "x"}},
                {"scoring": 0.5}, []):
        with pytest.raises(SettingsError):
            validate_overrides(bad)


def test_merge_patch_sets_and_removes():
    cur = {"scoring": {"family_min": 0.6, "family_margin": 0.2}, "quality": {"min_blur": 40}}
    assert merge(cur, {"scoring": {"family_min": 0.7}}) == {"scoring": {"family_min": 0.7, "family_margin": 0.2},
                                                           "quality": {"min_blur": 40}}
    assert merge(cur, {"scoring": {"family_min": None, "family_margin": None}}) == {"quality": {"min_blur": 40}}
    assert merge(cur, {"quality": None}) == {"scoring": {"family_min": 0.6, "family_margin": 0.2}}


def test_apply_overrides_returns_new_settings_and_keeps_base():
    base = Settings()
    new = apply_overrides(base, {"scoring": {"family_min": 0.77}})
    assert new.scoring.family_min == 0.77 and base.scoring.family_min != 0.77
    assert tunable_view(new)["scoring"]["family_min"] == 0.77


def test_weights_must_sum_to_one_and_types_are_checked():
    with pytest.raises(SettingsError):
        apply_overrides(Settings(), {"scoring": {"family_weights": {"text": 0.9, "embed": 0.2, "geometry": 0.3,
                                                                    "colour": 0.1}}})
    with pytest.raises(SettingsError):
        apply_overrides(Settings(), {"scoring": {"family_min": "high"}})
