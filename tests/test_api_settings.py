"""Runtime settings through the API: view, merge-patch, validation, reset - and their effect."""
from .conftest import ADMIN


def test_view(client):
    s = client.get("/v1/settings").json()
    assert s["overrides"] == {} and s["effective"] == s["defaults"]
    assert "family_min" in s["tunable"]["scoring"] and "orb_features" not in s["tunable"]["retrieval"]
    assert s["fixed"]["embedder"]["kind"] == "classic"


def test_patch_null_and_reset(client):
    r = client.patch("/v1/settings", json={"scoring": {"family_min": 0.62, "family_margin": 0.2},
                                           "quality": {"min_sharpness": 40}}, headers=ADMIN)
    assert r.status_code == 200
    s = r.json()
    assert s["effective"]["scoring"]["family_min"] == 0.62 and s["overrides"]["quality"] == {"min_sharpness": 40}
    assert client.get("/health/ready").json()["settings_overrides"] is True

    s = client.patch("/v1/settings", json={"scoring": {"family_min": None}}, headers=ADMIN).json()
    assert s["overrides"]["scoring"] == {"family_margin": 0.2}
    assert s["effective"]["scoring"]["family_min"] == s["defaults"]["scoring"]["family_min"]

    s = client.delete("/v1/settings", params={"section": "quality"}, headers=ADMIN).json()
    assert "quality" not in s["overrides"] and "scoring" in s["overrides"]
    assert client.delete("/v1/settings", headers=ADMIN).json()["overrides"] == {}


def test_invalid_patches_change_nothing(client):
    for bad in ({"retrieval": {"orb_features": 100}}, {"scoring": {"family_min": "x"}},
                {"scoring": {"variant_weights": {"colour": 1, "keyword": 1, "geometry": 0, "embed": 0, "prefix": 0}}}):
        assert client.patch("/v1/settings", json=bad, headers=ADMIN).status_code == 422
    assert client.get("/v1/settings").json()["overrides"] == {}
    assert client.patch("/v1/settings", json={"scoring": {"family_min": 0.6}}).status_code == 401
    assert client.delete("/v1/settings", params={"section": "nope"}, headers=ADMIN).status_code == 422
