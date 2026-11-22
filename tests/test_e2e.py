"""End-to-end tests on the synthetic samples (run tools/make_samples.py first) and the REST API."""
import json
from datetime import date
from pathlib import Path

import pytest

from cardagent.agent import CardAgent
from cardagent.schemas import ValidationResult
from cardagent.settings import get_settings

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "samples"
TODAY = date(2026, 11, 18)

pytestmark = pytest.mark.skipif(not (SAMPLES / "manifest.json").exists(),
                                reason="run tools/make_samples.py first")


@pytest.fixture(scope="module")
def agent():
    return CardAgent(get_settings())


def _case(name):
    return next(c for c in json.loads((SAMPLES / "manifest.json").read_text(encoding="utf-8")) if c["file"] == name)


@pytest.mark.parametrize("name", ["fazaa_gold.jpg", "esaad_privilege.jpg", "fazaa_gold_upside_down.jpg",
                                  "qa_demo01_gold.jpg"])
def test_accept_and_extract(agent, name):
    c = _case(name)
    r = agent.process((SAMPLES / name).read_bytes(), "t", TODAY)
    ValidationResult.model_validate(r)
    assert r["status"] == "ACCEPT", r["reasons"]
    assert r["card"]["family"] == c["family"] and r["card"]["variant"] == c["variant"]
    assert r["fields"]["card_number"]["value"] == c["fields"]["card_number"]
    assert r["quality"]["aligned_to_reference"] is True


def test_expired_rejected(agent):
    r = agent.process((SAMPLES / "esaad_gold_expired.jpg").read_bytes(), "t", TODAY)
    assert r["status"] == "REJECT" and r["reasons"][0] == "CARD_EXPIRED"


def test_unknown_card_rejected(agent):
    r = agent.process((SAMPLES / "unknown_card.jpg").read_bytes(), "t", TODAY)
    assert r["status"] == "REJECT" and "UNKNOWN_CARD_TYPE" in r["reasons"]


def test_bad_photo_asks_for_retake(agent):
    r = agent.process((SAMPLES / "fazaa_blurry.jpg").read_bytes(), "t", TODAY)
    assert r["status"] == "REJECT" and r["reasons"][0] == "RETAKE_REQUIRED"


def test_garbage_bytes(agent):
    r = agent.process(b"not an image", "t", TODAY)
    assert r["status"] == "REJECT" and r["reasons"] == ["INVALID_IMAGE"]


def test_hot_reload_keeps_version_when_unchanged(agent):
    before = agent.state.version
    agent.reload()
    assert agent.state.version == before


def test_api(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("CARDAGENT_API__ADMIN_TOKEN", "s3cret")
    get_settings.cache_clear()
    from cardagent.api.app import app
    with TestClient(app) as cl:
        assert cl.get("/health/ready").json()["status"] == "ready"
        with open(SAMPLES / "fazaa_silver.jpg", "rb") as fh:
            r = cl.post("/v1/cards/validate", files={"image": ("c.jpg", fh, "image/jpeg")},
                        data={"request_id": "req-1"})
        assert r.status_code == 200 and r.json()["card"]["variant"] == "silver"
        with open(SAMPLES / "fazaa_silver.jpg", "rb") as fh:   # idempotent replay
            r2 = cl.post("/v1/cards/validate", files={"image": ("c.jpg", fh, "image/jpeg")},
                         data={"request_id": "req-1"})
        assert r2.json()["processing_ms"] == r.json()["processing_ms"]
        with open(SAMPLES / "esaad_gold.jpg", "rb") as fh:     # same request_id, DIFFERENT photo
            r3 = cl.post("/v1/cards/validate", files={"image": ("c.jpg", fh, "image/jpeg")},
                         data={"request_id": "req-1"})
        assert r3.json()["card"]["family"] == "ae.esaad"         # processed fresh, not the cached Fazaa result
        bad = cl.post("/v1/cards/validate", files={"image": ("c.txt", b"x", "text/plain")})
        assert bad.status_code == 415
        assert cl.post("/v1/admin/registry/reload").status_code == 401
        assert cl.post("/v1/admin/registry/reload", headers={"X-Admin-Token": "s3cret"}).status_code == 200
        if cl.get("/health/ready").json()["vector_store"] == "faiss":           # card management needs Qdrant
            assert cl.get("/v1/admin/cards", headers={"X-Admin-Token": "s3cret"}).status_code == 501
        assert "cardagent_requests_total" in cl.get("/metrics").text
    get_settings.cache_clear()
