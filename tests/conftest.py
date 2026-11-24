"""Hermetic test setup: in-process Qdrant (location=":memory:"), the classic embedder (no model
download), and every data directory in a temporary folder. Each test module gets a fresh platform
(new in-memory Qdrant) through the `client` fixture. Needs Tesseract with the eng traineddata."""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="cardagent-tests-"))
os.environ.update({
    "CARDAGENT_QDRANT__LOCATION": ":memory:",
    "CARDAGENT_QDRANT__SYNC_ON_STARTUP": "false",
    "CARDAGENT_QDRANT__REFERENCE_STORE_DIR": str(_TMP / "reference_store"),
    "CARDAGENT_REGISTRY_DIR": str(_TMP / "registry"),
    "CARDAGENT_RUNS__STORE_DIR": str(_TMP / "run_store"),
    "CARDAGENT_EVALUATION__STORE_DIR": str(_TMP / "evaluations"),
    "CARDAGENT_EMBEDDER__KIND": "classic",
    "CARDAGENT_EMBEDDER__SIMILARITY_FLOOR": "0.55",
    "CARDAGENT_VLM__ENABLED": "false",
    "CARDAGENT_API__ADMIN_TOKEN": "test-admin-token",
    "CARDAGENT_API__CORS_ORIGINS": "[]",
    "CARDAGENT_LOGGING__JSON": "false",
    "CARDAGENT_LOGGING__LEVEL": "WARNING",
})

from fastapi.testclient import TestClient  # noqa: E402

from cardagent.api.app import app  # noqa: E402

from . import cards  # noqa: E402

ADMIN = {"X-Admin-Token": "test-admin-token"}
REGISTRY = _TMP / "registry"


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_TMP, ignore_errors=True)


@pytest.fixture(scope="module")
def client():
    """A fresh, empty platform (new in-memory Qdrant) per test module."""
    shutil.rmtree(REGISTRY, ignore_errors=True)
    (REGISTRY / "cards").mkdir(parents=True)
    with TestClient(app) as c:
        yield c


def add_card(c: TestClient, family_id: str, refs_per_variant: int = 3):
    r = c.put(f"/v1/cards/{family_id}", json=cards.FAMILIES[family_id], headers=ADMIN)
    assert r.status_code in (200, 201), r.text
    for v in cards.FAMILIES[family_id]["variants"]:
        files = [("images", (f"{v['id']}_{i}.png", data, "image/png"))
                 for i, data in enumerate(cards.reference_pngs(family_id, v["id"], refs_per_variant))]
        r = c.post(f"/v1/cards/{family_id}/variants/{v['id']}/references", files=files,
                   params={"detect_card": "false"}, headers=ADMIN)
        assert r.status_code == 201, r.text


@pytest.fixture(scope="module")
def loaded(client):
    """The platform with both test programmes and their reference images."""
    add_card(client, "ae.alpha")
    add_card(client, "sa.beta")
    return client


def validate(c: TestClient, image: bytes, request_id: str | None = None, **params):
    data = {"request_id": request_id} if request_id else {}
    return c.post("/v1/cards/validate", files={"image": ("photo.jpg", image, "image/jpeg")}, data=data, params=params)
