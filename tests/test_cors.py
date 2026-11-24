"""CORS for a browser app on another origin (the web console on the Vite dev server)."""
import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient
from pydantic import ValidationError

from cardagent.api.app import cors_options
from cardagent.settings import ApiCfg, Settings

ORIGIN = "http://localhost:5173"


def test_origins_from_env_json_list(monkeypatch):
    monkeypatch.setenv("CARDAGENT_API__CORS_ORIGINS", '["http://localhost:5173"]')
    assert Settings().api.cors_origins == [ORIGIN]


def test_origins_are_normalised_and_checked():
    assert ApiCfg(cors_origins=["http://localhost:5173/"]).cors_origins == [ORIGIN]
    assert ApiCfg(cors_origins="http://localhost:5173, http://127.0.0.1:5173").cors_origins == [
        ORIGIN, "http://127.0.0.1:5173"]
    for bad in (["localhost:5173"], ["http://localhost:5173/console"]):
        with pytest.raises(ValidationError):
            ApiCfg(cors_origins=bad)


@pytest.fixture(scope="module")
def cors_client():
    """The API's CORS policy (cors_options) on a tiny app, so the shared test app is not changed."""
    mini = FastAPI()
    mini.add_middleware(CORSMiddleware, **cors_options([ORIGIN]))

    @mini.get("/v1/x")
    def x():
        return {"ok": True}
    return TestClient(mini)


def test_preflight_for_admin_calls(cors_client):
    r = cors_client.options("/v1/x", headers={"Origin": ORIGIN, "Access-Control-Request-Method": "PATCH",
                                              "Access-Control-Request-Headers": "content-type,x-admin-token"})
    assert r.status_code == 200 and r.headers["access-control-allow-origin"] == ORIGIN
    assert r.headers["access-control-max-age"] == "3600"


def test_allowed_origin_can_read_retry_after_and_others_get_nothing(cors_client):
    r = cors_client.get("/v1/x", headers={"Origin": ORIGIN})
    assert r.headers["access-control-allow-origin"] == ORIGIN
    assert "Retry-After" in r.headers["access-control-expose-headers"]
    assert "access-control-allow-origin" not in cors_client.get("/v1/x", headers={"Origin": "http://evil.example"}).headers
