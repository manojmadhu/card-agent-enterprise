"""Card Agent Console (card-agent-web) compatibility API: /console/v1.

Unit tests for the format converters, then the API on both storage backends with a two-family
copy of the registry (the FAISS test edits files in a temp registry, never the repo's)."""
import shutil
from datetime import date
from pathlib import Path

import pytest
import yaml

from cardagent.console.schema import (ConsoleFamily, console_id, resolve_enterprise_id, to_console,
                                      to_console_result, to_enterprise)
from cardagent.pipeline.rules import run_rules
from cardagent.registry.models import CardFamily
from cardagent.settings import get_settings

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "samples"
H = {"X-Admin-Token": "s3cret"}
YAML_H = {**H, "Content-Type": "application/yaml"}
needs_samples = pytest.mark.skipif(not (SAMPLES / "fazaa_gold.jpg").exists(), reason="run tools/make_samples.py first")

# exactly what the console's seed data and family editor send (MVP schema)
CONSOLE_YAML = """\
family: testcard
family_name: Test Card
issuer: Test issuer
schema_version: 1
keywords: [TESTCARD]
negative_keywords: [ESAAD]
weights: {text: 0.55, visual: 0.30, colour: 0.15}
variants:
  - {id: gold, name: Test Gold, body_color_rgb: [201, 162, 39], keywords: [GOLD], number_prefix: "2"}
  - {id: silver, name: Test Silver, body_color_rgb: [192, 194, 198], keywords: [SILVER], number_prefix: "1"}
fields:
  - {name: holder_name, roi: [0.05, 0.55, 0.75, 0.68], regex: "^[A-Z][A-Z .'-]{2,40}$", required: true, normalise: upper}
  - {name: card_number, roi: [0.05, 0.40, 0.75, 0.54], regex: "^\\\\d{10}$", required: true, normalise: digits}
  - {name: issue_date, roi: [0.55, 0.72, 0.95, 0.86], regex: "^(0[1-9]|1[0-2])/\\\\d{2}$", normalise: date_mm_yy}
  - {name: expiry_date, roi: [0.05, 0.72, 0.50, 0.86], regex: "^(0[1-9]|1[0-2])/\\\\d{2}$", required: true, normalise: date_mm_yy, anchor: EXP}
rules:
  - {type: regex, field: card_number, critical: true}
  - {type: not_expired, field: expiry_date, critical: true}
  - {type: date_order, field: issue_date, other_field: expiry_date}
"""


def _enterprise(name):
    return CardFamily.model_validate(yaml.safe_load((ROOT / "registry" / "cards" / "ae" / f"{name}.yaml").read_text(encoding="utf-8")))


# ------------------------------------------------------------------------------ converters
def test_family_round_trip_keeps_enterprise_definition():
    for name in ("fazaa", "esaad"):
        ent = _enterprise(name)
        doc = to_console(ent, name)
        cf = ConsoleFamily.model_validate(doc)                     # valid against the console's schema
        assert cf.family == name and doc["weights"] == {"text": 0.55, "visual": 0.30, "colour": 0.15}
        back = to_enterprise(cf, ent.id, ent)
        assert back.model_dump() == ent.model_dump()               # nothing lost


def test_console_yaml_converts_and_new_rule_type_works():
    cf = ConsoleFamily.model_validate(yaml.safe_load(CONSOLE_YAML))
    ent = to_enterprise(cf, "ae.testcard")
    assert ent.country == "AE" and ent.variant("gold").number_prefix == "2"
    rule = next(r for r in ent.rules if r.type == "date_order")
    assert rule.params == {"other_field": "expiry_date"}
    f = lambda i, e: {"issue_date": {"value": i}, "expiry_date": {"value": e}, "card_number": {"value": "2000000000"}}  # noqa: E731
    ok = {r["rule"]: r["result"] for r in run_rules(ent, "gold", f("01/24", "12/28"), today=date(2026, 11, 23))}
    bad = {r["rule"]: r["result"] for r in run_rules(ent, "gold", f("12/28", "01/27"), today=date(2026, 11, 23))}
    assert ok["date_order"] == "pass" and bad["date_order"] == "fail"
    with pytest.raises(ValueError):                                # date_order must name its other field
        CardFamily.model_validate({**ent.model_dump(), "rules": [{"type": "date_order", "field": "issue_date"}]})


def test_console_schema_rules_match_the_web_app():
    base = yaml.safe_load(CONSOLE_YAML)
    for bad in ({"weights": {"text": 0.9, "visual": 0.3, "colour": 0.1}},       # must sum to 1
                {"family": "Test Card"}, {"keywords": []}, {"variants": []},
                {"variants": [{"id": "gold", "name": "G", "body_color_rgb": [300, 0, 0]}]}):
        with pytest.raises(ValueError):
            ConsoleFamily.model_validate({**base, **bad})


def test_id_mapping():
    ids = ["ae.fazaa", "ae.esaad", "sa.esaad"]
    assert console_id("ae.fazaa", ids) == "fazaa"
    assert console_id("sa.esaad", ids) == "sa-esaad"               # short name clashes -> country prefix
    for eid in ids:
        assert resolve_enterprise_id(console_id(eid, ids), ids, "ae") == eid
    assert resolve_enterprise_id("newcard", ids, "ae") == "ae.newcard"
    assert resolve_enterprise_id("qa-new-card", ids, "ae") == "qa.new_card"


def test_result_translation_matches_console_expectations():
    fams = {f.id: f for f in (_enterprise("fazaa"), _enterprise("esaad"))}
    cands = {"family": [{"id": "ae.esaad", "score": 0.48, "signals": {"text": 0.9}},
                        {"id": "ae.fazaa", "score": 0.42, "signals": {}}]}
    amb = to_console_result({"request_id": "r", "status": "REVIEW", "reasons": ["AMBIGUOUS_CARD_TYPE"],
                             "quality": {"blur": 1}, "candidates": cands}, fams)
    assert amb["reasons"] == ["AMBIGUOUS_FAMILY"]
    assert amb["candidates"][0] == {"family": "esaad", "name": "Esaad", "score": 0.48, "signals": {"text": 0.9}}
    assert amb["card"]["provisional"] is True and amb["card"]["family_name"] == "Esaad"   # console reads card
    unk = to_console_result({"request_id": "r", "status": "REJECT", "reasons": ["UNKNOWN_CARD_TYPE"],
                             "quality": {"blur": 1}, "candidates": {"family": []}}, fams)
    assert unk["reasons"] == ["UNKNOWN_CARD"] and unk["card"] is None and unk["candidates"] == []
    inv = to_console_result({"request_id": "r", "status": "REJECT", "reasons": ["INVALID_IMAGE"]}, fams)
    assert inv["reasons"][0] == "RETAKE_REQUIRED" and inv["quality"]["blur"] == 0.0     # never null


# ------------------------------------------------------------------------------ API helpers
def _small_registry(dst: Path) -> Path:
    src = ROOT / "registry"
    (dst / "cards" / "ae").mkdir(parents=True)
    for name in ("fazaa", "esaad"):
        shutil.copy(src / "cards" / "ae" / f"{name}.yaml", dst / "cards" / "ae" / f"{name}.yaml")
        shutil.copytree(src / "refs" / f"ae.{name}", dst / "refs" / f"ae.{name}")
    return dst


def _client(monkeypatch, env):
    from fastapi.testclient import TestClient
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    get_settings.cache_clear()
    from cardagent.api.app import app
    return TestClient(app)


def _validate(cl, name, rid="req_1"):
    with open(SAMPLES / name, "rb") as fh:
        r = cl.post("/console/v1/cards/validate", files={"image": ("card.jpg", fh, "image/jpeg")},
                    data={"request_id": rid})
    assert r.status_code == 200, r.text
    return r.json()


def _ref_files(reg, family, variant):
    return [("images", (p.name, p.read_bytes(), "image/png"))
            for p in sorted((reg / "refs" / family / variant).glob("*.png"))]


@pytest.fixture
def faiss_api(tmp_path, monkeypatch):
    reg = _small_registry(tmp_path / "registry")
    env = {"CARDAGENT_VECTOR_STORE__BACKEND": "faiss", "CARDAGENT_REGISTRY_DIR": str(reg),
           "CARDAGENT_INDEX_DIR": str(tmp_path / "index"), "CARDAGENT_API__ADMIN_TOKEN": "s3cret"}
    with _client(monkeypatch, env) as cl:
        yield cl, reg
    get_settings.cache_clear()


@pytest.fixture
def qdrant_api(tmp_path, monkeypatch):
    reg = _small_registry(tmp_path / "registry")
    env = {"CARDAGENT_VECTOR_STORE__BACKEND": "qdrant", "CARDAGENT_VECTOR_STORE__QDRANT__LOCATION": ":memory:",
           "CARDAGENT_VECTOR_STORE__QDRANT__SYNC_ON_STARTUP": "true",
           "CARDAGENT_VECTOR_STORE__QDRANT__REFERENCE_STORE_DIR": str(tmp_path / "blobs"),
           "CARDAGENT_REGISTRY_DIR": str(reg), "CARDAGENT_API__ADMIN_TOKEN": "s3cret"}
    with _client(monkeypatch, env) as cl:
        yield cl, reg
    get_settings.cache_clear()


# ------------------------------------------------------------------------------ API: routes the console calls
@needs_samples
@pytest.mark.parametrize("backend", ["faiss_api", "qdrant_api"])
def test_console_read_routes(backend, request):
    cl, _ = request.getfixturevalue(backend)
    reg = cl.get("/console/v1/registry").json()
    assert isinstance(reg["families"], list) and {f["family"] for f in reg["families"]} == {"fazaa", "esaad"}
    for f in reg["families"]:
        ConsoleFamily.model_validate({k: v for k, v in f.items()
                                      if k not in ("enterprise_id", "country", "reference_images")})

    r = _validate(cl, "fazaa_gold.jpg")
    assert r["status"] == "ACCEPT" and r["card"]["family"] == "fazaa" and r["card"]["variant"] == "gold"
    assert isinstance(r["candidates"], list) and r["candidates"][0]["family"] == "fazaa"
    assert {"blur", "brightness", "glare_ratio"} <= set(r["quality"])
    unknown = _validate(cl, "unknown_card.jpg", "req_2")
    assert unknown["status"] == "REJECT" and "UNKNOWN_CARD" in unknown["reasons"]
    retake = _validate(cl, "fazaa_blurry.jpg", "req_3")
    assert retake["reasons"][0] == "RETAKE_REQUIRED"

    v1 = cl.post("/v1/cards/validate", files={"image": ("c.jpg", (SAMPLES / "fazaa_gold.jpg").read_bytes(), "image/jpeg")})
    assert v1.json()["card"]["family"] == "ae.fazaa" and isinstance(v1.json()["candidates"], dict)  # /v1 unchanged

    y = cl.get("/console/v1/registry/families/fazaa")
    assert y.headers["content-type"].startswith("application/yaml")
    assert yaml.safe_load(y.text)["family"] == "fazaa"
    assert cl.get("/console/v1/registry/families/nope").status_code == 404


@needs_samples
def test_console_writes_on_faiss_edit_registry_files(faiss_api):
    cl, reg = faiss_api
    assert cl.put("/console/v1/registry/families/testcard", content=CONSOLE_YAML,
                  headers={"Content-Type": "application/yaml"}).status_code == 401        # token required
    assert cl.post("/console/v1/registry/reload").status_code == 401
    assert cl.put("/console/v1/registry/families/other", content=CONSOLE_YAML, headers=YAML_H).status_code == 422
    bad = CONSOLE_YAML.replace("text: 0.55", "text: 0.95")
    assert cl.put("/console/v1/registry/families/testcard", content=bad, headers=YAML_H).status_code == 422

    r = cl.put("/console/v1/registry/families/testcard", content=CONSOLE_YAML, headers=YAML_H)
    assert r.status_code == 201 and r.json()["enterprise_id"] == "ae.testcard"
    saved = reg / "cards" / "ae" / "testcard.yaml"
    assert CardFamily.model_validate(yaml.safe_load(saved.read_text(encoding="utf-8"))).id == "ae.testcard"
    assert cl.put("/console/v1/registry/families/testcard", content=CONSOLE_YAML, headers=YAML_H).status_code == 200
    assert "testcard" in {f["family"] for f in cl.get("/console/v1/registry").json()["families"]}

    r = cl.post("/console/v1/registry/families/testcard/variants/gold/references",
                files=_ref_files(reg, "ae.fazaa", "gold"), headers=H)
    assert r.status_code == 201 and r.json()["created"] == 3 and r.json()["reference_images_in_variant"] == 3
    refs = cl.get("/console/v1/registry/families/testcard/references").json()
    gold = next(v for v in refs["variants"] if v["id"] == "gold")["references"]
    assert len(gold) == 3
    r = cl.delete(f"/console/v1/registry/families/testcard/references/{gold[0]['ref_id']}", headers=H)
    assert r.status_code == 200 and r.json()["reference_images_in_variant"] == 2

    assert cl.post("/console/v1/registry/reload", headers=H).json()["families"] == 3
    r = cl.delete("/console/v1/registry/families/testcard", headers=H)
    assert r.status_code == 200 and not saved.exists() and any((reg / ".deleted").rglob("testcard.yaml"))
    assert cl.get("/console/v1/registry/families/testcard").status_code == 404
    assert _validate(cl, "fazaa_gold.jpg")["status"] == "ACCEPT"                          # others untouched


@needs_samples
def test_console_writes_on_qdrant_respect_registry_ownership(qdrant_api):
    cl, reg = qdrant_api
    fazaa = cl.get("/console/v1/registry/families/fazaa").text
    r = cl.put("/console/v1/registry/families/fazaa", content=fazaa, headers=YAML_H)
    assert r.status_code == 409                                                         # owned by git registry
    r = cl.put("/console/v1/registry/families/fazaa?force=true", content=fazaa, headers=YAML_H)
    assert r.status_code == 200 and r.json()["source"] == "api"
    assert cl.get("/v1/admin/cards/ae.fazaa", headers=H).json()["source"] == "api"      # same store as /v1/admin

    r = cl.put("/console/v1/registry/families/testcard", content=CONSOLE_YAML, headers=YAML_H)
    assert r.status_code == 201
    r = cl.post("/console/v1/registry/families/testcard/variants/gold/references",
                files=_ref_files(reg, "ae.fazaa", "gold"), headers=H)
    assert r.status_code == 201 and r.json()["created"] == 3
    again = cl.post("/console/v1/registry/families/testcard/variants/gold/references",
                    files=_ref_files(reg, "ae.fazaa", "gold")[:1], headers=H)
    assert again.json()["results"][0]["status"] == "exists"
    refs = cl.get("/console/v1/registry/families/testcard/references").json()
    ref = next(v for v in refs["variants"] if v["id"] == "gold")["references"][0]
    assert cl.delete(f"/console/v1/registry/families/testcard/references/{ref['ref_id']}", headers=H).status_code == 200
    assert cl.delete("/console/v1/registry/families/testcard", headers=H).status_code == 200
    assert cl.delete("/console/v1/registry/families/esaad", headers=H).status_code == 409   # registry-owned
    assert _validate(cl, "esaad_gold.jpg")["card"]["family"] == "esaad"


def test_console_api_can_be_disabled(faiss_api, monkeypatch):
    cl, _ = faiss_api
    monkeypatch.setenv("CARDAGENT_CONSOLE__ENABLED", "false")
    get_settings.cache_clear()
    assert cl.get("/console/v1/registry").status_code == 404
    assert cl.get("/v1/registry").status_code == 200
