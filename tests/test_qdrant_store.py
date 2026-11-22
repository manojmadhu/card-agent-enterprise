"""Qdrant backend + card management API, on an in-process Qdrant (no server needed).

Uses a two-family copy of the registry (ae.fazaa, ae.esaad) so each test run embeds ~21 images."""
import json
import shutil
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from cardagent.index.reference_index import build_or_load
from cardagent.registry.store import load_registry
from cardagent.settings import PROJECT_ROOT, QdrantCfg, get_settings
from cardagent.store.faiss_store import FaissRefView
from cardagent.store.qdrant_store import QdrantCardStore
from cardagent.store.sync import sync_registry

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "samples"
FAMILIES = ("ae.fazaa", "ae.esaad")
TODAY = date(2026, 11, 18)


class TinyEmbedder:
    """Fast, model-free, deterministic embedder for store tests (Lab colour layout + coarse gray)."""
    name = "test-tiny-v1"
    dim = 16 * 10 * 3 + 32 * 20

    def embed(self, img):
        lab = cv2.cvtColor(cv2.resize(img, (16, 10), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(cv2.resize(img, (32, 20), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        v = np.concatenate([(lab.astype(np.float32) - 128).ravel(), (gray.astype(np.float32) - 128).ravel()])
        return (v / (np.linalg.norm(v) or 1)).astype(np.float32)


def _small_registry(dst: Path) -> Path:
    src = ROOT / "registry"
    (dst / "cards" / "ae").mkdir(parents=True)
    for fid in FAMILIES:
        name = fid.split(".")[1]
        shutil.copy(src / "cards" / "ae" / f"{name}.yaml", dst / "cards" / "ae" / f"{name}.yaml")
        shutil.copytree(src / "refs" / fid, dst / "refs" / fid)
    return dst


def _store(tmp_path, embedder=None) -> QdrantCardStore:
    cfg = QdrantCfg(location=":memory:", reference_store_dir=str(tmp_path / "blobs"))
    return QdrantCardStore(cfg, embedder or TinyEmbedder(), 500, PROJECT_ROOT)


# ------------------------------------------------------------------------------ store + sync
def test_incremental_sync_only_touches_changes(tmp_path):
    reg = _small_registry(tmp_path / "registry")
    store = _store(tmp_path)
    first = sync_registry(store, reg)
    n_refs = sum(1 for _ in (reg / "refs").rglob("*.png"))
    assert first["families_created"] == 2 and first["references_added"] == n_refs
    v1 = store.current_version()

    again = sync_registry(store, reg)                                     # nothing changed
    assert again["changed"] is False and again["references_unchanged"] == n_refs
    assert store.current_version() == v1                                  # replicas see no change

    victim = sorted((reg / "refs" / "ae.esaad" / "gold").glob("*.png"))[0]
    victim.unlink()                                                       # remove ONE reference image
    r = sync_registry(store, reg)
    assert r["references_deleted"] == 1 and r["references_added"] == 0
    assert r["references_unchanged"] == n_refs - 1
    assert store.current_version() != v1

    y = reg / "cards" / "ae" / "esaad.yaml"                               # move a field ROI
    fam = yaml.safe_load(y.read_text(encoding="utf-8"))
    fam["fields"][0]["roi"] = [0.05, 0.60, 0.80, 0.74]
    y.write_text(yaml.safe_dump(fam, allow_unicode=True), encoding="utf-8")
    r = sync_registry(store, reg)
    assert r["families_updated"] == 1 and r["keypoints_refreshed"] == n_refs - 1 - 9   # only esaad's refs
    assert r["references_added"] == 0                                    # no re-embedding

    shutil.rmtree(reg / "refs" / "ae.esaad")                              # remove the whole family
    y.unlink()
    r = sync_registry(store, reg)
    assert r["families_deleted"] == 1
    snap = store.snapshot()
    assert set(snap.families) == {"ae.fazaa"} and snap.ref_count == 9
    assert sync_registry(store, reg, prune=True)["changed"] is False


def test_qdrant_scores_match_faiss(tmp_path):
    """Same embeddings, same cosine scores: identification cannot change with the backend."""
    reg = _small_registry(tmp_path / "registry")
    emb = TinyEmbedder()
    store = _store(tmp_path, emb)
    sync_registry(store, reg)
    snap = load_registry(reg)
    faiss_view = FaissRefView(build_or_load(snap, emb, tmp_path / "index", 500))
    rng = np.random.default_rng(1)
    q = faiss_view.idx.vectors[3] + 0.05 * rng.standard_normal(emb.dim).astype(np.float32)
    q = (q / np.linalg.norm(q)).astype(np.float32)
    key = lambda h: (h.family_id, h.variant_id, round(h.score, 4))  # noqa: E731
    fq = sorted(map(key, faiss_view.family_refs(q, set(FAMILIES), 500)))
    qq = sorted(map(key, store.family_refs(q, set(FAMILIES), 500)))
    assert fq == qq
    assert [key(h) for h in faiss_view.search(q, 5)] == [key(h) for h in store.search(q, 5)]
    kp = store.keypoints([h.ref_id for h in store.search(q, 3)])
    assert len(kp) == 3 and all(k.des is not None and len(k.pts) == len(k.des) for k in kp.values())


def test_embedder_change_is_detected(tmp_path):
    store = _store(tmp_path)

    class Other(TinyEmbedder):
        name = "other-embedder"

    with pytest.raises(RuntimeError, match="--reembed"):
        QdrantCardStore(store.cfg, Other(), 500, PROJECT_ROOT, client=store.client)


# ------------------------------------------------------------------------------ admin API
needs_samples = pytest.mark.skipif(not (SAMPLES / "esaad_gold.jpg").exists(), reason="run tools/make_samples.py first")


@pytest.fixture
def api(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    reg = _small_registry(tmp_path / "registry")
    env = {"CARDAGENT_VECTOR_STORE__BACKEND": "qdrant", "CARDAGENT_VECTOR_STORE__QDRANT__LOCATION": ":memory:",
           "CARDAGENT_VECTOR_STORE__QDRANT__SYNC_ON_STARTUP": "true",
           "CARDAGENT_VECTOR_STORE__QDRANT__REFERENCE_STORE_DIR": str(tmp_path / "blobs"),
           "CARDAGENT_REGISTRY_DIR": str(reg), "CARDAGENT_API__ADMIN_TOKEN": "s3cret"}
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    get_settings.cache_clear()
    from cardagent.api.app import app
    with TestClient(app) as cl:
        yield cl, reg
    get_settings.cache_clear()


H = {"X-Admin-Token": "s3cret"}


def _validate(cl, name):
    with open(SAMPLES / name, "rb") as fh:
        return cl.post("/v1/cards/validate", files={"image": (name, fh, "image/jpeg")}).json()


@needs_samples
def test_card_lifecycle_through_api(api, tmp_path):
    cl, reg = api
    assert cl.get("/v1/admin/cards").status_code == 401                               # token required
    lst = cl.get("/v1/admin/cards", headers=H).json()
    assert lst["families"] == 2 and {i["source"] for i in lst["items"]} == {"registry"}
    assert _validate(cl, "esaad_gold.jpg")["card"]["family"] == "ae.esaad"

    definition = yaml.safe_load((reg / "cards" / "ae" / "esaad.yaml").read_text(encoding="utf-8"))
    assert cl.put("/v1/admin/cards/ae.esaad", json=definition, headers=H).status_code == 409  # registry-owned
    assert cl.delete("/v1/admin/cards/ae.esaad", headers=H).status_code == 409

    # ---- delete a card: only its points go, the service stops recognising it immediately
    r = cl.delete("/v1/admin/cards/ae.esaad?force=true", headers=H)
    assert r.status_code == 200 and r.json()["references_deleted"] == 12 and "note" in r.json()
    assert cl.get("/v1/admin/cards/ae.esaad", headers=H).status_code == 404
    gone = _validate(cl, "esaad_gold.jpg")
    assert gone["status"] != "ACCEPT" and (gone.get("card") or {}).get("family") != "ae.esaad"
    assert _validate(cl, "fazaa_gold.jpg")["status"] == "ACCEPT"                       # others untouched

    # ---- add it back as an API-managed card: definition first, then reference images
    r = cl.put("/v1/admin/cards/ae.esaad", content=yaml.safe_dump(definition, allow_unicode=True),
               headers={**H, "Content-Type": "application/yaml"})
    assert r.status_code == 201 and r.json()["source"] == "api"
    assert sorted(r.json()["variants_without_references"]) == ["classic", "gold", "platinum", "privilege"]
    gold = sorted((reg / "refs" / "ae.esaad" / "gold").glob("*.png"))
    files = [("images", (p.name, p.read_bytes(), "image/png")) for p in gold]
    r = cl.post("/v1/admin/cards/ae.esaad/variants/gold/references", files=files, headers=H)
    assert r.status_code == 201 and r.json()["created"] == 3
    assert all(x["card_detected"] is False for x in r.json()["results"])               # flat scans used as-is
    assert len(list((tmp_path / "blobs" / "ae.esaad" / "gold").glob("*"))) == 3        # copies kept
    again = cl.post("/v1/admin/cards/ae.esaad/variants/gold/references", files=files[:1], headers=H)
    assert again.status_code == 200 and again.json()["results"][0]["status"] == "exists"  # idempotent

    back = _validate(cl, "esaad_gold.jpg")
    assert back["status"] == "ACCEPT" and back["card"]["family"] == "ae.esaad" and back["card"]["variant"] == "gold"

    # ---- delete one reference image
    card = cl.get("/v1/admin/cards/ae.esaad", headers=H).json()
    gold_refs = next(v for v in card["variants"] if v["id"] == "gold")["reference_images"]
    r = cl.delete(f"/v1/admin/cards/ae.esaad/references/{gold_refs[0]['ref_id']}", headers=H)
    assert r.status_code == 200 and r.json()["reference_images_in_variant"] == 2
    assert r.json()["stored_images_deleted"] == 1

    # ---- registry sync leaves the API-owned card alone
    s = cl.post("/v1/admin/cards/sync", headers=H).json()
    assert s["skipped_api_owned"] == ["ae.esaad"] and s["references_added"] == 0
    assert cl.get("/v1/admin/cards/ae.esaad", headers=H).json()["source"] == "api"


def test_admin_validation_errors(api):
    cl, reg = api
    definition = yaml.safe_load((reg / "cards" / "ae" / "fazaa.yaml").read_text(encoding="utf-8"))
    new = {**definition, "id": "ae.newcard", "keywords": ["NEWCARD"]}
    assert cl.put("/v1/admin/cards/ae.other", json=new, headers=H).status_code == 422    # id mismatch
    assert cl.put("/v1/admin/cards/ae.newcard", json={**new, "id": "Bad Id"}, headers=H).status_code == 422
    assert cl.put("/v1/admin/cards/ae.newcard", content=b"{not json", headers=H).status_code == 400
    assert cl.put("/v1/admin/cards/ae.newcard", json=new, headers=H).status_code == 201
    assert cl.put("/v1/admin/cards/ae.newcard", json=new, headers=H).status_code == 200    # replace
    png = (next((reg / "refs" / "ae.fazaa" / "gold").glob("*.png"))).read_bytes()
    assert cl.post("/v1/admin/cards/ae.newcard/variants/nope/references",
                   files=[("images", ("a.png", png, "image/png"))], headers=H).status_code == 404
    assert cl.post("/v1/admin/cards/ae.missing/variants/gold/references",
                   files=[("images", ("a.png", png, "image/png"))], headers=H).status_code == 404
    assert cl.post("/v1/admin/cards/ae.newcard/variants/gold/references",
                   files=[("images", ("a.txt", b"x", "text/plain"))], headers=H).status_code == 415
    bad = cl.post("/v1/admin/cards/ae.newcard/variants/gold/references",
                  files=[("images", ("a.png", b"not an image", "image/png"))], headers=H)
    assert bad.status_code == 422 and bad.json()["results"][0]["status"] == "rejected"
    # dropping a variant from the definition deletes its reference images
    cl.post("/v1/admin/cards/ae.newcard/variants/silver/references",
            files=[("images", ("s.png", png, "image/png"))], headers=H)
    slim = {**new, "variants": [v for v in new["variants"] if v["id"] != "silver"]}
    r = cl.put("/v1/admin/cards/ae.newcard", json=slim, headers=H)
    assert r.status_code == 200 and r.json()["references_deleted"] == 1
    assert cl.delete("/v1/admin/cards/ae.newcard", headers=H).status_code == 200          # api-owned: no force
    assert cl.delete("/v1/admin/cards/ae.newcard", headers=H).status_code == 404
