"""Cards and reference images through the API, registry summary and git-registry sync."""
import shutil

import yaml

from . import cards
from .conftest import ADMIN, REGISTRY, add_card


def test_platform_starts_empty(client):
    assert client.get("/health/live").json() == {"status": "ok"}
    ready = client.get("/health/ready").json()
    assert ready["status"] == "ready" and ready["families"] == 0 and ready["references"] == 0
    assert client.get("/v1/cards").json()["items"] == []
    assert "cardagent_requests_total" in client.get("/metrics").text or client.get("/metrics").status_code == 200


def test_admin_endpoints_need_the_token(client):
    assert client.put("/v1/cards/ae.alpha", json=cards.ALPHA).status_code == 401
    assert client.put("/v1/cards/ae.alpha", json=cards.ALPHA, headers={"X-Admin-Token": "wrong"}).status_code == 401
    assert client.get("/v1/runs").status_code == 401


def test_create_card_from_json_and_yaml(client):
    r = client.put("/v1/cards/ae.alpha", json=cards.ALPHA, headers=ADMIN)
    assert r.status_code == 201 and r.json()["created"] is True
    assert r.json()["variants_without_references"] == ["blue", "amber"]
    body = yaml.safe_dump(cards.BETA)
    r = client.put("/v1/cards/sa.beta", content=body, headers={**ADMIN, "Content-Type": "application/yaml"})
    assert r.status_code == 201
    r = client.put("/v1/cards/sa.beta", json=cards.BETA, headers=ADMIN)             # unchanged: 200
    assert r.status_code == 200 and r.json()["changed"] is False
    assert [i["family"] for i in client.get("/v1/cards").json()["items"]] == ["ae.alpha", "sa.beta"]
    as_yaml = client.get("/v1/cards/ae.alpha", params={"format": "yaml"})
    assert as_yaml.headers["content-type"].startswith("application/yaml")
    assert yaml.safe_load(as_yaml.text)["id"] == "ae.alpha"


def test_invalid_card_definitions(client):
    assert client.put("/v1/cards/ae.other", json=cards.ALPHA, headers=ADMIN).status_code == 422   # id mismatch
    assert client.put("/v1/cards/ae.alpha", json={**cards.ALPHA, "variants": "x"}, headers=ADMIN).status_code == 422
    assert client.put("/v1/cards/ae.alpha", content=b"{nope", headers={**ADMIN, "Content-Type": "application/json"}
                      ).status_code == 400
    assert client.get("/v1/cards/ae.missing").status_code == 404


def test_reference_images_lifecycle(client):
    pngs = cards.reference_pngs("ae.alpha", "blue", 2)
    files = [("images", (f"b{i}.png", d, "image/png")) for i, d in enumerate(pngs)]
    r = client.post("/v1/cards/ae.alpha/variants/blue/references", files=files, params={"detect_card": "false"},
                    headers=ADMIN)
    assert r.status_code == 201 and r.json()["created"] == 2 and r.json()["reference_images_in_variant"] == 2
    again = client.post("/v1/cards/ae.alpha/variants/blue/references", files=files[:1],
                        params={"detect_card": "false"}, headers=ADMIN)
    assert again.status_code == 200 and again.json()["results"][0]["status"] == "exists"
    assert client.post("/v1/cards/ae.alpha/variants/gold/references", files=files[:1], headers=ADMIN).status_code == 404
    bad_type = [("images", ("x.gif", b"GIF89a", "image/gif"))]
    assert client.post("/v1/cards/ae.alpha/variants/blue/references", files=bad_type, headers=ADMIN).status_code == 415

    refs = client.get("/v1/cards/ae.alpha/references", headers=ADMIN).json()
    blue = next(v for v in refs["variants"] if v["id"] == "blue")["references"]
    assert len(blue) == 2 and all(x["source"] == "api" for x in blue)
    img = client.get(f"/v1/cards/ae.alpha/references/{blue[0]['ref_id']}/image", headers=ADMIN)
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"

    r = client.delete(f"/v1/cards/ae.alpha/references/{blue[0]['ref_id']}", headers=ADMIN)
    assert r.status_code == 200 and r.json()["reference_images_in_variant"] == 1
    assert client.delete(f"/v1/cards/ae.alpha/references/{blue[0]['ref_id']}", headers=ADMIN).status_code == 404


def test_removing_a_variant_deletes_its_references(client):
    add_card(client, "ae.alpha", refs_per_variant=1)
    assert client.get("/v1/cards/ae.alpha").json()["variants"][1]["reference_images"] == 1
    one_variant = {**cards.ALPHA, "variants": cards.ALPHA["variants"][:1]}
    r = client.put("/v1/cards/ae.alpha", json=one_variant, headers=ADMIN)
    assert r.status_code == 200 and r.json()["references_deleted"] == 1


def test_registry_summary_and_reload(client):
    reg = client.get("/v1/registry").json()
    assert reg["families"] == 2 and reg["reference_images"] >= 1
    assert client.post("/v1/registry/reload", headers=ADMIN).json()["version"] == reg["version"]


def test_delete_card(client):
    r = client.delete("/v1/cards/sa.beta", headers=ADMIN)
    assert r.status_code == 200 and r.json()["family"] == "sa.beta"
    assert client.get("/v1/cards/sa.beta").status_code == 404
    assert client.delete("/v1/cards/sa.beta", headers=ADMIN).status_code == 404


def test_git_registry_sync_and_ownership(client):
    (REGISTRY / "cards" / "sa").mkdir(parents=True, exist_ok=True)
    (REGISTRY / "cards" / "sa" / "beta.yaml").write_text(yaml.safe_dump(cards.BETA), encoding="utf-8")
    refdir = REGISTRY / "refs" / "sa.beta" / "red"
    refdir.mkdir(parents=True, exist_ok=True)
    for i, d in enumerate(cards.reference_pngs("sa.beta", "red", 2)):
        (refdir / f"ref_{i}.png").write_bytes(d)
    out = client.post("/v1/registry/sync", headers=ADMIN).json()
    assert out["families_created"] == 1 and out["references_added"] == 2
    assert client.post("/v1/registry/sync", headers=ADMIN).json()["references_added"] == 0           # incremental
    assert client.get("/v1/cards/sa.beta").json()["source"] == "registry"

    # registry-owned cards are changed in git, unless forced
    assert client.put("/v1/cards/sa.beta", json=cards.BETA, headers=ADMIN).status_code == 409
    ref_id = client.get("/v1/cards/sa.beta/references", headers=ADMIN).json()["variants"][0]["references"][0]["ref_id"]
    assert client.delete(f"/v1/cards/sa.beta/references/{ref_id}", headers=ADMIN).status_code == 409
    assert client.delete("/v1/cards/sa.beta", headers=ADMIN).status_code == 409

    # removing the files prunes the card on the next sync; API-owned cards are left alone
    shutil.rmtree(REGISTRY / "cards" / "sa")
    out = client.post("/v1/registry/sync", headers=ADMIN).json()
    assert out["families_deleted"] == 1
    assert client.get("/v1/cards/sa.beta").status_code == 404
    assert client.get("/v1/cards/ae.alpha").status_code == 200


def test_invalid_registry_changes_nothing(client):
    (REGISTRY / "cards" / "bad.yaml").write_text("id: Bad\n", encoding="utf-8")
    r = client.post("/v1/registry/sync", headers=ADMIN)
    assert r.status_code == 422
    (REGISTRY / "cards" / "bad.yaml").unlink()
