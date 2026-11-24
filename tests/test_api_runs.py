"""Run history, review queue (with enrollment of a reviewed photo), statistics and pruning."""
from . import cards
from .conftest import ADMIN, validate


def photo(variant, number, seed):
    return cards.jpg(cards.photo(cards.render("ae.alpha", variant, "SARA MANSOOR", number, "05/30"), seed))


def test_runs_are_recorded_and_listed(loaded):
    ids = [validate(loaded, photo("blue", f"10000000{i:02d}", 20 + i)).json()["run_id"] for i in range(3)]
    page = loaded.get("/v1/runs", params={"limit": 2}, headers=ADMIN).json()
    assert [x["run_id"] for x in page["items"]] == ids[::-1][:2]                 # newest first
    nxt = loaded.get("/v1/runs", params={"limit": 2, "before": page["next_before"]}, headers=ADMIN).json()
    assert nxt["items"][0]["run_id"] == ids[0]
    run = loaded.get(f"/v1/runs/{ids[0]}", headers=ADMIN).json()
    assert run["status"] == "ACCEPT" and run["result"]["card"]["variant"] == "blue" and run["source"] == "api"
    img = loaded.get(f"/v1/runs/{ids[0]}/image", headers=ADMIN)
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    assert loaded.get("/v1/runs", params={"status": "REJECT"}, headers=ADMIN).json()["items"] == []
    assert loaded.get("/v1/runs", params={"since": "yesterday"}, headers=ADMIN).status_code == 422
    assert loaded.get("/v1/runs/does-not-exist", headers=ADMIN).status_code == 404


def test_review_queue_and_enroll(loaded):
    # a strict confidence threshold turns the next decision into REVIEW
    loaded.patch("/v1/settings", json={"scoring": {"field_min_confidence": 0.999}}, headers=ADMIN)
    body = validate(loaded, photo("amber", "2555555555", 31)).json()
    loaded.patch("/v1/settings", json={"scoring": {"field_min_confidence": None}}, headers=ADMIN)
    assert body["status"] == "REVIEW" and any(r.startswith("LOW_CONFIDENCE") for r in body["reasons"])

    queue = loaded.get("/v1/review", headers=ADMIN).json()
    assert queue["pending"] == 1 and queue["items"][0]["run_id"] == body["run_id"]

    before = loaded.get("/v1/cards/ae.alpha").json()["variants"][1]["reference_images"]
    bad = loaded.post(f"/v1/runs/{body['run_id']}/review", json={"decision": "reject", "enroll": True}, headers=ADMIN)
    assert bad.status_code == 422
    bad = loaded.post(f"/v1/runs/{body['run_id']}/review", json={"decision": "accept", "fields": {"nope": "1"}},
                      headers=ADMIN)
    assert bad.status_code == 422
    r = loaded.post(f"/v1/runs/{body['run_id']}/review", headers=ADMIN,
                    json={"decision": "accept", "reviewer": "qa1", "note": "fine", "enroll": True,
                          "fields": {"holder_name": "SARA MANSOOR"}})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["review_status"] == "accepted" and out["review"]["enrolled"]["status"] == "created"
    assert loaded.get("/v1/review", headers=ADMIN).json()["pending"] == 0
    assert loaded.get("/v1/cards/ae.alpha").json()["variants"][1]["reference_images"] == before + 1


def test_review_with_correction_of_an_unknown_card(loaded):
    run = validate(loaded, cards.not_a_card(9)).json()
    r = loaded.post(f"/v1/runs/{run['run_id']}/review", json={"decision": "accept"}, headers=ADMIN)
    assert r.status_code == 422                                            # needs family + variant
    r = loaded.post(f"/v1/runs/{run['run_id']}/review", json={"decision": "reject", "note": "not a card"},
                    headers=ADMIN)
    assert r.json()["review_status"] == "rejected"


def test_stats(loaded):
    s = loaded.get("/v1/stats", headers=ADMIN).json()
    assert s["runs"] >= 5 and s["by_status"]["ACCEPT"] >= 3
    assert 0 < s["auto_accept_rate"] <= 1 and "p95" in s["latency_ms"]
    assert "ae.alpha" in {f["family"] for f in s["by_family"]} and s["per_day"]
    assert s["truncated"] is False


def test_delete_and_prune(loaded):
    run_id = validate(loaded, cards.not_a_card(11)).json()["run_id"]
    assert loaded.delete(f"/v1/runs/{run_id}", headers=ADMIN).status_code == 200
    assert loaded.get(f"/v1/runs/{run_id}", headers=ADMIN).status_code == 404
    assert loaded.post("/v1/runs/prune", params={"older_than_days": 1}, headers=ADMIN).json()["deleted"] == 0
