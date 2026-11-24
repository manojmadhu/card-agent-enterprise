"""Evaluation sets (zip upload and from reviewed runs) and background evaluations, incl. what-if."""
import time

from . import cards
from .conftest import ADMIN, validate


def photo(fam, var, number, seed, expiry="05/30"):
    return cards.jpg(cards.photo(cards.render(fam, var, "KHALID RASHID", number, expiry), seed))


def wait(client, eid, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        doc = client.get(f"/v1/evaluations/{eid}", headers=ADMIN).json()
        if doc["status"] in ("done", "failed"):
            return doc
        time.sleep(0.3)
    raise AssertionError("evaluation did not finish")


def make_zip():
    return cards.evaluation_zip([
        ("a.jpg", photo("ae.alpha", "blue", "1222333444", 41),
         {"family": "ae.alpha", "variant": "blue", "expected_status": "ACCEPT", "fields": {"card_number": "1222333444"}}),
        ("b.jpg", photo("sa.beta", "red", "7222333444", 42),
         {"family": "sa.beta", "variant": "red", "expected_status": "ACCEPT"}),
        ("c.jpg", photo("ae.alpha", "amber", "2222333444", 43, expiry="02/22"),
         {"family": "ae.alpha", "variant": "amber", "expected_status": "REJECT"}),
        ("d.jpg", cards.not_a_card(44), {"family": None, "variant": None, "expected_status": "REJECT"}),
    ])


def test_upload_set_and_evaluate(loaded):
    r = loaded.post("/v1/evaluation-sets", files={"file": ("set.zip", make_zip(), "application/zip")},
                    data={"name": "smoke"}, headers=ADMIN)
    assert r.status_code == 201, r.text
    s = r.json()
    assert s["cases"] == 4 and s["expected"] == {"ACCEPT": 2, "REVIEW": 0, "REJECT": 2}
    assert loaded.get(f"/v1/evaluation-sets/{s['set_id']}", headers=ADMIN).json()["name"] == "smoke"

    r = loaded.post("/v1/evaluations", json={"set_id": s["set_id"], "today": "2026-11-24"}, headers=ADMIN)
    assert r.status_code == 202
    doc = wait(loaded, r.json()["evaluation_id"])
    assert doc["status"] == "done", doc.get("error")
    rep = doc["report"]
    assert rep["cases"] == 4 and rep["accept_precision"] == 1.0 and rep["status_agreement"] == 1.0
    assert rep["gate"]["passed"] is True and len(rep["rows"]) == 4
    assert doc["summary"]["wrong_accepts"] == 0

    # what-if: an impossible family threshold rejects everything, live settings are untouched
    r = loaded.post("/v1/evaluations", json={"set_id": s["set_id"], "today": "2026-11-24",
                                             "settings": {"scoring": {"family_min": 0.99, "reject_below": 0.99}}},
                    headers=ADMIN)
    doc = wait(loaded, r.json()["evaluation_id"])
    assert doc["report"]["automation_rate"] == 0 and doc["settings_overrides"]["scoring"]["family_min"] == 0.99
    assert loaded.get("/v1/settings").json()["overrides"] == {}
    no_rows = loaded.get(f"/v1/evaluations/{doc['evaluation_id']}", params={"rows": "false"}, headers=ADMIN).json()
    assert "rows" not in no_rows["report"]

    assert len(loaded.get("/v1/evaluations", params={"set_id": s["set_id"]}, headers=ADMIN).json()["items"]) == 2
    assert loaded.delete(f"/v1/evaluations/{doc['evaluation_id']}", headers=ADMIN).status_code == 200
    assert loaded.delete(f"/v1/evaluation-sets/{s['set_id']}", headers=ADMIN).status_code == 200
    assert loaded.get(f"/v1/evaluation-sets/{s['set_id']}", headers=ADMIN).status_code == 404


def test_bad_sets_are_rejected(loaded):
    bad = loaded.post("/v1/evaluation-sets", files={"file": ("x.zip", b"not a zip", "application/zip")}, headers=ADMIN)
    assert bad.status_code == 422
    no_manifest = cards.evaluation_zip([])
    import io, zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("../evil.txt", "x")
    assert loaded.post("/v1/evaluation-sets", files={"file": ("x.zip", buf.getvalue(), "application/zip")},
                       headers=ADMIN).status_code == 422
    assert loaded.post("/v1/evaluation-sets", files={"file": ("x.zip", no_manifest, "application/zip")},
                       headers=ADMIN).status_code in (201, 422)
    assert loaded.post("/v1/evaluations", json={"set_id": "missing"}, headers=ADMIN).status_code == 404


def test_set_from_reviewed_runs(loaded):
    good = validate(loaded, photo("ae.alpha", "blue", "1999888777", 51)).json()
    junk = validate(loaded, cards.not_a_card(52)).json()
    loaded.post(f"/v1/runs/{good['run_id']}/review", json={"decision": "accept"}, headers=ADMIN)
    loaded.post(f"/v1/runs/{junk['run_id']}/review", json={"decision": "reject"}, headers=ADMIN)
    r = loaded.post("/v1/evaluation-sets/from-runs", json={"name": "reviewed"}, headers=ADMIN)
    assert r.status_code == 201, r.text
    s = r.json()
    assert s["origin"] == "reviewed_runs" and s["cases"] >= 2 and s["expected"]["REJECT"] >= 1
    doc = wait(loaded, loaded.post("/v1/evaluations", json={"set_id": s["set_id"]}, headers=ADMIN).json()["evaluation_id"])
    assert doc["status"] == "done" and doc["report"]["status_agreement"] == 1.0
