"""Validation end to end: identification, field extraction, decisions, idempotency and the SSE stream."""
import json

from . import cards
from .conftest import validate


def card_photo(family_id, variant_id, number, expiry="05/30", name="MARIAM YOUSEF", seed=1):
    return cards.jpg(cards.photo(cards.render(family_id, variant_id, name, number, expiry), seed))


def test_accepts_and_identifies_each_variant(loaded):
    for fam, var, num, seed in [("ae.alpha", "blue", "1876543210", 1), ("ae.alpha", "amber", "2876543210", 2),
                                ("sa.beta", "red", "7876543210", 3)]:
        r = validate(loaded, card_photo(fam, var, num, seed=seed))
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ACCEPT", body["reasons"]
        assert (body["card"]["family"], body["card"]["variant"]) == (fam, var)
        assert body["fields"]["card_number"]["value"] == num
        assert body["fields"]["expiry_date"]["valid_until"] == "2030-05-31"
        assert body["run_id"]


def test_rejects_expired_card(loaded):
    body = validate(loaded, card_photo("ae.alpha", "blue", "1111222233", expiry="01/21", seed=5)).json()
    assert body["status"] == "REJECT" and body["reasons"][0] == "CARD_EXPIRED"


def test_rejects_unknown_image_and_bad_uploads(loaded):
    assert validate(loaded, cards.not_a_card()).json()["status"] == "REJECT"
    assert validate(loaded, b"not an image").json()["reasons"] == ["INVALID_IMAGE"]
    r = loaded.post("/v1/cards/validate", files={"image": ("a.gif", b"GIF89a", "image/gif")})
    assert r.status_code == 415


def test_idempotent_retry_returns_the_first_answer(loaded):
    img = card_photo("ae.alpha", "amber", "2000000001", seed=7)
    first = validate(loaded, img, request_id="retry-1").json()
    again = validate(loaded, img, request_id="retry-1").json()
    assert again == first
    other = validate(loaded, card_photo("ae.alpha", "amber", "2000000002", seed=8), request_id="retry-1").json()
    assert other["run_id"] != first["run_id"]


def test_store_false_does_not_record_a_run(loaded):
    body = validate(loaded, cards.not_a_card(3), store="false").json()
    assert body["run_id"] is None


def _events(text):
    out = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_stream_reports_every_step_then_the_result(loaded):
    steps = [s["id"] for s in loaded.get("/v1/pipeline/steps").json()["steps"]]
    assert len(steps) == 11 and steps[0] == "ingest" and steps[-1] == "decision"
    img = card_photo("sa.beta", "red", "7123123123", seed=9)
    r = loaded.post("/v1/cards/validate/stream", files={"image": ("p.jpg", img, "image/jpeg")})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    ev = _events(r.text)
    assert ev[0][0] == "snapshot" and all(s["status"] == "pending" for s in ev[0][1]["steps"])
    assert any(e == "image" for e, _ in ev)
    finished = {d["id"] for e, d in ev if e == "step" and d["status"] not in ("pending", "running")}
    assert {"ingest", "quality", "ocr", "family", "fields", "rules"} <= finished
    name, final = ev[-1]
    assert name == "result" and final["result"]["status"] == "ACCEPT" and final["result"]["run_id"]
    assert all(s["status"] not in ("pending", "running") for s in final["steps"])


def test_stream_snapshot_mode(loaded):
    r = loaded.post("/v1/cards/validate/stream", params={"mode": "snapshot"},
                    files={"image": ("p.jpg", cards.not_a_card(4), "image/jpeg")})
    ev = _events(r.text)
    assert {e for e, _ in ev} == {"snapshot", "result"}
    assert ev[-1][1]["result"]["status"] == "REJECT"
