"""Cards with information on both sides.

  ae.gamma  name + expiry on the front, card number on the back -> both sides required
  qa.delta  only a logo on the front, every field on the back    -> back required, front optional
  ae.alpha  everything on the front (the classic one-photo card) -> front only
"""
import io
import json
import zipfile

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from cardagent.registry.models import CardFamily

from . import cards
from .conftest import ADMIN, REGISTRY, add_card, add_two_sided_card, validate, validate_sides


def gamma(side, number="4123456789", seed=1, expiry="05/30", variant="gold"):
    return cards.side_photo("ae.gamma", variant, side, number, expiry=expiry, seed=seed)


def delta(side, seed=5):
    return cards.side_photo("qa.delta", "standard", side, "9111222333", seed=seed)


def blurred(jpeg: bytes) -> bytes:
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    return cards.jpg(cv2.GaussianBlur(img, (0, 0), 9))


@pytest.fixture(scope="module")
def platform(client):
    add_card(client, "ae.alpha")
    add_two_sided_card(client, "ae.gamma")
    add_two_sided_card(client, "qa.delta")
    return client


# ------------------------------------------------------------------------------ definition model
def test_sides_default_to_front_only():
    f = CardFamily.model_validate(cards.ALPHA)
    assert f.side_ids == ["front"] and f.required_sides() == ["front"]
    assert all(x.side == "front" for x in f.fields)


def test_required_sides_follow_the_fields():
    assert CardFamily.model_validate(cards.GAMMA).required_sides() == ["front", "back"]
    d = CardFamily.model_validate(cards.DELTA)
    assert d.required_sides() == ["back"] and [x.name for x in d.fields_on("back")] == [
        "holder_name", "card_number", "expiry_date"]
    forced = CardFamily.model_validate({**cards.DELTA, "sides": [{"id": "back"}, {"id": "front", "required": True}]})
    assert forced.side_ids == ["front", "back"] and forced.required_sides() == ["front", "back"]


@pytest.mark.parametrize("changes", [
    {"sides": [{"id": "front"}]},                                              # field on undeclared back side
    {"sides": [{"id": "front"}, {"id": "back", "required": False}]},           # required field on optional side
    {"sides": [{"id": "front"}, {"id": "front"}, {"id": "back"}]},             # duplicate side
    {"sides": []},
])
def test_invalid_side_definitions(changes):
    with pytest.raises(ValidationError):
        CardFamily.model_validate({**cards.GAMMA, **changes})


# ------------------------------------------------------------------------------ cards and references
def test_card_view_shows_sides_and_reference_counts(platform):
    g = platform.get("/v1/cards/ae.gamma").json()
    assert g["sides"] == [{"id": "front", "required": True, "fields": ["holder_name", "expiry_date"]},
                          {"id": "back", "required": True, "fields": ["card_number"]}]
    assert g["variants"][0]["reference_images"] == 6
    assert g["variants"][0]["reference_images_by_side"] == {"front": 3, "back": 3}
    assert g["missing_references"] == []
    refs = platform.get("/v1/cards/ae.gamma/references", headers=ADMIN).json()
    assert refs["sides"] == ["front", "back"]
    assert sorted(r["side"] for r in refs["variants"][0]["references"]) == ["back"] * 3 + ["front"] * 3
    alpha = platform.get("/v1/cards/ae.alpha").json()
    assert alpha["sides"][0]["id"] == "front" and alpha["variants"][0]["reference_images_by_side"] == {"front": 3}
    summary = platform.get("/v1/registry").json()
    gam = next(i for i in summary["items"] if i["family"] == "ae.gamma")
    assert gam["sides"] == ["front", "back"] and gam["variants"][0]["reference_images_by_side"]["back"] == 3


def test_back_references_need_a_back_side(platform):
    files = [("images", ("b.png", cards.side_reference_pngs("ae.gamma", "gold", "back", 1)[0], "image/png"))]
    r = platform.post("/v1/cards/ae.alpha/variants/blue/references", files=files, params={"side": "back"},
                      headers=ADMIN)
    assert r.status_code == 422 and "no back side" in r.json()["detail"]


# ------------------------------------------------------------------------------ validation
def test_front_and_back_are_combined(platform):
    b = validate_sides(platform, front=gamma("front"), back=gamma("back", seed=2)).json()
    assert b["status"] == "ACCEPT", b["reasons"]
    assert (b["card"]["family"], b["card"]["variant"]) == ("ae.gamma", "gold")
    assert {k: (f["value"], f["side"]) for k, f in b["fields"].items()} == {
        "holder_name": ("MARIAM YOUSEF", "front"), "expiry_date": ("05/30", "front"),
        "card_number": ("4123456789", "back")}
    assert [(p["upload"], p["side"], p["status"]) for p in b["photos"]] == [("front", "front", "used"),
                                                                            ("back", "back", "used")]
    assert b["photos"][1]["fields"] == ["card_number"] and b["missing_sides"] == []
    assert b["schema_version"] == "2.1"


def test_missing_back_asks_for_it(platform):
    b = validate_sides(platform, front=gamma("front")).json()
    assert b["status"] == "REJECT" and b["reasons"] == ["SIDE_REQUIRED:back"]
    assert b["card"]["family"] == "ae.gamma" and b["missing_sides"] == ["back"]
    assert b["fields"]["card_number"] == {"value": None, "confidence": 0.0, "source": "none",
                                          "valid_format": False, "valid_until": None, "side": "back"}
    rule = next(c for c in b["validation"] if c["field"] == "card_number")
    assert rule["result"] == "skip" and rule["detail"] == "SIDE_NOT_CAPTURED:back"


def test_swapped_labels_are_corrected(platform):
    b = validate_sides(platform, front=gamma("back", seed=2), back=gamma("front")).json()
    assert b["status"] == "ACCEPT"
    assert [(p["declared"], p["side"], p["side_corrected"]) for p in b["photos"]] == [
        ("front", "back", True), ("back", "front", True)]
    assert b["fields"]["card_number"]["value"] == "4123456789"


def test_unlabelled_photos_are_sorted_out(platform):
    b = validate_sides(platform, images=[gamma("back", seed=2), gamma("front")]).json()
    assert b["status"] == "ACCEPT"
    assert [p["side"] for p in b["photos"]] == ["back", "front"] and all(p["declared"] is None for p in b["photos"])


def test_variant_from_the_back_number_prefix(platform):
    """Both tiers share the back design: the number prefix printed on the back decides the variant."""
    b = validate_sides(platform, front=gamma("front", "3999888777", 3, variant="silver"),
                       back=gamma("back", "3999888777", 4, variant="silver")).json()
    assert b["status"] == "ACCEPT" and b["card"]["variant"] == "silver"


def test_card_with_everything_on_the_back(platform):
    b = validate_sides(platform, back=delta("back", 6)).json()
    assert b["status"] == "ACCEPT", b["reasons"]
    assert b["card"]["family"] == "qa.delta" and {f["side"] for f in b["fields"].values()} == {"back"}
    assert b["fields"]["card_number"]["value"] == "9111222333"
    both = validate_sides(platform, front=delta("front"), back=delta("back", 6)).json()
    assert both["status"] == "ACCEPT"
    only_front = validate_sides(platform, front=delta("front")).json()
    assert only_front["status"] == "REJECT" and only_front["reasons"] == ["SIDE_REQUIRED:back"]


def test_photos_of_two_different_cards(platform):
    alpha = cards.jpg(cards.photo(cards.render("ae.alpha", "blue", "OMAR KHALID", "1888777666", "05/30"), 7))
    b = validate_sides(platform, front=alpha, back=gamma("back", seed=2)).json()
    assert b["status"] == "REVIEW" and b["reasons"] == ["SIDES_MISMATCH"] and b["card"] is None


def test_the_same_side_twice_is_not_read_as_the_other(platform):
    b = validate_sides(platform, images=[gamma("front"), gamma("front", seed=9)]).json()
    assert b["status"] == "REJECT" and b["reasons"] == ["SIDE_REQUIRED:back"]
    assert sorted(p["status"] for p in b["photos"]) == ["not_used", "used"]


def test_blurry_back_asks_for_a_retake(platform):
    b = validate_sides(platform, front=gamma("front"), back=blurred(gamma("back", seed=2))).json()
    assert b["status"] == "REJECT"
    assert b["reasons"][0] == "RETAKE_REQUIRED" and "SIDE_REQUIRED:back" in b["reasons"]
    assert b["photos"][1]["status"] == "retake" and b["card"]["family"] == "ae.gamma"


def test_unneeded_blurry_photo_is_ignored(platform):
    """A front-only card: a bad extra photo does not block the decision."""
    alpha = cards.jpg(cards.photo(cards.render("ae.alpha", "amber", "SARA MANSOOR", "2555444333", "05/30"), 8))
    b = validate_sides(platform, front=alpha, back=blurred(alpha)).json()
    assert b["status"] == "ACCEPT" and b["photos"][1]["status"] == "retake"


def test_one_photo_requests_are_unchanged(platform):
    alpha = cards.jpg(cards.photo(cards.render("ae.alpha", "blue", "OMAR KHALID", "1888777666", "05/30"), 7))
    b = validate(platform, alpha).json()
    assert b["status"] == "ACCEPT" and b["missing_sides"] == []
    assert len(b["photos"]) == 1 and b["photos"][0]["upload"] == "image" and b["photos"][0]["side"] == "front"
    assert b["quality"]["aligned_to_reference"] is True


def test_upload_limits(platform):
    img = gamma("front")
    assert validate_sides(platform, images=[img, img, img]).status_code == 422
    assert validate_sides(platform, front=img, back=img, image=img).status_code == 422
    assert platform.post("/v1/cards/validate", data={"request_id": "x"}).status_code == 422


def test_idempotency_covers_every_photo(platform):
    f, b = gamma("front"), gamma("back", seed=2)

    def call(**photos):
        files = [(k, (f"{k}.jpg", v, "image/jpeg")) for k, v in photos.items()]
        return platform.post("/v1/cards/validate", files=files, data={"request_id": "two-sided-1"}).json()
    first = call(front=f, back=b)
    assert call(front=f, back=b)["run_id"] == first["run_id"]
    assert call(front=f)["run_id"] != first["run_id"]


# ------------------------------------------------------------------------------ stream
def _events(text):
    out = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_stream_with_two_photos(platform):
    files = [("front", ("f.jpg", gamma("front"), "image/jpeg")), ("back", ("b.jpg", gamma("back", seed=2), "image/jpeg"))]
    ev = _events(platform.post("/v1/cards/validate/stream", files=files).text)
    names = [e for e, _ in ev]
    assert names[0] == "snapshot" and names[-1] == "result"
    assert names.count("image") == 1 and names.count("photo_image") == 2
    final = ev[-1][1]
    assert final["result"]["status"] == "ACCEPT"
    assert [(p["label"], p["side"]) for p in final["photos"]] == [("front", "front"), ("back", "back")]
    ingest = final["steps"][0]
    assert ingest["status"] == "pass" and len(ingest["detail"]["photos"]) == 2 and "back:" in ingest["summary"]
    fields = next(s for s in final["steps"] if s["id"] == "fields")
    assert {r["name"]: r["side"] for r in fields["detail"]["fields"]}["card_number"] == "back"


# ------------------------------------------------------------------------------ runs, review, evaluation
def test_run_keeps_both_photos_and_enrolls_each_side(platform):
    platform.patch("/v1/settings", json={"scoring": {"field_min_confidence": 0.999}}, headers=ADMIN)
    b = validate_sides(platform, front=gamma("front", "4777000111", 21), back=gamma("back", "4777000111", 22)).json()
    platform.patch("/v1/settings", json={"scoring": {"field_min_confidence": None}}, headers=ADMIN)
    assert b["status"] == "REVIEW"
    run = platform.get(f"/v1/runs/{b['run_id']}", headers=ADMIN).json()
    assert run["photos"] == 2 and [i["side"] for i in run["images"]] == ["front", "back"]
    front = platform.get(f"/v1/runs/{b['run_id']}/image", headers=ADMIN)
    back = platform.get(f"/v1/runs/{b['run_id']}/image", params={"side": "back"}, headers=ADMIN)
    second = platform.get(f"/v1/runs/{b['run_id']}/image", params={"photo": 1}, headers=ADMIN)
    assert front.status_code == back.status_code == 200 and front.content != back.content
    assert second.content == back.content

    before = platform.get("/v1/cards/ae.gamma").json()["variants"][1]["reference_images_by_side"]
    r = platform.post(f"/v1/runs/{b['run_id']}/review", json={"decision": "accept", "enroll": True}, headers=ADMIN)
    assert r.status_code == 200, r.text
    enrolled = r.json()["review"]["enrolled_photos"]
    assert [(e["side"], e["status"]) for e in enrolled] == [("front", "created"), ("back", "created")]
    after = platform.get("/v1/cards/ae.gamma").json()["variants"][1]["reference_images_by_side"]
    assert after == {"front": before["front"] + 1, "back": before["back"] + 1}


def test_evaluation_with_two_photo_cases(platform):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("manifest.json", json.dumps([
            {"files": {"front": "p/1f.jpg", "back": "p/1b.jpg"}, "family": "ae.gamma", "variant": "gold",
             "expected_status": "ACCEPT", "fields": {"card_number": "4333222111"}},
            {"files": ["p/2b.jpg"], "family": "qa.delta", "variant": "standard", "expected_status": "ACCEPT"},
            {"file": "p/3f.jpg", "family": "ae.gamma", "variant": "gold", "expected_status": "REJECT"},
        ]))
        z.writestr("p/1f.jpg", gamma("front", "4333222111", 31))
        z.writestr("p/1b.jpg", gamma("back", "4333222111", 32))
        z.writestr("p/2b.jpg", delta("back", 33))
        z.writestr("p/3f.jpg", gamma("front", "4333222111", 34))
    s = platform.post("/v1/evaluation-sets", files={"file": ("s.zip", buf.getvalue(), "application/zip")},
                      headers=ADMIN)
    assert s.status_code == 201, s.text
    from .test_api_evaluations import wait
    eid = platform.post("/v1/evaluations", json={"set_id": s.json()["set_id"], "today": "2026-11-25"},
                        headers=ADMIN).json()["evaluation_id"]
    doc = wait(platform, eid)
    assert doc["status"] == "done", doc.get("error")
    rep = doc["report"]
    assert rep["status_agreement"] == 1.0 and rep["accept_precision"] == 1.0
    assert rep["rows"][0]["files"] == {"front": "p/1f.jpg", "back": "p/1b.jpg"}
    assert rep["field_accuracy"]["card_number"] == 1.0


def test_set_from_reviewed_two_photo_runs(platform):
    b = validate_sides(platform, front=gamma("front", "4999111222", 41), back=gamma("back", "4999111222", 42)).json()
    platform.post(f"/v1/runs/{b['run_id']}/review", json={"decision": "accept"}, headers=ADMIN)
    s = platform.post("/v1/evaluation-sets/from-runs", json={"family": "ae.gamma"}, headers=ADMIN)
    assert s.status_code == 201, s.text
    from cardagent.api.deps import STATE
    ev = STATE["agent"].evaluations
    manifest = json.loads((ev.set_dir(s.json()["set_id"]) / "manifest.json").read_text())
    case = next(c for c in manifest if c["run_id"] == b["run_id"])
    assert set(case["files"]) == {"front", "back"} and case["fields"]["card_number"] == "4999111222"


# ------------------------------------------------------------------------------ git registry
def test_registry_back_folder(platform):
    import yaml
    fam = {**cards.DELTA, "id": "qa.deltagit"}
    (REGISTRY / "cards" / "qa").mkdir(parents=True, exist_ok=True)
    (REGISTRY / "cards" / "qa" / "deltagit.yaml").write_text(yaml.safe_dump(fam), encoding="utf-8")
    base = REGISTRY / "refs" / "qa.deltagit" / "standard"
    (base / "back").mkdir(parents=True, exist_ok=True)
    (base / "f1.png").write_bytes(cards.side_reference_pngs("qa.delta", "standard", "front", 1)[0])
    (base / "back" / "b1.png").write_bytes(cards.side_reference_pngs("qa.delta", "standard", "back", 1)[0])
    out = platform.post("/v1/registry/sync", params={"prune": "false"}, headers=ADMIN).json()
    assert out["references_added"] == 2
    v = platform.get("/v1/cards/qa.deltagit").json()["variants"][0]
    assert v["reference_images_by_side"] == {"front": 1, "back": 1}

    alpha_back = REGISTRY / "refs" / "ae.alphagit" / "blue" / "back"
    (REGISTRY / "cards" / "ae").mkdir(parents=True, exist_ok=True)
    (REGISTRY / "cards" / "ae" / "alphagit.yaml").write_text(yaml.safe_dump({**cards.ALPHA, "id": "ae.alphagit"}))
    alpha_back.mkdir(parents=True, exist_ok=True)
    (alpha_back / "x.png").write_bytes(cards.reference_pngs("ae.alpha", "blue", 1)[0])
    r = platform.post("/v1/registry/sync", params={"prune": "false"}, headers=ADMIN)
    assert r.status_code == 422 and "back" in r.json()["detail"]
    (alpha_back / "x.png").unlink()


def test_example_templates_are_valid():
    import pathlib
    import yaml
    root = pathlib.Path(__file__).resolve().parents[1] / "examples" / "cards"
    files = sorted(root.rglob("*.yaml"))
    assert len(files) >= 4
    fams = {p.stem: CardFamily.model_validate(yaml.safe_load(p.read_text(encoding="utf-8"))) for p in files}
    assert fams["front_and_back"].required_sides() == ["front", "back"]
    assert fams["back_only_data"].required_sides() == ["back"]
