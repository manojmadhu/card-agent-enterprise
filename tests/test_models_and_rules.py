"""Card definition schema and validation rules (no Qdrant, no images)."""
from datetime import date

import pytest
from pydantic import ValidationError

from cardagent.pipeline.rules import luhn_ok, parse_expiry, run_rules
from cardagent.registry.models import CardFamily

from .cards import ALPHA


def fam(**changes) -> CardFamily:
    return CardFamily.model_validate({**ALPHA, **changes})


def fields(**values):
    return {k: {"value": v} for k, v in values.items()}


def test_valid_definition_uppercases_keywords():
    f = fam(keywords=["alphaclub"])
    assert f.keywords == ["ALPHACLUB"]
    assert f.variant("amber").number_prefix == "2"
    assert f.variant("nope") is None


@pytest.mark.parametrize("changes", [
    {"id": "Alpha"},                                                      # id format
    {"variants": ALPHA["variants"] + [ALPHA["variants"][0]]},             # duplicate variant id
    {"rules": [{"type": "regex", "field": "unknown"}]},                   # rule on unknown field
    {"rules": [{"type": "date_order", "field": "expiry_date"}]},          # date_order without other_field
    {"fields": [{**ALPHA["fields"][0], "roi": [0.5, 0.5, 0.4, 0.6]}]},    # inverted roi
    {"fields": [{**ALPHA["fields"][0], "regex": "([A-Z"}]},               # regex does not compile
])
def test_invalid_definitions_are_rejected(changes):
    with pytest.raises(ValidationError):
        fam(**changes)


def test_parse_expiry_and_luhn():
    assert parse_expiry("08/29") == date(2029, 8, 31)
    assert parse_expiry("02/2028") == date(2028, 2, 29)
    assert parse_expiry("13/29") is None
    assert luhn_ok("4539578763621486") and not luhn_ok("4539578763621487")


def test_rules_pass_and_fail():
    f = fam()
    ok = run_rules(f, "blue", fields(holder_name="OMAR KHALID", card_number="1234567890", expiry_date="08/29"),
                   today=date(2026, 11, 24))
    assert {c["rule"]: c["result"] for c in ok} == {"regex": "pass", "not_expired": "pass",
                                                    "prefix_matches_variant": "pass"}
    bad = run_rules(f, "blue", fields(holder_name="OMAR KHALID", card_number="2234567890", expiry_date="08/21"),
                    today=date(2026, 11, 24))
    res = {c["rule"]: c for c in bad}
    assert res["not_expired"]["result"] == "fail" and res["not_expired"]["detail"].startswith("CARD_EXPIRED")
    assert res["prefix_matches_variant"]["result"] == "fail"


def test_missing_required_field_fails_its_rules():
    out = run_rules(fam(), "blue", fields(holder_name="OMAR KHALID", card_number=None, expiry_date="08/29"),
                    today=date(2026, 11, 24))
    assert any(c["field"] == "card_number" and c["result"] == "fail" and c["detail"] == "FIELD_MISSING" for c in out)


def test_date_order_rule():
    f = fam(fields=ALPHA["fields"] + [{"name": "issue_date", "roi": [0.5, 0.72, 0.9, 0.86],
                                       "regex": "(0[1-9]|1[0-2])/(\\d{2}|\\d{4})", "normalise": "date_mm_yy"}],
            rules=[{"type": "date_order", "field": "issue_date", "params": {"other_field": "expiry_date"}}])
    good = run_rules(f, "blue", fields(issue_date="01/24", expiry_date="08/29"))
    bad = run_rules(f, "blue", fields(issue_date="09/30", expiry_date="08/29"))
    assert good[0]["result"] == "pass" and bad[0]["result"] == "fail"
