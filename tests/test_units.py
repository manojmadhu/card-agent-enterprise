"""Fast unit tests: registry validation, rules, normalisation, keyword matching, homography sanity."""
from datetime import date

import numpy as np
import pytest
import yaml

from cardagent.index.reference_index import skeleton
from cardagent.pipeline.extract import normalise_value
from cardagent.pipeline.rules import luhn_ok, parse_expiry, run_rules
from cardagent.registry.models import CardFamily
from cardagent.vision.features import _plausible_h

FAMILY = yaml.safe_load("""
id: ae.test
country: AE
issuer: Test
name: Test card
keywords: [TESTCARD]
variants:
  - {id: gold, name: Gold, body_color_rgb: [200, 160, 40], number_prefix: "2"}
fields:
  - {name: card_number, roi: [0.05, 0.4, 0.7, 0.5], regex: "\\\\d{10}", normalise: digits}
  - {name: expiry_date, roi: [0.05, 0.7, 0.5, 0.8], regex: "(0[1-9]|1[0-2])/\\\\d{2}", normalise: date_mm_yy}
rules:
  - {type: regex, field: card_number, critical: true}
  - {type: not_expired, field: expiry_date, critical: true}
  - {type: prefix_matches_variant, field: card_number}
  - {type: luhn, field: card_number}
  - {type: lookup, field: card_number}
""")


def test_registry_validation_rejects_bad_files():
    CardFamily.model_validate(FAMILY)
    bad = {**FAMILY, "id": "Fazaa"}
    with pytest.raises(ValueError):
        CardFamily.model_validate(bad)
    bad = {**FAMILY, "rules": [{"type": "regex", "field": "nope"}]}
    with pytest.raises(ValueError):
        CardFamily.model_validate(bad)
    bad = {**FAMILY, "fields": [{**FAMILY["fields"][0], "roi": [0.5, 0.4, 0.2, 0.5]}]}
    with pytest.raises(ValueError):
        CardFamily.model_validate(bad)


def test_expiry_month_end_and_formats():
    assert parse_expiry("02/28") == date(2028, 2, 29)
    assert parse_expiry("11/2027") == date(2027, 11, 30)
    assert parse_expiry("31/12/2026") == date(2026, 12, 31)
    assert parse_expiry("13/27") is None


def test_rules_engine():
    fam = CardFamily.model_validate(FAMILY)
    fields = {"card_number": {"value": "2000000008"}, "expiry_date": {"value": "05/27"}}
    res = {r["rule"]: r for r in run_rules(fam, "gold", fields, today=date(2026, 11, 18))}
    assert res["regex"]["result"] == "pass"
    assert res["not_expired"]["result"] == "pass"
    assert res["prefix_matches_variant"]["result"] == "pass"
    assert res["lookup"]["result"] == "skip" and res["lookup"]["detail"] == "NO_VERIFIER"
    expired = run_rules(fam, "gold", {**fields, "expiry_date": {"value": "10/26"}}, today=date(2026, 11, 18))
    assert next(r for r in expired if r["rule"] == "not_expired")["result"] == "fail"


def test_luhn():
    assert luhn_ok("79927398713")
    assert not luhn_ok("79927398710")


@pytest.mark.parametrize("raw,kind,anchor,expected", [
    ("2210 4577 83", "digits", None, "2210457783"),
    ("EXP 11/27", "date_mm_yy", "EXP", "11/27"),
    ("ID 784 1990 1234567 1", "emirates_id", "ID", "784-1990-1234567-1"),
    ("fatima  saeed!", "upper", None, "FATIMA SAEED"),
])
def test_normalise(raw, kind, anchor, expected):
    assert normalise_value(raw, kind, anchor) == expected


def test_ocr_confusable_skeleton():
    assert skeleton("DEMOQAO1") == skeleton("DEMOQA01")
    assert skeleton("FAZAA") == "FAZAA"


def test_homography_sanity():
    assert _plausible_h(np.eye(3))
    flip = np.diag([-1.0, 1.0, 1.0])
    assert not _plausible_h(flip)
