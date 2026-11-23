"""Stage 7: declarative rules engine. Rules are declared per family in YAML.

`lookup` rules call an issuer verifier registered at startup (e.g. a partner API or an internal
membership table). Without a verifier they are skipped with detail NO_VERIFIER — reading a card
is not the same as proving it is active.
"""
from __future__ import annotations

import calendar
import re
from datetime import date
from typing import Callable

from ..registry.models import CardFamily

# family_id -> callable(field_value, all_fields) -> (passed: bool, detail: str|None)
VERIFIERS: dict[str, Callable[[str, dict], tuple[bool, str | None]]] = {}


def register_verifier(family_id: str, fn: Callable[[str, dict], tuple[bool, str | None]]) -> None:
    VERIFIERS[family_id] = fn


def parse_expiry(value: str | None) -> date | None:
    if not value:
        return None
    m = re.fullmatch(r"(\d{2})/(\d{2}|\d{4})", value)
    if m:
        month, year = int(m.group(1)), int(m.group(2))
        year = year + 2000 if year < 100 else year
        if 1 <= month <= 12:
            return date(year, month, calendar.monthrange(year, month)[1])   # valid through month end
    m = re.fullmatch(r"(\d{2})/(\d{2})/(\d{4})", value)
    if m:
        try:
            return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            return None
    return None


def luhn_ok(num: str) -> bool:
    if not num.isdigit():
        return False
    total = 0
    for i, ch in enumerate(reversed(num)):
        d = int(ch)
        if i % 2:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
    return total % 10 == 0


def run_rules(fam: CardFamily, variant_id: str | None, fields: dict, today: date | None = None) -> list[dict]:
    today = today or date.today()
    specs = {f.name: f for f in fam.fields}
    out = []
    for rule in fam.rules:
        val = (fields.get(rule.field) or {}).get("value")
        res, detail = "skip", None
        if val is None:
            res, detail = ("fail" if specs[rule.field].required else "skip"), "FIELD_MISSING"
        elif rule.type == "regex":
            res = "pass" if re.fullmatch(specs[rule.field].regex, val) else "fail"
        elif rule.type == "length":
            n = len(re.sub(r"\D", "", val))
            res = "pass" if rule.params.get("min", 0) <= n <= rule.params.get("max", 99) else "fail"
        elif rule.type == "not_expired":
            exp = parse_expiry(val)
            if exp is None:
                res, detail = "fail", "UNPARSEABLE_DATE"
            elif exp >= today:
                res, detail = "pass", f"valid_until={exp.isoformat()}"
            else:
                res, detail = "fail", f"CARD_EXPIRED valid_until={exp.isoformat()}"
        elif rule.type == "emirates_id_format":
            res = "pass" if re.fullmatch(r"784-(19|20)\d{2}-\d{7}-\d", val) else "fail"
        elif rule.type == "luhn":
            res = "pass" if luhn_ok(re.sub(r"\D", "", val)) else "fail"
        elif rule.type == "prefix_matches_variant":
            v = fam.variant(variant_id)
            if v and v.number_prefix:
                res = "pass" if val.startswith(v.number_prefix) else "fail"
                detail = None if res == "pass" else "NUMBER_PREFIX_VARIANT_MISMATCH"
        elif rule.type == "date_order":                     # e.g. issue_date must not be after expiry_date
            other = (fields.get(rule.params.get("other_field")) or {}).get("value")
            a, b = parse_expiry(val), parse_expiry(other)
            if other is None:
                res, detail = "skip", "OTHER_FIELD_MISSING"
            elif a is None or b is None:
                res, detail = "fail", "UNPARSEABLE_DATE"
            else:
                res = "pass" if a <= b else "fail"
                detail = None if res == "pass" else f"DATE_ORDER {rule.field}>{rule.params['other_field']}"
        elif rule.type == "lookup":
            fn = VERIFIERS.get(fam.id)
            if fn is None:
                res, detail = "skip", "NO_VERIFIER"
            else:
                try:
                    ok, detail = fn(val, fields)
                    res = "pass" if ok else "fail"
                except Exception as e:                     # verifier outage -> review, never accept
                    res, detail = "fail", f"VERIFIER_ERROR {type(e).__name__}"
        out.append({"rule": rule.type, "field": rule.field, "critical": rule.critical,
                    "result": res, "detail": detail})
    return out
