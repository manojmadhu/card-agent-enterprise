"""Prometheus metrics + JSON logging with PII masking."""
from __future__ import annotations

import json
import logging
import sys
import time

from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter("cardagent_requests_total", "Validation requests", ["status"])
REASONS = Counter("cardagent_reasons_total", "Decision reason codes", ["reason"])
FAMILY = Counter("cardagent_family_total", "Identified card families", ["family", "variant"])
LATENCY = Histogram("cardagent_request_seconds", "End-to-end latency",
                    buckets=(0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 13, 30))
STAGE = Histogram("cardagent_stage_seconds", "Per-stage latency", ["stage"],
                  buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5))
VLM_CALLS = Counter("cardagent_vlm_fallback_total", "Fields resolved by VLM fallback")
REGISTRY_INFO = Gauge("cardagent_registry_reference_images", "Reference images loaded")
RELOADS = Counter("cardagent_registry_reloads_total", "Registry reloads", ["result"])

PII_FIELDS = {"card_number", "emirates_id", "holder_name", "member_id", "national_id", "phone"}


def mask(name: str, value):
    if value is None or name not in PII_FIELDS:
        return value
    s = str(value)
    return ("*" * max(0, len(s) - 4) + s[-4:]) if name != "holder_name" else (s[:1] + "***")


class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + "Z",
                   "level": record.levelname, "logger": record.name, "msg": record.getMessage()}
        if hasattr(record, "extra_json"):
            payload.update(record.extra_json)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str = "INFO", json_logs: bool = True):
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter() if json_logs else logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [h]
    root.setLevel(level)


def audit_record(result: dict, mask_pii: bool = True) -> dict:
    """Compact, PII-safe audit line for each decision."""
    fields = {k: (mask(k, v.get("value")) if mask_pii else v.get("value")) for k, v in result.get("fields", {}).items()}
    card = result.get("card") or {}
    return {"event": "card_decision", "request_id": result["request_id"], "status": result["status"],
            "reasons": result["reasons"], "family": card.get("family"), "variant": card.get("variant"),
            "family_confidence": card.get("family_confidence"), "fields": fields,
            "registry": result.get("model_versions", {}).get("registry"), "ms": result.get("processing_ms")}
