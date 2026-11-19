"""Self-hosted VLM fallback for unreadable required fields. Supports Ollama and any
OpenAI-compatible server (vLLM, llama.cpp server). Internal endpoints only.

A circuit breaker stops calling the VLM after repeated failures so an outage
degrades to REVIEW decisions instead of slow requests.
"""
from __future__ import annotations

import base64
import json
import logging
import threading
import time
import urllib.request

import cv2

from ..settings import VlmCfg

log = logging.getLogger(__name__)


class VlmClient:
    def __init__(self, cfg: VlmCfg):
        self.cfg = cfg
        self._fails = 0
        self._open_until = 0.0
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self.cfg.enabled and time.time() >= self._open_until

    def read_field(self, img_bgr, field_name: str, regex: str) -> str | None:
        if not self.available:
            return None
        ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            return None
        b64 = base64.b64encode(buf.tobytes()).decode()
        prompt = (f"Read the '{field_name}' printed on this membership/discount card. It must match the "
                  f"regular expression {regex}. Reply ONLY with JSON {{\"value\": \"...\"}}. "
                  "If it is not clearly visible reply {\"value\": null}. Never guess.")
        try:
            raw = self._call(prompt, b64)
            value = json.loads(raw).get("value")
            self._record(True)
            return value if isinstance(value, str) else None
        except Exception as e:
            log.warning("vlm call failed: %s", e)
            self._record(False)
            return None

    def _call(self, prompt: str, b64: str) -> str:
        c = self.cfg
        if c.kind == "ollama":
            url, body = f"{c.url}/api/generate", {
                "model": c.model, "prompt": prompt, "images": [b64], "stream": False,
                "format": "json", "options": {"temperature": 0}}
        else:
            url, body = f"{c.url}/v1/chat/completions", {
                "model": c.model, "temperature": 0, "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}]}
        req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=c.timeout_s) as r:
            data = json.loads(r.read())
        return data["response"] if c.kind == "ollama" else data["choices"][0]["message"]["content"]

    def _record(self, ok: bool):
        with self._lock:
            if ok:
                self._fails = 0
                return
            self._fails += 1
            if self._fails >= self.cfg.circuit_breaker_failures:
                self._open_until = time.time() + self.cfg.circuit_breaker_cooldown_s
                self._fails = 0
                log.error("vlm circuit breaker open for %ss", self.cfg.circuit_breaker_cooldown_s)
