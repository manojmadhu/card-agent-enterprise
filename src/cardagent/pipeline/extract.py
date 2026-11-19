"""Stage 6: field extraction.

The card is first ALIGNED to its best-matching reference image with the RANSAC homography,
so YAML regions (defined on the reference) land on the right pixels. Then per field:
  1) region OCR -> normalise -> regex
  2) regex search over full-card OCR text (layout drift)
  3) self-hosted VLM (if enabled) — output re-validated by the same regex
"""
from __future__ import annotations

import re

import cv2
import numpy as np

from ..ocr.engines import OcrEngine
from ..registry.models import CardFamily
from ..vision.geometry import CARD_H, CARD_W
from ..vlm.client import VlmClient

FULLTEXT_PATTERNS = {
    "digits": r"\b\d[\d ]{5,22}\d\b",
    "date_mm_yy": r"\b(0[1-9]|1[0-2])\s*/\s*(\d{4}|\d{2})\b",
    "date_dd_mm_yyyy": r"\b(0[1-9]|[12]\d|3[01])[/.-](0[1-9]|1[0-2])[/.-](\d{4})\b",
    "emirates_id": r"\b784-?\d{4}-?\d{7}-?\d\b",
}


def align(card: np.ndarray, H: np.ndarray | None) -> tuple[np.ndarray, bool]:
    if H is None:
        return card, False
    return cv2.warpPerspective(card, H, (CARD_W, CARD_H), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE), True


def normalise_value(raw: str, kind: str, anchor: str | None) -> str:
    s = (raw or "").upper().strip()
    if anchor:
        s = re.sub(rf"^\W*{re.escape(anchor)}\w*\W*", "", s)
    if kind == "digits":
        return re.sub(r"\D", "", s)
    if kind == "alnum":
        return re.sub(r"[^A-Z0-9]", "", s)
    if kind == "date_mm_yy":
        m = re.search(r"(0[1-9]|1[0-2])\s*/\s*(\d{4}|\d{2})", s)
        return f"{m.group(1)}/{m.group(2)}" if m else s
    if kind == "date_dd_mm_yyyy":
        m = re.search(r"(0[1-9]|[12]\d|3[01])[/.-](0[1-9]|1[0-2])[/.-](\d{4})", s)
        return f"{m.group(1)}/{m.group(2)}/{m.group(3)}" if m else s
    if kind == "emirates_id":
        d = re.sub(r"\D", "", s)
        return f"{d[:3]}-{d[3:7]}-{d[7:14]}-{d[14]}" if len(d) == 15 else s
    if kind == "upper":
        return re.sub(r"\s+", " ", re.sub(r"[^A-Z .'-]", "", s)).strip()
    return s


def extract_fields(fam: CardFamily, img: np.ndarray, full_ocr: dict, ocr: OcrEngine,
                   vlm: VlmClient | None) -> dict:
    out = {}
    for f in fam.fields:
        rx = re.compile(f.regex)
        x0, y0, x1, y1 = f.roi
        pad = 0.01
        crop = img[int(max(0, y0 - pad) * CARD_H):int(min(1, y1 + pad) * CARD_H),
                   int(max(0, x0 - pad) * CARD_W):int(min(1, x1 + pad) * CARD_W)]
        raw, conf = ocr.read_line(crop, f.script)
        val = normalise_value(raw, f.normalise, f.anchor)
        r = {"value": val or None, "confidence": round(conf, 3), "source": "ocr_roi",
             "valid_format": bool(val and rx.fullmatch(val)), "required": f.required}

        if not r["valid_format"] and f.normalise in FULLTEXT_PATTERNS:
            for m in re.finditer(FULLTEXT_PATTERNS[f.normalise], full_ocr["text"]):
                cand = normalise_value(m.group(0), f.normalise, None)
                if rx.fullmatch(cand):
                    r.update(value=cand, source="ocr_fulltext", confidence=0.7, valid_format=True)
                    break

        if not r["valid_format"] and f.required and vlm is not None and vlm.available:
            v = vlm.read_field(img, f.name, f.regex)
            cand = normalise_value(v or "", f.normalise, None)
            if v and rx.fullmatch(cand):
                r.update(value=cand, source="vlm", confidence=0.6, valid_format=True)

        if r["value"] is None:
            r["source"] = "none"
        out[f.name] = r
    return out
