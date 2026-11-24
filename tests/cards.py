"""Test cards drawn in code: two fictional programmes, nothing read from disk.

ALPHA CLUB (ae.alpha) has two variants that differ in colour, tier word and number prefix;
BETA REWARDS (sa.beta) has a different layout. photo() simulates a phone photo of a card.
"""
from __future__ import annotations

import io
import json
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1012, 638
_BOLD = ("DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "arialbd.ttf",
         "/System/Library/Fonts/Supplemental/Arial Bold.ttf")
_MONO = ("DejaVuSansMono-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", "courbd.ttf",
         "/System/Library/Fonts/Supplemental/Courier New Bold.ttf")

FIELDS = [
    {"name": "holder_name", "roi": [0.05, 0.55, 0.75, 0.68], "regex": "[A-Z][A-Z .'-]{2,40}", "normalise": "upper"},
    {"name": "card_number", "roi": [0.05, 0.40, 0.75, 0.54], "regex": "\\d{10}", "normalise": "digits"},
    {"name": "expiry_date", "roi": [0.05, 0.72, 0.50, 0.86], "regex": "(0[1-9]|1[0-2])/(\\d{2}|\\d{4})",
     "normalise": "date_mm_yy", "anchor": "EXP"},
]
RULES = [{"type": "regex", "field": "card_number", "critical": True},
         {"type": "regex", "field": "holder_name"},
         {"type": "not_expired", "field": "expiry_date", "critical": True},
         {"type": "prefix_matches_variant", "field": "card_number"}]

ALPHA = {"id": "ae.alpha", "country": "AE", "issuer": "Alpha Test Issuer", "name": "Alpha Club",
         "keywords": ["ALPHACLUB"], "negative_keywords": ["BETAREWARDS"],
         "variants": [{"id": "blue", "name": "Alpha Blue", "body_color_rgb": [28, 64, 160], "keywords": ["BLUE"],
                       "number_prefix": "1"},
                      {"id": "amber", "name": "Alpha Amber", "body_color_rgb": [214, 150, 40], "keywords": ["AMBER"],
                       "number_prefix": "2"}],
         "fields": FIELDS, "rules": RULES}
BETA = {"id": "sa.beta", "country": "SA", "issuer": "Beta Test Issuer", "name": "Beta Rewards",
        "keywords": ["BETAREWARDS"], "negative_keywords": ["ALPHACLUB"],
        "variants": [{"id": "red", "name": "Beta Red", "body_color_rgb": [160, 36, 44], "keywords": ["RED"],
                      "number_prefix": "7"}],
        "fields": FIELDS, "rules": RULES}
FAMILIES = {"ae.alpha": ALPHA, "sa.beta": BETA}
NAMES = ["AHMED ALI HASSAN", "FATIMA SAEED", "OMAR KHALID", "MARIAM YOUSEF", "SARA MANSOOR", "KHALID RASHID"]


def _font(cands, size):
    for c in cands:
        try:
            return ImageFont.truetype(c, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def render(family_id: str, variant_id: str, name: str = "AHMED ALI HASSAN", number: str | None = None,
           expiry: str = "08/29") -> np.ndarray:
    fam = FAMILIES[family_id]
    v = next(x for x in fam["variants"] if x["id"] == variant_id)
    number = number or (v["number_prefix"] + "234567890")[:10]
    bg = tuple(v["body_color_rgb"])
    fg = (255, 255, 255) if 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2] < 140 else (20, 20, 25)
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    if family_id == "ae.alpha":                          # ring + triangle logo, stripe
        d.ellipse([W - 230, 30, W - 70, 190], outline=fg, width=10)
        d.polygon([(W - 150, 55), (W - 105, 160), (W - 195, 160)], outline=fg, width=6)
        d.rectangle([0, int(H * .31), W, int(H * .34)], fill=fg)
    else:                                                # square logo with diagonal, corner wedge
        d.rectangle([W - 220, 40, W - 80, 180], outline=fg, width=8)
        d.line([(W - 220, 40), (W - 80, 180)], fill=fg, width=8)
        d.polygon([(0, H), (int(W * .25), H), (0, int(H * .85))], fill=fg)
    d.text((int(W * .06), int(H * .07)), fam["keywords"][0], font=_font(_BOLD, 76), fill=fg)
    d.text((int(W * .06), int(H * .22)), v["keywords"][0], font=_font(_BOLD, 38), fill=fg)
    spaced = " ".join(number[i:i + 4] for i in range(0, len(number), 4))
    d.text((int(W * .06), int(H * .42)), spaced, font=_font(_MONO, 56), fill=fg)
    d.text((int(W * .06), int(H * .57)), name, font=_font(_BOLD, 44), fill=fg)
    d.text((int(W * .06), int(H * .74)), f"EXP {expiry}", font=_font(_MONO, 38), fill=fg)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def photo(card: np.ndarray, seed: int = 0) -> np.ndarray:
    """The card on a grey table, slightly rotated and in perspective, with sensor noise."""
    rnd = np.random.default_rng(seed)
    bw, bh = 1600, 1200
    bg = np.full((bh, bw, 3), int(rnd.integers(80, 140)), np.uint8)
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    a = np.deg2rad(rnd.uniform(-8, 8))
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    dst = (((src - [W / 2, H / 2]) * rnd.uniform(1.0, 1.1)) @ rot.T + [bw / 2, bh / 2]
           + rnd.uniform(-15, 15, (4, 2))).astype(np.float32)
    m = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(card, m, (bw, bh))
    mask = cv2.warpPerspective(np.full((H, W), 255, np.uint8), m, (bw, bh))
    out = np.where(mask[..., None] > 0, warped, bg).astype(np.float32)
    return np.clip(out + rnd.normal(0, 3, out.shape), 0, 255).astype(np.uint8)


def png(img: np.ndarray) -> bytes:
    return cv2.imencode(".png", img)[1].tobytes()


def jpg(img: np.ndarray) -> bytes:
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()


def reference_pngs(family_id: str, variant_id: str, n: int = 3) -> list[bytes]:
    v = next(x for x in FAMILIES[family_id]["variants"] if x["id"] == variant_id)
    return [png(render(family_id, variant_id, NAMES[i], v["number_prefix"] + f"{i}" * 9)) for i in range(n)]


def not_a_card(seed: int = 1) -> bytes:
    """A textured, in-focus photo with no card in it."""
    rnd = np.random.default_rng(seed)
    img = cv2.resize(rnd.integers(0, 255, (60, 80, 3), dtype=np.uint8), (1600, 1200), interpolation=cv2.INTER_NEAREST)
    return jpg(img)


def evaluation_zip(cases: list[tuple[str, bytes, dict]]) -> bytes:
    """cases: (filename, image bytes, manifest entry without 'file')."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("manifest.json", json.dumps([{"file": f"photos/{fn}", **entry} for fn, _, entry in cases]))
        for fn, data, _ in cases:
            z.writestr(f"photos/{fn}", data)
    return buf.getvalue()
