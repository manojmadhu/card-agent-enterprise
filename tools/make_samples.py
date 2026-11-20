"""Generate SYNTHETIC data for development and CI (fictional designs marked SAMPLE).

  * reference images for every variant in registry/cards/**.yaml
  * optional fictional "demo" programmes across GCC countries (--demo-families N) to exercise
    retrieval at scale (look-alike colours and identical layouts on purpose)
  * simulated phone photos + labelled manifest in data/samples/

Replace with real, consented images before calibrating thresholds for production.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
W, H = 1012, 638
# Linux (DejaVu) first, then Windows / macOS equivalents. Bare names are resolved by Pillow from the OS font dir.
FONT_B = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVuSans-Bold.ttf", "arialbd.ttf",
          "/System/Library/Fonts/Supplemental/Arial Bold.ttf")
# Courier New before Consolas: Consolas draws a dotted zero that Tesseract misreads as 6.
FONT_M = ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", "DejaVuSansMono-Bold.ttf", "courbd.ttf",
          "consolab.ttf", "/System/Library/Fonts/Supplemental/Courier New Bold.ttf")
NAMES = ["AHMED ALI HASSAN", "FATIMA SAEED", "OMAR KHALID", "MARIAM YOUSEF", "SARA AL MANSOORI",
         "KHALID RASHID", "NOURA SALEM", "YOUSEF IBRAHIM", "HESSA MOHAMMED", "RASHID OBAID",
         "LATIFA HAMAD", "SALEM JABER", "AISHA NASSER", "HAMDAN SAIF"]
COUNTRIES = ["sa", "qa", "kw", "bh", "om", "ae"]


def font(candidates, s):
    for p in candidates:
        try:
            return ImageFont.truetype(p, s)
        except OSError:
            continue
    raise OSError(f"none of these fonts were found: {candidates}")


def fy(fam, name, default):
    f = next((x for x in fam["fields"] if x["name"] == name), None)
    return f["roi"] if f else default


def render(fam: dict, v: dict, name, number, exp, eid=None, style: int = 0, title=None) -> np.ndarray:
    bg = tuple(v["body_color_rgb"])
    lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    fg = (255, 255, 255) if lum < 120 else (20, 20, 25)
    accent = tuple(min(255, c + 40) if lum < 120 else max(0, c - 40) for c in bg)
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    s = style % 4
    if s == 0:
        d.ellipse([W - 230, 30, W - 70, 190], outline=fg, width=10)
        d.polygon([(W - 150, 55), (W - 105, 160), (W - 195, 160)], fill=accent, outline=fg)
        d.rectangle([0, int(H * .30), W, int(H * .33)], fill=accent)
    elif s == 1:
        cx, cy = W - 150, 115
        pts = [(cx + 80 * np.cos(a) * (1 if i % 2 == 0 else .45), cy + 80 * np.sin(a) * (1 if i % 2 == 0 else .45))
               for i, a in enumerate(np.linspace(-np.pi / 2, 1.5 * np.pi, 11)[:-1])]
        d.polygon(pts, fill=accent, outline=fg)
        for k in range(4):
            d.line([(int(W * .55) + 40 * k, int(H * .34)), (int(W * .75) + 40 * k, int(H * .40))], fill=accent, width=6)
    elif s == 2:
        d.rectangle([W - 220, 40, W - 80, 180], outline=fg, width=8)
        d.line([(W - 220, 40), (W - 80, 180)], fill=accent, width=8)
        d.rectangle([int(W * .55), int(H * .31), W, int(H * .36)], fill=accent)
    else:
        for k in range(3):
            d.ellipse([W - 260 + 50 * k, 60, W - 160 + 50 * k, 160], outline=fg, width=6)
        d.polygon([(0, H), (int(W * .25), H), (0, int(H * .85))], fill=accent)

    d.text((int(W * .06), int(H * .08)), title or fam["_title"], font=font(FONT_B, 76), fill=fg)
    d.text((int(W * .06), int(H * .25)), v.get("label", v["id"].upper()), font=font(FONT_B, 36), fill=fg)
    d.text((int(W * .62), int(H * .27)), "SAMPLE", font=font(FONT_B, 22), fill=accent)
    num_y = fy(fam, "card_number", [0, .40])[1] + 0.02
    name_y = fy(fam, "holder_name", [0, .55])[1] + 0.02
    exp_roi = fy(fam, "expiry_date", [.05, .72])
    spaced = " ".join(number[i:i + 4] for i in range(0, len(number), 4))
    d.text((int(W * .06), int(H * num_y)), spaced, font=font(FONT_M, 56), fill=fg)
    d.text((int(W * .06), int(H * name_y)), name, font=font(FONT_B, 44), fill=fg)
    d.text((int(W * (exp_roi[0] + .01)), int(H * (exp_roi[1] + .02))), f"EXP {exp}", font=font(FONT_M, 38), fill=fg)
    if eid and any(f["name"] == "emirates_id" for f in fam["fields"]):
        r = fy(fam, "emirates_id", [.05, .78])
        d.text((int(W * (r[0] + .01)), int(H * (r[1] + .02))), f"ID {eid}", font=font(FONT_M, 32), fill=fg)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def photograph(card, seed, upside_down=False, blur=0.0, glare=False):
    rnd = np.random.default_rng(seed)
    if upside_down:
        card = cv2.rotate(card, cv2.ROTATE_180)
    BW, BH = 1600, 1200
    bg = np.full((BH, BW, 3), int(rnd.integers(70, 150)), np.uint8)
    bg = cv2.GaussianBlur(cv2.add(bg, rnd.normal(0, 12, bg.shape).clip(0, 40).astype(np.uint8)), (9, 9), 0)
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    cx, cy, s = BW / 2 + rnd.uniform(-80, 80), BH / 2 + rnd.uniform(-60, 60), rnd.uniform(.95, 1.15)
    a = np.deg2rad(rnd.uniform(-12, 12))
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    dst = (((src - [W / 2, H / 2]) * s) @ R.T + [cx, cy] + rnd.uniform(-25, 25, (4, 2))).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(card, M, (BW, BH))
    mask = cv2.warpPerspective(np.full((H, W), 255, np.uint8), M, (BW, BH))
    out = np.where(mask[..., None] > 0, warped, bg).astype(np.float32)
    out = np.clip(out * rnd.uniform(.85, 1.12) + rnd.normal(0, 4, out.shape), 0, 255).astype(np.uint8)
    if glare:
        cv2.circle(out, (int(cx), int(cy)), 260, (255, 255, 255), -1)
    if blur:
        out = cv2.GaussianBlur(out, (0, 0), blur)
    return out


def demo_families(n: int, rnd: random.Random) -> list[dict]:
    palette = [[201, 162, 39], [192, 194, 198], [58, 60, 66], [24, 92, 160], [150, 30, 40], [20, 120, 90],
               [90, 50, 140], [230, 120, 30], [22, 22, 26], [196, 152, 46], [120, 180, 200], [240, 240, 235]]
    tiers = [("silver", "SILVER"), ("gold", "GOLD"), ("platinum", "PLATINUM"), ("classic", "CLASSIC"),
             ("elite", "ELITE"), ("family", "FAMILY")]
    fams = []
    for i in range(n):
        cc = COUNTRIES[i % len(COUNTRIES)]
        code = f"{cc.upper()}{i:02d}"
        k = rnd.choice([2, 3, 3, 4])
        vs = []
        for j, (vid, lab) in enumerate(rnd.sample(tiers, k)):
            vs.append({"id": vid, "name": f"Demo {code} {lab.title()}", "body_color_rgb": rnd.choice(palette),
                       "keywords": [lab], "number_prefix": str(j + 1)})
        # make sure variants inside a family have distinct colours
        seen = set()
        for v in vs:
            while tuple(v["body_color_rgb"]) in seen:
                v["body_color_rgb"] = rnd.choice(palette)
            seen.add(tuple(v["body_color_rgb"]))
        fams.append({
            "id": f"{cc}.demo{i:02d}", "country": cc.upper(), "issuer": f"Demo issuer {code}",
            "name": f"Demo programme {code}", "schema_version": 1,
            "keywords": [f"DEMO{code}"], "negative_keywords": [], "variants": vs,
            "fields": [
                {"name": "holder_name", "roi": [0.05, 0.55, 0.75, 0.68], "regex": "[A-Z][A-Z .'-]{2,40}", "normalise": "upper"},
                {"name": "card_number", "roi": [0.05, 0.40, 0.75, 0.54], "regex": "\\d{10}", "normalise": "digits"},
                {"name": "expiry_date", "roi": [0.05, 0.72, 0.50, 0.86], "regex": "(0[1-9]|1[0-2])/(\\d{2}|\\d{4})",
                 "normalise": "date_mm_yy", "anchor": "EXP"}],
            "rules": [{"type": "regex", "field": "card_number", "critical": True},
                      {"type": "not_expired", "field": "expiry_date", "critical": True},
                      {"type": "prefix_matches_variant", "field": "card_number"}],
            "_style": i % 4})
    return fams


def num_for(fam, v, rnd):
    f = next(x for x in fam["fields"] if x["name"] == "card_number")
    n = int(f["regex"].split("{")[1].split("}")[0])
    p = v.get("number_prefix") or ""
    return p + "".join(rnd.choice("0123456789") for _ in range(n - len(p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-families", type=int, default=30)
    ap.add_argument("--refs-per-variant", type=int, default=3)
    ap.add_argument("--scale-photos-per-variant", type=int, default=1,
                    help="extra photos for EVERY variant into data/samples_scale (0 = skip)")
    a = ap.parse_args()
    rnd = random.Random(7)

    demo_dir = ROOT / "registry" / "cards" / "demo"
    shutil.rmtree(demo_dir, ignore_errors=True)
    shutil.rmtree(ROOT / "registry" / "refs", ignore_errors=True)
    demos = demo_families(a.demo_families, rnd)
    if demos:
        demo_dir.mkdir(parents=True)
        for f in demos:
            (demo_dir / f"{f['id'].replace('.', '_')}.yaml").write_text(
                "# FICTIONAL demo programme for scale testing — delete for production\n"
                + yaml.safe_dump({k: v for k, v in f.items() if not k.startswith("_")}, sort_keys=False, allow_unicode=True),
                encoding="utf-8")

    fams = []
    for p in sorted((ROOT / "registry" / "cards").rglob("*.yaml")):
        f = yaml.safe_load(p.read_text(encoding="utf-8"))
        f["_title"] = f["keywords"][0] if not f["id"].startswith(tuple(f"{c}.demo" for c in COUNTRIES)) else f"DEMO{f['id'].split('demo')[0][:2].upper()}{f['id'][-2:]}"
        f["_style"] = {"ae.fazaa": 0, "ae.esaad": 1}.get(f["id"], next((d["_style"] for d in demos if d["id"] == f["id"]), 2))
        fams.append(f)

    for f in fams:
        for v in f["variants"]:
            d = ROOT / "registry" / "refs" / f["id"] / v["id"]
            d.mkdir(parents=True, exist_ok=True)
            for i in range(a.refs_per_variant):
                img = render(f, v, rnd.choice(NAMES), num_for(f, v, rnd), "06/29", "784-1988-7654321-2", f["_style"])
                cv2.imwrite(str(d / f"ref_{i}.png"), img)

    sd = ROOT / "data" / "samples"
    shutil.rmtree(sd, ignore_errors=True)
    sd.mkdir(parents=True)
    by_id = {f["id"]: f for f in fams}
    V = lambda fid, vid: next(v for v in by_id[fid]["variants"] if v["id"] == vid)  # noqa: E731
    cases = [
        ("fazaa_silver", "ae.fazaa", "silver", "AHMED ALI HASSAN", "1042873391", "08/28", None, {}, "ACCEPT"),
        ("fazaa_gold", "ae.fazaa", "gold", "FATIMA SAEED", "2210457783", "11/27", None, {}, "ACCEPT"),
        ("fazaa_platinum", "ae.fazaa", "platinum", "OMAR KHALID", "3198822450", "02/29", None, {}, "ACCEPT"),
        ("fazaa_gold_upside_down", "ae.fazaa", "gold", "MARIAM YOUSEF", "2877001234", "05/28", None, {"upside_down": True}, "ACCEPT"),
        ("fazaa_gold_prefix_mismatch", "ae.fazaa", "gold", "KHALID RASHID", "1877001299", "05/28", None, {}, "REVIEW"),
        ("esaad_classic", "ae.esaad", "classic", "SARA AL MANSOORI", "104455667788", "12/27", "784-1990-1234567-1", {}, "ACCEPT"),
        ("esaad_gold", "ae.esaad", "gold", "NOURA SALEM", "209988776655", "07/28", "784-1985-7654321-3", {}, "ACCEPT"),
        ("esaad_platinum", "ae.esaad", "platinum", "YOUSEF IBRAHIM", "301122334455", "09/27", "784-1979-1112223-4", {}, "ACCEPT"),
        ("esaad_privilege", "ae.esaad", "privilege", "HESSA MOHAMMED", "405566778899", "01/29", "784-1992-3334445-5", {}, "ACCEPT"),
        ("esaad_gold_expired", "ae.esaad", "gold", "RASHID OBAID", "201234567890", "03/25", "784-1980-9998887-6", {}, "REJECT"),
        ("fazaa_blurry", "ae.fazaa", "silver", "AHMED ALI HASSAN", "1042873391", "08/28", None, {"blur": 9}, "REJECT"),
        ("fazaa_glare", "ae.fazaa", "gold", "FATIMA SAEED", "2210457783", "11/27", None, {"glare": True}, "REJECT"),
    ]
    # scale test: photos of several demo programmes, including look-alikes of Fazaa Gold
    for k, f in enumerate([d for d in demos][:12]):
        v = f["variants"][k % len(f["variants"])]
        cases.append((f"{f['id'].replace('.', '_')}_{v['id']}", f["id"], v["id"], rnd.choice(NAMES),
                      num_for(f, v, rnd), "10/28", None, {}, "ACCEPT"))
    manifest = []
    for i, (fn, fid, vid, name, num, exp, eid, kw, exp_status) in enumerate(cases):
        card = render(by_id[fid], V(fid, vid), name, num, exp, eid, by_id[fid]["_style"])
        cv2.imwrite(str(sd / f"{fn}.jpg"), photograph(card, 100 + i, **kw), [cv2.IMWRITE_JPEG_QUALITY, 90])
        manifest.append({"file": f"{fn}.jpg", "family": fid, "variant": vid, "expected_status": exp_status,
                         "fields": {"card_number": num, "holder_name": name, "expiry_date": exp}})
    unk = render({"fields": [], "_title": "ACME REWARDS"}, {"id": "member", "label": "MEMBER", "body_color_rgb": [120, 30, 60]},
                 "JOHN SMITH", "5555444433", "10/28", None, 3)
    cv2.imwrite(str(sd / "unknown_card.jpg"), photograph(unk, 999))
    manifest.append({"file": "unknown_card.jpg", "family": None, "variant": None, "expected_status": "REJECT", "fields": {}})
    (sd / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if a.scale_photos_per_variant:
        sc = ROOT / "data" / "samples_scale"
        shutil.rmtree(sc, ignore_errors=True)
        sc.mkdir(parents=True)
        sm, k = [], 0
        for f in fams:
            for v in f["variants"]:
                for j in range(a.scale_photos_per_variant):
                    k += 1
                    name, num = rnd.choice(NAMES), num_for(f, v, rnd)
                    eid = "784-1991-2345678-9" if f["id"] == "ae.esaad" else None
                    fn = f"{f['id'].replace('.', '_')}_{v['id']}_{j}.jpg"
                    card = render(f, v, name, num, "12/28", eid, f["_style"])
                    cv2.imwrite(str(sc / fn), photograph(card, 5000 + k, upside_down=(k % 7 == 0)),
                                [cv2.IMWRITE_JPEG_QUALITY, 88])
                    sm.append({"file": fn, "family": f["id"], "variant": v["id"], "expected_status": "ACCEPT",
                               "fields": {"card_number": num, "holder_name": name, "expiry_date": "12/28"}})
        (sc / "manifest.json").write_text(json.dumps(sm, indent=2), encoding="utf-8")
        print(f"scale set: {len(sm)} photos in {sc}")
    nv = sum(len(f["variants"]) for f in fams)
    print(f"{len(fams)} families, {nv} variants, {nv * a.refs_per_variant} reference images, {len(manifest)} test photos")


if __name__ == "__main__":
    main()
