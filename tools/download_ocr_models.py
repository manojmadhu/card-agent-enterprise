"""Download the PaddleOCR models that are not shipped in the rapidocr package (run once, online).

  python tools/download_ocr_models.py              # Arabic recogniser (PP-OCRv5 Arabic, ~8 MB)
  python tools/download_ocr_models.py --medium     # + PP-OCRv6 medium detection/recognition (GPU hosts)
  python tools/download_ocr_models.py --to D:\\models\\paddleocr

The PP-OCRv6 small models (English + Latin scripts) ship inside the rapidocr package and need no
download. Files are checked against the SHA-256 sums published with the installed rapidocr version
and saved to models/paddleocr/ (git-ignored). For an offline host: run this on a connected PC and copy
the folder. The service never downloads anything itself.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ARABIC = ("PP-OCRv5", "rec", "arabic_PP-OCRv5_rec_mobile")
MEDIUM = [("PP-OCRv6", "det", "multi_PP-OCRv6_det_medium"), ("PP-OCRv6", "rec", "multi_PP-OCRv6_rec_medium")]
DICTS = {"arabic_PP-OCRv5_rec_mobile": ("mnn", "PP-OCRv5", "rec", "arabic_PP-OCRv5_rec_mobile")}


def catalogue() -> dict:
    import rapidocr
    import yaml
    return yaml.safe_load((Path(rapidocr.__file__).parent / "default_models.yaml").read_text(encoding="utf-8"))


def fetch(url: str, dst: Path, sha256: str | None):
    if dst.exists() and sha256 and hashlib.sha256(dst.read_bytes()).hexdigest() == sha256:
        print(f"  ok (already there) {dst.name}")
        return
    print(f"  downloading {url}")
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as fh:
        while chunk := r.read(1 << 20):
            fh.write(chunk)
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
    if sha256 and digest != sha256:
        tmp.unlink()
        sys.exit(f"ERROR: checksum mismatch for {dst.name}: got {digest}, expected {sha256}")
    tmp.replace(dst)
    print(f"  saved {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--medium", action="store_true", help="also PP-OCRv6 medium det + rec (more accurate, slower)")
    ap.add_argument("--no-arabic", action="store_true")
    ap.add_argument("--to", default=str(ROOT / "models" / "paddleocr"))
    a = ap.parse_args()
    out = Path(a.to)
    out.mkdir(parents=True, exist_ok=True)
    cat = catalogue()
    wanted = ([] if a.no_arabic else [ARABIC]) + (MEDIUM if a.medium else [])
    for version, task, key in wanted:
        entry = cat["onnxruntime"][version][task][key]
        name = key.removeprefix("multi_") + ".onnx"
        fetch(entry["model_dir"], out / name, entry.get("SHA256"))
        if key in DICTS:                                    # the character list, in case the model has none
            eng, v, t, k = DICTS[key]
            url = cat[eng][v][t][k].get("dict_url")
            if url:
                fetch(url, out / url.rsplit("/", 1)[-1], None)
    rel = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    print("\nconfig/settings.yaml -> ocr:")
    if not a.no_arabic:
        print(f"  paddle_rec_model_ar: {rel / 'arabic_PP-OCRv5_rec_mobile.onnx'}")
    if a.medium:
        print(f"  paddle_det_model: {rel / 'PP-OCRv6_det_medium.onnx'}")
        print(f"  paddle_rec_model: {rel / 'PP-OCRv6_rec_medium.onnx'}")


if __name__ == "__main__":
    main()
