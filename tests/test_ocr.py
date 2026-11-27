"""OCR engines and the quality gate that sits in front of them."""
import cv2
import numpy as np
import pytest

from cardagent.settings import PROJECT_ROOT, OcrCfg, QualityCfg
from cardagent.ocr.engines import PaddleOcrEngine, build_ocr
from cardagent.vision import normalise, quality

from . import cards

W, H = 1012, 638
ROIS = {"card_number": (0.05, 0.40, 0.75, 0.54), "holder_name": (0.05, 0.55, 0.75, 0.68),
        "expiry_date": (0.05, 0.72, 0.50, 0.86)}


@pytest.fixture(scope="module")
def paddle():
    return build_ocr(OcrCfg(engine="paddle"), PROJECT_ROOT)


def card_photo(seed=3, f=None):
    img = cards.photo(cards.render("ae.alpha", "amber", "FATIMA SAEED", "2345678901", "07/31"), seed).astype(np.float32)
    return np.clip(f(img) if f else img, 0, 255).astype(np.uint8)


def crop(card, roi):
    x0, y0, x1, y1 = roi
    return card[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)]


def test_paddle_is_the_default_engine(paddle):
    assert OcrCfg().engine == "paddle" and isinstance(paddle, PaddleOcrEngine)
    assert paddle.name.startswith("paddleocr-onnx(PP-OCRv6_det_small+PP-OCRv6_rec_small")


def test_paddle_reads_the_whole_card(paddle):
    card, _ = normalise.normalise(card_photo())
    r = paddle.read_full(card)
    assert {"ALPHACLUB", "AMBER", "FATIMA", "SAEED", "07/31"} <= set(r["tokens"])
    assert "2345 6789 01" in r["text"] and r["mean_conf"] > 0.9


@pytest.mark.parametrize("condition", [None, lambda f: f * 0.3 + 12, lambda f: cv2.GaussianBlur(f, (0, 0), 2.5)],
                         ids=["clean", "dim", "blurred"])
def test_paddle_reads_field_crops(paddle, condition):
    card, _ = normalise.normalise(card_photo(f=condition))
    assert paddle.read_line(crop(card, ROIS["card_number"]))[0].replace(" ", "") == "2345678901"
    assert paddle.read_line(crop(card, ROIS["holder_name"]))[0] == "FATIMA SAEED"
    assert paddle.read_line(crop(card, ROIS["expiry_date"]))[0].endswith("07/31")


def test_paddle_crop_with_two_lines_and_empty_crop(paddle):
    card, _ = normalise.normalise(card_photo())
    text, conf = paddle.read_line(card[int(.40 * H):int(.68 * H), int(.05 * W):int(.75 * W)])
    assert text.replace(" ", "").startswith("2345678901") and "FATIMA SAEED" in text and conf > 0.5
    assert paddle.read_line(card[0:0, 0:0]) == ("", 0.0)


def test_arabic_model_is_optional(tmp_path):
    eng = build_ocr(OcrCfg(engine="paddle", paddle_rec_model_ar=str(tmp_path / "missing.onnx")), PROJECT_ROOT)
    assert eng.arabic is False and eng.read_line(np.zeros((40, 200, 3), np.uint8), "arabic") == ("", 0.0)


def test_missing_custom_model_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_ocr_models"):
        build_ocr(OcrCfg(engine="paddle", paddle_det_model=str(tmp_path / "nope.onnx")), PROJECT_ROOT)


@pytest.mark.skipif(not (PROJECT_ROOT / OcrCfg().paddle_rec_model_ar).is_file(),
                    reason="Arabic model not downloaded (python tools/download_ocr_models.py)")
def test_arabic_reading(paddle):
    from PIL import Image, ImageDraw, ImageFont, features
    if not features.check("raqm"):
        pytest.skip("Pillow without raqm cannot shape Arabic text for the test image")
    font = next((f for f in ("DejaVuSans.ttf", "arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
                 if _font_ok(f)), None)
    img = Image.new("RGB", (700, 110), "white")
    ImageDraw.Draw(img).text((20, 15), "بطاقة فزعة", font=ImageFont.truetype(font, 64), fill="black",
                             direction="rtl", language="ar")
    text, conf = paddle.read_line(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), "arabic")
    assert "فزعة" in text and conf > 0.5


def _font_ok(f):
    from PIL import ImageFont
    try:
        ImageFont.truetype(f, 10)
        return True
    except OSError:
        return False


def test_tesseract_still_available():
    try:
        eng = build_ocr(OcrCfg(engine="tesseract"), PROJECT_ROOT)
    except Exception as e:                                   # noqa: BLE001 - optional on a PC
        pytest.skip(f"Tesseract not installed ({type(e).__name__})")
    card, _ = normalise.normalise(card_photo())
    assert eng.name.startswith("tesseract") and "ALPHACLUB" in eng.read_full(card)["tokens"]


# ------------------------------------------------------------------------------ quality gate
def test_dim_but_sharp_photo_is_not_called_blurry():
    q = quality.assess(card_photo(f=lambda f: f * 0.38 + 18), QualityCfg(), "paddle")
    assert q["passed"], q["issues"]
    assert q["sharpness"] > 200 and q["blur"] < 60          # the old raw measure would have rejected it


def test_blur_limit_depends_on_the_ocr_engine():
    soft = card_photo(f=lambda f: cv2.GaussianBlur(f, (0, 0), 2.2))
    assert quality.assess(soft, QualityCfg(), "paddle")["passed"]                 # PaddleOCR still reads it
    assert "IMAGE_BLURRY" in quality.assess(soft, QualityCfg(), "tesseract")["issues"]
    very = card_photo(f=lambda f: cv2.GaussianBlur(f, (0, 0), 7))
    assert "IMAGE_BLURRY" in quality.assess(very, QualityCfg(), "paddle")["issues"]


def test_explicit_and_legacy_thresholds():
    img = card_photo()
    assert "IMAGE_BLURRY" in quality.assess(img, QualityCfg(min_sharpness=1e6), "paddle")["issues"]
    dim = card_photo(f=lambda f: f * 0.38 + 18)
    assert "IMAGE_BLURRY" in quality.assess(dim, QualityCfg(min_blur=60), "paddle")["issues"]   # legacy, opt-in
