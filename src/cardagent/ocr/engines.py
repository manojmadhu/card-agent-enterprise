"""OCR engines behind one interface, both fully offline from local files.

paddle     PaddleOCR models run on ONNX Runtime (via RapidOCR): PP-OCRv6 text detection and
           recognition (English and 45 other Latin-script languages, plus Chinese/Japanese), and the
           PP-OCRv5 Arabic recogniser. Deep-learning detection + recognition: robust to uneven
           lighting, glare, low contrast, coloured backgrounds and small text. CPU or GPU (CUDA).
           The default engine.
tesseract  Tesseract 5 (eng+ara), classic OCR on a binarised image. Works well on flat, evenly lit,
           high-contrast scans; much weaker on phone photos.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from ..settings import OcrCfg

log = logging.getLogger("cardagent.ocr")


class OcrEngine(Protocol):
    name: str

    def read_full(self, img_bgr: np.ndarray) -> dict: ...          # {"tokens","text","mean_conf"}

    def read_line(self, img_bgr: np.ndarray, script: str = "latin") -> tuple[str, float]: ...


def _binarise(img_bgr, invert: bool, upscale: float = 1.0) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if upscale != 1.0:
        gray = cv2.resize(gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    if invert:
        gray = 255 - gray
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def _find_tesseract(explicit: str = "") -> str | None:
    """Explicit setting wins; else PATH; else the standard Windows installer locations
    (the UB-Mannheim installer does not add itself to PATH)."""
    if explicit:
        if not Path(explicit).is_file():
            raise FileNotFoundError(f"ocr.tesseract_cmd points to a missing file: {explicit}")
        return explicit
    if shutil.which("tesseract"):
        return None                                   # pytesseract's default lookup works
    for base in (os.getenv("ProgramFiles"), os.getenv("ProgramFiles(x86)"), os.getenv("LOCALAPPDATA")):
        if base:
            for sub in ("Tesseract-OCR", os.path.join("Programs", "Tesseract-OCR")):
                cand = Path(base) / sub / "tesseract.exe"
                if cand.is_file():
                    return str(cand)
    return None


class TesseractOcr:
    def __init__(self, lang: str, cmd: str = ""):
        import pytesseract
        found = _find_tesseract(cmd)
        if found:
            pytesseract.pytesseract.tesseract_cmd = found
        self._t = pytesseract
        self.lang = lang
        self.name = f"tesseract-{pytesseract.get_tesseract_version()}"

    def read_full(self, img_bgr):
        tokens, confs = [], []
        for inv in (False, True):   # dark-on-light and light-on-dark cards
            d = self._t.image_to_data(_binarise(img_bgr, inv), lang=self.lang, config="--psm 11",
                                      output_type=self._t.Output.DICT)
            for t, c in zip(d["text"], d["conf"]):
                t = t.strip()
                if t and float(c) > 30:
                    tokens.append(t.upper())
                    confs.append(float(c) / 100)
        return {"tokens": tokens, "text": " ".join(tokens), "mean_conf": float(np.mean(confs)) if confs else 0.0}

    def read_line(self, img_bgr, script="latin"):
        lang = "ara" if script == "arabic" else "eng"
        best = ("", 0.0)
        for inv in (False, True):
            d = self._t.image_to_data(_binarise(img_bgr, inv, 2.0), lang=lang, config="--psm 7",
                                      output_type=self._t.Output.DICT)
            words = [(t.strip(), float(c)) for t, c in zip(d["text"], d["conf"]) if t.strip() and float(c) >= 0]
            if words:
                cand = (" ".join(w for w, _ in words), float(np.mean([c for _, c in words])) / 100)
                if cand[1] > best[1]:
                    best = cand
        return best


_ARABIC = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")


def _model_kind(path: str, version: str, size: str) -> tuple[str, str]:
    """PP-OCR version and size from a model file name (PP-OCRv6_rec_medium.onnx -> PP-OCRv6, medium)."""
    name = Path(path).name
    m = re.search(r"PP-OCRv(\d)", name)
    version = f"PP-OCRv{m.group(1)}" if m else version
    size = next((s for s in ("tiny", "small", "medium", "mobile", "server") if s in name), size)
    return version, size


def _keys_for(model: str, explicit: str | None) -> str | None:
    """The character dictionary to pass: explicit, else none when the model embeds it (ONNX metadata
    'character'), else a *dict*.txt next to the model (tools/download_ocr_models.py saves one)."""
    if explicit:
        return explicit
    try:
        import onnxruntime as ort
        meta = ort.InferenceSession(model, providers=["CPUExecutionProvider"]).get_modelmeta().custom_metadata_map
        if meta.get("character"):
            return None
    except Exception:                                   # noqa: BLE001 - fall back to a dict file
        pass
    found = sorted(Path(model).parent.glob("*dict*.txt"))
    if not found:
        raise FileNotFoundError(f"{model} has no embedded character list and no *dict*.txt next to it")
    return str(found[0])


def _crop_quad(img: np.ndarray, box) -> np.ndarray:
    """Perspective-crop one detected text line (4 corner points) to an upright strip."""
    pts = np.asarray(box, dtype=np.float32).reshape(4, 2)
    w = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
    h = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
    if w < 2 or h < 2:
        return img[0:0, 0:0]
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    crop = cv2.warpPerspective(img, cv2.getPerspectiveTransform(pts, dst), (w, h),
                               borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    return np.rot90(crop) if h / max(w, 1) >= 1.5 else crop


class PaddleOcrEngine:
    """PaddleOCR models on ONNX Runtime (RapidOCR), loaded from local files only - never downloads.

    Default models: PP-OCRv6 small detection + recognition, shipped inside the rapidocr package.
    Arabic: the PP-OCRv5 Arabic recogniser (tools/download_ocr_models.py fetches it once); without
    it Arabic text is not read (a warning is logged) and Latin text works as usual.
    read_full: detect every text line, recognise it with the Latin model, and - when the Arabic model
    is present - also with the Arabic one, keeping the Arabic reading for lines that come out in
    Arabic script with good confidence. read_line (field ROIs): detect inside the crop and join the
    lines; if nothing is detected, recognise the whole crop as one line."""

    def __init__(self, cfg: OcrCfg, root: Path):
        try:
            import rapidocr
            from rapidocr import RapidOCR
            from rapidocr.utils.typings import LangRec, ModelType, OCRVersion
        except ImportError as e:             # pragma: no cover - dependency of the package
            raise RuntimeError("ocr.engine=paddle needs the 'rapidocr' package: pip install -e .") from e
        bundled = Path(rapidocr.__file__).parent / "models"

        def path(value: str, default: Path | None) -> str | None:
            if not value:
                return str(default) if default else None
            q = Path(value)
            q = q if q.is_absolute() else root / q
            if not q.is_file():
                raise FileNotFoundError(f"OCR model file not found: {q} (python tools/download_ocr_models.py)")
            return str(q)

        def params(task_det: bool, rec: str, keys: str | None, lang: str, version: str, size: str) -> dict:
            version, size = _model_kind(rec, version, size)
            p = {"Global.use_cls": False, "Global.log_level": "error", "Global.text_score": cfg.paddle_min_score,
                 "Global.max_side_len": 2000, "Rec.model_path": rec, "Rec.lang_type": LangRec(lang),
                 "Rec.ocr_version": OCRVersion(version), "Rec.model_type": ModelType(size),
                 "EngineConfig.onnxruntime.use_cuda": cfg.use_gpu}
            if task_det:
                p.update({"Det.model_path": det, "Det.box_thresh": cfg.paddle_box_thresh,
                          "Det.unclip_ratio": cfg.paddle_unclip_ratio})
            if keys:
                p["Rec.rec_keys_path"] = keys
            return p

        det = path(cfg.paddle_det_model, bundled / "PP-OCRv6_det_small.onnx")
        rec = path(cfg.paddle_rec_model, bundled / "PP-OCRv6_rec_small.onnx")
        keys = path(cfg.paddle_rec_keys, None)
        self._latin = RapidOCR(params=params(True, rec, keys, "ch", "PP-OCRv6", "small"))   # v6: one model, 50 languages
        # field crops: detection capped at 960 px on the LONG side (the default scales the SHORT side up to
        # 736 px, which turns a 700x90 strip into ~5800x736 and takes seconds)
        self._roi = RapidOCR(params={**params(True, rec, keys, "ch", "PP-OCRv6", "small"),
                                     "Det.limit_type": "max", "Det.limit_side_len": 960})
        self._ar = None
        ar_model = Path(cfg.paddle_rec_model_ar) if cfg.paddle_rec_model_ar else None
        if ar_model is not None and not ar_model.is_absolute():
            ar_model = root / ar_model
        if ar_model is not None and ar_model.is_file():
            self._ar = RapidOCR(params=params(False, str(ar_model), _keys_for(str(ar_model), path(cfg.paddle_rec_keys_ar, None)),
                                              "arabic", "PP-OCRv5", "mobile"))
        elif ar_model is not None:
            log.warning("Arabic OCR model not found at %s: Arabic text will not be read "
                        "(python tools/download_ocr_models.py)", ar_model)
        self.min_score = cfg.paddle_min_score
        self.name = f"paddleocr-onnx({Path(det).stem}+{Path(rec).stem}{'+' + ar_model.stem if self._ar else ''})"
        self.arabic = self._ar is not None

    def _arabic(self, crop: np.ndarray) -> tuple[str, float]:
        if self._ar is None or crop.size == 0:
            return "", 0.0
        r = self._ar(crop, use_det=False, use_cls=False, use_rec=True)
        if not r.txts:
            return "", 0.0
        return str(r.txts[0]).strip(), float(r.scores[0])

    def _lines(self, img_bgr: np.ndarray, arabic: bool) -> list[tuple[str, float]]:
        r = self._latin(img_bgr, use_det=True, use_cls=False, use_rec=True)
        out = []
        for txt, score, box in zip(r.txts or (), r.scores or (), r.boxes if r.boxes is not None else ()):
            txt, score = str(txt).strip(), float(score)
            if arabic and self._ar is not None:
                at, asc = self._arabic(_crop_quad(img_bgr, box))
                if _ARABIC.search(at) and asc >= max(self.min_score, score - 0.05):
                    txt, score = at, asc
            if txt:
                out.append((txt, score))
        return out

    def read_full(self, img_bgr):
        lines = self._lines(img_bgr, arabic=True)
        tokens = [w.upper() for t, c in lines if c >= self.min_score for w in t.split()]
        confs = [c for _, c in lines]
        return {"tokens": tokens, "text": " ".join(tokens), "mean_conf": float(np.mean(confs)) if confs else 0.0}

    def read_line(self, img_bgr, script="latin"):
        """A field crop (normally one printed line): recognise it directly; when that gives nothing
        confident (two lines, text off-centre, extra margin), detect the lines inside and join them."""
        if img_bgr is None or img_bgr.size == 0:
            return "", 0.0
        if script == "arabic":
            if self._ar is None:
                return "", 0.0
            direct = self._arabic(img_bgr)
            if direct[0] and direct[1] >= self.min_score:
                return direct
            pad = cv2.copyMakeBorder(img_bgr, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
            r = self._roi(pad, use_det=True, use_cls=False, use_rec=False)
            boxes = list(r.boxes) if r.boxes is not None else []
            reads = [x for x in (self._arabic(_crop_quad(pad, b))           # right-to-left
                                 for b in sorted(boxes, key=lambda b: -min(p[0] for p in b))) if x[0]]
            return (" ".join(t for t, _ in reads), float(np.mean([s for _, s in reads]))) if reads else direct
        r = self._latin(img_bgr, use_det=False, use_cls=False, use_rec=True)
        direct = (str(r.txts[0]).strip(), float(r.scores[0])) if r.txts else ("", 0.0)
        if direct[0] and direct[1] >= self.min_score:
            return direct
        pad = cv2.copyMakeBorder(img_bgr, 8, 8, 8, 8, cv2.BORDER_REPLICATE)     # room for the detector
        r = self._roi(pad, use_det=True, use_cls=False, use_rec=True)
        rows = sorted(zip(r.txts or (), r.scores or (), r.boxes if r.boxes is not None else ()),
                      key=lambda x: (round(min(p[1] for p in x[2]) / 20), min(p[0] for p in x[2])))
        reads = [(str(t).strip(), float(s)) for t, s, _ in rows if str(t).strip()]
        if not reads:
            return direct
        return " ".join(t for t, _ in reads), float(np.mean([s for _, s in reads]))


def build_ocr(cfg: OcrCfg, root: Path) -> OcrEngine:
    return PaddleOcrEngine(cfg, root) if cfg.engine == "paddle" else TesseractOcr(cfg.tesseract_lang, cfg.tesseract_cmd)
