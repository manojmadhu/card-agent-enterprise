"""OCR engines behind one interface. Tesseract (default, CPU) or PaddleOCR (GPU-capable,
better on worn/embossed cards). Both run fully offline from local model files."""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from ..settings import OcrCfg


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


class PaddleOcrEngine:
    """PaddleOCR with LOCAL model dirs only (never auto-downloads). Arabic lines come back in
    visual left-to-right order: reverse word order per line for logical RTL order."""

    def __init__(self, cfg: OcrCfg, root: Path):
        from paddleocr import PaddleOCR  # lazy import; optional dependency

        def p(x):
            q = Path(x)
            return str(q if q.is_absolute() else root / q)

        common = dict(use_angle_cls=False, use_gpu=cfg.use_gpu, show_log=False, det_model_dir=p(cfg.paddle_det_dir))
        self._en = PaddleOCR(lang="en", rec_model_dir=p(cfg.paddle_rec_dir_en), **common)
        self._ar = PaddleOCR(lang="ar", rec_model_dir=p(cfg.paddle_rec_dir_ar), **common)
        self.name = "paddleocr"

    @staticmethod
    def _lines(res):
        return [(ln[1][0], float(ln[1][1])) for ln in (res[0] or [])] if res else []

    def read_full(self, img_bgr):
        lines = self._lines(self._en.ocr(img_bgr, cls=False)) + self._lines(self._ar.ocr(img_bgr, cls=False))
        tokens = [w.upper() for t, c in lines if c > 0.3 for w in t.split()]
        confs = [c for _, c in lines]
        return {"tokens": tokens, "text": " ".join(tokens), "mean_conf": float(np.mean(confs)) if confs else 0.0}

    def read_line(self, img_bgr, script="latin"):
        eng = self._ar if script == "arabic" else self._en
        lines = self._lines(eng.ocr(img_bgr, cls=False))
        if not lines:
            return "", 0.0
        text = " ".join(t for t, _ in lines)
        if script == "arabic":
            text = " ".join(reversed(text.split()))
        return text, float(np.mean([c for _, c in lines]))


def build_ocr(cfg: OcrCfg, root: Path) -> OcrEngine:
    return PaddleOcrEngine(cfg, root) if cfg.engine == "paddle" else TesseractOcr(cfg.tesseract_lang, cfg.tesseract_cmd)
