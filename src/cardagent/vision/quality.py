"""Stage 1 — image quality gate.

Sharpness is the Laplacian variance of the grayscale image after stretching its 1st-99th percentile
to the full range. The raw Laplacian variance (`blur`, kept for reference) falls with contrast too,
so a sharp photo taken in a dim room used to be rejected as blurry."""
from __future__ import annotations

import cv2
import numpy as np

from ..settings import SHARPNESS_DEFAULT, QualityCfg


def sharpness(gray: np.ndarray) -> float:
    g = gray.astype(np.float32)
    lo, hi = np.percentile(g, (1, 99))
    g = np.clip((g - lo) * (255.0 / max(float(hi - lo), 1.0)), 0, 255)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def assess(img_bgr: np.ndarray, cfg: QualityCfg, ocr_engine: str = "paddle") -> dict:
    h, w = img_bgr.shape[:2]
    scale = 1000.0 / max(h, w)
    small = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharp = sharpness(gray)
    min_sharp = cfg.min_sharpness if cfg.min_sharpness is not None else SHARPNESS_DEFAULT.get(ocr_engine, 80.0)
    brightness = float(gray.mean() / 255.0)
    glare = float((gray > 250).mean())
    issues = []
    if min(h, w) < cfg.min_side:
        issues.append("LOW_RESOLUTION")
    if sharp < min_sharp or (cfg.min_blur > 0 and blur < cfg.min_blur):
        issues.append("IMAGE_BLURRY")
    if not cfg.brightness_min <= brightness <= cfg.brightness_max:
        issues.append("BAD_EXPOSURE")
    if glare > cfg.max_glare:
        issues.append("GLARE")
    return {"width": w, "height": h, "sharpness": round(sharp, 1), "min_sharpness": min_sharp,
            "blur": round(blur, 1), "brightness": round(brightness, 3),
            "glare_ratio": round(glare, 4), "passed": not issues, "issues": issues}
