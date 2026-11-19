"""Stage 1 — image quality gate."""
from __future__ import annotations

import cv2
import numpy as np

from ..settings import QualityCfg


def assess(img_bgr: np.ndarray, cfg: QualityCfg) -> dict:
    h, w = img_bgr.shape[:2]
    scale = 1000.0 / max(h, w)
    small = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean() / 255.0)
    glare = float((gray > 250).mean())
    issues = []
    if min(h, w) < cfg.min_side:
        issues.append("LOW_RESOLUTION")
    if blur < cfg.min_blur:
        issues.append("IMAGE_BLURRY")
    if not cfg.brightness_min <= brightness <= cfg.brightness_max:
        issues.append("BAD_EXPOSURE")
    if glare > cfg.max_glare:
        issues.append("GLARE")
    return {"width": w, "height": h, "blur": round(blur, 1), "brightness": round(brightness, 3),
            "glare_ratio": round(glare, 4), "passed": not issues, "issues": issues}
