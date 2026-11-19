"""Stage 2 — find the card in the photo and warp it to a canonical 1012x638 view."""
from __future__ import annotations

import cv2
import numpy as np

from .geometry import CARD_W, CARD_H

ID1_ASPECT = 85.6 / 54.0


def _order(pts: np.ndarray) -> np.ndarray:
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.float32([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]])


def _plausible(quad: np.ndarray, frame_area: float) -> bool:
    q = _order(quad)
    w = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
    h = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
    ar = max(w, h) / max(1.0, min(w, h))
    area = cv2.contourArea(q)
    return 1.3 <= ar <= 1.9 and 0.10 * frame_area <= area <= 0.92 * frame_area


def _quad_from_contour(c: np.ndarray) -> np.ndarray:
    hull = cv2.convexHull(c)
    for eps in (0.01, 0.02, 0.03, 0.05):
        approx = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    return cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)


def _by_colour(small: np.ndarray) -> np.ndarray | None:
    """Card vs background: distance from the median border colour in Lab."""
    lab = cv2.cvtColor(cv2.GaussianBlur(small, (7, 7), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.concatenate([lab[:12].reshape(-1, 3), lab[-12:].reshape(-1, 3),
                             lab[:, :12].reshape(-1, 3), lab[:, -12:].reshape(-1, 3)])
    diff = np.linalg.norm(lab - np.median(border, axis=0), axis=2)
    thr = max(12.0, float(np.percentile(np.linalg.norm(border - np.median(border, axis=0), axis=1), 99)) * 1.2)
    mask = (diff > thr).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    q = _quad_from_contour(max(cnts, key=cv2.contourArea))
    return q if _plausible(q, small.shape[0] * small.shape[1]) else None


def _by_edges(small: np.ndarray) -> np.ndarray | None:
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    frame, best, best_area = small.shape[0] * small.shape[1], None, 0.0
    for lo, hi in ((30, 100), (60, 180)):
        edges = cv2.dilate(cv2.Canny(gray, lo, hi), np.ones((3, 3), np.uint8), iterations=2)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area <= best_area:
                continue
            q = _quad_from_contour(c)
            if _plausible(q, frame):
                best, best_area = q, area
    return best


def find_card_quad(img: np.ndarray) -> tuple[np.ndarray | None, str]:
    h, w = img.shape[:2]
    scale = 800.0 / max(h, w)
    small = cv2.resize(img, (int(w * scale), int(h * scale)))
    for method, fn in (("colour", _by_colour), ("edges", _by_edges)):
        q = fn(small)
        if q is not None:
            return _order(q / scale), method
    return None, "none"


def normalise(img: np.ndarray) -> tuple[np.ndarray, dict]:
    quad, method = find_card_quad(img)
    info = {"card_detected": quad is not None, "detector": method}
    if quad is None:
        warped = img  # card fills the frame (e.g. fixed scanner) — use as is
    else:
        tl, tr, br, bl = quad
        width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
        height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
        if height > width:  # card captured in portrait -> rotate corner order
            quad = np.float32([bl, tl, tr, br])
        dst = np.float32([[0, 0], [CARD_W, 0], [CARD_W, CARD_H], [0, CARD_H]])
        warped = cv2.warpPerspective(img, cv2.getPerspectiveTransform(quad, dst), (CARD_W, CARD_H))
    return cv2.resize(warped, (CARD_W, CARD_H)), info
