"""Local features: body colour (Lab), ORB keypoints, RANSAC homography verification."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# ---------- colour ----------
def body_colour_rgb(img_bgr: np.ndarray) -> tuple[int, int, int]:
    """Median colour of the card body, sampled from border bands (avoids printed text)."""
    h, w = img_bgr.shape[:2]
    band = np.concatenate([
        img_bgr[int(h * .03):int(h * .12), int(w * .03):int(w * .97)].reshape(-1, 3),
        img_bgr[int(h * .90):int(h * .97), int(w * .03):int(w * .97)].reshape(-1, 3),
        img_bgr[int(h * .15):int(h * .85), int(w * .90):int(w * .97)].reshape(-1, 3),
    ])
    b, g, r = np.median(band, axis=0)
    return int(r), int(g), int(b)


def _lab(rgb) -> np.ndarray:
    return cv2.cvtColor(np.uint8([[list(rgb)[::-1]]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float)


def colour_similarity(rgb_a, rgb_b, scale: float = 30.0) -> float:
    return float(np.exp(-np.linalg.norm(_lab(rgb_a) - _lab(rgb_b)) / scale))


def design_mask(rois, w: int, h: int, pad: float = 0.015) -> np.ndarray:
    m = np.full((h, w), 255, np.uint8)
    for x0, y0, x1, y1 in rois:
        m[int(max(0, y0 - pad) * h):int(min(1, y1 + pad) * h), int(max(0, x0 - pad) * w):int(min(1, x1 + pad) * w)] = 0
    return m


# ---------- keypoints ----------
@dataclass
class Keypoints:
    pts: np.ndarray          # (N, 2) float32 pixel coords
    des: np.ndarray | None   # (N, 32) uint8 ORB descriptors


def extract_keypoints(img_bgr: np.ndarray, n_features: int = 1000, mask: np.ndarray | None = None) -> Keypoints:
    """mask (uint8, same size): 0 = ignore. References mask out personal-data regions so that
    verification matches the card DESIGN (logo, artwork, fixed print), not names and numbers."""
    orb = cv2.ORB_create(nfeatures=n_features)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)  # lighting robustness
    kp, des = orb.detectAndCompute(gray, mask)
    pts = np.float32([k.pt for k in kp]) if kp else np.zeros((0, 2), np.float32)
    return Keypoints(pts=pts, des=des)


@dataclass
class GeoMatch:
    inliers: int
    good_matches: int
    H: np.ndarray | None     # maps QUERY pixel coords -> REFERENCE pixel coords


_bf = cv2.BFMatcher(cv2.NORM_HAMMING)


def verify(query: Keypoints, ref: Keypoints, ratio: float = 0.75) -> GeoMatch:
    """Lowe ratio test, then RANSAC homography; inliers = geometrically consistent matches."""
    if query.des is None or ref.des is None or len(query.des) < 8 or len(ref.des) < 8:
        return GeoMatch(0, 0, None)
    pairs = _bf.knnMatch(query.des, ref.des, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < ratio * p[1].distance]
    if len(good) < 8:
        return GeoMatch(0, len(good), None)
    src = query.pts[[m.queryIdx for m in good]].reshape(-1, 1, 2)
    dst = ref.pts[[m.trainIdx for m in good]].reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None or not _plausible_h(H):
        return GeoMatch(0, len(good), None)
    return GeoMatch(int(mask.sum()), len(good), H)


def _plausible_h(H: np.ndarray) -> bool:
    """Reject degenerate/flipping homographies (a card photo is only mildly warped after normalising)."""
    det = np.linalg.det(H[:2, :2])
    return 0.3 < det < 3.0 and abs(H[2, 0]) < 2e-3 and abs(H[2, 1]) < 2e-3
