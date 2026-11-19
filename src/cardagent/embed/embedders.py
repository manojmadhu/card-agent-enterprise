"""Global image embedders: image -> L2-normalised vector for the vector index.

- ClassicEmbedder: no model weights; Lab colour layout + HOG shape descriptor. Works offline
  out of the box, good for tens of programmes; use for dev/CI.
- Dinov2OnnxEmbedder: DINOv2 (Apache-2.0) exported to ONNX, run with ONNX Runtime on CPU or
  GPU (T4: CUDAExecutionProvider). Recommended for production / GCC scale.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from ..settings import EmbedderCfg


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, img_bgr: np.ndarray) -> np.ndarray: ...


def _l2(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32).ravel()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class ClassicEmbedder:
    name = "classic-lab-hog-v1"

    def __init__(self):
        self._hog = cv2.HOGDescriptor((128, 80), (16, 16), (8, 8), (8, 8), 9)
        self.dim = len(self.embed(np.zeros((80, 128, 3), np.uint8)))

    def embed(self, img_bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(cv2.resize(img_bgr, (16, 10), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2LAB)
        colour = _l2(lab.astype(np.float32) - 128.0)
        gray = cv2.cvtColor(cv2.resize(img_bgr, (128, 80), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        shape = _l2(self._hog.compute(gray))
        return _l2(np.concatenate([0.6 * colour, 0.8 * shape]))


class Dinov2OnnxEmbedder:
    """Expects an ONNX graph with input 'pixel_values' (1,3,S,S) and output 'embedding' (1,D)
    — produced by tools/export_dinov2_onnx.py."""

    _MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    _STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, onnx_path: Path, providers: list[str], size: int = 224):
        import onnxruntime as ort  # lazy: only needed in production mode

        if not onnx_path.exists():
            raise FileNotFoundError(f"DINOv2 ONNX model not found at {onnx_path}. "
                                    "Run tools/export_dinov2_onnx.py on a machine with the weights.")
        avail = ort.get_available_providers()
        self.session = ort.InferenceSession(str(onnx_path), providers=[p for p in providers if p in avail])
        self.size = size
        self.input_name = self.session.get_inputs()[0].name
        self.dim = int(self.session.get_outputs()[0].shape[-1])
        self.name = f"dinov2-onnx:{onnx_path.name}"

    def embed(self, img_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(cv2.resize(img_bgr, (self.size, self.size), interpolation=cv2.INTER_AREA),
                           cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = ((rgb - self._MEAN) / self._STD).transpose(2, 0, 1)[None]
        out = self.session.run(None, {self.input_name: x})[0]
        return _l2(out)


def build_embedder(cfg: EmbedderCfg, root: Path) -> Embedder:
    if cfg.kind == "dinov2_onnx":
        p = Path(cfg.onnx_path)
        return Dinov2OnnxEmbedder(p if p.is_absolute() else root / p, cfg.onnx_providers, cfg.input_size)
    return ClassicEmbedder()
