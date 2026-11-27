# OCR: PaddleOCR (default) and Tesseract

## Which engine, and why
| | PaddleOCR models on ONNX Runtime (`engine: paddle`, default) | Tesseract 5 (`engine: tesseract`) |
|---|---|---|
| How it reads | deep-learning text **detection** (PP-OCRv6) + **recognition** per line (PP-OCRv6; Arabic: PP-OCRv5 Arabic) | classic OCR on a black-and-white (Otsu) image |
| Phone photos (shadow, glare, soft focus, dim light, small text) | robust | weak: thresholding breaks on uneven light and blur |
| Languages | English and 45 other Latin-script languages, Chinese, Japanese; Arabic with the extra Arabic model | `eng+ara` |
| GPU | yes (ONNX Runtime CUDA, `use_gpu: true`; the T4 image sets it) | no |
| Install | `pip install -e .` (the `rapidocr` package ships the PP-OCRv6 models); Arabic model: one download | system package / Windows installer |

**OpenCV is not an alternative OCR engine.** It is the image library the pipeline already uses
for card detection, straightening, alignment and the quality gate. Its text add-on needs separate
models, and none are maintained for Arabic.

**Why ONNX Runtime and not the `paddleocr` Python package.** Same models, same accuracy. The official
package pulls in the PaddlePaddle framework and PaddleX, and force-installs a second OpenCV
(`opencv-contrib-python 4.10`) next to `opencv-python-headless`: two packages owning `cv2`.
RapidOCR (Apache-2.0) runs the PaddleOCR models on ONNX Runtime, which the service already uses for
DINOv2, and uses the same `opencv-python-headless`.

## Measured (synthetic test cards, 2-core CPU, no GPU)
40 card photos (front-only and two-sided cards), 5 each under: clean, strong shadow, glare spot,
blur (σ 2.2), low resolution (560 px wide), dim light, JPEG quality 22, sensor noise. Same cards,
references and settings; only the OCR engine differs (`tools/compare_ocr.py`).

| Result | PaddleOCR | Tesseract |
|---|---|---|
| Right decision (status agreement) | **0.95** | 0.675 |
| Wrong ACCEPTs | **0** | 1 |
| Card number / name / expiry read correctly | **0.975 / 0.975 / 0.975** | 0.88 / 0.91 / 0.91 |
| Latency p50 / p95 (whole request, CPU) | 1.34 s / 3.40 s | 1.19 s / 2.34 s |

Per condition (correct ACCEPTs of 5): PaddleOCR 5/5 on clean, shadow, blur, low-res, JPEG, noise;
4/5 on glare (one photo rejected by the glare check) and dim (one variant colour ambiguity).
Tesseract: 0/5 on blur, 1/5 on glare. With the quality gate switched off, Tesseract made 4 wrong
ACCEPTs out of 40; PaddleOCR none. Field crops read directly in ~20 ms each with PaddleOCR.

These are synthetic cards. **Measure on your own photos before switching production**:

```bash
python tools/compare_ocr.py --manifest data/eval/manifest.json --today 2026-11-27
```

## Setup
1. `pip install -e ".[dev]"`: installs `rapidocr` with the PP-OCRv6 small models (no download).
2. Arabic, once, on a PC with internet: `python tools/download_ocr_models.py`. This saves
   `models/paddleocr/arabic_PP-OCRv5_rec_mobile.onnx` (SHA-256 checked); copy the folder to offline
   hosts. Without it the service logs a warning and reads Latin text only.
3. Optional, GPU hosts: `python tools/download_ocr_models.py --medium` and set `paddle_det_model` /
   `paddle_rec_model` to the PP-OCRv6 medium files (more accurate, slower on CPU).
4. Docker: the images contain the PP-OCRv6 models; compose mounts `models/paddleocr` for Arabic.
   The GPU image sets `CARDAGENT_OCR__USE_GPU=true`.

Settings (`config/settings.yaml` → `ocr`): `engine`, `paddle_det_model`, `paddle_rec_model`,
`paddle_rec_model_ar` (`""` = off), `paddle_min_score` (drop lines below this confidence),
`paddle_box_thresh`, `paddle_unclip_ratio`, `use_gpu`, `tesseract_lang`, `tesseract_cmd`.
Changing the OCR engine needs a restart; the OCR engine does not affect stored vectors.

## Quality gate (in front of the OCR)
The gate's sharpness measure is now contrast-normalised: the Laplacian variance of the image after
stretching its 1st–99th percentile. The old raw Laplacian variance falls with contrast, so sharp
photos taken in dim light were rejected as `IMAGE_BLURRY` before any OCR ran.

The threshold depends on the engine: `quality.min_sharpness: null` means PaddleOCR 15 (it still reads
blur of about σ 3.5) and Tesseract 80 (it misreads from about σ 1.5). Set a number to override; it is
also tunable at runtime (`PATCH /v1/settings`). The raw measure is still reported as `quality.blur`;
`quality.min_blur` (default 0 = off) re-enables the old check.

## Troubleshooting
| Symptom | Check |
|---|---|
| Many `RETAKE_REQUIRED` + `IMAGE_BLURRY` | `quality.sharpness` in the result vs `quality.min_sharpness`; the capture app's focus/resolution. Lower `min_sharpness` only after a what-if evaluation |
| Startup warning "Arabic OCR model not found" | `python tools/download_ocr_models.py` |
| `FileNotFoundError … OCR model file not found` | a path in `ocr.paddle_*_model` is wrong (paths are relative to the project folder) |
| Slow OCR on CPU | expected ~1 s per card side on 2 cores; use the GPU image, or more CPU cores |
| Fields read with spaces (`2345 6789 01`) | normal: the field's `normalise` (e.g. `digits`) removes them before the regex |
