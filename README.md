# Card Validation Agent — Enterprise edition (v1.0.0)

Identifies GCC discount/membership cards from a photo (family → variant), extracts the printed
fields, validates them, and returns a strict JSON decision: `ACCEPT` / `REVIEW` / `REJECT`.
**Fully offline — no public AI APIs.** All models run inside your network.

- Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Deployment, GPU sizing (NVIDIA T4), onboarding cards, runbook: [docs/OPERATIONS.md](docs/OPERATIONS.md)

## Quick start (CPU, no model downloads)
```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt pytest httpx      # + OS package: tesseract-ocr tesseract-ocr-ara
python tools/make_samples.py                      # SYNTHETIC registry refs + test photos
python tools/build_index.py --force               # validate registry, build reference index
python tools/evaluate.py --today 2026-11-18       # scorecard + PASS/FAIL gate
pytest -q                                         # 20 tests
make run                                          # API on :8080
curl -F image=@data/samples/fazaa_gold.jpg -F request_id=abc http://localhost:8080/v1/cards/validate
```

## API
| Method | Path | Notes |
|---|---|---|
| POST | `/v1/cards/validate` | multipart `image` (jpeg/png/webp ≤ 8 MB) + optional `request_id` (idempotent) |
| GET | `/v1/registry` | loaded families/variants/reference counts + registry version |
| POST | `/v1/admin/registry/reload` | header `X-Admin-Token`; validates + rebuilds index, atomic swap, keeps old version on failure |
| GET | `/health/live`, `/health/ready` | liveness / readiness (registry, embedder, OCR, VLM status) |
| GET | `/metrics` | Prometheus |

Output contract: `src/cardagent/schemas.py` (`python -m cardagent.schemas` prints the JSON Schema).

## Project layout
```
config/settings.yaml          all thresholds/paths; override with CARDAGENT_<SECTION>__<KEY>
registry/cards/<cc>/*.yaml    one file per card programme (validated on load)
registry/refs/<family>/<variant>/*.png   reference images
src/cardagent/
  agent.py                    orchestrator + hot-reloadable engine state
  vision/                     quality gate, card detection/warp, colour, ORB + RANSAC
  embed/                      ClassicEmbedder (no model) | Dinov2OnnxEmbedder (production)
  index/                      FAISS vector index + keypoints + keyword inverted index (cached per version)
  ocr/                        Tesseract | PaddleOCR (local models only)
  pipeline/                   identify (2-stage), extract (homography-aligned), rules
  vlm/                        self-hosted VLM fallback (Ollama / OpenAI-compatible) + circuit breaker
  api/                        FastAPI app
  observability/              metrics, JSON logs, PII masking
tools/                        make_samples, build_index, evaluate (CI gate), export_dinov2_onnx
deploy/                       Dockerfile (CPU), Dockerfile.gpu (T4), docker-compose.yml (agent + VLM)
```

## Results on the synthetic data (this repo, CPU, classic embedder)
| Set | Photos | Correct decision | Wrong ACCEPTs | Field accuracy | p50 / p95 |
|---|---|---|---|---|---|
| Main (Fazaa, Esaad, expired, blurry, glare, unknown, look-alikes) | 25 | 25 / 25 | 0 | 100 % | 1.5 s / 1.9 s |
| Scale (every variant of 32 programmes, 279 references) | 93 | 90 / 93 (3 → REVIEW) | 0 | 100 % | 1.4 s / 1.9 s |

These are **synthetic** cards. Real accuracy must be measured on real, consented photos (see OPERATIONS §5).

## Not verified in this build environment
- The DINOv2 ONNX path and the GPU image (no GPU or model weights were available here) — the code is
  in place; validate with `tools/export_dinov2_onnx.py` + `tools/evaluate.py` on your T4 host.
- PaddleOCR engine and the VLM fallback (code paths present; need local models/endpoint).
- Card layouts/colours/number formats in `registry/cards/ae/*.yaml` are placeholders.
