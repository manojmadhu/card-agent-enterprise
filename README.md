# Card Validation Agent — Enterprise edition (v1.0.0)

Identifies GCC discount/membership cards from a photo (family → variant), extracts the printed
fields, validates them, and returns a strict JSON decision: `ACCEPT` / `REVIEW` / `REJECT`.
**Fully offline — no public AI APIs.** All models run inside your network.

- Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Deployment, GPU sizing (NVIDIA T4), onboarding cards, runbook: [docs/OPERATIONS.md](docs/OPERATIONS.md)

## Quick start (CPU)
```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"                           # + OS package: tesseract-ocr tesseract-ocr-ara
docker run -d --name cardagent-qdrant -p 127.0.0.1:6333:6333 \
  -v cardagent_qdrant:/qdrant/storage qdrant/qdrant:v1.18.1          # vector store
python tools/make_samples.py                      # SYNTHETIC registry refs + test photos
python tools/build_index.py                       # validate registry, sync it into Qdrant (incremental)
python tools/evaluate.py --today 2026-11-18       # scorecard + PASS/FAIL gate
pytest -q                                         # no Qdrant server needed (in-process Qdrant for its tests)
uvicorn cardagent.api.app:app --port 8080         # API
curl -F image=@data/samples/fazaa_gold.jpg http://127.0.0.1:8080/v1/cards/validate
```
No Docker? Set `vector_store.backend: faiss` in `config/settings.yaml` (file index, no card-management API).

## API
| Method | Path | Notes |
|---|---|---|
| POST | `/v1/cards/validate` | multipart `image` (jpeg/png/webp ≤ 8 MB) + optional `request_id` (idempotent: a retry with the same id **and** same image returns the cached result; a different image is always processed) |
| GET | `/v1/registry` | loaded families/variants/reference counts + registry version |
| GET | `/health/live`, `/health/ready` | liveness / readiness (vector store, registry, embedder, OCR, VLM status) |
| GET | `/metrics` | Prometheus |

**Card management** (header `X-Admin-Token`; Qdrant backend). Each call touches only that card — no re-index:

| Method | Path | Notes |
|---|---|---|
| GET | `/v1/admin/cards` | all card families with owner (`registry`/`api`) and reference counts |
| GET | `/v1/admin/cards/{family_id}` | definition + reference images (ids) per variant |
| PUT | `/v1/admin/cards/{family_id}` | create (201) / replace (200) a definition; JSON or YAML body, same schema as `registry/cards/*.yaml` |
| DELETE | `/v1/admin/cards/{family_id}` | remove the card and all its reference vectors |
| POST | `/v1/admin/cards/{family_id}/variants/{variant_id}/references` | multipart `images` (1–20); embedded and indexed immediately; re-upload = no-op |
| DELETE | `/v1/admin/cards/{family_id}/references/{ref_id}` | remove one reference image |
| POST | `/v1/admin/cards/sync` | incremental sync from `registry/` (same as `tools/build_index.py`) |
| POST | `/v1/admin/registry/reload` | re-read definitions now (replicas also poll every `refresh_seconds`) |

Registry-owned cards answer `409` to API changes unless `?force=true` — see [OPERATIONS §4](docs/OPERATIONS.md).

**Card Agent Console** (card-agent-web): set the console's API base URL to `http://<host>:8080/console/v1`.
That API speaks the console's MVP formats (family YAML, result envelope) on top of the same pipeline and card
store; `/v1` is unchanged. Routes, translation rules and security: [docs/CONSOLE_API.md](docs/CONSOLE_API.md).

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
  store/                      card store: Qdrant (production) | FAISS (dev), keyword index, incremental sync
  index/                      FAISS file index (faiss backend)
  ocr/                        Tesseract | PaddleOCR (local models only)
  pipeline/                   identify (2-stage), extract (homography-aligned), rules
  vlm/                        self-hosted VLM fallback (Ollama / OpenAI-compatible) + circuit breaker
  api/                        FastAPI app + card management router (admin_cards.py)
  observability/              metrics, JSON logs, PII masking
tools/                        make_samples, build_index, evaluate (CI gate), export_dinov2_onnx
deploy/                       Dockerfile (CPU), Dockerfile.gpu (T4), docker-compose.yml (Qdrant + sync + agent + VLM)
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
