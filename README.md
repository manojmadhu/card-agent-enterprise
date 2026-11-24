# Card Validation Agent — Enterprise edition (v1.0.0)

Identifies GCC discount/membership cards from a photo (family → variant), extracts the printed
fields, validates them, and returns a strict JSON decision: `ACCEPT` / `REVIEW` / `REJECT`.
**Fully offline — no public AI APIs.** All models run inside your network; **Qdrant** is the only store.

- API reference (every endpoint): [docs/API.md](docs/API.md) · OpenAPI at `/docs` on a running server
- Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Deployment, GPU sizing (NVIDIA T4), card onboarding, runbook: [docs/OPERATIONS.md](docs/OPERATIONS.md)
- Windows step by step: [docs/SETUP_WINDOWS.md](docs/SETUP_WINDOWS.md)

## What the platform does
| Feature | Endpoints |
|---|---|
| Validate a card photo, plain or as a live step-by-step stream (SSE) | `POST /v1/cards/validate`, `POST /v1/cards/validate/stream` |
| Manage card programmes, variants, fields, rules and reference images | `/v1/cards…` |
| Import Git-managed cards from `registry/` (optional) | `POST /v1/registry/sync` |
| Run history, human review queue, enroll reviewed photos as references | `/v1/runs…`, `/v1/review` |
| Operating statistics (automation rate, review rate, latency, reasons) | `GET /v1/stats` |
| Change thresholds at runtime, no restart | `GET/PATCH/DELETE /v1/settings` |
| Accuracy evaluation on labelled sets, incl. what-if settings | `/v1/evaluation-sets…`, `/v1/evaluations…` |
| Health and Prometheus metrics | `/health/live`, `/health/ready`, `/metrics` |

The platform **starts empty**: no demo cards. Add your programmes through the API (or `registry/`),
upload a few reference images per variant, and it is ready. `examples/cards/` has templates.

## Run on a local PC
**You need:** Python 3.10–3.12 (64-bit), Git, Tesseract OCR with English + Arabic data, Docker (for Qdrant).
Windows: follow [docs/SETUP_WINDOWS.md](docs/SETUP_WINDOWS.md).

```bash
git clone -b feature/qdrant-platform https://github.com/manojmadhu/card-agent-enterprise.git
cd card-agent-enterprise
python3 -m venv .venv && . .venv/bin/activate        # Windows: py -3.12 -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"                              # + OS packages: tesseract-ocr tesseract-ocr-ara

# Qdrant (data kept in the docker volume cardagent_qdrant)
docker run -d --name cardagent-qdrant --restart unless-stopped \
  -p 127.0.0.1:6333:6333 -v cardagent_qdrant:/qdrant/storage qdrant/qdrant:v1.18.1
```

**Matcher model.** `config/settings.yaml` uses DINOv2 (`models/dinov2/dinov2_vits14.onnx`, ~90 MB,
not in Git). Copy it from another install, or create it once:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers onnx onnxscript huggingface_hub
hf download facebook/dinov2-small --local-dir models/dinov2-small
python tools/export_dinov2_onnx.py --model-dir models/dinov2-small --out models/dinov2/dinov2_vits14.onnx
```
Without the model, use the built-in matcher: `export CARDAGENT_EMBEDDER__KIND=classic CARDAGENT_EMBEDDER__SIMILARITY_FLOOR=0.55`
(Windows: `$env:CARDAGENT_EMBEDDER__KIND = "classic"`). Use one matcher per Qdrant: switching later
needs `python tools/sync_registry.py --reembed`.

**Start the API:**
```bash
export CARDAGENT_API__ADMIN_TOKEN=choose-a-long-secret     # Windows: $env:CARDAGENT_API__ADMIN_TOKEN = "..."
python -m uvicorn cardagent.api.app:app --host 127.0.0.1 --port 8080
curl http://127.0.0.1:8080/health/ready                    # Windows: curl.exe
```

**Add a card and validate** (second terminal):
```bash
H="X-Admin-Token: choose-a-long-secret"
curl -X PUT -H "$H" -H "Content-Type: application/yaml" --data-binary @examples/cards/ae/fazaa.yaml \
     http://127.0.0.1:8080/v1/cards/ae.fazaa                       # edit the template for the real design first
curl -X POST -H "$H" -F images=@gold_ref1.jpg -F images=@gold_ref2.jpg \
     http://127.0.0.1:8080/v1/cards/ae.fazaa/variants/gold/references
curl -F image=@photo.jpg -F request_id=t1 http://127.0.0.1:8080/v1/cards/validate
curl -N -F image=@photo.jpg http://127.0.0.1:8080/v1/cards/validate/stream
```
Repeat the reference upload for every variant (3–20 clean images each). Then measure accuracy with an
evaluation set (`docs/API.md` §6) before trusting automatic ACCEPTs.

**Web console** (card-agent-web): API base URL `http://127.0.0.1:8080/v1`; for a console served from
another origin, add it to `api.cors_origins`. The stream format is in `docs/API.md` §1.

## Tests
```bash
python -m pytest -q
```
Self-contained: an in-process Qdrant, the built-in matcher and test cards drawn in code — no server,
no model, no sample files. Needs Tesseract. 43 tests covering the rules, settings and every API group.

## Project layout
```
config/settings.yaml          all settings; override with CARDAGENT_<SECTION>__<KEY>
examples/cards/               card definition templates (not loaded)
registry/cards, registry/refs optional Git-managed cards -> POST /v1/registry/sync or tools/sync_registry.py
src/cardagent/
  agent.py                    orchestrator; hot-reloads cards and settings from Qdrant
  api/app.py, api/routes/     FastAPI app: validate, cards, registry, runs, settings, evaluations
  store/                      Qdrant: card store, runs, evaluations, runtime settings, registry sync
  evaluation.py               evaluation metrics and gate (API and CLI)
  vision/                     quality gate, card detection/warp, colour, ORB + RANSAC
  embed/                      ClassicEmbedder (no model) | Dinov2OnnxEmbedder (production)
  ocr/                        Tesseract | PaddleOCR (local models only)
  pipeline/                   identify (2-stage), extract (homography-aligned), rules, step trace (stream)
  vlm/                        self-hosted VLM fallback (Ollama / OpenAI-compatible) + circuit breaker
  observability/              metrics, JSON logs, PII masking
tools/                        sync_registry, evaluate (CI gate), export_dinov2_onnx
deploy/                       Dockerfile (CPU), Dockerfile.gpu (T4), docker-compose.yml (Qdrant + sync + agent + VLM)
tests/                        self-contained test suite
```

## Not verified in this build environment
- The DINOv2 ONNX path and the GPU image (no GPU or model weights here); the tests use the built-in matcher.
- PaddleOCR engine and the VLM fallback (code paths present; need local models/endpoint).
- Layouts, colours and number formats in `examples/cards/` are placeholders.
