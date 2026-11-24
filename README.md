# Card Validation Agent — Enterprise edition (v1.0.0)

Identifies GCC discount/membership cards from a photo (family → variant), extracts the printed
fields, validates them, and returns a strict JSON decision: `ACCEPT` / `REVIEW` / `REJECT`.
**Fully offline — no public AI APIs.** All models run inside your network.

- Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Deployment, GPU sizing (NVIDIA T4), onboarding cards, runbook: [docs/OPERATIONS.md](docs/OPERATIONS.md)

## Run on a local PC
**Windows: follow [docs/SETUP_WINDOWS.md](docs/SETUP_WINDOWS.md)** — full step-by-step guide with a
troubleshooting table. The short version is below.

**You need:** Python 3.10–3.12 (64-bit), Git, Tesseract OCR with English + Arabic data, and — for the
full setup only — Docker (for Qdrant). There are two ways to run it:

| | Quick check | Full setup (production-like) |
|---|---|---|
| Matcher | built-in `classic` (no model file) | DINOv2 (`models/dinov2/dinov2_vits14.onnx`, ~90 MB) |
| Card store | `faiss` (files, no server) | Qdrant server (Docker) |
| Use for | first run, CI, a PC without Docker | real accuracy, card management API, console writes |

### Windows (PowerShell)
```powershell
git clone -b feature/console-api https://github.com/manojmadhu/card-agent-enterprise.git
cd card-agent-enterprise
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1                      # every new terminal: activate first
pip install -e ".[dev]"
python tools/make_samples.py                      # synthetic test photos -> data/samples (keeps registry/refs)

# quick check (no Docker, no model)
$env:CARDAGENT_VECTOR_STORE__BACKEND = "faiss"
$env:CARDAGENT_EMBEDDER__KIND = "classic"
python tools/build_index.py --force
python tools/evaluate.py --today 2026-11-24

# run the API, then in a second terminal: curl.exe http://127.0.0.1:8081/health/ready
python -m uvicorn cardagent.api.app:app --host 127.0.0.1 --port 8081
```

### Linux / macOS
```bash
git clone -b feature/console-api https://github.com/manojmadhu/card-agent-enterprise.git
cd card-agent-enterprise
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"                           # + OS packages: tesseract-ocr tesseract-ocr-ara
python tools/make_samples.py
export CARDAGENT_VECTOR_STORE__BACKEND=faiss CARDAGENT_EMBEDDER__KIND=classic   # quick check
python tools/build_index.py --force
python tools/evaluate.py --today 2026-11-24
python -m uvicorn cardagent.api.app:app --host 127.0.0.1 --port 8081
```

### Full setup (both systems)
1. **DINOv2 model** — copy `models/dinov2/dinov2_vits14.onnx` from another install, or create it once:
   `pip install torch --index-url https://download.pytorch.org/whl/cpu` ·
   `pip install transformers onnx onnxscript huggingface_hub` ·
   `hf download facebook/dinov2-small --local-dir models/dinov2-small` ·
   `python tools/export_dinov2_onnx.py --model-dir models/dinov2-small --out models/dinov2/dinov2_vits14.onnx`
2. **Qdrant** — `docker run -d --name cardagent-qdrant --restart unless-stopped -p 127.0.0.1:6333:6333 -v cardagent_qdrant:/qdrant/storage qdrant/qdrant:v1.18.1`
3. **Default settings** (clear the quick-check variables) — `python tools/build_index.py` loads the
   registry into Qdrant (a new Qdrant is empty), then `python tools/evaluate.py` and start the API.

### Try it
```bash
curl.exe -F image=@data/samples/fazaa_gold.jpg -F request_id=t1 http://127.0.0.1:8081/v1/cards/validate
curl.exe -N -F image=@data/samples/fazaa_gold.jpg http://127.0.0.1:8081/console/v1/cards/validate/stream
python -m pytest -q        # no Qdrant server needed (in-process Qdrant)
```
(`curl.exe` on Windows; plain `curl` on Linux/macOS.) Set `CARDAGENT_API__ADMIN_TOKEN` before starting
the API to use the admin endpoints. Expected on a quick check: `ACCEPT precision 1.0`; a few demo cards
go to REVIEW and the gate may show FAIL, because the built-in matcher is simpler than DINOv2.

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
store, including a live step-by-step stream (`POST /console/v1/cards/validate/stream`, Server-Sent Events in the
console's StepState format); `/v1` is unchanged. Routes, translation rules and security: [docs/CONSOLE_API.md](docs/CONSOLE_API.md).

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
