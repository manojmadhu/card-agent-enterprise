# Run card-agent-enterprise on a Windows PC

Clean setup from a fresh clone, in **PowerShell**. The platform keeps everything in **Qdrant** and
starts empty: you add your card programmes and reference images through the API.

## 1. Install once per PC
| Tool | Get it | Check |
|---|---|---|
| Python 3.12 (64-bit) | python.org → tick **Add python.exe to PATH** | `py -3.12 --version` |
| Git | git-scm.com | `git --version` |
| Tesseract OCR (optional) | only for `ocr.engine: tesseract`; the default OCR (PaddleOCR) is installed by pip. UB-Mannheim installer, tick **Additional language data → Arabic** | `& "C:\Program Files\Tesseract-OCR\tesseract.exe" --list-langs` |
| Docker Desktop | docker.com (runs Qdrant) | `docker --version` |

## 2. Clone
```powershell
cd C:\Users\$env:USERNAME\source\repos
git clone -b feature/qdrant-platform https://github.com/manojmadhu/card-agent-enterprise.git
cd card-agent-enterprise
```

## 3. Python environment
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # prompt now starts with (.venv)
python -m pip install --upgrade pip
pip install -e ".[dev]"                 # OpenCV is pinned <5; includes PaddleOCR (PP-OCRv6) via rapidocr
python tools/download_ocr_models.py     # once: the Arabic OCR model (~8 MB, needs internet)
```
If `Activate.ps1 cannot be loaded`: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate again.
**Every new terminal:** `cd` into the project and run `.\.venv\Scripts\Activate.ps1` first.

## 4. Qdrant
```powershell
docker run -d --name cardagent-qdrant --restart unless-stopped `
  -p 127.0.0.1:6333:6333 -v cardagent_qdrant:/qdrant/storage qdrant/qdrant:v1.18.1
curl.exe http://127.0.0.1:6333/healthz                # "healthz check passed"
```
Keep the URL `127.0.0.1` (the default) — with `localhost`, Windows tries IPv6 first and every call waits about 2 s.

## 5. Matcher model
**DINOv2 (recommended, the default in `config\settings.yaml`).** Copy `models\dinov2\dinov2_vits14.onnx`
(~90 MB) from another install, or create it once:
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers onnx onnxscript huggingface_hub
hf download facebook/dinov2-small --local-dir models\dinov2-small
python tools/export_dinov2_onnx.py --model-dir models\dinov2-small --out models\dinov2\dinov2_vits14.onnx
```
`models\` is git-ignored. Afterwards you can `pip uninstall torch transformers`.

**No model:** use the built-in matcher in every terminal that runs the API:
```powershell
$env:CARDAGENT_EMBEDDER__KIND = "classic"
$env:CARDAGENT_EMBEDDER__SIMILARITY_FLOOR = "0.55"
```
Pick one and keep it: Qdrant remembers which matcher made its vectors and the API refuses to start
with the other one. Switching later: `python tools/sync_registry.py --reembed`.

## 6. Run the API
```powershell
$env:CARDAGENT_API__ADMIN_TOKEN = "choose-a-long-secret"     # needed for every admin endpoint
$env:CARDAGENT_API__CORS_ORIGINS = '["http://localhost:5173"]'   # only for the web console on the Vite dev server
python -m uvicorn cardagent.api.app:app --host 127.0.0.1 --port 8080
```
Wait for `Uvicorn running on http://127.0.0.1:8080` (if the port is taken, use `--port 8081`).
OpenAPI page: http://127.0.0.1:8080/docs

## 7. Add a card and validate
In a **second** terminal (project folder, venv active):
```powershell
$H = "X-Admin-Token: choose-a-long-secret"
# 1. the card definition (edit the template to the real design first: keywords, colours, ROIs, rules)
curl.exe -X PUT -H $H -H "Content-Type: application/yaml" --data-binary "@examples/cards/ae/fazaa.yaml" `
  http://127.0.0.1:8080/v1/cards/ae.fazaa
# 2. reference images, per variant (3-20 clean photos or scans each)
curl.exe -X POST -H $H -F "images=@C:\cards\gold1.jpg" -F "images=@C:\cards\gold2.jpg" `
  http://127.0.0.1:8080/v1/cards/ae.fazaa/variants/gold/references
# 3. validate
curl.exe -F "image=@C:\cards\photo.jpg" -F request_id=t1 http://127.0.0.1:8080/v1/cards/validate
curl.exe -N -F "image=@C:\cards\photo.jpg" http://127.0.0.1:8080/v1/cards/validate/stream
curl.exe -H $H http://127.0.0.1:8080/v1/runs
```
Use `curl.exe`, not `curl` (in PowerShell `curl` is `Invoke-WebRequest`). All endpoints: `docs/API.md`.

## 8. Tests (optional)
```powershell
python -m pytest -q
```
Self-contained (in-process Qdrant, built-in matcher, test cards drawn in code); expect `87 passed` (a few skip when Tesseract or the Arabic model is not installed).
To run them against your Qdrant server: `$env:CARDAGENT_TEST_QDRANT_URL = "http://127.0.0.1:6333"` first
(the tests use their own temporary collections and delete them).

## Troubleshooting
| Message | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'cardagent'` or `'cv2'` | The venv is not active: `.\.venv\Scripts\Activate.ps1`. If it persists: `pip install -e ".[dev]"` |
| `module 'cv2' has no attribute 'HOGDescriptor'` | `pip install "opencv-python-headless>=4.9,<5"` |
| `FileNotFoundError … dinov2_vits14.onnx` | Do step 5, or use the built-in matcher |
| Startup error "built with embedder …" | You changed the matcher: `python tools/sync_registry.py --reembed` (re-embeds every reference) |
| `TesseractNotFoundError` | Only with `ocr.engine: tesseract`: install Tesseract, or set `$env:CARDAGENT_OCR__TESSERACT_CMD` |
| Warning "Arabic OCR model not found" / Arabic text never read | `python tools/download_ocr_models.py` (once, with internet) |
| Many `IMAGE_BLURRY` | see docs/OCR.md (quality gate) |
| `/health/ready` 503 or connection refused `127.0.0.1:6333` | Start Docker Desktop, then `docker start cardagent-qdrant` |
| Every card `UNKNOWN_CARD_TYPE` | No cards or no reference images yet: `curl.exe http://127.0.0.1:8080/v1/cards` |
| Every request takes 2 s or more | The Qdrant URL says `localhost`; use `127.0.0.1` |
| Admin call answers 403 | `CARDAGENT_API__ADMIN_TOKEN` was not set in the terminal that started the API |
| Admin call answers 401 | The `X-Admin-Token` header does not match the token |
| Server exits at once / `address already in use` | The port is taken: use another `--port` |
| Browser console: "blocked by CORS policy" / "No 'Access-Control-Allow-Origin' header" | Set `$env:CARDAGENT_API__CORS_ORIGINS = '["http://localhost:5173"]'` (single quotes outside, double inside) in the terminal that starts the API, then restart it. The origin must match the address bar exactly: `localhost` ≠ `127.0.0.1`, no trailing `/` |
| `make : The term 'make' is not recognized` | The Makefile is for Linux; use the commands in this guide |
