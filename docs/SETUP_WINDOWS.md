# Run card-agent-enterprise on a new Windows PC

Clean setup from a fresh clone, in **PowerShell**. Tested end to end on a fresh clone of
`feature/console-api` in a new virtual environment (24 Nov 2026).

## Why a plain clone fails
| Problem | Cause | Fixed in step |
|---|---|---|
| Tesseract not found, sample generator crashes on fonts, DINOv2 export fails | `git clone` gives **`main`**, which doesn't have the Windows fixes (they're on the feature branches) | 2 |
| `AttributeError: module 'cv2' has no attribute 'HOGDescriptor'` | pip installed **OpenCV 5.x**, which removed it — now pinned to `<5` in this branch | 3 |
| `FileNotFoundError … models/dinov2/dinov2_vits14.onnx` | the settings use DINOv2 by default, but `models/` is not in Git | 5 |
| DINOv2 export fails on a `dinov2-small/` folder from Git | older commits hold 133-byte **Git LFS placeholders**, not the 88 MB weights — removed in this branch | 5 |
| Connection refused on port 6333 / every card UNKNOWN | the default storage is a **Qdrant server**, and a new PC has none (and nothing synced into it) | 6 |
| `data/samples/...` missing | sample photos are generated, not stored in Git | 4 |
| `git status` shows 279 changed reference PNGs | older `make_samples.py` rewrote them (fonts differ per OS) — now kept by default | 4 |

---

## 1. Install once per PC
| Tool | Get it | Check |
|---|---|---|
| Python 3.12 (64-bit) | python.org → tick **Add python.exe to PATH** | `py -3.12 --version` |
| Git | git-scm.com | `git --version` |
| Tesseract OCR | UB-Mannheim installer (github.com/UB-Mannheim/tesseract/wiki). During setup tick **Additional language data → Arabic**. Keep the default folder `C:\Program Files\Tesseract-OCR` (the agent finds it there; no PATH change needed) | `& "C:\Program Files\Tesseract-OCR\tesseract.exe" --list-langs` shows `ara` and `eng` |
| Docker Desktop | docker.com (only for the full setup, step 6) | `docker --version` |

## 2. Clone the right branch
```powershell
cd C:\Users\$env:USERNAME\source\repos
git clone -b feature/console-api https://github.com/manojmadhu/card-agent-enterprise.git
cd card-agent-enterprise
```
`feature/console-api` = `main` + Windows fixes + Qdrant card store + console API (newest).

## 3. Python environment
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # prompt now starts with (.venv)
python -m pip install --upgrade pip
pip install -e ".[dev]"                 # installs OpenCV 4.x (pinned <5: OpenCV 5 breaks the built-in matcher)
```
If `Activate.ps1 cannot be loaded`: run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate again.
**Every new terminal:** `cd` into the project and run `.\.venv\Scripts\Activate.ps1` first.

## 4. Sample photos
```powershell
python tools/make_samples.py
```
Writes 25 labelled test photos to `data/samples` and 93 to `data/samples_scale`. The committed
reference images in `registry/refs` are kept, so every PC uses the same references
(`--regenerate-refs` rewrites them — only for changing the synthetic designs).

---

Choose **A** (quick check, 10 minutes) or **B** (the real setup, as on your first PC).

## 5A. Quick check — no Docker, no model
```powershell
$env:CARDAGENT_VECTOR_STORE__BACKEND = "faiss"
$env:CARDAGENT_EMBEDDER__KIND = "classic"
python tools/build_index.py --force
python tools/evaluate.py --today 2026-11-24
```
Expected: the whole pipeline runs, `ACCEPT precision 1.0`, field accuracy 100 %. The built-in
matcher is simpler than DINOv2, so a few demo cards go to REVIEW and the gate can show `FAIL` —
that is expected here. These two `$env:` lines apply to the current terminal only; set them again
in every terminal that should use quick-check mode. Then go to step 7.

## 5B. DINOv2 model (full setup)
**Easiest:** copy `models\dinov2\dinov2_vits14.onnx` (~90 MB) from your first PC into the same
folder here, and skip the rest of this step.

Otherwise download and convert it once:
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers onnx onnxscript huggingface_hub
hf download facebook/dinov2-small --local-dir models\dinov2-small
python tools/export_dinov2_onnx.py --model-dir models\dinov2-small --out models\dinov2\dinov2_vits14.onnx
```
`models\` is git-ignored, so the download and the model stay out of Git. Check:
`models\dinov2\dinov2_vits14.onnx` is about 90 MB.
After exporting you can `pip uninstall torch transformers` — the service only needs onnxruntime.

## 6. Qdrant server (full setup)
```powershell
docker run -d --name cardagent-qdrant --restart unless-stopped `
  -p 127.0.0.1:6333:6333 -v cardagent_qdrant:/qdrant/storage qdrant/qdrant:v1.18.1
curl.exe http://127.0.0.1:6333/healthz                # "healthz check passed"

Remove-Item Env:CARDAGENT_VECTOR_STORE__BACKEND, Env:CARDAGENT_EMBEDDER__KIND -ErrorAction SilentlyContinue
python tools/build_index.py                           # validates the registry and syncs it into Qdrant
python tools/evaluate.py --today 2026-11-24
```
A new Qdrant is empty: `build_index.py` loads the 32 card families and 279 references into it.
Run it again after changing `registry/`. Keep the URL `127.0.0.1` — with `localhost`, Windows tries
IPv6 first and every call waits about 2 s.

## 7. Run the API
```powershell
$env:CARDAGENT_API__ADMIN_TOKEN = "choose-a-long-secret"     # needed for admin/write endpoints
python -m uvicorn cardagent.api.app:app --host 127.0.0.1 --port 8081
```
Wait for `Uvicorn running on http://127.0.0.1:8081`. In a **second** terminal (in the project folder):
```powershell
curl.exe http://127.0.0.1:8081/health/ready
curl.exe -F image=@data/samples/fazaa_gold.jpg -F request_id=t1 http://127.0.0.1:8081/v1/cards/validate
curl.exe -N -F image=@data/samples/fazaa_gold.jpg http://127.0.0.1:8081/console/v1/cards/validate/stream
```
Use `curl.exe`, not `curl` (in PowerShell `curl` is `Invoke-WebRequest`). Port 8081 is used because
8080 is often taken by another container; any free port works. Stop the server with `Ctrl+C`.
Web console: set its API base URL to `http://127.0.0.1:8081/console/v1` (see `docs/CONSOLE_API.md`).

## 8. Tests (optional)
```powershell
python -m pytest -q
```
No Qdrant server needed (the tests use an in-process Qdrant). With the quick-check settings from 5A,
expect 3–4 failures in `test_accept_and_extract`, all on `aligned_to_reference`. They are a known
issue on the branch itself, not a setup problem.

---

## Troubleshooting
| Message | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'cardagent'` or `'cv2'` | The venv is not active, or plain `python` is your global Python: `.\.venv\Scripts\Activate.ps1`. If it persists: `pip install -e ".[dev]"` |
| `module 'cv2' has no attribute 'HOGDescriptor'` | `pip install "opencv-python-headless>=4.9,<5"` |
| `FileNotFoundError … dinov2_vits14.onnx` | Do step 5B, or use quick-check mode (5A) |
| `TesseractNotFoundError` | Install Tesseract (step 1). Installed elsewhere: `$env:CARDAGENT_OCR__TESSERACT_CMD = "D:\Tools\Tesseract-OCR\tesseract.exe"` |
| Arabic text never read | Re-run the Tesseract installer and tick Arabic language data |
| Connection refused `127.0.0.1:6333` | Start Docker Desktop, then `docker start cardagent-qdrant` |
| Every card `UNKNOWN_CARD_TYPE` on Qdrant | Qdrant is empty: `python tools/build_index.py` |
| Every request takes 2 s or more | The Qdrant URL says `localhost`; use `127.0.0.1` |
| Export error mentions `safetensors` / invalid header | You pointed at the repo's LFS placeholders; download into `models\dinov2-small` (5B) |
| Server exits at once / `address already in use` | The port is taken: use another `--port` |
| `curl: … -F` not recognised | Use `curl.exe` |
| `make : The term 'make' is not recognized` | The Makefile is for Linux; use the commands in this guide |
| `git status` shows 279 changed PNGs | You ran `make_samples.py --regenerate-refs` (or an old version): `git restore registry ; git clean -fd registry` |
