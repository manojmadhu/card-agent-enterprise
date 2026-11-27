# Operations guide

## 1. Hardware sizing — NVIDIA T4 (Azure NCasT4_v3)

The T4 has 16 GB memory and compute capability 7.5 (Turing). Turing matters for model serving:
**no bfloat16** (run models in float16 or 4-bit) and **no FlashAttention-2** (Ampere+ only).

| Workload on one T4 | GPU memory (approx.) | Notes |
|---|---|---|
| DINOv2 ViT-S/14 ONNX (fp16) | < 1 GB | few ms per card |
| PaddleOCR PP-OCRv6 small det + rec, PP-OCRv5 Arabic (ONNX Runtime CUDA) | small (model files ~10–21 MB each, plus the CUDA runtime); not measured on a T4 yet | the default OCR; the biggest latency win on GPU |
| Qwen2.5-VL-7B, 4-bit (Ollama/llama.cpp) | ~6–8 GB incl. image tokens/context | fallback only (a few % of cards); seconds per call |
| **Total** | **well under 16 GB** | fits on one T4 with headroom; measure with `nvidia-smi` under load |

Recommended VM sizes:
* `Standard_NC8as_T4_v3` (8 vCPU, 56 GB RAM, 1×T4) — production start point. CPU cores matter: card
  detection and ORB/RANSAC run on CPU; 2 API workers + Ollama fit comfortably.
* `Standard_NC4as_T4_v3` (4 vCPU) — dev/test, or without the VLM.
* Scale out horizontally (more VMs behind the gateway) rather than up.

Measured on this repo's 2-core build box *without* GPU, PaddleOCR engine: p50 ≈ 1.3 s, p95 ≈ 3.4 s
per request (synthetic cards, including two-photo requests); a full-card OCR pass is ~0.9 s, a field
crop ~20 ms. Re-measure on your T4 with `tools/evaluate.py` (it reports `timings_ms` per stage) —
expect OCR to dominate unless it runs on the GPU.

**VLM serving on T4:** prefer **Ollama / llama.cpp with a 4-bit GGUF**. vLLM supports compute
capability 7.5 but newer vLLM engines and some vision models target Ampere+; if you use vLLM, force
`--dtype float16`, use an AWQ checkpoint, and test your exact version on the T4 first.

## 2. Offline installation
1. Mirror Python packages (Artifactory/Nexus) and container images (Harbor/ACR) internally; pin digests.
2. Build images with `PIP_INDEX_URL` pointing at the mirror:
   `make docker` (CPU) or `make docker-gpu` (T4).
3. Models — fetch once on a build box, verify checksums, store internally:
   * DINOv2: `python tools/export_dinov2_onnx.py --model-dir ./dinov2-small --out models/dinov2/dinov2_vits14.onnx --fp16`
   * OCR: the PP-OCRv6 models ship in the `rapidocr` package; the Arabic model: `python tools/download_ocr_models.py` (add `--medium` for the larger PP-OCRv6 models on GPU) → `models/paddleocr/`. The GPU image sets `CARDAGENT_OCR__USE_GPU=true`. See [OCR.md](OCR.md).
   * VLM (optional): on a connected box `ollama pull qwen2.5vl:7b`, then copy the `~/.ollama/models` folder into the `ollama` volume on the T4 host. Confirm the license on the exact model card.
4. Inference hosts get **no outbound internet route** (compose `internal: true` network; NSG deny-all egress).

## 3. Deploy (single T4 host)
```bash
export CARDAGENT_ADMIN_TOKEN=$(openssl rand -hex 24)
export QDRANT_API_KEY=$(openssl rand -hex 24)
docker compose -f deploy/docker-compose.yml up -d     # qdrant -> registry-sync (one-shot) -> agent
CARDAGENT_VLM_ENABLED=true docker compose -f deploy/docker-compose.yml --profile vlm up -d
curl -s localhost:8080/health/ready                    # registry version, families, reference count
```
Qdrant runs on the private network only (not published on the host), with an API key. Several
agent replicas/workers can share it: each picks up card changes within
`qdrant.refresh_seconds` (default 5 s). For HA run a 3-node Qdrant cluster with
replication factor 2 (see the Qdrant docs) and point `qdrant.url` at it. With several replicas, mount
`run_store`, `evaluations` and `reference_store` on shared storage (photos and reports live there).

Switching embedder (e.g. classic → DINOv2) changes the embedding space: the service refuses to start
until you run `python tools/sync_registry.py --reembed` (or `POST /v1/registry/sync?reembed=true`
from a replica already running the new embedder); then re-run the evaluation and recalibrate
`embedder.similarity_floor`.

**Backups.** Qdrant holds the cards, vectors, runs, review decisions, settings overrides and
evaluations: snapshot all four collections (`POST /collections/cardagent_refs/snapshots`, and the same
for `cardagent_families`, `cardagent_runs`, `cardagent_evaluations` on the Qdrant API) on a schedule.
Back up the `reference_store` volume (reference images uploaded through the API — needed to re-embed
after an embedder change) and, if you keep them, `run_store` and `evaluations`. Registry-owned cards
can always be rebuilt from Git with `tools/sync_registry.py`.

**Data retention.** Runs and their photos are deleted after `runs.retention_days` (default 30, checked
hourly; `POST /v1/runs/prune` on demand). Set `runs.store_images=false` to keep no photos at all (then
review enrollment and evaluation sets from runs are not available).

## 4. Onboarding, changing and removing cards
The platform starts empty. Two ways to manage cards; both touch **only the affected card** (no full
re-index) and go live on all replicas within `refresh_seconds`. Endpoint details: [API.md](API.md).

Before either: collect consented photos — 3–20 clean references per variant, plus 50–200 realistic
test photos per variant (lighting, glare, angles, worn cards, app screenshots if accepted) for the
evaluation set. Write the definition (keywords EN+AR, negative keywords for look-alikes, variant
colours/keywords/prefixes, field ROIs on the reference image, rules); `examples/cards/` has templates.

**Data on both sides?** List `sides: [{id: front}, {id: back}]`, give every field its `side`, measure
each ROI on a reference of that side, and upload references per variant **and side**
(`…/references?side=back`; Git: `registry/refs/<family>/<variant>/back/`). Collect test photos of both
sides, and evaluation cases with `"files": {"front": …, "back": …}`. Check
`GET /v1/cards/{id}` → `missing_references` is empty. See [TWO_SIDED_CARDS.md](TWO_SIDED_CARDS.md).

**A. Through the API (default)**
```bash
H="X-Admin-Token: $CARDAGENT_ADMIN_TOKEN"
curl -X PUT -H "$H" -H "Content-Type: application/yaml" --data-binary @new_card.yaml \
     localhost:8080/v1/cards/ae.newcard                                     # create / replace
curl -X POST -H "$H" -F images=@gold_1.jpg -F images=@gold_2.jpg \
     localhost:8080/v1/cards/ae.newcard/variants/gold/references            # per variant
curl localhost:8080/v1/cards/ae.newcard                                     # variants_without_references must be []
curl -H "$H" localhost:8080/v1/cards/ae.newcard/references                  # reference ids
curl -X DELETE -H "$H" localhost:8080/v1/cards/ae.newcard/references/<ref_id>   # one image
curl -X DELETE -H "$H" localhost:8080/v1/cards/ae.newcard                       # whole card
```
Then run an evaluation with test photos of the new card **and** existing cards (a new programme can
steal matches from an old one) before relying on automatic ACCEPTs. Every admin action is written to
the JSON log (`"event": "admin"`).

**B. Git-managed (for programmes with an approval process)**
1. Add `registry/cards/<cc>/<programme>.yaml` and the references in `registry/refs/<family>/<variant>/`.
2. CI: `python tools/sync_registry.py --validate-only` → sync a staging Qdrant →
   `python tools/evaluate.py --manifest <labelled set>/manifest.json`. Merge only when the gate passes.
3. Deploy: `docker compose run --rm registry-sync` (or `POST /v1/registry/sync`). Only new/changed/removed
   files are processed; to **remove** a card, delete its YAML and `registry/refs/<family>/` and sync.
   Roll back = sync the previous Git revision.

Registry-owned cards answer `409` to API changes unless `?force=true` (a forced PUT moves ownership to
the API; a forced DELETE lasts until the next sync unless the files are removed too). API-owned cards
are never modified by a sync.

**After go-live:** watch the per-family REVIEW rate (`GET /v1/stats?family=…`), work the review queue,
enroll good photos of new print runs (`enroll: true`), and rebuild an evaluation set from reviewed runs
every few weeks.

## 5. Calibration and the evaluation gate
Evaluations run through the API (`/v1/evaluation-sets`, `/v1/evaluations`, see API.md §6) or in CI with
`tools/evaluate.py`. The report has: status agreement, **ACCEPT precision** (share of accepts that were
correct card + correct fields), automation rate, per-field accuracy, family confusion, latency p50/p95.
Gate defaults: ACCEPT precision ≥ 0.99 and status agreement ≥ 0.90 (tune per business risk).
Thresholds to tune on real data: `family_min/margin`, `variant_min/margin`, `inliers_baseline/full_score`,
`similarity_floor`, `geometry_required_without_text`, `field_min_confidence`. Workflow: run a what-if
evaluation (`POST /v1/evaluations` with `"settings": {...}`), compare with the baseline, then apply with
`PATCH /v1/settings` (no restart) or put the values in `config/settings.yaml`.
Recommended rollout: shadow mode → assisted (staff confirm) → automatic ACCEPT.

## 6. Monitoring & alerts (Prometheus)
| Metric | Alert idea |
|---|---|
| `cardagent_requests_total{status}` | REVIEW share > baseline + 10 pts for 30 min |
| `cardagent_reasons_total{reason}` | spike in `RETAKE_REQUIRED` (capture app issue) or `UNKNOWN_CARD_TYPE` (new design in the wild) |
| `cardagent_request_seconds` | p95 > 3 s |
| `cardagent_stage_seconds{stage}` | find the slow stage |
| `cardagent_vlm_fallback_total` | sudden rise = OCR degradation |
| `cardagent_registry_reloads_total{result="failed"}` | any |

## 7. Runbook
| Symptom | Check / action |
|---|---|
| `/health/ready` 503 | startup failed: logs; usually a missing ONNX file or Qdrant unreachable |
| Any call returns 503 "Qdrant unavailable" | Qdrant down/unreachable: check the `qdrant` container, API key, network; the agent recovers automatically |
| Startup error "built with embedder … Run: python tools/sync_registry.py --reembed" | embedder/model changed: run that command once |
| Card change not visible on a replica | wait `refresh_seconds`; `GET /v1/registry` shows the revision each replica serves |
| Sync returns 422 | YAML validation error in the message; nothing was changed |
| Settings change not applied | `GET /v1/settings` on that replica (`overrides`); replicas apply them within `refresh_seconds` |
| Many `SIDE_REQUIRED:back` for one programme | the capture app does not ask for the back: prompt for it when the API answers `SIDE_REQUIRED`, or always capture both sides for that programme |
| `SIDES_MISMATCH` on genuine cards | the back references are missing or show another design: check `reference_images_by_side`; raise `scoring.sides_mismatch_geometry` only after a what-if evaluation |
| Review queue grows | `GET /v1/stats` → `top_reasons`; tune thresholds with a what-if evaluation, or add references for the family |
| Many `AMBIGUOUS_CARD_TYPE` for two programmes | add negative keywords; add more distinctive references; check colours |
| Many `FIELD_UNREADABLE` for one variant | ROI wrong for a new print run — fix ROI on the reference; check `ocr.engine` is `paddle`; enable the VLM |
| Many `RETAKE_REQUIRED` + `IMAGE_BLURRY` | compare `quality.sharpness` with `quality.min_sharpness` in the results; see [OCR.md](OCR.md) |
| VLM timeouts | circuit breaker opens automatically (decisions degrade to REVIEW); check Ollama GPU memory |
| Latency regression | `cardagent_stage_seconds`; reduce `retrieval.max_verify_refs`; add CPU cores/workers |

## 8. Issuer verification (`lookup` rule)
Register a verifier at startup to check membership status against an issuer API or internal table:
```python
from cardagent.pipeline.rules import register_verifier
register_verifier("ae.fazaa", lambda number, fields: (my_membership_db.is_active(number), None))
```
Verifier errors fail the rule (→ REVIEW), never pass it.
