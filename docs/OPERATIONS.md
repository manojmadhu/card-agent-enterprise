# Operations guide

## 1. Hardware sizing — NVIDIA T4 (Azure NCasT4_v3)

The T4 has 16 GB memory and compute capability 7.5 (Turing). Turing matters for model serving:
**no bfloat16** (run models in float16 or 4-bit) and **no FlashAttention-2** (Ampere+ only).

| Workload on one T4 | GPU memory (approx.) | Notes |
|---|---|---|
| DINOv2 ViT-S/14 ONNX (fp16) | < 1 GB | few ms per card |
| PaddleOCR det + rec (en, ar) | 1–2 GB | optional; biggest latency win vs CPU Tesseract |
| Qwen2.5-VL-7B, 4-bit (Ollama/llama.cpp) | ~6–8 GB incl. image tokens/context | fallback only (a few % of cards); seconds per call |
| **Total** | **~9–11 GB of 16 GB** | fits on one T4 with headroom |

Recommended VM sizes:
* `Standard_NC8as_T4_v3` (8 vCPU, 56 GB RAM, 1×T4) — production start point. CPU cores matter: card
  detection, ORB/RANSAC and Tesseract run on CPU; 2 API workers + Ollama fit comfortably.
* `Standard_NC4as_T4_v3` (4 vCPU) — dev/test, or without the VLM.
* Scale out horizontally (more VMs behind the gateway) rather than up.

Measured on this repo's 2-core build box *without* GPU: p50 ≈ 1.4 s per card, of which OCR ≈ 0.95 s,
identification ≈ 0.2 s (279 references). Re-measure on your T4 with `tools/evaluate.py` (it reports
`timings_ms` per stage) — expect OCR to dominate unless PaddleOCR runs on the GPU.

**VLM serving on T4:** prefer **Ollama / llama.cpp with a 4-bit GGUF**. vLLM supports compute
capability 7.5 but newer vLLM engines and some vision models target Ampere+; if you use vLLM, force
`--dtype float16`, use an AWQ checkpoint, and test your exact version on the T4 first.

## 2. Offline installation
1. Mirror Python packages (Artifactory/Nexus) and container images (Harbor/ACR) internally; pin digests.
2. Build images with `PIP_INDEX_URL` pointing at the mirror:
   `make docker` (CPU) or `make docker-gpu` (T4).
3. Models — fetch once on a build box, verify checksums, store internally:
   * DINOv2: `python tools/export_dinov2_onnx.py --model-dir ./dinov2-small --out models/dinov2/dinov2_vits14.onnx --fp16`
   * PaddleOCR (optional): download det/rec_en/rec_ar inference models → `models/paddleocr/...`; set `CARDAGENT_OCR__ENGINE=paddle`, `CARDAGENT_OCR__USE_GPU=true`.
   * VLM (optional): on a connected box `ollama pull qwen2.5vl:7b`, then copy the `~/.ollama/models` folder into the `ollama` volume on the T4 host. Confirm the license on the exact model card.
4. Inference hosts get **no outbound internet route** (compose `internal: true` network; NSG deny-all egress).

## 3. Deploy (single T4 host)
```bash
export CARDAGENT_ADMIN_TOKEN=$(openssl rand -hex 24)
export QDRANT_API_KEY=$(openssl rand -hex 24)
docker compose -f deploy/docker-compose.yml up -d     # qdrant -> registry-sync (one-shot) -> agent
CARDAGENT_VLM_ENABLED=true docker compose -f deploy/docker-compose.yml --profile vlm up -d
curl -s localhost:8080/health/ready                    # "vector_store": "qdrant", reference count
```
Qdrant runs on the private network only (not published on the host), with an API key. Several
agent replicas/workers can share it: each picks up card changes within
`vector_store.qdrant.refresh_seconds` (default 5 s). For HA run a 3-node Qdrant cluster with
replication factor 2 (see the Qdrant docs) and point `vector_store.qdrant.url` at it.

Switching embedder (e.g. classic → DINOv2) changes the embedding space: the service refuses to start
until you run `python tools/build_index.py --reembed`; then re-run the evaluation and recalibrate
`embedder.similarity_floor`.

**Backups.** Qdrant holds the vectors *and* API-created card definitions: snapshot both collections
(`POST /collections/cardagent_refs/snapshots`, `.../cardagent_families/snapshots` on the Qdrant API) on a
schedule, and back up the `reference_store` volume (original images uploaded through the API — needed
to re-embed after an embedder change). Registry-owned cards can always be rebuilt from Git with
`tools/build_index.py`.

## 4. Onboarding, changing and removing cards
Two ways; both touch **only the affected card** (no full re-index) and go live on all replicas within
`refresh_seconds`.

**A. Git-managed (recommended for programmes with an approval process)**
1. Collect consented photos: ≥ 5–20 clean references per variant + 50–200 realistic test photos per variant
   (lighting, glare, angles, worn cards, app screenshots if accepted).
2. Add/extend `registry/cards/<cc>/<programme>.yaml` (keywords EN+AR, negative keywords for look-alikes,
   variant colours/keywords/prefixes, field ROIs on the reference image, rules).
3. Put references in `registry/refs/<family>/<variant>/`.
4. CI: `python tools/build_index.py --validate-only` → sync a staging Qdrant → `python tools/evaluate.py --manifest <labelled set>`
   on **new and existing** cards (a new programme can steal matches from an old one). Merge only on `GATE: PASS`.
5. Deploy: `docker compose run --rm registry-sync` (or `POST /v1/admin/cards/sync`). Only new/changed/removed
   YAML and images are processed; to **remove** a card, delete its YAML and `registry/refs/<family>/` and sync.
6. Watch for a week: per-family REVIEW rate, `cardagent_reasons_total`, confusion from human reviews.
   Roll back = sync the previous Git revision.

**B. Admin API (header `X-Admin-Token`)**
```bash
H="X-Admin-Token: $CARDAGENT_ADMIN_TOKEN"
# create / replace a definition (same schema as the YAML files; JSON or YAML body)
curl -X PUT -H "$H" -H "Content-Type: application/yaml" --data-binary @new_card.yaml \
     localhost:8080/v1/admin/cards/ae.newcard
# upload reference images for a variant (flat scans or photos; the card is detected and straightened)
curl -X POST -H "$H" -F images=@gold_1.jpg -F images=@gold_2.jpg \
     localhost:8080/v1/admin/cards/ae.newcard/variants/gold/references
curl -H "$H" localhost:8080/v1/admin/cards/ae.newcard                    # definition + reference ids
curl -X DELETE -H "$H" localhost:8080/v1/admin/cards/ae.newcard/references/<ref_id>   # one image
curl -X DELETE -H "$H" localhost:8080/v1/admin/cards/ae.newcard                       # whole card
```
Cards created through the API are owned by the API: registry syncs never modify or delete them. Cards
from `registry/` are owned by Git: the API answers `409` for changes unless `?force=true`
(a forced PUT moves ownership to the API; a forced DELETE lasts until the next sync unless the YAML is
removed too). Every admin action is written to the JSON log (`"event": "card_admin"`).
Run the evaluation gate after API changes too, before relying on automatic ACCEPTs for the new card.

### From the Card Agent Console
The web console manages cards through `/console/v1` (see [CONSOLE_API.md](CONSOLE_API.md)). On Qdrant it uses
the same card store and ownership rule as `/v1/admin/cards`: cards it creates are API-owned; editing a
registry-owned card returns 409 until it is forced (`?force=true`). On FAISS it edits `registry/` files and
keeps replaced or deleted files under `registry/.deleted/` — commit console-made changes to git yourself.

## 5. Calibration and the evaluation gate
`tools/evaluate.py` reports: status agreement, **ACCEPT precision** (share of accepts that were
correct card + correct fields), automation rate, per-field accuracy, family confusion, latency p50/p95.
Gate defaults: ACCEPT precision ≥ 0.99 and status agreement ≥ 0.90 (tune per business risk).
Thresholds to tune on real data (`config/settings.yaml → scoring/retrieval`): `family_min/margin`,
`variant_min/margin`, `inliers_baseline/full_score`, `similarity_floor`, `geometry_required_without_text`.
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
| `/health/ready` 503 | startup failed: logs; usually invalid YAML, missing ONNX file, or Qdrant unreachable |
| Validate returns 503 "vector store unavailable" | Qdrant down/unreachable: check the `qdrant` container, API key, network; the agent recovers automatically |
| Startup error "built with embedder … Run: build_index.py --reembed" | embedder/model changed since the last sync: run `python tools/build_index.py --reembed` |
| Card change not visible on a replica | wait `refresh_seconds`; `GET /v1/registry` shows the revision each replica serves |
| Reload returns 422 | YAML validation error in message; old version still serving |
| Many `AMBIGUOUS_CARD_TYPE` for two programmes | add negative keywords; add more distinctive references; check colours |
| Many `FIELD_UNREADABLE` for one variant | ROI wrong for a new print run — fix ROI on the reference; enable PaddleOCR/VLM |
| VLM timeouts | circuit breaker opens automatically (decisions degrade to REVIEW); check Ollama GPU memory |
| Latency regression | `cardagent_stage_seconds`; reduce `retrieval.max_verify_refs`; add CPU cores/workers |

## 8. Issuer verification (`lookup` rule)
Register a verifier at startup to check membership status against an issuer API or internal table:
```python
from cardagent.pipeline.rules import register_verifier
register_verifier("ae.fazaa", lambda number, fields: (my_membership_db.is_active(number), None))
```
Verifier errors fail the rule (→ REVIEW), never pass it.
