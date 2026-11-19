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
docker compose -f deploy/docker-compose.yml up -d                       # agent only
CARDAGENT_VLM_ENABLED=true docker compose -f deploy/docker-compose.yml --profile vlm up -d
curl -s localhost:8080/health/ready
```
Switching to DINOv2 changes the embedding space: rebuild the index (`tools/build_index.py --force`),
re-run the evaluation, and recalibrate `embedder.similarity_floor` (start ~0.30).

## 4. Onboarding a new card programme / variant
1. Collect consented photos: ≥ 5–20 clean references per variant + 50–200 realistic test photos per variant
   (lighting, glare, angles, worn cards, app screenshots if accepted).
2. Add/extend `registry/cards/<cc>/<programme>.yaml` (keywords EN+AR, negative keywords for look-alikes,
   variant colours/keywords/prefixes, field ROIs on the reference image, rules).
3. Put references in `registry/refs/<family>/<variant>/`.
4. CI: `python tools/build_index.py` (schema + missing-reference check) → `python tools/evaluate.py --manifest <labelled set>`
   on **new and existing** cards (a new programme can steal matches from an old one). Merge only on `GATE: PASS`.
5. Deploy the registry (Git → volume) and `POST /v1/admin/registry/reload`.
6. Watch for a week: per-family REVIEW rate, `cardagent_reasons_total`, confusion from human reviews. Roll back = reload the previous Git revision.

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
| `/health/ready` 503 | startup failed: logs; usually invalid YAML or missing ONNX file |
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
