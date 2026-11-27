# Architecture — Card Validation Agent v1.0

## 1. Goals and constraints
* Identify any card programme across the GCC (hundreds of families, each with several variants).
* Extract printed fields and validate them; return a strict, versioned JSON decision.
* **No public AI APIs.** Every model is self-hosted; no runtime internet access.
* Safe failure: when unsure, return `REVIEW`, never a confident wrong `ACCEPT`.
* Adding a card = configuration + images, no code change, no redeploy.

## 2. Pipeline
```
image ─► [1] quality gate ─► [2] detect + warp card ─► [3] full-card OCR (0°/180°)
                                                           │
                    ┌──────────────── query features ──────┴───────────────┐
                    │ global embedding   ORB keypoints   body colour (Lab)  │
                    └───────────────────────────────────────────────────────┘
 [4] IDENTIFY — stage 1 (recall):   vector top-k over ALL references  ∪  keyword inverted index
               stage 2 (precision): filtered vector query = similarity to every ref of the shortlisted
                                    families; ORB + RANSAC on ≤24 of them (2 per variant)
               family score = w·(text, embed, geometry, colour); needs score ≥ min AND margin ≥ δ
               variant score (inside family) = colour, variant keyword, geometry, embed, number prefix
 [5] ALIGN to best reference with its homography  →  YAML regions land on exact pixels
 [6] EXTRACT fields: region OCR → regex  |  full-text regex  |  self-hosted VLM (re-validated)
 [7] RULES (card definition): regex, not_expired, date_order, emirates_id_format, luhn, length,
     prefix_matches_variant, lookup
 [8] DECIDE: ACCEPT / REVIEW / REJECT + reason codes → JSON (schema 2.1) + run record (Qdrant) + audit log + metrics
     REVIEW → human review queue → accept/reject (+ enroll photo as reference) → evaluation sets
```

### Two-sided cards
A request carries one card as 1–2 photos (front/back, or unlabelled). Stages 1–4a run per photo;
identification pools the OCR text of both photos and takes the best visual evidence of either; each
photo is assigned the side it matches (references are stored per side; labels are trusted unless the
geometry clearly says otherwise); stage 5–6 align each photo to a reference of its side and read only
that side's fields. A required side without a usable photo → `REJECT SIDE_REQUIRED:<side>`; photos of
two different cards → `REVIEW SIDES_MISMATCH`. Full specification: [TWO_SIDED_CARDS.md](TWO_SIDED_CARDS.md).

### Design choices that matter
| Decision | Why |
|---|---|
| Two-stage retrieval | Comparing a photo to 10k+ references with keypoints is too slow; vector search shortlists in ms, RANSAC verifies a bounded set (cost independent of registry size). |
| Score **and** margin | Prevents confident errors between look-alike programmes; ambiguity → REVIEW. |
| Reference keypoints exclude personal-data regions | Otherwise names/numbers in the same font match across programmes. Measured: wrong cards with a shared template reach ~45–85 inliers vs 100–200 for the right one → `inliers_baseline`. |
| OCR-confusable keyword skeleton (O↔0, I/L↔1, S↔5, B↔8) | Real OCR misreads; "DEMOQAO1" must still match "DEMOQA01". |
| Exact keyword hit suppresses fuzzy credit | A word that exactly names one programme is not evidence for its look-alikes. |
| Missing text ≠ negative text | If no programme keyword is readable (glare), weights renormalise to visual signals but ACCEPT requires strong geometry (`geometry_required_without_text`). |
| Homography alignment before field OCR | Field regions are defined on the reference; alignment makes them exact under tilt/crop. |
| VLM only as fallback, output re-validated | VLMs can hallucinate plausible numbers; regex + rules have the final word; VLM-sourced fields force REVIEW. |
| `lookup` rule hook | Reading a card ≠ proving it is active. Register an issuer verifier when available. |

## 3. Qdrant: the single store
Everything the platform keeps lives in Qdrant (`src/cardagent/store/`), so any number of API replicas
can share one Qdrant and stay consistent. Collections (prefix `qdrant.collection_prefix`, default `cardagent`):

| Collection | Points |
|---|---|
| `_refs` | one per reference image: cosine vector + payload `family`, `variant`, `side` (front/back), `country`, `sha256`, `source`, ORB keypoints |
| `_families` | card definitions (validated JSON), a `meta` point (revision, embedder, vector size) and the runtime settings overrides |
| `_runs` | every validation (payload only): decision summary, full response, review; photo on disk (`runs.store_dir`) |
| `_evaluations` | evaluation sets and evaluations; zips and reports on disk (`evaluation.store_dir`) |

* **Point ids are content addressed**: `uuid5(family/variant/sha256(image))` (back side: `family/variant/back/sha256`). Re-adding an image is a no-op; an image never gets two vectors.
* **Changes touch only the affected card**: add/replace/delete a family or a reference image in seconds, no re-index. Each write bumps the store revision; every replica checks it every `qdrant.refresh_seconds` and reloads (cards + settings) when it changed.
* **Ownership**: every family has `source = api | registry`. Cards are normally managed through the API. Optionally, Git-managed files in `registry/` are imported incrementally (`POST /v1/registry/sync`, `tools/sync_registry.py`); a sync never modifies or prunes API-owned cards, and the API refuses (409) to change registry-owned cards unless `?force=true`.
* **ROI changes** re-extract keypoints for that family only (personal-data regions are masked out of reference keypoints); vectors are untouched. **Embedder changes** require a re-embed (`tools/sync_registry.py --reembed`): the service refuses to start on a mismatch. Uploaded reference images are kept in `qdrant.reference_store_dir` for that purpose.
* Definitions are validated by `registry/models.py` (pydantic): id format, ROI bounds, regex compiles, rules reference real fields, unique variants.
* **Hot reload**: a new snapshot is built off to the side → single reference swap; in-flight requests finish on the state they started with; failures keep the old state (metric `cardagent_registry_reloads_total{result="failed"}`).
* **Runtime settings**: decision thresholds can be overridden through `PATCH /v1/settings` (stored in Qdrant, applied by every replica). Effective settings = file + environment + overrides. `process(settings=...)` runs a single request with other settings — used by what-if evaluations without touching live traffic.

## 3a. Human review loop
`REVIEW` decisions enter a queue (`GET /v1/review`). A reviewer accepts (optionally correcting the card
and fields) or rejects. Accepting with `enroll=true` adds the photo as a new reference image, so the
design is recognised next time. Reviewed runs can be turned into an evaluation set
(`POST /v1/evaluation-sets/from-runs`), closing the loop: production feedback → ground truth → gate
for the next threshold or card change.

## 4. Models (all self-hosted)
| Role | Default (dev/CI) | Production | License |
|---|---|---|---|
| Global embedding | Classic (Lab layout + HOG), no weights | DINOv2 ViT-S/14 via ONNX Runtime (CUDA) | Apache-2.0 |
| Vector store | Qdrant 1.18 (in-process `:memory:` for tests) | Qdrant 1.18 server (self-hosted; HNSW, filtered search, snapshots, replication) | Apache-2.0 |
| Local features | OpenCV ORB + RANSAC | same | Apache-2.0 |
| OCR | PaddleOCR PP-OCRv6 det + rec (+ PP-OCRv5 Arabic) on ONNX Runtime, CPU | same on GPU (CUDA); Tesseract 5 as an alternative ([OCR.md](OCR.md)) | Apache-2.0 |
| VLM fallback | off | Qwen2.5-VL-7B (4-bit) via Ollama/llama.cpp | Apache-2.0 (7B; verify model card) |

## 5. Output contract (excerpt)
```json
{
  "schema_version": "2.1", "request_id": "abc", "run_id": "5c0f…", "status": "ACCEPT", "reasons": [],
  "card": {"family": "ae.fazaa", "family_name": "Fazaa", "country": "AE", "issuer": "Fazaa",
           "family_confidence": 0.93, "variant": "gold", "variant_name": "Fazaa Gold", "variant_confidence": 0.88},
  "fields": {"card_number": {"value": "2210457783", "confidence": 0.95, "source": "ocr_roi", "valid_format": true},
             "expiry_date": {"value": "11/27", "valid_until": "2027-11-30", "confidence": 0.95, "source": "ocr_roi", "valid_format": true}},
  "validation": [{"rule": "not_expired", "field": "expiry_date", "critical": true, "result": "pass", "detail": "valid_until=2027-11-30"}],
  "quality": {"passed": true, "card_detected": true, "rotation": 0, "aligned_to_reference": true, "...": "..."},
  "photos": [{"photo": 0, "upload": "image", "side": "front", "status": "used", "fields": ["card_number", "expiry_date"], "...": "..."}],
  "missing_sides": [],
  "candidates": {"family": [{"id": "ae.fazaa", "score": 0.93, "signals": {"text": 1.0, "embed": 0.7, "geometry": 1.0, "colour": 0.9}}]},
  "model_versions": {"pipeline": "1.0.0", "registry": "…", "embedder": "…", "ocr": "…"},
  "timings_ms": {"quality": 5, "normalise": 18, "ocr_full": 397, "features": 15, "identify": 208, "extract": 558},
  "processing_ms": 1400
}
```

## 6. Security & privacy
* No outbound network from inference containers (`internal: true` network); models loaded from local paths; `HF_HUB_OFFLINE=1`.
* PII masked in logs (`card_number`, `emirates_id`, `holder_name`, …); full values only in the API response.
* Admin endpoints require a token (constant-time compare); disabled when unset.
* Non-root container, read-only root filesystem, upload type/size limits.
* Store images encrypted with a retention policy aligned to UAE PDPL / applicable GCC data-protection law.
