# API reference (v1)

One API for every operation. Interactive OpenAPI docs: `http://<host>:8080/docs`.

* **Open** endpoints need no credentials. **Admin** endpoints need the header `X-Admin-Token: <CARDAGENT_API__ADMIN_TOKEN>`.
  Without a configured token every admin endpoint answers `403`; with a wrong token, `401`.
* Errors are `{"detail": "..."}`. `422` = invalid input (nothing changed), `409` = ownership conflict,
  `503` = Qdrant unreachable (retry; header `Retry-After`).
* Images: `image/jpeg`, `image/png` or `image/webp`, at most `api.max_image_bytes` (8 MB). Otherwise `415` / `413`.
* **CORS.** A browser app on another origin (e.g. the web console on the Vite dev server) must be listed in
  `api.cors_origins`, e.g. `CARDAGENT_API__CORS_ORIGINS='["http://localhost:5173"]'`. Allowed: methods GET, POST,
  PUT, PATCH, DELETE; request headers `Content-Type`, `X-Admin-Token`, `Accept`, `Cache-Control`, `Last-Event-ID`;
  the browser may read `Retry-After`. No cookies/credentials. `localhost` and `127.0.0.1` are different origins.
  Not needed when the console is served through the same-origin reverse proxy.

| Group | Method | Path | Access |
|---|---|---|---|
| Validation | POST | `/v1/cards/validate` | open |
| | POST | `/v1/cards/validate/stream` | open |
| | GET | `/v1/pipeline/steps` | open |
| Cards | GET | `/v1/cards` | open |
| | GET | `/v1/cards/{family_id}` | open |
| | PUT | `/v1/cards/{family_id}` | admin |
| | DELETE | `/v1/cards/{family_id}` | admin |
| Reference images | GET | `/v1/cards/{family_id}/references` | admin |
| | POST | `/v1/cards/{family_id}/variants/{variant_id}/references` | admin |
| | GET | `/v1/cards/{family_id}/references/{ref_id}/image` | admin |
| | DELETE | `/v1/cards/{family_id}/references/{ref_id}` | admin |
| Registry | GET | `/v1/registry` | open |
| | POST | `/v1/registry/sync` | admin |
| | POST | `/v1/registry/reload` | admin |
| Runs & review | GET | `/v1/runs` | admin |
| | GET | `/v1/runs/{run_id}` | admin |
| | GET | `/v1/runs/{run_id}/image` | admin |
| | DELETE | `/v1/runs/{run_id}` | admin |
| | POST | `/v1/runs/{run_id}/review` | admin |
| | GET | `/v1/review` | admin |
| | POST | `/v1/runs/prune` | admin |
| Statistics | GET | `/v1/stats` | admin |
| Settings | GET | `/v1/settings` | open |
| | PATCH | `/v1/settings` | admin |
| | DELETE | `/v1/settings` | admin |
| Evaluation | POST | `/v1/evaluation-sets` | admin |
| | POST | `/v1/evaluation-sets/from-runs` | admin |
| | GET | `/v1/evaluation-sets` | admin |
| | GET / DELETE | `/v1/evaluation-sets/{set_id}` | admin |
| | POST | `/v1/evaluations` | admin |
| | GET | `/v1/evaluations` | admin |
| | GET / DELETE | `/v1/evaluations/{evaluation_id}` | admin |
| Health | GET | `/health/live`, `/health/ready`, `/metrics` | open |

---

## 1. Validation

### POST /v1/cards/validate
Multipart form, **one card per request, 1 or 2 photos** (one per side):

| Field | Meaning |
|---|---|
| `image` | one photo; the side it shows is detected (the original field, unchanged) |
| `front`, `back` | a photo of that side; the label is trusted unless the photo clearly shows the other side |
| `images` | 1–2 photos (repeat the field); the sides are detected |
| `request_id` | text, optional |

Query: `store` (default `true`). No photo, or more than 2, answers `422`. Cards printed on both sides
are explained in [Two-sided cards](TWO_SIDED_CARDS.md).

* A retry with the same `request_id` **and** the same photos returns the first answer (idempotency
  cache per process, `api.idempotency_cache_size`). Different photos are always processed.
* Each decision is recorded as a run (history, review queue, statistics) unless `runs.enabled=false` or
  `?store=false`. The response then carries `run_id`.

Response (`src/cardagent/schemas.py`; `python -m cardagent.schemas` prints the JSON Schema):
```json
{
  "schema_version": "2.1", "request_id": "t1", "run_id": "5c0f…", "status": "ACCEPT", "reasons": [],
  "card": {"family": "ae.fazaa", "family_name": "Fazaa", "country": "AE", "issuer": "Fazaa",
           "family_confidence": 0.93, "variant": "gold", "variant_name": "Fazaa Gold", "variant_confidence": 0.88},
  "fields": {"card_number": {"value": "2210457783", "confidence": 0.95, "source": "ocr_roi", "valid_format": true, "side": "back"},
             "expiry_date": {"value": "11/27", "valid_until": "2027-11-30", "confidence": 0.95, "source": "ocr_roi", "valid_format": true, "side": "front"}},
  "validation": [{"rule": "not_expired", "field": "expiry_date", "critical": true, "result": "pass", "detail": "valid_until=2027-11-30"}],
  "quality": {"passed": true, "card_detected": true, "rotation": 0, "aligned_to_reference": true},
  "photos": [{"photo": 0, "upload": "front", "declared": "front", "side": "front", "status": "used", "side_corrected": false,
              "quality": {"…": "…"}, "reference_id": "…", "inliers": 150, "fields": ["holder_name", "expiry_date"]},
             {"photo": 1, "upload": "back", "declared": "back", "side": "back", "status": "used", "side_corrected": false,
              "quality": {"…": "…"}, "reference_id": "…", "inliers": 185, "fields": ["card_number"]}],
  "missing_sides": [],
  "candidates": {"family": [{"id": "ae.fazaa", "score": 0.93, "signals": {"text": 1.0, "embed": 0.7, "geometry": 1.0, "colour": 0.9}}]},
  "model_versions": {"pipeline": "1.0.0", "registry": "…", "embedder": "…", "ocr": "…"},
  "timings_ms": {"quality": 5, "normalise": 18, "ocr_full": 397, "features": 15, "identify": 208, "extract": 558},
  "processing_ms": 1400
}
```
`status`: `ACCEPT` (automatic), `REVIEW` (a person must decide; the run enters the review queue),
`REJECT`. Reason codes: `RETAKE_REQUIRED` (+ quality issues), `INVALID_IMAGE`, `UNKNOWN_CARD_TYPE`,
`AMBIGUOUS_CARD_TYPE`, `AMBIGUOUS_VARIANT`, `FIELD_UNREADABLE:<field>`, `LOW_CONFIDENCE:<field>`,
`RULE_FAILED:<rule>:<field>`, `CARD_EXPIRED`, and for two-sided cards `SIDE_REQUIRED:<side>`
(REJECT: the card needs a photo of that side — ask for it) and `SIDES_MISMATCH` (REVIEW: the two photos
show different cards).

Schema 2.1 is 2.0 plus `photos`, `missing_sides` and `fields.*.side` (additive). `quality` is the main
photo's (the front when there is one). `photos[].status`: `used`, `not_used` (a second photo of the same
side, or a side the card does not have), `retake` (failed the quality gate), `invalid` (not an image).

### POST /v1/cards/validate/stream
Same form fields and `store` as above, plus query `mode=delta|snapshot` (default `delta`). Answers with
**Server-Sent Events** (`text/event-stream`) while the card is processed. Never served from the cache.

`StepState = {id, status, summary, ms, detail}` with `status` one of
`pending | running | pass | warn | fail | skip`, over the 11 steps of `GET /v1/pipeline/steps`:
`ingest, quality, normalise, ocr, embed, search, family, variant, fields, rules, decision`.

| Event (mode=delta) | Data |
|---|---|
| `snapshot` | first event: `{"steps": [StepState…]}`, all pending |
| `step` | one StepState, each time a step starts or finishes |
| `image` | once: `{"normalisedUrl": "data:image/jpeg;base64,…"}` (the straightened card) |
| `result` | last event: `{"steps": […], "normalisedUrl": "…", "result": <validation response>}` |
| `photo_image` | two photos only, once per photo: `{"photo": 0, "label": "front", "normalisedUrl": "…"}` |
| `error` | `{"detail": "…"}` if processing failed after the stream started |

**Two photos.** Steps `ingest`, `quality`, `normalise`, `ocr` and `embed` run once per photo and are
merged into the same step: `status` is the worst finished photo (`running` while any photo is in it),
`summary` reads `front: … · back: …`, and `detail` is the first photo's detail plus
`photos: [{photo, label, status, summary, detail}]`. Snapshots also carry
`photos: [{photo, label, side, normalisedUrl}]`; labels become the detected side once known. The
`fields` step lists each field's `side` and adds `photos`, `missing_sides` and `side_assignment`.
A one-photo stream is unchanged.

With `mode=snapshot` every event is `snapshot` with the full `{steps, normalisedUrl?}`, the last one is
`result`. `detail` per step: quality checks, OCR tokens, embedding (first 768 values), shortlisted and
verified references, family and variant candidates with their signals, extracted fields, rule results.

```bash
curl -N -F image=@card.jpg http://127.0.0.1:8080/v1/cards/validate/stream
curl -N -F front=@card_front.jpg -F back=@card_back.jpg http://127.0.0.1:8080/v1/cards/validate/stream
```

---

## 2. Cards and reference images
A **card family** (programme) has variants, field definitions and rules; the schema is
`src/cardagent/registry/models.py`, examples in `examples/cards/`. Every change touches only that
card's points in Qdrant and is live on all replicas within `qdrant.refresh_seconds`.

**Sides.** `sides` lists the card's sides (default `[{"id": "front"}]`) and every field has a `side`
(default `front`); its `roi` is measured on a reference image of that side. A side is required when a
required field is printed on it (override with `required: true|false`). Reference images are stored per
variant **and side**. Details: [Two-sided cards](TWO_SIDED_CARDS.md).

**Ownership.** Cards imported from `registry/` (`source: registry`) are managed in Git. Changing or
deleting them (or their reference images) through the API answers `409` unless `?force=true`; a forced
change hands the card to the API, and later syncs leave it alone. Cards created through the API are
`source: api` and are never touched by a sync.

| Endpoint | Notes |
|---|---|
| `GET /v1/cards` | every family with `sides`, variants, reference counts (total and `reference_images_by_side`), owner and last change |
| `GET /v1/cards/{id}` | definition + `sides` (`[{id, required, fields}]`) + `variants` (reference counts, total and by side) + `variants_without_references` + `missing_references` (`[{variant, side}]` without any reference image); `?format=yaml` returns the definition as YAML |
| `PUT /v1/cards/{id}` | create (`201`) or replace (`200`). JSON body, or YAML with `Content-Type: application/yaml`. The id in the body must match the URL. Removing a variant (or a side) deletes its reference images; changing field ROIs or sides re-extracts this card's keypoints. Response includes `created`, `changed`, `references_deleted`, `keypoints_refreshed` |
| `DELETE /v1/cards/{id}` | removes the definition and all its reference vectors and stored images |
| `GET /v1/cards/{id}/references` | `sides`, and reference images per variant: `ref_id`, `side`, `filename`, `sha256`, `source`, `added_at` |
| `POST /v1/cards/{id}/variants/{variant}/references` | multipart `images` (1–20 files). Query `side` (`front` default, or `back`; `422` if the card has no such side) and `detect_card` (default `true`: find and straighten the card in a photo; use `false` for flat scans). Response adds `side` and `reference_images_in_side`. Per file: `created`, `exists` (same image already stored) or `rejected` with an error. `201` if any was created, `422` if all were rejected |
| `GET /v1/cards/{id}/references/{ref_id}/image` | the stored reference image |
| `DELETE /v1/cards/{id}/references/{ref_id}` | removes one reference image |

```bash
H="X-Admin-Token: $CARDAGENT_API__ADMIN_TOKEN"
curl -X PUT -H "$H" -H "Content-Type: application/yaml" --data-binary @examples/cards/ae/fazaa.yaml \
     http://127.0.0.1:8080/v1/cards/ae.fazaa
curl -X POST -H "$H" -F images=@gold_1.jpg -F images=@gold_2.jpg \
     http://127.0.0.1:8080/v1/cards/ae.fazaa/variants/gold/references
curl -X POST -H "$H" -F images=@gold_back_1.jpg "http://127.0.0.1:8080/v1/cards/ae.fazaa/variants/gold/references?side=back"
```

---

## 3. Registry
| Endpoint | Notes |
|---|---|
| `GET /v1/registry` | `version` (Qdrant revision; changes on every card or settings write), family/variant/reference counts |
| `POST /v1/registry/sync` | incremental import of `registry/cards/**/*.yaml` and `registry/refs/<family>/<variant>/*` (front; back images in `<variant>/back/`) (same as `tools/sync_registry.py`). Query `prune` (default `true`: delete registry-owned cards whose files are gone), `reembed` (default `false`: re-embed every reference after an embedder change). An invalid YAML answers `422` and changes nothing |
| `POST /v1/registry/reload` | reload cards and settings from Qdrant now (replicas also check every `refresh_seconds`) |

---

## 4. Runs, review queue and statistics
A **run** is one stored validation: summary fields, the full response (`result`), the photos (if
`runs.store_images`; one, or front and back), and the review. Runs older than `runs.retention_days`
are deleted hourly. Run list items carry `photos` (the number of uploaded photos); the run itself has
`images: [{photo, side, upload, path}]`.

| Endpoint | Notes |
|---|---|
| `GET /v1/runs` | newest first, without `result`. Filters: `status`, `family`, `review_status` (`none`, `pending`, `accepted`, `rejected`), `since`, `until` (ISO-8601, e.g. `2026-11-24` or `2026-11-24T08:00:00Z`), `limit` (≤ 200). Paging: pass `next_before` from the response as `before` |
| `GET /v1/review` | the review queue: runs with `review_status=pending`, plus the `pending` count. Filter `family`, paging as above |
| `GET /v1/runs/{id}` | everything, including the full `result` |
| `GET /v1/runs/{id}/image` | an uploaded photo: by default the main one (front when present); `?side=back` the photo used as that side, `?photo=1` the photo at that upload position |
| `DELETE /v1/runs/{id}` | deletes the run and its photos |
| `POST /v1/runs/{id}/review` | body below; works on any run (e.g. a wrong ACCEPT found later) |
| `POST /v1/runs/prune` | delete runs older than `older_than_days` (default `runs.retention_days`) now |
| `GET /v1/stats` | from stored runs, default the last 7 days. Query `since`, `until`, `family`. Returns `by_status`, `auto_accept_rate`, `review_rate`, `review_queue`, `review_agreement` (share of reviews that accepted the card), `latency_ms` (p50/p95/max), `top_reasons`, `by_family`, `per_day`, `truncated` |

Review body:
```json
{"decision": "accept", "family": "ae.fazaa", "variant": "gold",
 "fields": {"card_number": "2210457783"}, "reviewer": "agent.42", "note": "glare on number", "enroll": true}
```
* `decision`: `accept` (the card is genuine and valid) or `reject`.
* `family` / `variant`: the correct identification; default what the agent found. Required for `accept`
  when the agent could not identify the card.
* `fields`: corrected values (only fields defined for the family).
* `enroll`: with `accept`, adds the run's photos as reference images of that variant, each as the side it
  shows, so the same design is recognised next time. Not allowed with `reject` (`422`).
* `photo_sides` (optional, with `enroll`): the side of each photo by position, e.g. `{"0": "back"}`, when
  the agent did not assign it (unknown card) or assigned it wrongly.
* Response `review.enrolled` (first photo, as before) and `review.enrolled_photos`
  (`[{photo, side, ref_id, status}]`).

---

## 5. Settings
Effective settings = `config/settings.yaml` + `CARDAGENT_*` environment variables + **overrides** stored
in Qdrant. Overrides reach every replica within `refresh_seconds` and need no restart. Only decision
thresholds and weights can be overridden; models, OCR, storage and credentials stay in the file/env.

| Endpoint | Notes |
|---|---|
| `GET /v1/settings` | `effective`, `defaults` (file/env), `overrides`, `tunable` (keys that PATCH accepts), `fixed` (the rest, read-only) |
| `PATCH /v1/settings` | merge-patch the overrides: a value sets it, `null` removes it (back to the default), a section set to `null` removes all of that section. The result is validated as a whole (types, weights summing to 1) before it is stored |
| `DELETE /v1/settings` | remove all overrides, or one section with `?section=scoring` |

Tunable sections: `scoring.*` (family/variant weights, `family_min`, `family_margin`, `variant_min`,
`variant_margin`, `reject_below`, `geometry_required_without_text`, `field_min_confidence`,
`sides_mismatch_geometry`),
`quality.*`, `retrieval.*` except `orb_features`, and `embedder.similarity_floor`.

```bash
curl -X PATCH -H "$H" -H "Content-Type: application/json" \
     -d '{"scoring": {"family_min": 0.55}}' http://127.0.0.1:8080/v1/settings
```
Test a change first with a what-if evaluation (section 6).

---

## 6. Evaluation
Measures accuracy on labelled photos. Use it before a threshold change, after adding cards, and in CI.

**Evaluation set as a zip**: `manifest.json` plus the photos it lists (paths relative to the manifest):
```json
[
  {"file": "photos/001.jpg", "family": "ae.fazaa", "variant": "gold", "expected_status": "ACCEPT",
   "fields": {"card_number": "2210457783", "expiry_date": "11/27"}},
  {"file": "photos/002.jpg", "family": "ae.esaad", "variant": "platinum", "expected_status": "REJECT"},
  {"file": "photos/003.jpg", "family": null, "variant": null, "expected_status": "REJECT"}
]
```
`fields` is optional (checked when given). Up to `evaluation.max_set_bytes` (500 MB).
Cards photographed on both sides use `files` instead of `file`: `{"front": "photos/004f.jpg",
"back": "photos/004b.jpg"}`, or a list of 1–2 paths whose sides are detected. Sets built from reviewed
runs include every photo of the run. Report rows carry `files` for such cases.

| Endpoint | Notes |
|---|---|
| `POST /v1/evaluation-sets` | multipart `file` (the zip) + `name` → the set (`set_id`, `cases`, `expected` counts, `families`) |
| `POST /v1/evaluation-sets/from-runs` | JSON `{"name", "since", "until", "family"}` (all optional): a set from reviewed runs — accepted reviews become expected `ACCEPT` with the reviewed card and corrected fields, rejected reviews expected `REJECT` |
| `GET /v1/evaluation-sets`, `GET/DELETE /v1/evaluation-sets/{id}` | list, read, delete |
| `POST /v1/evaluations` | JSON `{"set_id", "settings": {…}, "today": "2026-11-24"}` → `202` with `evaluation_id`. Runs in the background (one at a time per process). `settings` = what-if overrides on top of the live ones (same shape as PATCH; live traffic is not affected). `today` fixes the date for expiry rules |
| `GET /v1/evaluations` | list (`?set_id=`) |
| `GET /v1/evaluations/{id}` | `status` (`queued`, `running`, `done`, `failed`), `progress`, `summary`, and when done the `report`; `?rows=false` leaves out the per-photo rows |
| `DELETE /v1/evaluations/{id}` | `409` while it is still running |

Report: `cases`, `status_agreement`, `automation_rate`, `accept_precision` (share of ACCEPTs with the
right card and fields), `wrong_accepts`, `field_accuracy`, `latency_ms`, `family_confusion`,
`identification_errors`, `gate` (`passed` when ACCEPT precision ≥ `evaluation.min_accept_precision`
and status agreement ≥ `evaluation.min_status_agreement`), `registry_version`, `embedder`, `rows`.

The same evaluation from the command line (CI): `python tools/evaluate.py --manifest <set>/manifest.json`
(exit code 1 when the gate fails).

---

## 7. Health and metrics
* `GET /health/live` — the process is up.
* `GET /health/ready` — `200` when the agent is loaded and Qdrant answers: registry version, family and
  reference counts, embedder, OCR, VLM, whether runs are recorded and settings overridden. `503` otherwise.
* `GET /metrics` — Prometheus: `cardagent_requests_total{status}`, `cardagent_reasons_total{reason}`,
  `cardagent_request_seconds`, `cardagent_stage_seconds{stage}`, `cardagent_vlm_fallback_total`,
  `cardagent_registry_reloads_total{result}`, and more.

## 8. Storage (all in Qdrant)
| Collection | Content |
|---|---|
| `<prefix>_refs` | one point per reference image: vector + family, variant, side, country, sha256, source, ORB keypoints |
| `<prefix>_families` | card definitions, the store revision/embedder meta point, the settings overrides |
| `<prefix>_runs` | runs (payload only; photos in `runs.store_dir`) |
| `<prefix>_evaluations` | evaluation sets and evaluations (files and reports in `evaluation.store_dir`) |

`prefix` = `qdrant.collection_prefix` (default `cardagent`). Uploaded reference images are also kept
in `qdrant.reference_store_dir`, so they can be re-embedded after an embedder change.
