# Two-sided cards — design and API specification (v1, schema 2.1)

Branch `feature/two-sided-cards`. This document is the specification for client changes, in
particular the web console (card-agent-web). Every change is **additive**: one-photo clients, card
definitions, reference images, runs and the admin token work exactly as before.

1. [The problem and the approach](#1-the-problem-and-the-approach)
2. [Card definition: sides](#2-card-definition-sides)
3. [Reference images per side](#3-reference-images-per-side)
4. [Validation with one or two photos](#4-validation-with-one-or-two-photos)
5. [Response schema 2.1](#5-response-schema-21)
6. [Live stream with two photos](#6-live-stream-with-two-photos)
7. [Runs, review and enrolment](#7-runs-review-and-enrolment)
8. [Evaluation sets](#8-evaluation-sets)
9. [Web console: changes per screen](#9-web-console-changes-per-screen)
10. [Compatibility](#10-compatibility)

---

## 1. The problem and the approach

Card programmes print their data in different places:

| Card | Front | Back | Definition | Photos needed |
|---|---|---|---|---|
| A | name, expiry | card number | `sides: [front, back]`, `card_number.side: back` | front **and** back |
| B | everything | (nothing needed) | no `sides` (default: front only) | front |
| C | logo only | name, number, expiry | `sides: [front, back]`, every field `side: back` | back (front optional) |

**Decision: the validation endpoint accepts up to two photos of one card in one request** (one per
side). Alternatives that were rejected:

* *One request per side, combined later.* Needs server-side session state, a timeout policy and a
  merge step; a lost second request leaves a half-validated card. One request keeps the API stateless,
  idempotent and atomic: one card → one decision → one run.
* *Stitching both sides into one image on the client.* Breaks card detection, orientation, alignment
  and the reference matching.
* *Always asking for both sides.* Unnecessary friction for card B; the card definition knows which
  sides carry required data.

How a request with two photos is processed:

1. Each photo goes through the quality gate, card detection, OCR and feature extraction on its own.
2. **Identification pools the evidence**: the OCR text of both photos counts for the same card (the
   programme name on the front and the number prefix on the back both help), and each photo is matched
   against the reference images of **both** sides. Scores are the best over the photos.
3. **Side assignment.** For every photo the agent measures how strongly it matches each side's
   references (RANSAC geometry first, embedding second). A `front`/`back` label is trusted unless the
   photo clearly shows the other side (swapped uploads are corrected, `side_corrected: true`);
   unlabelled photos (`image`, `images`) are sorted out the same way. A second photo of the same side is
   left unused rather than read as the other side.
4. **Field extraction per side.** Each photo is aligned to the best reference of *its* side and only the
   fields printed on that side are read from it.
5. **Decision.**

| Situation | Status | Reasons |
|---|---|---|
| All required sides present, fields valid | ACCEPT | — |
| A required side has no photo | REJECT | `SIDE_REQUIRED:<side>` — ask the user for that side, then send **both** photos again |
| The photo of a required side failed the quality gate | REJECT | `RETAKE_REQUIRED`, its issues, `SIDE_REQUIRED:<side>` |
| A photo that is not needed failed the quality gate (e.g. a blurry back of card B) | (decided on the rest) | — (`photos[].status: retake`) |
| The two photos strongly match **different** cards | REVIEW | `SIDES_MISMATCH` |
| Anything else | unchanged from schema 2.0 | |

Rules on fields of a side that was not photographed are reported as `skip` with
`detail: SIDE_NOT_CAPTURED:<side>` (not as failures), and such fields do not produce
`FIELD_UNREADABLE`: the single reason is `SIDE_REQUIRED`.

Variant scoring on a photo of the back ignores the body colour (the variant colours describe the front,
and most programmes use one back for every tier); the tier word and the card-number prefix decide.

---

## 2. Card definition: sides

New in the card definition (`src/cardagent/registry/models.py`):

```yaml
sides:                      # optional; default [{id: front}]
  - id: front               # front | back
  - id: back
    required: true          # optional; default: required when a required field is printed on it,
                            # or when it is the card's only side. false is refused if a required field is on it
fields:
  - name: card_number
    side: back              # optional; default front. The roi is on a reference image of THIS side
    roi: [0.05, 0.60, 0.75, 0.74]
    regex: "\\d{10}"
    normalise: digits
```

Validation (`422` on `PUT /v1/cards/{id}`): sides listed once each; every field's side must be listed;
`required: false` on a side that carries a required field is refused.

Card A, B and C as definitions (complete templates in `examples/cards/two_sided/`):

```yaml
# Card A: name + expiry on the front, number on the back
sides: [{id: front}, {id: back}]
fields:
  - {name: holder_name, side: front, roi: [...], ...}
  - {name: expiry_date, side: front, roi: [...], ...}
  - {name: card_number, side: back,  roi: [...], ...}
# Card B: unchanged, no sides
# Card C: everything on the back; the front (logo) is optional but helps identification
sides: [{id: front}, {id: back}]
fields:
  - {name: holder_name, side: back, ...}
  - {name: card_number, side: back, ...}
  - {name: expiry_date, side: back, ...}
```

`GET /v1/cards/{id}` adds:

```json
"sides": [{"id": "front", "required": true, "fields": ["holder_name", "expiry_date"]},
          {"id": "back",  "required": true, "fields": ["card_number"]}],
"variants": [{"id": "gold", "name": "Gamma Gold", "reference_images": 6,
              "reference_images_by_side": {"front": 3, "back": 3}}],
"missing_references": [{"variant": "silver", "side": "back"}]
```

`GET /v1/cards` and `GET /v1/registry` items add `sides` and `variants[].reference_images_by_side`.

---

## 3. Reference images per side

| Endpoint | Change |
|---|---|
| `POST /v1/cards/{id}/variants/{v}/references` | new query `side=front|back` (default `front`). `422` if the card has no such side. Response adds `side`, `reference_images_in_side`; each result adds `side` |
| `GET /v1/cards/{id}/references` | adds `sides`; every reference adds `side` |
| `DELETE /v1/cards/{id}/references/{ref}` | response adds `side` |
| `PUT /v1/cards/{id}` | removing a side deletes that side's reference images (`references_deleted`) |

Onboarding: 3–20 clean images **per variant and side**. Backs that are identical across tiers still
need images under every variant (upload the same images to each; they are stored once per variant).
Reference keypoints mask the ROIs of the fields printed on that side.

Git registry: front images stay in `registry/refs/<family>/<variant>/`, back images go to
`registry/refs/<family>/<variant>/back/` (an explicit `front/` folder also works). Back images for a card
without a back side make the sync fail (`422`, nothing changed). `tools/sync_registry.py` reports
missing references per side.

---

## 4. Validation with one or two photos

`POST /v1/cards/validate` and `POST /v1/cards/validate/stream`, multipart:

| Field | Count | Meaning |
|---|---|---|
| `image` | 0–1 | one photo, side detected — the 2.0 field, unchanged |
| `front` | 0–1 | photo of the front |
| `back` | 0–1 | photo of the back |
| `images` | 0–2 | photos with sides detected (repeat the field) |
| `request_id` | 0–1 | text |

At least one and at most **two** photos in total, otherwise `422`:
`"send a card photo: field 'image', or 'front' and/or 'back', or 'images'"` /
`"at most 2 photos per request (one card, front and back)"`. Each file: JPEG/PNG/WebP, ≤ 8 MB
(`415` / `413` as before). Photos are processed in the order front, back, image, images; `photo`
indexes in the response follow that order.

Idempotency: a retry with the same `request_id` **and** the same photos (same fields, same bytes)
returns the first answer; any other combination is processed.

Recommended capture flow (stateless):
1. Capture what the user offers (or both sides for programmes known to be two-sided) and send it.
2. If the answer is `REJECT` with `SIDE_REQUIRED:back` (or `:front`), prompt for that side and send
   **both** photos again in a new request. The agent keeps no partial state between requests.
3. `RETAKE_REQUIRED` + `photos[i].status: "retake"` tells which photo to retake.

```bash
curl -F front=@front.jpg -F back=@back.jpg -F request_id=t7 http://127.0.0.1:8080/v1/cards/validate
curl -F images=@a.jpg -F images=@b.jpg http://127.0.0.1:8080/v1/cards/validate      # sides detected
curl -F image=@front.jpg http://127.0.0.1:8080/v1/cards/validate                    # as before
```

---

## 5. Response schema 2.1

`schema_version: "2.1"`; everything from 2.0 is unchanged. New:

```ts
type Side = "front" | "back";

interface FieldResult {            // existing type, one new member
  value: string | null; confidence: number; source: "ocr_roi" | "ocr_fulltext" | "vlm" | "none";
  valid_format: boolean; valid_until?: string | null;
  side: Side;                      // NEW: side the field is printed on
}

interface PhotoInfo {              // NEW
  photo: number;                   // 0-based position (order: front, back, image, images)
  upload: "image" | "front" | "back" | "images";
  declared: Side | null;           // side named by the upload field
  side: Side | null;               // side the photo was used as; null = not used
  status: "used" | "not_used" | "retake" | "invalid";
  side_corrected: boolean;         // photo shows the other side than declared
  quality: QualityInfo | null;     // same shape as the top-level quality
  reference_id: string | null;     // reference image it was aligned to
  inliers: number | null;          // RANSAC inliers against that reference
  fields: string[];                // fields read from this photo
}

interface ValidationResult {       // existing type, new members
  // ...all 2.0 members...
  quality: QualityInfo | null;     // the MAIN photo's (front when present)
  photos: PhotoInfo[];             // NEW: one entry per uploaded photo (also for one-photo requests)
  missing_sides: Side[];           // NEW: required sides without a usable photo
}
```

New reason codes for the label table: `SIDE_REQUIRED:<side>` ("Photo of the back/front needed"),
`SIDES_MISMATCH` ("The photos show different cards"). New rule detail: `SIDE_NOT_CAPTURED:<side>`.
`candidates.family[]` and `candidates.variant[]` may carry `photo` (the photo that gave the best score)
when two photos were sent.

Example (card A, labels swapped by the user, corrected by the agent):

```json
{
  "schema_version": "2.1", "status": "ACCEPT", "reasons": [],
  "card": {"family": "ae.gamma", "variant": "gold", "family_confidence": 0.957, "variant_confidence": 0.967, "...": "..."},
  "fields": {"holder_name": {"value": "MARIAM YOUSEF", "side": "front", "...": "..."},
             "expiry_date": {"value": "05/30", "valid_until": "2030-05-31", "side": "front", "...": "..."},
             "card_number": {"value": "4123456789", "side": "back", "...": "..."}},
  "photos": [{"photo": 0, "upload": "front", "declared": "front", "side": "back", "status": "used",
              "side_corrected": true, "inliers": 185, "fields": ["card_number"], "...": "..."},
             {"photo": 1, "upload": "back", "declared": "back", "side": "front", "status": "used",
              "side_corrected": true, "inliers": 150, "fields": ["holder_name", "expiry_date"], "...": "..."}],
  "missing_sides": []
}
```

Example (card A, front only):

```json
{"status": "REJECT", "reasons": ["SIDE_REQUIRED:back"], "missing_sides": ["back"],
 "card": {"family": "ae.gamma", "...": "..."},
 "fields": {"card_number": {"value": null, "confidence": 0.0, "source": "none", "valid_format": false, "side": "back"}},
 "validation": [{"rule": "regex", "field": "card_number", "critical": true, "result": "skip",
                 "detail": "SIDE_NOT_CAPTURED:back"}]}
```

---

## 6. Live stream with two photos

Same 11 steps and events; with **one** photo the stream is byte-for-byte the 2.0 stream.

With two photos:

| | Change |
|---|---|
| Per-photo steps | `ingest`, `quality`, `normalise`, `ocr`, `embed` run per photo and are **merged**: `status` = worst finished photo (`fail > warn > pass > skip`), `running` while any photo is still in the step; `summary` = `"front: … · back: …"`; `ms` = sum; `detail` = the first photo's detail **plus** `photos: [{photo, label, status, summary, detail}]` |
| Shared steps | `search`, `family`, `variant` describe the combined identification (the `search` detail is that of the photo with the best match, plus `photos: [{photo, top, verified}]` and `sides_mismatch`) |
| `fields` step | rows add `side`; detail adds `photos: [{photo, side, fields, aligned}]`, `missing_sides`, `side_assignment: [{photo, affinity: {front, back}, value: {front, back}}]` |
| Event `image` | still sent once, with the first photo |
| Event `photo_image` (new) | once per photo: `{"photo": 0, "label": "front", "normalisedUrl": "data:image/jpeg;base64,…"}` |
| Snapshots (`snapshot`, `result`) | add `photos: [{photo, label, side, normalisedUrl}]`; `label` starts as the declared side or `photo 1`/`photo 2` and becomes the detected side once known |

Parsers that ignore unknown events and members keep working unchanged.

---

## 7. Runs, review and enrolment

| Endpoint | Change |
|---|---|
| `GET /v1/runs`, `GET /v1/review` | items add `photos` (number of uploaded photos) |
| `GET /v1/runs/{id}` | adds `images: [{photo, side, upload, path}]`; `result` is schema 2.1 |
| `GET /v1/runs/{id}/image` | default: the main photo (front when present) — as before for one-photo runs. New query `side=front|back` (the photo used as that side) or `photo=0|1` (upload position). `404` if there is no such photo |
| `POST /v1/runs/{id}/review` | `enroll: true` now enrolls **every** used photo as a reference of its side. New optional `photo_sides: {"0": "front", "1": "back"}` to set or correct the sides (needed when the agent could not identify the card). Response `review.enrolled` (first photo, as before) plus `review.enrolled_photos: [{photo, side, ref_id, status}]` |
| `DELETE /v1/runs/{id}`, `POST /v1/runs/prune` | delete all photos of a run |

Runs stored before this version have one photo and no `images`; `?side=front`/`?photo=0` return it.

---

## 8. Evaluation sets

Manifest entries may use `files` instead of `file`:

```json
[{"files": {"front": "p/0001f.jpg", "back": "p/0001b.jpg"}, "family": "ae.gamma", "variant": "gold",
  "expected_status": "ACCEPT", "fields": {"card_number": "4123456789"}},
 {"files": ["p/0002a.jpg", "p/0002b.jpg"], "family": "qa.delta", "variant": "standard", "expected_status": "ACCEPT"},
 {"file": "p/0003.jpg", "family": "ae.alpha", "variant": "blue", "expected_status": "ACCEPT"}]
```

A `files` list means "sides detected". Report rows add `files` for such cases. Sets built from reviewed
runs (`POST /v1/evaluation-sets/from-runs`) copy every photo and write `files` by side. A case that ends
in `SIDE_REQUIRED` is excluded from the identification-error table, like `RETAKE_REQUIRED`.

New tunable setting: `scoring.sides_mismatch_geometry` (default `0.30`): geometry score above which two
photos matching different families count as `SIDES_MISMATCH`.

---

## 9. Web console: changes per screen

| Screen | What to change | API |
|---|---|---|
| Validate › Live | Two drop zones, **Front** and **Back** (each optional), plus "I don't know which side" (sends `images`). Show both straightened photos from `photo_image` events with their side badge; a notice when `side_corrected`. Per-photo sub-rows in merged steps from `detail.photos`. When `missing_sides` is not empty, a call to action "Add a photo of the back" that re-sends **both** photos | §4, §6 |
| Validate › Batch | Pairing rule for multiple files: by file name (`*_front.*` / `*_back.*`, or the same stem with `_1`/`_2`), else one card per file | §4 |
| Result view (all) | Field rows show a side badge; fields of a missing side show "not photographed" instead of "unreadable"; a Photos panel from `photos[]` (status, quality, reference, inliers, fields); new reason labels | §5 |
| Programme › Definition (YAML) | zod mirror: `sides: {id: "front"|"back", required?: boolean}[]` (default front, unique), `fields[].side` (default front, must be in `sides`), side with required fields cannot be `required: false`. Map server 422 messages as today | §2 |
| Add programme wizard | New step "Sides": front only / front and back; for each field a side selector; show which sides become required | §2 |
| Field layout editor | Choose the side; the background image list is `GET …/references` filtered by `side`; regions belong to that side | §2, §3 |
| Programme › Reference images | Tabs or columns per side with counts from `reference_images_by_side`; upload sends `?side=`; a warning per entry of `missing_references` | §3 |
| Card programmes list / Registry | Show `sides` and per-side reference counts | §2 |
| Run detail / Review queue | Show both photos (`/image?side=front`, `/image?side=back`, or `?photo=n`); enrol shows which sides will be added; when a photo has no side (unknown card), let the reviewer pick it and send `photo_sides` | §7 |
| Evaluations | Upload help text for `files`; report rows show both file names | §8 |
| Decision settings | New slider `scoring.sides_mismatch_geometry` (0–1) | §8 |

Type changes for `src/api/types.ts`: `Side`, `FieldResult.side`, `PhotoInfo`, `ValidationResult.photos`,
`ValidationResult.missing_sides`, `CardFamily.sides`, `FieldSpec.side`, `VariantCounts.reference_images_by_side`,
`Reference.side`, `RunDetail.images`, `RunListItem.photos`, `ReviewBody.photo_sides`,
`Review.enrolled_photos`, SSE `PhotoImageEvent`, `Snapshot.photos`, `StepDetail.photos`.

---

## 10. Compatibility

* **Authentication unchanged**: open endpoints stay open, admin endpoints still need `X-Admin-Token`.
* **One-photo requests**: the `image` field, the response members of 2.0 and the one-photo stream are
  unchanged; the response adds `photos` (one entry) and `missing_sides: []` and says `schema_version: "2.1"`.
* **Card definitions without `sides`** are front-only cards and behave as before. Re-saving one does not
  reprocess its references.
* **Stored data**: references stored before this version have no side and count as front; their point ids
  are unchanged. The new payload index `side_key` is created at startup on existing Qdrant collections.
  Old runs keep their single photo.
* Verified on this branch: all earlier tests pass, plus 26 new tests for two-sided cards, on an
  in-process Qdrant and on a Qdrant 1.18.1 server; an upgrade test read a database written by
  `feature/qdrant-platform` (references, a run, its review and enrolment) with the new code.
