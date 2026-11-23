# Console API — Card Agent Console (card-agent-web) ↔ card-agent-enterprise

The **Card Agent Console** (React web platform: dashboard, validate, runs & review, card
families, family editor, reference cards, settings) was built against the card-agent **MVP**.
It speaks the MVP formats, which differ from the enterprise API:

| | Console expects (MVP) | Enterprise `/v1` returns |
|---|---|---|
| Family id | `fazaa` | `ae.fazaa` |
| Family YAML | `family`, `family_name`, `weights: {text, visual, colour}` | `id`, `country`, `name`, global scoring weights |
| `candidates` | flat list `[{family, score}]` | `{family: [{id, score, signals}], variant: [...]}` |
| Reason codes | `UNKNOWN_CARD`, `AMBIGUOUS_FAMILY` | `UNKNOWN_CARD_TYPE`, `AMBIGUOUS_CARD_TYPE` |
| Reload / save | `POST /registry/reload`, `PUT /registry/families/{id}` | `POST /v1/admin/registry/reload`, `PUT /v1/admin/cards/{id}` |

Pointing the console straight at `/v1` therefore breaks it (it calls `candidates.map(...)` on an
object). The service now serves a **compatibility API at `/console/v1`** that speaks the console's
formats exactly. The `/v1` API is unchanged.

## Connect the console
1. Console → **Thresholds & settings → Engine → card-agent API**.
2. **API base URL:** `http://<host>:8080/console/v1`, then **Test connection** (shows "connected · N families loaded").
3. If the console is served from another origin (e.g. `npm run dev` on :5173), either use its Vite
   proxy (`CARD_AGENT_API=http://<host>:8080 npm run dev`) or allow the origin:
   `CARDAGENT_CONSOLE__CORS_ORIGINS='["http://localhost:5173"]'`.

## Routes (`/console/v1`)
Called by the console today:

| Method | Path | Console feature | Notes |
|---|---|---|---|
| POST | `/cards/validate` | Validate a card (API mode) | multipart `image` + `request_id`; same pipeline, cache and audit log as `/v1/cards/validate`, answer in the MVP envelope |
| GET | `/registry` | Settings → Test connection | `families` = list of console-schema families (+ `enterprise_id`, `country`, `reference_images` per variant) |
| POST | `/registry/reload` | after saving a family | 🔒 |
| PUT | `/registry/families/{id}` | Family editor → Save | 🔒 console YAML (`application/yaml`) or JSON; validated against the console's schema, then the enterprise schema |

For the family editor and reference-card pages:

| Method | Path | Notes |
|---|---|---|
| GET | `/registry/families/{id}` | the family as console YAML |
| DELETE | `/registry/families/{id}` | 🔒 family + its reference images |
| GET | `/registry/families/{id}/references` | reference images per variant (`ref_id`, filename, sha256, source) |
| POST | `/registry/families/{id}/variants/{variant}/references` | 🔒 enroll / upload: multipart `images` (1–20); photos are straightened when `detect_card=true`; re-upload = no-op |
| DELETE | `/registry/families/{id}/references/{ref_id}` | 🔒 one reference image |

🔒 = needs header `X-Admin-Token` (see *Security*).

## How the result is translated
* `candidates` → flat list `{family, name, score, signals}`; variant candidates in `variant_candidates`.
* `UNKNOWN_CARD_TYPE` → `UNKNOWN_CARD`, `AMBIGUOUS_CARD_TYPE` → `AMBIGUOUS_FAMILY`, `INVALID_IMAGE` adds `RETAKE_REQUIRED`.
* `quality` is always an object (zeros for an unreadable upload).
* `card.family` uses the console id; `card.enterprise_family` keeps `ae.fazaa`. For an ambiguous
  family the top candidate is returned as `card` with `provisional: true` (the decision stays REVIEW).

## How families are translated
* Ids: `ae.fazaa` ↔ `fazaa` (short name when unique; `sa-esaad` style when two countries share a
  name). A new console id is created in `console.default_country` (default `ae`).
* Console-only `weights` are accepted but not stored — the enterprise service scores every family
  with the global `scoring.family_weights` (4 signals). `GET` returns the MVP defaults.
* Enterprise-only attributes (country, a field's `script`, extra rule params) are kept when the
  console re-saves an existing family.
* New rule type `date_order` (console feature): `field` must not be later than `other_field`.

## Storage behaviour
| `vector_store.backend` | Writes from the console |
|---|---|
| `qdrant` | Through the card store, incremental (only that card), shared by all replicas. **Same ownership rule as `/v1/admin/cards`:** a family that comes from `registry/cards` (git) answers `409`; retry with `?force=true` to hand it over to the API. Cards created in the console are API-owned. |
| `faiss` (dev) | Edits `registry/` (YAML written atomically; replaced/deleted files moved to `registry/.deleted/<timestamp>/`), then reloads. A change the service cannot load is rolled back (`422`). A reload re-embeds all references, so use Qdrant beyond development. |

## Security
* Writes need `X-Admin-Token` = `api.admin_token`. The console does not send this header yet, so in
  production put the console and API behind your internal gateway and let it inject the header for
  authenticated operators, or add the header to the console's API client.
* `console.require_admin_token: false` disables the check — local development only.
* `console.enabled: false` turns the whole `/console/v1` API off (404).

## Settings
```yaml
console:
  enabled: true
  default_country: ae
  require_admin_token: true
  cors_origins: []
```
Environment: `CARDAGENT_CONSOLE__ENABLED`, `CARDAGENT_CONSOLE__DEFAULT_COUNTRY`,
`CARDAGENT_CONSOLE__REQUIRE_ADMIN_TOKEN`, `CARDAGENT_CONSOLE__CORS_ORIGINS`.

## Quick check
```bash
curl -s http://localhost:8080/console/v1/registry | jq '.families | length'
curl -s -F image=@data/samples/fazaa_gold.jpg -F request_id=t1 http://localhost:8080/console/v1/cards/validate | jq '.status, .card.family, .candidates[0]'
curl -s -X PUT -H "X-Admin-Token: $TOKEN" -H "Content-Type: application/yaml" \
     --data-binary @mycard.yaml http://localhost:8080/console/v1/registry/families/mycard
```
