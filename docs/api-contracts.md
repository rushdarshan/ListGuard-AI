# API contracts (v1)

All responses are JSON. IDs are UUIDv4 (serialized as strings). Errors use
`{ "error": { "code": ..., "message": ... } }`.

## Listings

- `POST /v1/listings` → `201` — `{ title (1..200), description? (≤5000), category? (≤120), seller_id }`
- `GET /v1/listings?limit=20&offset=0&status&seller_id&q` → `{ items, total, limit, offset }`
  (`q` is a literal-substring pre-filter — `%`, `_`, `\` match literally, never as wildcards.
  Unknown `status` → `422`. Postgres FTS/GIN index exists in migration `0001`.)
- `GET /v1/listings/{id}` → `200` / `404`
- `PATCH /v1/listings/{id}` → `200` (title/description/category/status)
- `DELETE /v1/listings/{id}` → `204` (soft delete via `deleted_at`; default queries exclude)

## Images — metadata REGISTRATION (not upload)

> Registers an image that has already been uploaded to object/local storage.
> This endpoint never accepts binary bytes. Actual upload is deferred.

- `POST /v1/listings/{id}/images` → `201`
  `{ storage_key, filename, content_type (image/jpeg|png|webp), byte_size>0, width?, height?, sha256 (64 lowercase-hex; non-hex → 422) }`
  Duplicate `(listing_id, sha256)` → `409`.
- `GET /v1/listings/{id}/images` → `200`

## Model versions

- `POST /v1/model-versions` → `201` — `{ name, version, config? }`; duplicate name+version → `409`
- `GET /v1/model-versions` → `200`

## Predictions (storage only — no inference in scope)

- `POST /v1/listings/{id}/predictions` → `201`
  `{ model_version_id, prediction_type (attribute_extraction|duplicate_detection|price_estimation|quality_check, default attribute_extraction), latency_ms?, attributes: [{ key, value|null, confidence 0..1, abstained=false, evidence: [{ source (text|image|multimodal|ocr), image_id? (must belong to listing), text_span? {field,start,end}, bounding_box? {x,y,w,h}, note? }] }] }`
  Invariants: `abstained=true ⇒ value=null`; `abstained=false ⇒ value!=null` (`422` otherwise); unknown model version → `404`.
  Evidence provenance enforced: `source=image` requires `image_id`; `source=text` and `source=ocr` require `text_span` with `field/start/end` (`422` otherwise); DB CHECK `ck_evidence_source` is the backstop (migration `0003` added `ocr`).
  Size caps (DoS guard): ≤100 attributes/prediction, ≤20 evidence/attribute (`422` otherwise).
- `GET /v1/listings/{id}/predictions` → `200`, newest first with attributes + evidence.

## Retrieval — executes the product pipeline (reviewer item 3)

- `POST /v1/retrieval/search` → `200`
  `{ query_text? (≤5000 chars), query_image_sha256? (64-hex), modality (text|image|multimodal, default multimodal), top_k (1..100, default 10) }`
  → `[{ listing_id, score, text_score?, image_score? }]` ranked desc; soft-deleted listings excluded; image modality dedupes multiple images per listing.
  `modality=image` without `query_image_sha256` → `422`; `text`/`multimodal` without blank-stripped `query_text` → `422`.
  Vectors are deterministic fixtures (`app/retrieval.py`, 64-dim) until a learned model lands; route, ranking, evidence, and latency shape are the production contract.

## Reviews (append-only)

- `POST /v1/listings/{id}/reviews` → `201`
  `{ decision (accepted|rejected|corrected), prediction_id? (must belong to listing), corrected_attributes? (required when corrected), note?, reviewer_id }`
- `GET /v1/listings/{id}/reviews` → `200`, chronological (complete decision history).
- No PUT/DELETE review routes exist by design.

## Health

- `GET /health` → `{ status: "ok" }` (no dependencies)
- `GET /ready` → `{ status: ready|not_ready, checks: { db, redis } }`; `503` unless DB ok and redis ok/skipped.
  Redis is `skipped` when `REDIS_URL` is empty (local/test); compose sets it so readiness gates on redis there.
