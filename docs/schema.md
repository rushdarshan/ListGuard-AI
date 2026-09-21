# Schema

Migrations `apps/api/alembic/versions/0001_initial.py` + `0002_embeddings.py` + `0003_evidence_ocr.py` are the source of truth.
`infra/postgres/init.sql` creates `vector`, `pg_trgm`, `pgcrypto` extensions.

## Tables

- `listings(id, seller_id, title, description, category, status, deleted_at, created_at, updated_at)`
  `status ∈ draft|pending_review|approved|rejected`. Soft delete via `deleted_at`.
  Indexes: `(seller_id)`, `(status)`, Postgres-only FTS GIN + title trgm GIN.
- `listing_images(id, listing_id→listings CASCADE, storage_key, filename, content_type, byte_size>0, width?, height?, sha256, created_at)`
  Metadata REGISTRATION only — no bytes stored. `UNIQUE(listing_id, sha256)`.
- `model_versions(id, name, version, config JSON, created_at)` — `UNIQUE(name, version)`.
- `predictions(id, listing_id→listings CASCADE, model_version_id→model_versions RESTRICT, prediction_type, latency_ms?, created_at)`
  `prediction_type ∈ attribute_extraction|duplicate_detection|price_estimation|quality_check`. Index `(listing_id, created_at)`.
- `prediction_attributes(id, prediction_id→predictions CASCADE, key, value?, confidence 0..1, abstained)`
  `UNIQUE(prediction_id, key)`. Invariant: `(abstained=false AND value NOT NULL) OR (abstained=true AND value NULL)` (DB CHECK + API 422).
- `prediction_evidence(id, prediction_attribute_id→prediction_attributes CASCADE, source, image_id→listing_images SET NULL?, text_span JSON? {field,start,end}, bounding_box JSON? {x,y,w,h}, note?, created_at)`
  `source ∈ text|image|multimodal|ocr` (`ocr` added by migration `0003`). First-class provenance: confidence lives on the attribute, model version on the prediction, image/text grounding here.
- `review_events(id, listing_id→listings CASCADE, prediction_id→predictions SET NULL?, reviewer_id, decision, corrected_attributes JSON?, note?, created_at)`
  `decision ∈ accepted|rejected|corrected`. Append-only: no update/delete routes; history endpoint reads chronologically.

## Approved deferrals (explicitly NOT in schema yet)

- Embedding vector columns: ADDED in migration 0002 as 64-dim deterministic fixture vectors
  (`listings.text_embedding`, `listing_images.image_embedding`; pgvector `vector(64)` + HNSW on
  PostgreSQL, JSON on SQLite). Learned weights (CLIP/SigLIP/text-embedding) swap in at the
  `text_embed`/`image_embed` seam; see `docs/testing/phase2-retrieval.tdd.md`.
- No inference, OCR, duplicate/price/quality logic — prediction rows are written by tests/seed scripts only.
- No auth, object storage, workers, dashboard, monitoring.
