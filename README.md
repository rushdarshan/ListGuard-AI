# ListGuard AI

Multimodal marketplace listing-intelligence platform. Sellers upload product
images plus a rough description; ListGuard extracts **grounded product
attributes** (brand, category, color, condition) with confidence scores and
evidence, **abstains** when support is thin, flags **duplicates**,
**contradictions**, and **pricing anomalies**, and routes everything through a
**human-review workflow** with a complete decision history.

> **Status: backend foundation + deterministic pipelines (v0.1.0).**
> Learned models (CLIP/SigLIP, OCR, price estimation) are explicitly deferred.
> Retrieval, attribute extraction, risk detection, and eval run on
> deterministic fixture implementations behind production-shaped seams, so the
> API, ranking, evidence, and latency contracts are real and tested while the
> weights are still placeholders. There is no Next.js frontend yet — review
> happens in a static dashboard served by the API. There is no auth yet —
> **run dev-local only.**

## What works today

- **Listing CRUD** with soft delete (`deleted_at`; default queries exclude)
- **Image metadata registration** (not upload — registers an object-storage
  key + sha256; binary upload is deferred)
- **Prediction storage** with `prediction_type`
  (`attribute_extraction | duplicate_detection | price_estimation | quality_check`),
  per-attribute confidence, abstention invariant, and first-class evidence
  (`source | image_id | text_span | bounding_box`), backstopped by DB CHECKs
- **Human review**: append-only `review_events`
  (`accepted | rejected | corrected`), correction payloads, chronological history
- **Retrieval**: `POST /v1/retrieval/search` over text vectors, image vectors,
  and full-text search, with weighted multimodal fusion, per-hit scores, and
  measured latency
- **Attribute extraction**: closed-vocabulary grounded matching with
  abstention + contradiction flags and adversarial tests
- **Risk engine**: contradictions, missing attributes, duplicate descriptions,
  tiered abnormal pricing, unusable images, seller-claim keywords — every
  finding carries category, severity, confidence, evidence, and recommended action
- **Reviewer dashboard** (static, served at `/web`) with risk inputs,
  price-interval derivation, and decision history
- **Eval harness** for duplicate detection (Recall@K / MRR / mAP against
  hand-computed values, modality ablation, dataset + model versioning)
- **Health**: `/health` (liveness) + `/ready` (DB + Redis checks)
- **Migrations**: Alembic `0001` (schema) → `0002` (pgvector embeddings + HNSW)
  → `0003` (OCR evidence source)

## Architecture

```
apps/api        FastAPI backend (Python >= 3.12)
  app/          routers, schemas, models, retrieval, attributes, risk,
                dashboard derivation, eval harness
  alembic/      migrations 0001–0003 (Postgres + SQLite paths)
  tests/        pytest suite (see Verification)
apps/web        reviewer dashboard (dashboard.html + dashboard.ts → dashboard.js),
                Playwright E2E (e2e_dashboard.py), node unit tests
infra/          docker-compose (pgvector:pg16 + redis:7 + api),
                postgres init (vector, pg_trgm, pgcrypto)
docs/           api-contracts.md, schema.md, testing/phase2-retrieval.tdd.md
```

PostgreSQL + pgvector is the production store (HNSW cosine indexes, FTS GIN);
SQLite is the test/local fallback (JSON columns, token-overlap FTS).
Redis is probed at `/ready` and reserved for future background workers.

## Quickstart

### Docker (production-like)

```bash
cp .env.example infra/.env   # dev defaults only; generate real secrets for prod
docker compose -f infra/docker-compose.yml up --build
# API migrates itself on boot (alembic upgrade head), then serves on :8000
curl localhost:8000/health
```

### Local (SQLite, no Docker)

```bash
cd apps/api
pip install -e ".[test]"        # fastapi, uvicorn, sqlalchemy, alembic,
                                # pydantic, psycopg, redis, httpx, pgvector
python -m alembic upgrade head
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
# Dashboard: http://127.0.0.1:8000/web/dashboard.html
```

## API (v1)

| Method & path | Notes |
|---|---|
| `GET /health` | liveness |
| `GET /ready` | DB + Redis checks; `503` when not ready |
| `GET/POST /v1/listings` | `q` (LIKE-escaped), `status` (strict), pagination; soft delete |
| `GET/PATCH/DELETE /v1/listings/{id}` | `204` on delete (soft) |
| `POST /v1/listings/{id}/images` | **metadata registration**, `201`; dup `(listing, sha256)` → `409` |
| `GET /v1/listings/{id}/images` | `200` |
| `POST/GET /v1/model-versions` | `UNIQUE(name, version)`; dup → `409` |
| `POST /v1/listings/{id}/predictions` | storage only, no inference; abstention invariant `422`; unknown model → `404` |
| `GET /v1/listings/{id}/predictions` | newest first, attributes + evidence |
| `POST /v1/listings/{id}/reviews` | `accepted\|rejected\|corrected`; corrections required when `corrected` |
| `GET /v1/listings/{id}/reviews` / `/history` | chronological decision trail |
| `POST /v1/retrieval/search` | `{query_text?, query_image_sha256?, modality, top_k}` → ranked hits with scores |

Full contracts: [`docs/api-contracts.md`](docs/api-contracts.md).
Schema + deferrals: [`docs/schema.md`](docs/schema.md).

## Verification

```bash
cd apps/api
python -m pytest tests -q --cov=app --cov-branch --cov-report=term-missing
python -m ruff check app tests alembic
python -m mypy app --ignore-missing-imports
```

Last verified: **108/108 tests, 100% statement + 100% branch coverage**
(1006 stmts / 234 branches), ruff and mypy clean. Web: `dashboard.test.ts`
(8 tests) + Playwright E2E against a live server.

## Measurement honesty

- Vectors are **deterministic fixtures** (signed-hash text, hash-derived
  image, 64-dim), not learned — duplicate-detection scores (e.g. Recall 1.0
  on fixtures) measure pipeline correctness on trivially-separable data, not
  model quality.
- Scores named `confidence` on fixture paths are **heuristic**, not calibrated;
  calibration is roadmap work.
- Latency is measured per call (`measure_latency`) but only meaningful on your
  own hardware — no latency numbers are claimed here.

## Roadmap (explicitly deferred)

Learned CLIP/SigLIP + text-embedding weights (swap in at the
`text_embed`/`image_embed` seam) · real OCR · price-estimation + quality-check
models · embedding-metadata table · Next.js dashboard · auth/RBAC ·
rate limiting · background workers · monitoring (confidence/latency/drift/error
taxonomy) · DB-enforced append-only trigger. The API contract is frozen; none
of these require schema churn for predictions, evidence, or review events.
