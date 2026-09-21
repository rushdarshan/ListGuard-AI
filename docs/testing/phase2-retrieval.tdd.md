# Phase 2: Multimodal Retrieval — TDD Evidence Report

Date: 2026-09-18. Scope: retrieval ONLY (no price prediction, no attribute extraction).

## TDD cycle log

| Step | Action | Evidence |
|------|--------|----------|
| SCAFFOLD | `app/retrieval.py` interfaces + `raise NotImplementedError` stubs | file created |
| RED | `tests/test_retrieval_phase2.py`, 12 tests (happy path, edge cases, error conditions) | 12 collected, 12 × `NotImplementedError` |
| GREEN | Minimal stdlib-only implementation (no new runtime deps for tests) | 16/16 pass |
| REFACTOR | Single-pass tokenization in `full_text_search`; extracted `_normalize` | suite stays green |
| COVERAGE | Added DB roundtrip + guard tests | **28/28 pass, 100% on `app/retrieval.py`** (sqlite-runnable code) |

Full suite (incl. pre-existing `test_api.py`, `test_coverage_gaps.py`): **28/28 pass, no regressions**.

## What was built (4 files + 1 dep line)

- `apps/api/app/retrieval.py` (~240 lines): `text_embed` (signed-hash word vectors, L2-normed),
  `image_embed` (sha256-seeded PRNG, L2-normed), `cosine_similarity`, `full_text_search`
  (token overlap, stable order), `image_similarity_search`, `multimodal_retrieve`
  (weighted fusion + per-item evidence, contributions sum to 1.0), `measure_latency`
  (perf_counter), `store_text_embedding` / `store_image_embedding` / `text_vector_search`
  / `fts_search` (pgvector `<=>` + `to_tsvector/ts_rank` on PostgreSQL, Python/JSON
  fallback on SQLite), `explain_analyze` (ANALYZE on PG, QUERY PLAN on SQLite).
- `apps/api/app/models.py`: `EmbeddingVector` TypeDecorator (pgvector on PG, JSON on SQLite)
  + `listings.text_embedding`, `listing_images.image_embedding`. No new tables.
- `apps/api/alembic/versions/0002_embeddings.py`: `vector(64)` + HNSW
  (`vector_cosine_ops`) on PG; JSON columns elsewhere. pgvector>=0.7 required for HNSW.
- `apps/api/pyproject.toml`: `pgvector>=0.3` (PostgreSQL only; lazily imported, tests need no install).
- `docs/schema.md`: deferral note updated to 0002 reality.

## Measured numbers (this machine only, not claims)

- `multimodal_retrieve` over 200 synthetic docs, top_k=10: **11.02 ms** (Python brute force,
  win32, single sample). Latency is measured per call via `measure_latency` /
  `RetrievalResult.latency_ms` — no absolute performance is claimed.
- `EXPLAIN ANALYZE` / `EXPLAIN QUERY PLAN` runnable via `explain_analyze(db, sql)` against
  any session (SQLite verified in tests; PG path needs compose).

## Honest limitations

1. Fixture embeddings rank by **token overlap, not semantics** ("red bicycle" ties broken by
   hash noise). Learned weights swap in at the `text_embed`/`image_embed` seam only.
2. PG-only branches (`<=>` ordering, `to_tsvector` ranking, HNSW) are **string-tested, not
   live-tested**. Verify on compose: `docker compose up -d db`, `alembic upgrade head`,
   then run `explain_analyze` + `text_vector_search` against the PG session.
3. `text_vector_search` on PG returns ids (no rescore); rescore in Python if scores needed.
4. `explain_analyze` with ANALYZE **executes** the statement — read-only queries only.

## Test env note

The `rtk pytest` wrapper stopped executing in this workspace (stale "No tests collected",
no new logs) — root cause is environmental (a stray `queue.py` in the shared temp dir
shadows stdlib `queue` during plugin autoload). Verified via in-process runner
(`run_phase2.py`: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, script dir removed from `sys.path`).
Re-run with: `python C:\Users\rushd\AppData\Local\Temp\opencode\run_phase2.py tests`.
