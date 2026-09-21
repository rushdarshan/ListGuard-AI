"""Phase 2 multimodal retrieval — RED tests (written BEFORE implementation).

Covers: text/image embeddings, pgvector storage helpers, FTS,
image similarity, combined retrieval, scores, evidence,
deterministic fixtures, latency, EXPLAIN ANALYZE.
Explicitly NOT covered: price prediction, attribute extraction.
"""

import pytest

from app.retrieval import (
    RetrievedItem,
    RetrievalEvidence,
    RetrievalResult,
    cosine_similarity,
    explain_analyze,
    full_text_search,
    image_embed,
    image_similarity_search,
    measure_latency,
    multimodal_retrieve,
    text_embed,
)


def test_text_embed_deterministic_and_normalized():
    a = text_embed("red bicycle")
    b = text_embed("red bicycle")
    assert a == b  # deterministic fixture
    norm = sum(x * x for x in a) ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-6)
    assert len(a) == 64


def test_text_embed_empty_raises():
    with pytest.raises(ValueError):
        text_embed("")


def test_image_embed_deterministic_and_normalized():
    a = image_embed(sha256="a" * 64)
    b = image_embed(sha256="a" * 64)
    assert a == b
    norm = sum(x * x for x in a) ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-6)


def test_image_embed_invalid_sha_raises():
    with pytest.raises(ValueError):
        image_embed(sha256="zzz")


def test_cosine_similarity_identical_is_one():
    v = text_embed("hello world")
    assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)


def test_cosine_similarity_range_and_mismatch_raises():
    v = text_embed("hello world")
    w = text_embed("totally different query xyz")
    s = cosine_similarity(v, w)
    assert -1.0 <= s <= 1.0
    with pytest.raises(ValueError):
        cosine_similarity(v, [1.0, 2.0])


def test_full_text_search_finds_and_empty_query():
    docs = [
        {"id": "1", "title": "Red bicycle", "description": "fast road bike"},
        {"id": "2", "title": "Blue kayak", "description": "river boat"},
    ]
    assert full_text_search("", docs) == []
    hits = full_text_search("bicycle", docs)
    assert [h["id"] for h in hits] == ["1"]


def test_image_similarity_search_ranked():
    query = image_embed(sha256="a" * 64)
    cands = [
        {"id": "far", "vector": image_embed(sha256="b" * 64)},
        {"id": "near", "vector": image_embed(sha256="a" * 64)},
    ]
    ranked = image_similarity_search(query, cands, top_k=2)
    assert ranked[0][0] == "near"
    assert ranked[0][1] >= ranked[1][1]


def test_multimodal_retrieve_combined_scores_and_evidence():
    docs = [
        {"id": "1", "title": "Red bicycle fast", "description": "", "image_sha256": "a" * 64},
        {"id": "2", "title": "Blue kayak", "description": "", "image_sha256": "b" * 64},
    ]
    res = multimodal_retrieve("red bicycle", image_sha256="a" * 64, docs=docs, top_k=2)
    assert isinstance(res, RetrievalResult)
    assert res.results[0].id == "1"
    assert res.results[0].score >= res.results[1].score
    ev: RetrievalEvidence = res.results[0].evidence
    assert ev.text_contribution + ev.image_contribution == pytest.approx(1.0, abs=1e-6)
    assert isinstance(res.results[0], RetrievedItem)


def test_multimodal_retrieve_text_only_mode():
    docs = [{"id": "1", "title": "Red bicycle", "description": ""}]
    res = multimodal_retrieve("red bicycle", docs=docs, top_k=1)
    assert res.results[0].evidence.image_contribution == pytest.approx(0.0)


def test_measure_latency_reports_ms():
    out, ms = measure_latency(lambda: text_embed("hello world probe"))
    assert len(out) == 64
    assert ms >= 0.0  # measured, never claimed


def test_explain_analyze_returns_plan_string():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("sqlite:///:memory:")
    with sessionmaker(bind=engine)() as db:
        plan = explain_analyze(db, "SELECT 1")
    assert isinstance(plan, str) and len(plan) > 0


def _sqlite_db():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.core.db import Base

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


def test_store_and_vector_search_roundtrip():
    from app import models as _models
    from app.retrieval import fts_search, store_text_embedding, text_vector_search

    db = _sqlite_db()
    row = _models.Listing(seller_id="s", title="Red bicycle", description="fast road bike")
    db.add(row)
    db.commit()
    store_text_embedding(db, row.id, text_embed("red bicycle"))
    hits = text_vector_search(db, text_embed("red bicycle"), top_k=5)
    assert hits[0][0] == row.id
    assert hits[0][1] == pytest.approx(1.0, abs=1e-6)
    assert fts_search(db, "bicycle") == [row.id]
    assert fts_search(db, "") == []
    with pytest.raises(ValueError):
        store_text_embedding(db, "nope", text_embed("red bicycle"))
    db.close()


def test_text_vector_search_skips_unembedded_listings():
    # Regression: rows with no stored vector must be skipped, not crash with
    # `list(None)` (live 500: ORM wrote the TEXT string 'null', which passes
    # `IS NOT NULL` and deserializes to None). Covers both the SQL filter
    # (real NULL) and the deserialized-None guard (legacy 'null' marker).
    from sqlalchemy import text as sql_text

    from app import models as _models
    from app.retrieval import store_text_embedding, text_embed, text_vector_search

    db = _sqlite_db()
    plain = _models.Listing(seller_id="s", title="No vector here", description="plain")
    db.add(plain)
    db.commit()
    assert text_vector_search(db, text_embed("anything"), top_k=5) == []
    db.execute(sql_text(
        "INSERT INTO listings (id, seller_id, title, status, created_at, updated_at, text_embedding) "
        "VALUES ('legacy1','s','legacy row','draft','2026-01-01T00:00:00','2026-01-01T00:00:00','null')"))
    db.commit()
    hits = text_vector_search(db, text_embed("anything"), top_k=5)
    assert hits == []
    store_text_embedding(db, plain.id, text_embed("vector here"))
    ranked = text_vector_search(db, text_embed("vector here"), top_k=5)
    assert [lid for lid, _ in ranked] == [plain.id]  # legacy row skipped, vector row hit
    db.close()


def test_store_image_embedding_roundtrip():
    from app import models as _models
    from app.retrieval import store_image_embedding

    db = _sqlite_db()
    listing = _models.Listing(seller_id="s", title="Bike")
    db.add(listing)
    db.commit()
    img = _models.ListingImage(
        listing_id=listing.id, storage_key="k", filename="f.jpg",
        content_type="image/jpeg", byte_size=10, sha256="c" * 64,
    )
    db.add(img)
    db.commit()
    store_image_embedding(db, img.id, image_embed("c" * 64))
    fetched = db.get(_models.ListingImage, img.id)
    assert fetched is not None and fetched.image_embedding is not None
    with pytest.raises(ValueError):
        store_image_embedding(db, "nope", image_embed("c" * 64))
    db.close()


def test_postgres_sql_uses_pgvector_and_fts():
    from app.retrieval import _PG_FTS_SQL, _PG_VECTOR_SQL

    assert "<=>" in _PG_VECTOR_SQL and "text_embedding" in _PG_VECTOR_SQL
    assert "deleted_at IS NULL" in _PG_VECTOR_SQL  # soft-deleted rows stay buried
    assert "to_tsvector" in _PG_FTS_SQL and "plainto_tsquery" in _PG_FTS_SQL and "ts_rank" in _PG_FTS_SQL


def test_multimodal_rejects_bad_weight_and_empty_docs():
    from app.retrieval import multimodal_retrieve

    with pytest.raises(ValueError):
        multimodal_retrieve("bike", docs=[], text_weight=2.0)
    res = multimodal_retrieve("bike", docs=[], top_k=5)
    assert res.results == [] and res.latency_ms >= 0.0


def test_zero_vectors_and_untokenizable_text_raise():
    from app.retrieval import cosine_similarity, text_embed

    with pytest.raises(ValueError):
        text_embed("!!!")
    with pytest.raises(ValueError):
        cosine_similarity([0.0] * 64, [0.0] * 64)


def test_vector_search_excludes_soft_deleted():
    from datetime import datetime, timezone

    from app import models as _models
    from app.retrieval import fts_search, store_text_embedding, text_embed, text_vector_search

    db = _sqlite_db()
    live = _models.Listing(seller_id="s", title="Red bicycle", description="fast")
    gone = _models.Listing(seller_id="s", title="Red bicycle", description="fast")
    db.add_all([live, gone])
    db.commit()
    store_text_embedding(db, live.id, text_embed("red bicycle"))
    store_text_embedding(db, gone.id, text_embed("red bicycle"))
    gone.deleted_at = datetime.now(timezone.utc)
    db.commit()
    assert [hit[0] for hit in text_vector_search(db, text_embed("red bicycle"))] == [live.id]
    assert fts_search(db, "bicycle") == [live.id]
    db.close()


def test_multimodal_survives_empty_and_none_text():
    docs = [
        {"id": "empty", "title": "", "description": ""},
        {"id": "none", "title": None, "description": None},
        {"id": "punct", "title": "!!!", "description": "???"},
        {"id": "good", "title": "Red bicycle", "description": ""},
    ]
    res = multimodal_retrieve("bicycle", docs=docs, top_k=4)
    assert len(res.results) == 4  # no crash; weak docs score ~0, good doc wins
    assert res.results[0].id == "good"
    by_id = {r.id: r for r in res.results}
    for weak in ("empty", "none", "punct"):
        assert by_id[weak].score == pytest.approx(0.0)
        ev = by_id[weak].evidence
        assert ev.text_contribution + ev.image_contribution == pytest.approx(1.0, abs=1e-6)


def test_full_text_search_ignores_none_fields():
    docs = [{"id": "1", "title": "Bike", "description": None}]
    assert [h["id"] for h in full_text_search("bike", docs)] == ["1"]
    assert full_text_search("none", docs) == []  # no literal "None" token


def test_store_embeddings_reject_bad_vectors():
    from app import models as _models
    from app.retrieval import store_image_embedding, store_text_embedding

    db = _sqlite_db()
    listing = _models.Listing(seller_id="s", title="Bike")
    db.add(listing)
    db.commit()
    img = _models.ListingImage(
        listing_id=listing.id, storage_key="k", filename="f.jpg",
        content_type="image/jpeg", byte_size=10, sha256="d" * 64,
    )
    db.add(img)
    db.commit()
    with pytest.raises(ValueError):
        store_text_embedding(db, listing.id, [1.0] * 32)  # wrong dim
    with pytest.raises(ValueError):
        store_text_embedding(db, listing.id, [float("nan")] * 64)  # non-finite
    with pytest.raises(ValueError):
        store_image_embedding(db, img.id, [1.0] * 128)  # wrong dim
    db.close()


def test_top_k_must_be_positive():
    from app.retrieval import (
        fts_search,
        full_text_search,
        image_similarity_search,
        multimodal_retrieve,
        text_vector_search,
    )

    docs = [{"id": "1", "title": "Bike", "description": ""}]
    with pytest.raises(ValueError):
        full_text_search("bike", docs, top_k=0)
    with pytest.raises(ValueError):
        full_text_search("bike", docs, top_k=-1)
    with pytest.raises(ValueError):
        image_similarity_search([1.0] * 64, [{"id": "1", "vector": [1.0] * 64}], top_k=0)
    with pytest.raises(ValueError):
        multimodal_retrieve("bike", docs, top_k=0)
    db = _sqlite_db()
    with pytest.raises(ValueError):
        text_vector_search(db, [1.0] * 64, top_k=-2)
    with pytest.raises(ValueError):
        fts_search(db, "bike", top_k=0)
    db.close()


def test_punctuation_only_query_returns_empty():
    from app.retrieval import fts_search

    docs = [{"id": "1", "title": "Bike", "description": ""}]
    assert full_text_search("!!!", docs) == []
    db = _sqlite_db()
    assert fts_search(db, "!!!") == []
    db.close()


def test_multimodal_malformed_doc_sha_degrades_to_text_only():
    docs = [
        {"id": "bad", "title": "Red bicycle", "description": "", "image_sha256": "zzz"},
        {"id": "good", "title": "Red bicycle", "description": "", "image_sha256": "a" * 64},
    ]
    res = multimodal_retrieve("bicycle", image_sha256="a" * 64, docs=docs, top_k=2)
    assert len(res.results) == 2 and res.results[0].id == "good"
    bad = next(r for r in res.results if r.id == "bad")
    assert bad.evidence.image_contribution == pytest.approx(0.0)
    with pytest.raises(ValueError):  # malformed QUERY sha stays a hard error
        multimodal_retrieve("bike", image_sha256="not-hex", docs=docs)


def test_top_k_rejects_bool():
    with pytest.raises(ValueError):
        full_text_search("bike", [{"id": "1", "title": "x", "description": ""}], top_k=True)


def test_explain_analyze_rejects_non_select():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("sqlite:///:memory:")
    with sessionmaker(bind=engine)() as db:
        with pytest.raises(ValueError):
            explain_analyze(db, "DELETE FROM listings")
