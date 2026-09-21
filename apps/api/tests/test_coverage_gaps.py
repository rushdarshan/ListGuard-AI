"""Gap-fill tests for coverage: get_db, redis_status branches, /ready 503 paths."""

import pytest

import app.core.db as dbmod
import app.core.redis_client as redis_mod
from app.core.db import get_db


def test_get_db_yields_and_closes():
    gen = get_db()
    db = next(gen)
    assert db is not None
    with pytest.raises(StopIteration):
        next(gen)


def test_redis_status_branches(monkeypatch):
    monkeypatch.setattr(redis_mod.settings, "redis_url", "")
    assert redis_mod.redis_status()["state"] == "skipped"

    monkeypatch.setattr(redis_mod.settings, "redis_url", "redis://localhost:6379/0")

    class FakeClient:
        def ping(self):
            return True

        def close(self):
            return True

    monkeypatch.setattr(redis_mod.redis.Redis, "from_url", lambda *a, **k: FakeClient())
    assert redis_mod.redis_status()["state"] == "ok"

    def boom(*a, **k):
        raise ConnectionError("down")

    monkeypatch.setattr(redis_mod.redis.Redis, "from_url", boom)
    assert redis_mod.redis_status()["state"] == "fail"


def test_ready_503_on_db_failure(client, monkeypatch):
    class BadEngine:
        def connect(self):
            raise RuntimeError("db down")

    monkeypatch.setattr(dbmod, "engine", BadEngine())
    r = client.get("/ready")
    assert r.status_code == 503
    assert r.json()["checks"]["db"] == "fail"  # generic: no driver text leaks to callers


def test_ready_503_on_redis_failure(client, monkeypatch):
    import app.main as main_mod

    monkeypatch.setattr(main_mod, "redis_status", lambda: {"state": "fail", "detail": "down"})
    r = client.get("/ready")
    assert r.status_code == 503


def test_embedding_vector_uses_pgvector_on_postgres():
    from sqlalchemy.dialects import postgresql

    from app.models import EmbeddingVector
    from app.retrieval import EMBEDDING_DIM

    assert str(EmbeddingVector().load_dialect_impl(postgresql.dialect())) == f"VECTOR({EMBEDDING_DIM})"
