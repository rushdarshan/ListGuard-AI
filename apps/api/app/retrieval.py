"""Phase 2: multimodal retrieval.

Deterministic, dependency-free baseline (stdlib only):
- Text vectors: signed-hashing word vectorizer, L2-normalized.
- Image vectors: hash-derived deterministic vectors (no PRNG, no model weights).
- Fusion: weighted text + image cosine scores with per-item evidence.
- Storage: ``EmbeddingVector`` columns (see app.models) — pgvector ``vector``
  on PostgreSQL, JSON on SQLite. Live-model weights (CLIP/SigLIP/...) swap in
  by replacing ``text_embed``/``image_embed`` only; the storage, fusion and
  evidence code is model-agnostic.
- Latency is MEASURED with perf_counter, never claimed.
- ``explain_analyze`` runs EXPLAIN (ANALYZE on postgres, QUERY PLAN on
  sqlite). NOTE: ANALYZE executes the statement — read-only queries only.

Explicitly NOT in scope: price prediction, attribute extraction.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from sqlalchemy import text as sql_text

EMBEDDING_DIM = 64
Vector = list[float]

T = TypeVar("T")

_TOKEN = re.compile(r"[a-z0-9]+")
_SHA = re.compile(r"^[0-9a-fA-F]{64}$")

# Actual SQL executed on PostgreSQL (unit-tested as strings; live runs need PG).
_PG_VECTOR_SQL = (
    "SELECT id, text_embedding <=> CAST(:q AS vector) AS dist FROM listings "
    "WHERE deleted_at IS NULL AND text_embedding IS NOT NULL "
    "ORDER BY dist ASC LIMIT :k"
)
_PG_FTS_SQL = (
    "SELECT id FROM listings WHERE deleted_at IS NULL AND "
    "to_tsvector('english', title || ' ' || coalesce(description, '')) "
    "@@ plainto_tsquery('english', :q) "
    "ORDER BY ts_rank(to_tsvector('english', title || ' ' || coalesce(description, '')), "
    "plainto_tsquery('english', :q)) DESC LIMIT :k"
)


@dataclass
class RetrievalEvidence:
    """Explains each modality's contribution; contributions sum to 1.0."""

    text_contribution: float
    image_contribution: float
    text_score: float
    image_score: float | None


@dataclass
class RetrievedItem:
    id: str
    score: float
    evidence: RetrievalEvidence


@dataclass
class RetrievalResult:
    results: list[RetrievedItem] = field(default_factory=list)
    latency_ms: float = 0.0


def _normalize(vec: Vector) -> Vector:
    norm = sum(v * v for v in vec) ** 0.5
    if norm == 0:  # pragma: no cover - defensive; hash/prng vectors are never exactly zero
        raise ValueError("zero vector has no direction")
    return [v / norm for v in vec]


def text_embed(text: str, dim: int = EMBEDDING_DIM) -> Vector:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    toks = _TOKEN.findall(text.lower())
    if not toks:
        raise ValueError("text has no indexable tokens")
    vec = [0.0] * dim
    for tok in toks:  # ponytail: signed hashing; deterministic, no deps/model download
        digest = hashlib.sha256(tok.encode()).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        vec[idx] += 1.0 if digest[4] % 2 == 0 else -1.0
    return _normalize(vec)


def image_embed(sha256: str, dim: int = EMBEDDING_DIM) -> Vector:
    if not isinstance(sha256, str) or not _SHA.match(sha256):
        raise ValueError("sha256 must be 64 hex chars")
    # ponytail: hash-derived fixture, no PRNG/no model weights.
    # Deliberately NOT random.gauss: gauss output is not guaranteed stable
    # across Python versions, while hashlib is.
    vec = [0.0] * dim
    for i in range(dim):
        digest = hashlib.sha256(f"{sha256}:{i}".encode()).digest()
        vec[i] = (1.0 if digest[0] % 2 == 0 else -1.0) * (digest[1] / 255.0 + 0.5)
    return _normalize(vec)


def cosine_similarity(a: Vector, b: Vector) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("vectors must be non-empty and same length")
    denom = (sum(v * v for v in a) ** 0.5) * (sum(v * v for v in b) ** 0.5)
    if denom == 0:
        raise ValueError("zero vector has no direction")
    return max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b)) / denom))


def _doc_text(d: dict[str, Any]) -> str:
    # `or ''`: a present-but-None field must not become the literal token "None".
    return f"{d.get('title') or ''} {d.get('description') or ''}"


def _check_vector(vector: Vector) -> list[float]:
    vals = list(vector)
    if len(vals) != EMBEDDING_DIM or not all(math.isfinite(v) for v in vals):
        raise ValueError(f"vector must be {EMBEDDING_DIM} finite floats")
    return vals


def _check_top_k(top_k: int) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")


def full_text_search(query: str, docs: list[dict[str, Any]], top_k: int = 10) -> list[dict[str, Any]]:
    _check_top_k(top_k)
    if not query.strip():
        return []
    qtoks = set(_TOKEN.findall(query.lower()))
    if not qtoks:
        return []  # non-blank but untokenizable query (e.g. "!!!"): avoids division by zero
    doc_toks = [(i, d, set(_TOKEN.findall(_doc_text(d).lower()))) for i, d in enumerate(docs)]
    scored = [(len(qtoks & dt) / len(qtoks), i, d) for i, d, dt in doc_toks if qtoks & dt]
    scored.sort(key=lambda t: (-t[0], t[1]))  # stable: score desc, input order on ties
    return [d for _, _, d in scored[:top_k]]


def image_similarity_search(
    query_vector: Vector, candidates: list[dict[str, Any]], top_k: int = 10
) -> list[tuple[str, float]]:
    _check_top_k(top_k)
    ranked = sorted(
        ((c["id"], cosine_similarity(query_vector, list(c["vector"]))) for c in candidates),
        key=lambda t: t[1],
        reverse=True,
    )
    return ranked[:top_k]


def multimodal_retrieve(
    query_text: str,
    docs: list[dict[str, Any]],
    image_sha256: str | None = None,
    top_k: int = 10,
    text_weight: float = 0.7,
) -> RetrievalResult:
    if not 0.0 <= text_weight <= 1.0:
        raise ValueError("text_weight must be in [0, 1]")
    _check_top_k(top_k)
    started = time.perf_counter()
    q_text = text_embed(query_text)
    q_img = image_embed(image_sha256) if image_sha256 else None
    items = []
    for d in docs:
        try:
            t_score = cosine_similarity(q_text, text_embed(_doc_text(d)))
        except ValueError:
            t_score = 0.0  # empty/untokenizable doc: no text evidence, item stays listed
        try:
            i_score = (
                cosine_similarity(q_img, image_embed(d["image_sha256"]))
                if q_img and d.get("image_sha256")
                else None
            )
        except ValueError:
            i_score = None  # malformed doc sha: text-only scoring for that item
        if i_score is None:
            combined, t_contrib, i_contrib = t_score, 1.0, 0.0
        else:
            combined = text_weight * t_score + (1 - text_weight) * i_score
            t_contrib, i_contrib = text_weight, 1 - text_weight
        items.append(
            RetrievedItem(
                id=d["id"],
                score=combined,
                evidence=RetrievalEvidence(t_contrib, i_contrib, t_score, i_score),
            )
        )
    items.sort(key=lambda r: r.score, reverse=True)
    return RetrievalResult(results=items[:top_k], latency_ms=(time.perf_counter() - started) * 1000.0)


def measure_latency(fn: Callable[[], T]) -> tuple[T, float]:
    started = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - started) * 1000.0


# ---- DB-backed retrieval (pgvector/FTS on postgres, portable fallback on sqlite) ----


def store_text_embedding(db: Any, listing_id: str, vector: Vector) -> Any:
    from . import models as _models  # ponytail: lazy import, avoids models<->retrieval cycle

    row = db.get(_models.Listing, str(listing_id))
    if row is None:
        raise ValueError("listing not found")
    row.text_embedding = _check_vector(vector)
    db.commit()
    return row


def store_image_embedding(db: Any, image_id: str, vector: Vector) -> Any:
    from . import models as _models

    row = db.get(_models.ListingImage, str(image_id))
    if row is None:
        raise ValueError("image not found")
    row.image_embedding = _check_vector(vector)
    db.commit()
    return row


def text_vector_search(db: Any, query_vector: Vector, top_k: int = 10) -> list[tuple[str, float]]:
    """Nearest listings. Postgres score is negative Euclidean distance
    (higher = closer); sqlite score is cosine similarity. Rank order agrees
    across backends; scales differ, so threshold on ranks, not raw scores."""
    from . import models as _models

    _check_top_k(top_k)
    _check_vector(query_vector)  # both backends reject NaN/inf uniformly

    if db.get_bind().dialect.name == "postgresql":  # pragma: no cover - needs live PG; SQL verified by string test
        rows = db.execute(
            sql_text(_PG_VECTOR_SQL),
            {"q": "[" + ",".join(str(v) for v in query_vector) + "]", "k": top_k},
        ).all()
        return [(r[0], -r[1]) for r in rows]  # negative distance: higher = closer
    rows = (
        db.query(_models.Listing)
        .filter(_models.Listing.deleted_at.is_(None), _models.Listing.text_embedding.is_not(None))
        .all()
    )
    ranked = sorted(
        # Belt-and-braces: rows written before the EmbeddingVector None fix
        # can hold legacy non-NULL marker values that deserialize to None.
        ((r.id, cosine_similarity(query_vector, list(r.text_embedding))) for r in rows
         if r.text_embedding),
        key=lambda t: t[1],
        reverse=True,
    )
    return ranked[:top_k]


def fts_search(db: Any, query: str, top_k: int = 10) -> list[str]:
    """Full-text listing ids. Postgres uses to_tsvector/ts_rank; sqlite reuses token overlap."""
    from . import models as _models

    _check_top_k(top_k)
    if not query.strip():
        return []
    if db.get_bind().dialect.name == "postgresql":  # pragma: no cover - needs live PG; SQL verified by string test
        return [
            r[0] for r in db.execute(sql_text(_PG_FTS_SQL), {"q": query, "k": top_k}).all()
        ]
    rows = db.query(_models.Listing).filter(_models.Listing.deleted_at.is_(None)).all()
    docs = [{"id": r.id, "title": r.title, "description": r.description or ""} for r in rows]
    return [d["id"] for d in full_text_search(query, docs, top_k=top_k)]


def explain_analyze(db: Any, sql: str) -> str:
    """Run EXPLAIN on read-only ``sql`` and return the plan as text (measured, not claimed)."""
    first = sql.strip().split(None, 1)
    if not first or first[0].lower() != "select":
        # EXPLAIN ANALYZE executes on Postgres: never allow non-SELECT statements.
        raise ValueError("explain_analyze accepts read-only SELECT statements only")
    dialect = db.get_bind().dialect.name
    stmt = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) {sql}" if dialect == "postgresql" else f"EXPLAIN QUERY PLAN {sql}"
    return "\n".join(str(r[0]) for r in db.execute(sql_text(stmt)).all())
