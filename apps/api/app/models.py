"""SQLAlchemy models. Portable across SQLite (tests) and PostgreSQL (prod).

Approved modification #1: NO hard-coded embedding dims from model selection.
Phase 2 added 64-dim deterministic fixture vectors (migration 0002):
listings.text_embedding + listing_images.image_embedding via EmbeddingVector
(pgvector on PostgreSQL, JSON on SQLite).
Approved modification #3: predictions carry prediction_type.
Approved modification #4: evidence is first-class (prediction_evidence table)
with structured provenance, not arbitrary JSON.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, CheckConstraint, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator

from .core.db import Base

UUID_LEN = 36


class EmbeddingVector(TypeDecorator):
    """Phase 2 storage: pgvector ``vector`` on PostgreSQL, JSON on SQLite/tests.

    pgvector is imported lazily so SQLite test runs need no extra dependency.
    Dim must match the migration (0002) and retrieval.EMBEDDING_DIM.
    """

    impl = JSON
    cache_ok = True

    def process_bind_param(self, value, dialect):
        # Without this, None flows into the JSON bind processor and is stored
        # as the TEXT string 'null' (NOT NULL) on SQLite — which then slips
        # past `IS NOT NULL` filters and crashes vector search with
        # `list(None)`. Force real SQL NULL for missing embeddings.
        return None if value is None else value

    def load_dialect_impl(self, dialect):
        from .retrieval import EMBEDDING_DIM

        if dialect.name == "postgresql":
            from pgvector.sqlalchemy import Vector

            return dialect.type_descriptor(Vector(EMBEDDING_DIM))
        return dialect.type_descriptor(JSON())


def new_uuid() -> str:
    return str(uuid.uuid4())


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Listing(Base):
    __tablename__ = "listings"

    id: Mapped[str] = mapped_column(String(UUID_LEN), primary_key=True, default=new_uuid)
    seller_id: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    category: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="draft")
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    text_embedding: Mapped[list | None] = mapped_column(EmbeddingVector, nullable=True)  # Phase 2
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    __table_args__ = (CheckConstraint("status IN ('draft','pending_review','approved','rejected')", name="ck_listings_status"),)

    images: Mapped[list["ListingImage"]] = relationship(cascade="all, delete-orphan", back_populates="listing")
    predictions: Mapped[list["Prediction"]] = relationship(cascade="all, delete-orphan", back_populates="listing")
    review_events: Mapped[list["ReviewEvent"]] = relationship(cascade="all, delete-orphan", back_populates="listing")


class ListingImage(Base):
    """Metadata REGISTRATION for an image already in object/local storage.

    This table never receives binary bytes. See POST /v1/listings/{id}/images.
    """

    __tablename__ = "listing_images"

    id: Mapped[str] = mapped_column(String(UUID_LEN), primary_key=True, default=new_uuid)
    listing_id: Mapped[str] = mapped_column(String(UUID_LEN), ForeignKey("listings.id", ondelete="CASCADE"), nullable=False)
    storage_key: Mapped[str] = mapped_column(Text, nullable=False)
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    content_type: Mapped[str] = mapped_column(Text, nullable=False)
    byte_size: Mapped[int] = mapped_column(Integer, nullable=False)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    image_embedding: Mapped[list | None] = mapped_column(EmbeddingVector, nullable=True)  # Phase 2
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        UniqueConstraint("listing_id", "sha256", name="uq_images_listing_sha"),
        CheckConstraint("byte_size > 0", name="ck_images_bytes"),
    )

    listing: Mapped[Listing] = relationship(back_populates="images")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[str] = mapped_column(String(UUID_LEN), primary_key=True, default=new_uuid)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    version: Mapped[str] = mapped_column(Text, nullable=False)
    config: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (UniqueConstraint("name", "version", name="uq_model_name_version"),)


PREDICTION_TYPES = ("attribute_extraction", "duplicate_detection", "price_estimation", "quality_check")


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(String(UUID_LEN), primary_key=True, default=new_uuid)
    listing_id: Mapped[str] = mapped_column(String(UUID_LEN), ForeignKey("listings.id", ondelete="CASCADE"), nullable=False)
    model_version_id: Mapped[str] = mapped_column(String(UUID_LEN), ForeignKey("model_versions.id"), nullable=False)
    prediction_type: Mapped[str] = mapped_column(Text, nullable=False, default="attribute_extraction")
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        CheckConstraint(
            "prediction_type IN ('attribute_extraction','duplicate_detection','price_estimation','quality_check')",
            name="ck_predictions_type",
        ),
    )

    listing: Mapped[Listing] = relationship(back_populates="predictions")
    attributes: Mapped[list["PredictionAttribute"]] = relationship(cascade="all, delete-orphan", back_populates="prediction")


class PredictionAttribute(Base):
    __tablename__ = "prediction_attributes"

    id: Mapped[str] = mapped_column(String(UUID_LEN), primary_key=True, default=new_uuid)
    prediction_id: Mapped[str] = mapped_column(String(UUID_LEN), ForeignKey("predictions.id", ondelete="CASCADE"), nullable=False)
    key: Mapped[str] = mapped_column(Text, nullable=False)
    value: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    abstained: Mapped[bool] = mapped_column(nullable=False, default=False)

    __table_args__ = (
        UniqueConstraint("prediction_id", "key", name="uq_pred_attr_key"),
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="ck_attr_confidence"),
        # Abstention invariant: abstained=true => value IS NULL; abstained=false => value IS NOT NULL
        # Boolean-logic form (no integer comparison) so it type-checks on both PostgreSQL and SQLite.
        CheckConstraint(
            "(NOT abstained AND value IS NOT NULL) OR (abstained AND value IS NULL)",
            name="ck_attr_abstention",
        ),
    )

    prediction: Mapped[Prediction] = relationship(back_populates="attributes")
    evidence: Mapped[list["PredictionEvidence"]] = relationship(cascade="all, delete-orphan", back_populates="attribute")


EVIDENCE_SOURCES = ("text", "image", "multimodal")


class PredictionEvidence(Base):
    """First-class evidence row: structured provenance for one attribute."""

    __tablename__ = "prediction_evidence"

    id: Mapped[str] = mapped_column(String(UUID_LEN), primary_key=True, default=new_uuid)
    prediction_attribute_id: Mapped[str] = mapped_column(
        String(UUID_LEN), ForeignKey("prediction_attributes.id", ondelete="CASCADE"), nullable=False
    )
    source: Mapped[str] = mapped_column(Text, nullable=False)
    image_id: Mapped[str | None] = mapped_column(String(UUID_LEN), ForeignKey("listing_images.id", ondelete="SET NULL"), nullable=True)
    text_span: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # {field, start, end}
    bounding_box: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # {x, y, w, h}
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        CheckConstraint("source IN ('text','image','multimodal','ocr')", name="ck_evidence_source"),
    )

    attribute: Mapped[PredictionAttribute] = relationship(back_populates="evidence")


class ReviewEvent(Base):
    """Append-only reviewer decision. No UPDATE/DELETE routes exist by design."""

    __tablename__ = "review_events"

    id: Mapped[str] = mapped_column(String(UUID_LEN), primary_key=True, default=new_uuid)
    listing_id: Mapped[str] = mapped_column(String(UUID_LEN), ForeignKey("listings.id", ondelete="CASCADE"), nullable=False)
    prediction_id: Mapped[str | None] = mapped_column(String(UUID_LEN), ForeignKey("predictions.id", ondelete="SET NULL"), nullable=True)
    reviewer_id: Mapped[str] = mapped_column(Text, nullable=False)
    decision: Mapped[str] = mapped_column(Text, nullable=False)
    corrected_attributes: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        CheckConstraint("decision IN ('accepted','rejected','corrected')", name="ck_review_decision"),
    )

    listing: Mapped[Listing] = relationship(back_populates="review_events")
