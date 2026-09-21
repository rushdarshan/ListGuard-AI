"""Initial schema: listings, listing_images, model_versions, predictions,
prediction_attributes, prediction_evidence, review_events.

Revision ID: 0001
"""

from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "listings",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("seller_id", sa.Text, nullable=False),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("category", sa.Text, nullable=True),
        sa.Column("status", sa.Text, nullable=False, server_default="draft"),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.func.now()),
        sa.CheckConstraint("status IN ('draft','pending_review','approved','rejected')", name="ck_listings_status"),
    )
    op.create_index("ix_listings_seller", "listings", ["seller_id"])
    op.create_index("ix_listings_status", "listings", ["status"])

    op.create_table(
        "listing_images",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("listing_id", sa.String(36), sa.ForeignKey("listings.id", ondelete="CASCADE"), nullable=False),
        sa.Column("storage_key", sa.Text, nullable=False),
        sa.Column("filename", sa.Text, nullable=False),
        sa.Column("content_type", sa.Text, nullable=False),
        sa.Column("byte_size", sa.Integer, nullable=False),
        sa.Column("width", sa.Integer, nullable=True),
        sa.Column("height", sa.Integer, nullable=True),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.func.now()),
        sa.UniqueConstraint("listing_id", "sha256", name="uq_images_listing_sha"),
        sa.CheckConstraint("byte_size > 0", name="ck_images_bytes"),
    )

    op.create_table(
        "model_versions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("version", sa.Text, nullable=False),
        sa.Column("config", sa.JSON, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.func.now()),
        sa.UniqueConstraint("name", "version", name="uq_model_name_version"),
    )

    op.create_table(
        "predictions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("listing_id", sa.String(36), sa.ForeignKey("listings.id", ondelete="CASCADE"), nullable=False),
        sa.Column("model_version_id", sa.String(36), sa.ForeignKey("model_versions.id"), nullable=False),
        sa.Column("prediction_type", sa.Text, nullable=False, server_default="attribute_extraction"),
        sa.Column("latency_ms", sa.Integer, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.func.now()),
        sa.CheckConstraint(
            "prediction_type IN ('attribute_extraction','duplicate_detection','price_estimation','quality_check')",
            name="ck_predictions_type",
        ),
    )
    op.create_index("ix_predictions_listing_created", "predictions", ["listing_id", "created_at"])

    op.create_table(
        "prediction_attributes",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("prediction_id", sa.String(36), sa.ForeignKey("predictions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("key", sa.Text, nullable=False),
        sa.Column("value", sa.Text, nullable=True),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("abstained", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.UniqueConstraint("prediction_id", "key", name="uq_pred_attr_key"),
        sa.CheckConstraint("confidence >= 0 AND confidence <= 1", name="ck_attr_confidence"),
        sa.CheckConstraint(
            "(NOT abstained AND value IS NOT NULL) OR (abstained AND value IS NULL)",
            name="ck_attr_abstention",
        ),
    )

    op.create_table(
        "prediction_evidence",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("prediction_attribute_id", sa.String(36), sa.ForeignKey("prediction_attributes.id", ondelete="CASCADE"), nullable=False),
        sa.Column("source", sa.Text, nullable=False),
        sa.Column("image_id", sa.String(36), sa.ForeignKey("listing_images.id", ondelete="SET NULL"), nullable=True),
        sa.Column("text_span", sa.JSON, nullable=True),
        sa.Column("bounding_box", sa.JSON, nullable=True),
        sa.Column("note", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.func.now()),
        sa.CheckConstraint("source IN ('text','image','multimodal')", name="ck_evidence_source"),
    )

    op.create_table(
        "review_events",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("listing_id", sa.String(36), sa.ForeignKey("listings.id", ondelete="CASCADE"), nullable=False),
        sa.Column("prediction_id", sa.String(36), sa.ForeignKey("predictions.id", ondelete="SET NULL"), nullable=True),
        sa.Column("reviewer_id", sa.Text, nullable=False),
        sa.Column("decision", sa.Text, nullable=False),
        sa.Column("corrected_attributes", sa.JSON, nullable=True),
        sa.Column("note", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.func.now()),
        sa.CheckConstraint("decision IN ('accepted','rejected','corrected')", name="ck_review_decision"),
    )
    op.create_index("ix_reviews_listing_created", "review_events", ["listing_id", "created_at"])

    # Postgres-only retrieval helpers (no-op on SQLite): FTS + trgm.
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(
            "CREATE INDEX IF NOT EXISTS ix_listings_fts ON listings "
            "USING gin (to_tsvector('english', title || ' ' || coalesce(description, '')))"
        )
        op.execute("CREATE INDEX IF NOT EXISTS ix_listings_title_trgm ON listings USING gin (title gin_trgm_ops)")


def downgrade() -> None:
    for table in ("review_events", "prediction_evidence", "prediction_attributes", "predictions", "model_versions", "listing_images", "listings"):
        op.drop_table(table)
