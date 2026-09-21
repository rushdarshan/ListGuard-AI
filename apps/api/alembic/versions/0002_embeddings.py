"""Phase 2: embedding columns (pgvector on postgres, JSON elsewhere).

Revision ID: 0002
"""

from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        # Extension already ensured by infra/postgres/init.sql; HNSW needs pgvector >= 0.7.
        op.execute("ALTER TABLE listings ADD COLUMN IF NOT EXISTS text_embedding vector(64)")
        op.execute("ALTER TABLE listing_images ADD COLUMN IF NOT EXISTS image_embedding vector(64)")
        op.execute(
            "CREATE INDEX IF NOT EXISTS ix_listings_text_vec ON listings "
            "USING hnsw (text_embedding vector_cosine_ops)"
        )
        op.execute(
            "CREATE INDEX IF NOT EXISTS ix_images_vec ON listing_images "
            "USING hnsw (image_embedding vector_cosine_ops)"
        )
    else:
        op.add_column("listings", sa.Column("text_embedding", sa.JSON(), nullable=True))
        op.add_column("listing_images", sa.Column("image_embedding", sa.JSON(), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("DROP INDEX IF EXISTS ix_images_vec")
        op.execute("DROP INDEX IF EXISTS ix_listings_text_vec")
        op.execute("ALTER TABLE listing_images DROP COLUMN IF EXISTS image_embedding")
        op.execute("ALTER TABLE listings DROP COLUMN IF EXISTS text_embedding")
    else:
        with op.batch_alter_table("listing_images") as batch:
            batch.drop_column("image_embedding")
        with op.batch_alter_table("listings") as batch:
            batch.drop_column("text_embedding")
