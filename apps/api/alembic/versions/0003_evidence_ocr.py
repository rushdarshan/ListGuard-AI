"""Reviewer item 6: allow source='ocr' in prediction_evidence CHECK.

Revision ID: 0003
"""

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

WIDE = "source IN ('text','image','multimodal','ocr')"
NARROW = "source IN ('text','image','multimodal')"


def upgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        op.drop_constraint("ck_evidence_source", "prediction_evidence", type_="check")
        op.create_check_constraint("ck_evidence_source", "prediction_evidence", WIDE)
    else:
        # SQLite cannot ALTER CHECKs: recreate the table via batch mode.
        with op.batch_alter_table("prediction_evidence", recreate="always") as batch_op:
            batch_op.drop_constraint("ck_evidence_source", type_="check")
            batch_op.create_check_constraint("ck_evidence_source", WIDE)


def downgrade() -> None:
    # Fails if 'ocr' rows exist (by design: narrowing would orphan them).
    if op.get_bind().dialect.name == "postgresql":
        op.drop_constraint("ck_evidence_source", "prediction_evidence", type_="check")
        op.create_check_constraint("ck_evidence_source", "prediction_evidence", NARROW)
    else:
        with op.batch_alter_table("prediction_evidence", recreate="always") as batch_op:
            batch_op.drop_constraint("ck_evidence_source", type_="check")
            batch_op.create_check_constraint("ck_evidence_source", NARROW)
