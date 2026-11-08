"""create the decisions table and the trigger that forbids update, delete and truncate

Revision ID: 0001_decisions
Revises:
Create Date: 2026-10-16
"""

from alembic import op
import sqlalchemy as sa


revision = "0001_decisions"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "decisions",
        sa.Column("seq", sa.BigInteger, primary_key=True),
        sa.Column("t", sa.Float, nullable=False),
        sa.Column("payload", sa.Text, nullable=False),
        sa.Column("prev", sa.CHAR(64), nullable=False, unique=True),
        sa.Column("hash", sa.CHAR(64), nullable=False, unique=True),
    )
    op.execute(
        """
        CREATE OR REPLACE FUNCTION decisions_forbid_change() RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION 'decisions is append-only';
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        CREATE TRIGGER decisions_no_change
        BEFORE UPDATE OR DELETE ON decisions
        FOR EACH ROW EXECUTE FUNCTION decisions_forbid_change();
        """
    )
    op.execute(
        """
        CREATE TRIGGER decisions_no_truncate
        BEFORE TRUNCATE ON decisions
        FOR EACH STATEMENT EXECUTE FUNCTION decisions_forbid_change();
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS decisions_no_truncate ON decisions")
    op.execute("DROP TRIGGER IF EXISTS decisions_no_change ON decisions")
    op.execute("DROP FUNCTION IF EXISTS decisions_forbid_change()")
    op.drop_table("decisions")
