from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "68677cc4a4b9"
down_revision: str | Sequence[str] | None = "99d67646f026"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "document_sources",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "organization",
            sa.Enum("openmfp", "platform-mesh", name="organization", native_enum=False, length=50),
            nullable=False,
        ),
        sa.Column("repository", sa.String(length=100), nullable=False),
        sa.Column("source_path", sa.String(length=500), nullable=False),
        sa.Column("content_sha256", sa.String(length=64), nullable=False),
        sa.Column("embedding_model", sa.String(length=100), nullable=False),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_document_sources")),
        sa.UniqueConstraint(
            "organization", "repository", "source_path", name="uq_document_sources_source"
        ),
    )


def downgrade() -> None:
    op.drop_table("document_sources")
