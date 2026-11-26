from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "99d67646f026"
down_revision: str | Sequence[str] | None = "ecbc8e885e43"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "document_chunks",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "organization",
            sa.Enum("openmfp", "platform-mesh", name="organization", native_enum=False, length=50),
            nullable=False,
        ),
        sa.Column("repository", sa.String(length=100), nullable=False),
        sa.Column("source_path", sa.String(length=500), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(1024), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_document_chunks")),
        sa.UniqueConstraint(
            "organization",
            "repository",
            "source_path",
            "chunk_index",
            name="uq_document_chunks_source_chunk",
        ),
    )
    op.create_index(
        "ix_document_chunks_embedding",
        "document_chunks",
        ["embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index(
        "ix_document_chunks_embedding",
        table_name="document_chunks",
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.drop_table("document_chunks")
