from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Enum, Index, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from openmfp_copilot.db.base import Base
from openmfp_copilot.libraries.models import Organization

EMBEDDING_DIMENSIONS = 1024
SHA256_HEX_LENGTH = 64


def organization_type() -> Enum:
    return Enum(
        Organization,
        native_enum=False,
        length=50,
        values_callable=lambda members: [member.value for member in members],
    )


class DocumentSource(Base):
    __tablename__ = "document_sources"
    __table_args__ = (
        UniqueConstraint(
            "organization", "repository", "source_path", name="uq_document_sources_source"
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    organization: Mapped[Organization] = mapped_column(organization_type())
    repository: Mapped[str] = mapped_column(String(100))
    source_path: Mapped[str] = mapped_column(String(500))
    content_sha256: Mapped[str] = mapped_column(String(SHA256_HEX_LENGTH))
    embedding_model: Mapped[str] = mapped_column(String(100))
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint(
            "organization",
            "repository",
            "source_path",
            "chunk_index",
            name="uq_document_chunks_source_chunk",
        ),
        Index(
            "ix_document_chunks_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    organization: Mapped[Organization] = mapped_column(organization_type())
    repository: Mapped[str] = mapped_column(String(100))
    source_path: Mapped[str] = mapped_column(String(500))
    chunk_index: Mapped[int]
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSIONS))
