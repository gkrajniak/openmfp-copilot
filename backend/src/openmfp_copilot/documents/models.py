from pgvector.sqlalchemy import Vector
from sqlalchemy import Enum, Index, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from openmfp_copilot.db.base import Base
from openmfp_copilot.libraries.models import Organization

EMBEDDING_DIMENSIONS = 1024


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
    organization: Mapped[Organization] = mapped_column(
        Enum(
            Organization,
            native_enum=False,
            length=50,
            values_callable=lambda members: [member.value for member in members],
        )
    )
    repository: Mapped[str] = mapped_column(String(100))
    source_path: Mapped[str] = mapped_column(String(500))
    chunk_index: Mapped[int]
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSIONS))
