from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from openmfp_copilot.documents.models import DocumentChunk, DocumentSource
from openmfp_copilot.libraries.models import Organization


@dataclass(frozen=True)
class SourceKey:
    organization: Organization
    repository: str
    source_path: str


@dataclass(frozen=True)
class SourceFingerprint:
    content_sha256: str
    embedding_model: str


@dataclass(frozen=True)
class EmbeddedChunk:
    content: str
    embedding: list[float]


@dataclass(frozen=True)
class ChunkMatch:
    source: SourceKey
    chunk_index: int
    content: str
    distance: float


class DocumentChunkRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def fingerprint(self, source: SourceKey) -> SourceFingerprint | None:
        row = (
            await self._session.execute(
                select(DocumentSource.content_sha256, DocumentSource.embedding_model).where(
                    DocumentSource.organization == source.organization,
                    DocumentSource.repository == source.repository,
                    DocumentSource.source_path == source.source_path,
                )
            )
        ).one_or_none()
        if row is None:
            return None
        return SourceFingerprint(content_sha256=row[0], embedding_model=row[1])

    async def nearest(self, embedding: list[float], limit: int | None) -> list[ChunkMatch]:
        distance = DocumentChunk.embedding.cosine_distance(embedding).label("distance")
        rows = await self._session.execute(
            select(DocumentChunk, distance).order_by(distance, DocumentChunk.id).limit(limit)
        )
        return [
            ChunkMatch(
                source=SourceKey(chunk.organization, chunk.repository, chunk.source_path),
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                distance=chunk_distance,
            )
            for chunk, chunk_distance in rows.tuples()
        ]

    async def replace(
        self,
        source: SourceKey,
        fingerprint: SourceFingerprint,
        chunks: Sequence[EmbeddedChunk],
    ) -> None:
        await self._session.execute(
            delete(DocumentChunk).where(
                DocumentChunk.organization == source.organization,
                DocumentChunk.repository == source.repository,
                DocumentChunk.source_path == source.source_path,
            )
        )
        self._session.add_all(
            DocumentChunk(
                organization=source.organization,
                repository=source.repository,
                source_path=source.source_path,
                chunk_index=index,
                content=chunk.content,
                embedding=chunk.embedding,
            )
            for index, chunk in enumerate(chunks)
        )
        fingerprint_values = {
            "content_sha256": fingerprint.content_sha256,
            "embedding_model": fingerprint.embedding_model,
        }
        await self._session.execute(
            insert(DocumentSource)
            .values(
                organization=source.organization,
                repository=source.repository,
                source_path=source.source_path,
                **fingerprint_values,
            )
            .on_conflict_do_update(
                constraint="uq_document_sources_source",
                set_={**fingerprint_values, "ingested_at": func.now()},
            )
        )
        await self._session.flush()
