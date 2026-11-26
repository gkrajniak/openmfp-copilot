from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from openmfp_copilot.documents.models import DocumentChunk
from openmfp_copilot.libraries.models import Organization


@dataclass(frozen=True)
class SourceDocument:
    organization: Organization
    repository: str
    source_path: str


@dataclass(frozen=True)
class EmbeddedChunk:
    content: str
    embedding: list[float]


class DocumentChunkRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def replace(self, source: SourceDocument, chunks: Sequence[EmbeddedChunk]) -> None:
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
        await self._session.flush()
