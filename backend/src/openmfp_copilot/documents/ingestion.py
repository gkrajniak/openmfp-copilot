import hashlib
from dataclasses import dataclass

from openmfp_copilot.db.session import SessionFactory
from openmfp_copilot.documents.chunking import split_markdown
from openmfp_copilot.documents.embeddings import Embedder
from openmfp_copilot.documents.repository import (
    DocumentChunkRepository,
    EmbeddedChunk,
    SourceFingerprint,
    SourceKey,
)
from openmfp_copilot.github.client import GitHubClient
from openmfp_copilot.libraries.models import Library

README_SOURCE = "README"


@dataclass(frozen=True)
class IngestionResult:
    unchanged: bool
    chunks: int = 0


class Ingestion:
    def __init__(
        self, github: GitHubClient, embedder: Embedder, session_factory: SessionFactory
    ) -> None:
        self._github = github
        self._embedder = embedder
        self._session_factory = session_factory

    async def ingest_readme(self, library: Library, *, force: bool = False) -> IngestionResult:
        markdown = await self._github.get_readme(library.organization, library.name)
        source = SourceKey(library.organization, library.name, README_SOURCE)
        fingerprint = SourceFingerprint(
            content_sha256=hashlib.sha256(markdown.encode()).hexdigest(),
            embedding_model=self._embedder.model,
        )
        if not force and await self._stored_fingerprint(source) == fingerprint:
            return IngestionResult(unchanged=True)

        sections = split_markdown(markdown, f"{library.organization}/{library.name}")
        texts = [section.text for section in sections]
        embeddings = await self._embedder.embed_documents(texts)
        chunks = [
            EmbeddedChunk(content=text, embedding=embedding)
            for text, embedding in zip(texts, embeddings, strict=True)
        ]
        async with self._session_factory() as session, session.begin():
            await DocumentChunkRepository(session).replace(source, fingerprint, chunks)
        return IngestionResult(unchanged=False, chunks=len(chunks))

    async def _stored_fingerprint(self, source: SourceKey) -> SourceFingerprint | None:
        async with self._session_factory() as session:
            return await DocumentChunkRepository(session).fingerprint(source)
