from openmfp_copilot.db.session import SessionFactory
from openmfp_copilot.documents.chunking import split_markdown
from openmfp_copilot.documents.embeddings import Embedder
from openmfp_copilot.documents.repository import (
    DocumentChunkRepository,
    EmbeddedChunk,
    SourceDocument,
)
from openmfp_copilot.github.client import GitHubClient
from openmfp_copilot.libraries.models import Library

README_SOURCE = "README"


class Ingestion:
    def __init__(
        self, github: GitHubClient, embedder: Embedder, session_factory: SessionFactory
    ) -> None:
        self._github = github
        self._embedder = embedder
        self._session_factory = session_factory

    async def ingest_readme(self, library: Library) -> int:
        markdown = await self._github.get_readme(library.organization, library.name)
        sections = split_markdown(markdown, f"{library.organization}/{library.name}")
        texts = [section.text for section in sections]
        embeddings = await self._embedder.embed_documents(texts)

        source = SourceDocument(library.organization, library.name, README_SOURCE)
        chunks = [
            EmbeddedChunk(content=text, embedding=embedding)
            for text, embedding in zip(texts, embeddings, strict=True)
        ]
        async with self._session_factory() as session, session.begin():
            await DocumentChunkRepository(session).replace(source, chunks)
        return len(chunks)
