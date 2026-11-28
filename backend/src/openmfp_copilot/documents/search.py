from openmfp_copilot.db.session import SessionFactory
from openmfp_copilot.documents.embeddings import Embedder
from openmfp_copilot.documents.repository import ChunkMatch, DocumentChunkRepository


class DocumentSearch:
    def __init__(self, embedder: Embedder, session_factory: SessionFactory) -> None:
        self._embedder = embedder
        self._session_factory = session_factory

    async def search(self, question: str, limit: int | None) -> list[ChunkMatch]:
        embedding = await self._embedder.embed_query(question)
        async with self._session_factory() as session:
            return await DocumentChunkRepository(session).nearest(embedding, limit)
