import pytest
from sqlalchemy import select

from openmfp_copilot.db.session import SessionFactory
from openmfp_copilot.documents.models import EMBEDDING_DIMENSIONS, DocumentChunk
from openmfp_copilot.documents.repository import (
    DocumentChunkRepository,
    EmbeddedChunk,
    SourceDocument,
)
from openmfp_copilot.libraries.models import Organization

pytestmark = pytest.mark.anyio

UI_README = SourceDocument(Organization.OPENMFP, "portal-ui-lib", "README")
MESH_UI_README = SourceDocument(Organization.PLATFORM_MESH, "portal-ui-lib", "README")


def chunk(content: str, position: int = 0) -> EmbeddedChunk:
    embedding = [0.0] * EMBEDDING_DIMENSIONS
    embedding[position] = 1.0
    return EmbeddedChunk(content=content, embedding=embedding)


async def stored(session_factory: SessionFactory) -> list[tuple[str, str, int, str]]:
    async with session_factory() as session:
        result = await session.execute(
            select(
                DocumentChunk.organization,
                DocumentChunk.repository,
                DocumentChunk.chunk_index,
                DocumentChunk.content,
            ).order_by(
                DocumentChunk.organization, DocumentChunk.repository, DocumentChunk.chunk_index
            )
        )
        return [(str(org), repo, index, content) for org, repo, index, content in result.all()]


async def replace(
    session_factory: SessionFactory, source: SourceDocument, chunks: list[EmbeddedChunk]
) -> None:
    async with session_factory() as session, session.begin():
        await DocumentChunkRepository(session).replace(source, chunks)


async def test_replace_stores_chunks_in_order_with_their_embeddings(
    session_factory: SessionFactory,
) -> None:
    await replace(session_factory, UI_README, [chunk("Install", 0), chunk("Configure", 1)])

    assert await stored(session_factory) == [
        ("openmfp", "portal-ui-lib", 0, "Install"),
        ("openmfp", "portal-ui-lib", 1, "Configure"),
    ]
    async with session_factory() as session:
        embedding = await session.scalar(
            select(DocumentChunk.embedding).where(DocumentChunk.chunk_index == 1)
        )
    assert embedding is not None
    assert len(embedding) == EMBEDDING_DIMENSIONS
    assert embedding[1] == 1.0


async def test_replace_removes_previous_chunks_of_the_same_source_only(
    session_factory: SessionFactory,
) -> None:
    await replace(session_factory, UI_README, [chunk("old 0"), chunk("old 1"), chunk("old 2")])
    await replace(session_factory, MESH_UI_README, [chunk("mesh")])

    await replace(session_factory, UI_README, [chunk("new 0")])

    assert await stored(session_factory) == [
        ("openmfp", "portal-ui-lib", 0, "new 0"),
        ("platform-mesh", "portal-ui-lib", 0, "mesh"),
    ]
