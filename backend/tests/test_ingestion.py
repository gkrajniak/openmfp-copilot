from dataclasses import dataclass, field
from typing import Any, cast

import httpx2
import pytest
from sqlalchemy import select
from voyageai.client_async import AsyncClient

from openmfp_copilot.db.session import SessionFactory
from openmfp_copilot.documents.embeddings import Embedder
from openmfp_copilot.documents.ingestion import Ingestion
from openmfp_copilot.documents.models import EMBEDDING_DIMENSIONS, DocumentChunk
from openmfp_copilot.github.client import GitHubClient, ReadmeNotFoundError
from openmfp_copilot.libraries.repository import KNOWN_LIBRARIES

pytestmark = pytest.mark.anyio

PORTAL_UI_LIB = next(
    library
    for library in KNOWN_LIBRARIES
    if (library.organization, library.name) == ("openmfp", "portal-ui-lib")
)
README = "# Portal UI Library\nAngular library.\n## Install\nnpm install @openmfp/portal-ui-lib\n"


@dataclass
class FakeVoyage:
    texts: list[str] = field(default_factory=list)

    async def embed(self, texts: list[str], **kwargs: Any) -> Any:
        self.texts += texts

        def one_hot(position: int) -> list[float]:
            vector = [0.0] * EMBEDDING_DIMENSIONS
            vector[position] = 1.0
            return vector

        return type("Result", (), {"embeddings": [one_hot(i) for i in range(len(texts))]})


def ingestion(
    session_factory: SessionFactory, response: httpx2.Response, voyage: FakeVoyage
) -> Ingestion:
    http = httpx2.AsyncClient(transport=httpx2.MockTransport(lambda request: response))
    embedder = Embedder(cast(AsyncClient, voyage), "voyage-4")
    return Ingestion(GitHubClient(http), embedder, session_factory)


async def test_ingest_readme_stores_one_embedded_chunk_per_section(
    session_factory: SessionFactory,
) -> None:
    voyage = FakeVoyage()

    count = await ingestion(
        session_factory, httpx2.Response(200, text=README), voyage
    ).ingest_readme(PORTAL_UI_LIB)

    assert count == 2
    expected = [
        "openmfp/portal-ui-lib > Portal UI Library\n\nAngular library.",
        "openmfp/portal-ui-lib > Portal UI Library > Install\n\nnpm install @openmfp/portal-ui-lib",
    ]
    assert voyage.texts == expected
    async with session_factory() as session:
        rows = (
            await session.execute(
                select(
                    DocumentChunk.repository,
                    DocumentChunk.source_path,
                    DocumentChunk.chunk_index,
                    DocumentChunk.content,
                ).order_by(DocumentChunk.chunk_index)
            )
        ).all()
    assert [tuple(row) for row in rows] == [
        ("portal-ui-lib", "README", 0, expected[0]),
        ("portal-ui-lib", "README", 1, expected[1]),
    ]


async def test_ingest_readme_keeps_existing_chunks_when_readme_is_missing(
    session_factory: SessionFactory,
) -> None:
    await ingestion(session_factory, httpx2.Response(200, text=README), FakeVoyage()).ingest_readme(
        PORTAL_UI_LIB
    )
    voyage = FakeVoyage()

    with pytest.raises(ReadmeNotFoundError):
        await ingestion(session_factory, httpx2.Response(404), voyage).ingest_readme(PORTAL_UI_LIB)

    assert voyage.texts == []
    async with session_factory() as session:
        assert len((await session.execute(select(DocumentChunk.id))).all()) == 2
