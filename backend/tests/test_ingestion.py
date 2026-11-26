from dataclasses import dataclass, field
from typing import Any, cast

import httpx2
import pytest
from sqlalchemy import select
from voyageai.client_async import AsyncClient

from openmfp_copilot.db.session import SessionFactory
from openmfp_copilot.documents.embeddings import Embedder
from openmfp_copilot.documents.ingestion import Ingestion, IngestionResult
from openmfp_copilot.documents.models import EMBEDDING_DIMENSIONS, DocumentChunk, DocumentSource
from openmfp_copilot.github.client import GitHubClient, ReadmeNotFoundError
from openmfp_copilot.ingest import parse_args
from openmfp_copilot.libraries.repository import KNOWN_LIBRARIES

pytestmark = pytest.mark.anyio

PORTAL_UI_LIB = next(
    library
    for library in KNOWN_LIBRARIES
    if (library.organization, library.name) == ("openmfp", "portal-ui-lib")
)
README = "# Portal UI Library\nAngular library.\n## Install\nnpm install @openmfp/portal-ui-lib\n"
README_SHA256 = "364c23a410c63037eca2c828c037f70309ebd07949f6171ad1d3ae35ed1b0872"


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


@dataclass
class Setup:
    session_factory: SessionFactory
    voyage: FakeVoyage = field(default_factory=FakeVoyage)

    async def ingest(
        self,
        response: httpx2.Response | None = None,
        *,
        model: str = "voyage-4",
        force: bool = False,
    ) -> IngestionResult:
        served = response or httpx2.Response(200, text=README)
        http = httpx2.AsyncClient(transport=httpx2.MockTransport(lambda request: served))
        embedder = Embedder(cast(AsyncClient, self.voyage), model)
        ingestion = Ingestion(GitHubClient(http), embedder, self.session_factory)
        return await ingestion.ingest_readme(PORTAL_UI_LIB, force=force)

    async def chunk_ids(self) -> list[int]:
        async with self.session_factory() as session:
            return list(
                (await session.scalars(select(DocumentChunk.id).order_by(DocumentChunk.id))).all()
            )


@pytest.fixture
def setup(session_factory: SessionFactory) -> Setup:
    return Setup(session_factory)


async def test_ingest_readme_stores_one_embedded_chunk_per_section(setup: Setup) -> None:
    result = await setup.ingest()

    assert result == IngestionResult(unchanged=False, chunks=2)
    expected = [
        "openmfp/portal-ui-lib > Portal UI Library\n\nAngular library.",
        "openmfp/portal-ui-lib > Portal UI Library > Install\n\nnpm install @openmfp/portal-ui-lib",
    ]
    assert setup.voyage.texts == expected
    async with setup.session_factory() as session:
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


async def test_ingest_readme_records_sha256_of_the_readme_and_the_model(setup: Setup) -> None:
    await setup.ingest()

    async with setup.session_factory() as session:
        source = (await session.scalars(select(DocumentSource))).one()
    assert (source.repository, source.source_path) == ("portal-ui-lib", "README")
    assert source.content_sha256 == README_SHA256
    assert source.embedding_model == "voyage-4"
    assert source.ingested_at is not None


async def test_ingest_readme_skips_unchanged_readme_without_embedding(setup: Setup) -> None:
    await setup.ingest()
    ids_before = await setup.chunk_ids()
    setup.voyage.texts.clear()

    result = await setup.ingest()

    assert result == IngestionResult(unchanged=True)
    assert setup.voyage.texts == []
    assert await setup.chunk_ids() == ids_before


async def test_ingest_readme_reingests_when_readme_changes(setup: Setup) -> None:
    await setup.ingest()

    result = await setup.ingest(httpx2.Response(200, text=README + "## License\nApache-2.0\n"))

    assert result == IngestionResult(unchanged=False, chunks=3)


async def test_ingest_readme_reingests_when_embedding_model_changes(setup: Setup) -> None:
    await setup.ingest()

    result = await setup.ingest(model="voyage-4-large")

    assert result == IngestionResult(unchanged=False, chunks=2)


async def test_ingest_readme_reingests_unchanged_readme_when_forced(setup: Setup) -> None:
    await setup.ingest()
    ids_before = await setup.chunk_ids()

    result = await setup.ingest(force=True)

    assert result == IngestionResult(unchanged=False, chunks=2)
    assert await setup.chunk_ids() != ids_before


async def test_ingest_readme_keeps_existing_chunks_when_readme_is_missing(setup: Setup) -> None:
    await setup.ingest()
    ids_before = await setup.chunk_ids()
    setup.voyage.texts.clear()

    with pytest.raises(ReadmeNotFoundError):
        await setup.ingest(httpx2.Response(404))

    assert setup.voyage.texts == []
    assert await setup.chunk_ids() == ids_before


def test_ingest_command_accepts_force_flag() -> None:
    assert parse_args([]).force is False
    assert parse_args(["--force"]).force is True
