from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import SecretStr
from voyageai.client_async import AsyncClient

from openmfp_copilot.config import Settings
from openmfp_copilot.documents.embeddings import (
    MAX_TEXTS_PER_REQUEST,
    Embedder,
    create_voyage_client,
)

pytestmark = pytest.mark.anyio


@dataclass
class FakeVoyage:
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def embed(self, texts: list[str], **kwargs: Any) -> SimpleNamespace:
        self.calls.append({"texts": texts, **kwargs})
        return SimpleNamespace(embeddings=[[float(len(text)), 1] for text in texts])

    def embedder(self) -> Embedder:
        return Embedder(cast(AsyncClient, self), "voyage-4")


async def test_embed_documents_requests_document_embeddings_without_truncation() -> None:
    voyage = FakeVoyage()

    embeddings = await voyage.embedder().embed_documents(["ab", "abcd"])

    assert embeddings == [[2.0, 1.0], [4.0, 1.0]]
    [call] = voyage.calls
    assert call == {
        "texts": ["ab", "abcd"],
        "model": "voyage-4",
        "input_type": "document",
        "truncation": False,
        "output_dimension": 1024,
    }


async def test_embed_documents_splits_large_inputs_into_batches() -> None:
    voyage = FakeVoyage()
    texts = [str(index) for index in range(MAX_TEXTS_PER_REQUEST + 1)]

    embeddings = await voyage.embedder().embed_documents(texts)

    assert [len(call["texts"]) for call in voyage.calls] == [MAX_TEXTS_PER_REQUEST, 1]
    assert len(embeddings) == len(texts)


async def test_embed_query_requests_query_embedding() -> None:
    voyage = FakeVoyage()

    embedding = await voyage.embedder().embed_query("How do I install it?")

    assert embedding == [20.0, 1.0]
    assert voyage.calls[0]["input_type"] == "query"


def test_voyage_client_uses_api_key_and_retries_twice(settings: Settings) -> None:
    settings = settings.model_copy(update={"voyage_api_key": SecretStr("voyage-test-key")})

    client = create_voyage_client(settings)

    assert client.api_key == "voyage-test-key"
    assert client.max_retries == 3
