from collections.abc import Sequence
from typing import Literal

from voyageai.client_async import AsyncClient

from openmfp_copilot.config import Settings
from openmfp_copilot.documents.models import EMBEDDING_DIMENSIONS

MAX_TEXTS_PER_REQUEST = 1000
MAX_ATTEMPTS = 3

type InputType = Literal["document", "query"]


class Embedder:
    def __init__(self, client: AsyncClient, model: str) -> None:
        self._client = client
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), MAX_TEXTS_PER_REQUEST):
            batch = texts[start : start + MAX_TEXTS_PER_REQUEST]
            embeddings += await self._embed(list(batch), "document")
        return embeddings

    async def embed_query(self, text: str) -> list[float]:
        [embedding] = await self._embed([text], "query")
        return embedding

    async def _embed(self, texts: list[str], input_type: InputType) -> list[list[float]]:
        result = await self._client.embed(
            texts,
            model=self._model,
            input_type=input_type,
            truncation=False,
            output_dimension=EMBEDDING_DIMENSIONS,
        )
        return [[float(value) for value in embedding] for embedding in result.embeddings]


def create_voyage_client(settings: Settings) -> AsyncClient:
    api_key = settings.voyage_api_key.get_secret_value() if settings.voyage_api_key else None
    return AsyncClient(api_key=api_key, max_retries=MAX_ATTEMPTS)


def create_embedder(settings: Settings) -> Embedder:
    return Embedder(create_voyage_client(settings), settings.voyage_model)
