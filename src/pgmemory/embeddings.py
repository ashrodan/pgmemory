"""Pluggable embedding providers.

The store needs vectors. How you generate them is your business.
Implement EmbeddingProvider or use one of the built-in ones.
"""

from __future__ import annotations

import abc
from typing import Sequence


class EmbeddingProvider(abc.ABC):
    """Abstract base. Implement `embed` and `dimensions`."""

    @abc.abstractmethod
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
        ...

    @property
    @abc.abstractmethod
    def dimensions(self) -> int:
        """Dimensionality of the output vectors."""
        ...


class VertexEmbeddingProvider(EmbeddingProvider):
    """Google Vertex AI text-embedding model.

    pip install pgmemory[vertex]
    """

    def __init__(
        self,
        model: str = "text-embedding-004",
        *,
        task_type: str = "RETRIEVAL_DOCUMENT",
        dimensionality: int = 768,
    ):
        self._model = model
        self._task_type = task_type
        self._dimensions = dimensionality

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        from google import genai

        client = genai.Client()
        response = await client.aio.models.embed_content(
            model=self._model,
            contents=list(texts),
            config=genai.types.EmbedContentConfig(
                task_type=self._task_type,
                output_dimensionality=self._dimensions,
            ),
        )
        assert response.embeddings is not None
        return [e.values for e in response.embeddings if e.values is not None]


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Local Ollama (e.g. nomic-embed-text).

    pip install pgmemory[ollama]
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        *,
        dimensionality: int = 768,
        host: str | None = None,
    ):
        self._model = model
        self._dimensions = dimensionality
        self._host = host

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        import ollama as _ollama

        client = _ollama.AsyncClient(host=self._host) if self._host else _ollama.AsyncClient()
        response = await client.embed(model=self._model, input=list(texts))
        return response["embeddings"]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI / Azure OpenAI embeddings.

    pip install pgmemory[openai]
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        dimensionality: int = 1536,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self._model = model
        self._dimensions = dimensionality
        self._api_key = api_key
        self._base_url = base_url

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)
        response = await client.embeddings.create(
            model=self._model,
            input=list(texts),
            dimensions=self._dimensions,
        )
        return [item.embedding for item in response.data]
