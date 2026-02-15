"""pgmemory — Opinionated multi-user agent memory on PostgreSQL + pgvector.

Core (no framework deps):
    from pgmemory import MemoryStore, Category, Memory, SearchQuery

Embedding providers:
    from pgmemory import VertexEmbeddingProvider, OllamaEmbeddingProvider, OpenAIEmbeddingProvider

Adapters (optional deps):
    from pgmemory.adapters.adk import ADKMemoryService, build_adk_tools
    from pgmemory.adapters.langchain import LangChainMemory, build_langchain_tools
"""

from .embeddings import (
    EmbeddingProvider,
    OllamaEmbeddingProvider,
    OpenAIEmbeddingProvider,
    VertexEmbeddingProvider,
)
from .store import MemoryStore
from .types import Category, Memory, SearchQuery, SearchResult

__all__ = [
    # Core
    "MemoryStore",
    "Memory",
    "Category",
    "SearchQuery",
    "SearchResult",
    # Embeddings
    "EmbeddingProvider",
    "VertexEmbeddingProvider",
    "OllamaEmbeddingProvider",
    "OpenAIEmbeddingProvider",
]
