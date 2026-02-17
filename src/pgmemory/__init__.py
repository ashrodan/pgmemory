"""pgmemory — Opinionated multi-user agent memory on PostgreSQL + pgvector.

Core (no framework deps):
    from pgmemory import MemoryStore, Category, Memory, SearchQuery

Embedding providers:
    from pgmemory import VertexEmbeddingProvider, OllamaEmbeddingProvider, OpenAIEmbeddingProvider, VoyageEmbeddingProvider

Adapters (optional deps):
    from pgmemory.adapters.adk import ADKMemoryService, build_adk_tools, build_context_injection
    from pgmemory.adapters.langchain import LangChainMemory, build_langchain_tools
"""

from importlib.metadata import version

__version__ = version("pgmemory")

from .embeddings import (
    EmbeddingProvider,
    OllamaEmbeddingProvider,
    OpenAIEmbeddingProvider,
    VertexEmbeddingProvider,
    VoyageEmbeddingProvider,
)
from .store import MemoryStore
from .types import Category, Memory, SearchQuery, SearchResult

MEMORY_INSTRUCTIONS = """\
You have long-term memory tools. Use them proactively.

**search_memory** — Search past memories. Use before answering questions that \
might relate to stored knowledge.
**remember_fact** — Store preferences, facts, project context, rules, or events \
the user shares. Automatically supersedes outdated memories on the same topic.
**forget_memory** — Expire a memory by ID when it's wrong or outdated.

Categories: fact, preference, skill, context, rule, event.
Importance: 1 (low) to 5 (critical). Default to 3 for explicit requests.

Reinforce memories the user confirms. Forget memories the user corrects.\
"""

__all__ = [
    # Version
    "__version__",
    # Core
    "MemoryStore",
    "Memory",
    "Category",
    "SearchQuery",
    "SearchResult",
    "MEMORY_INSTRUCTIONS",
    # Embeddings
    "EmbeddingProvider",
    "VertexEmbeddingProvider",
    "OllamaEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "VoyageEmbeddingProvider",
]
