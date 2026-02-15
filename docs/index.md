# pgmemory

**Opinionated multi-user agent memory on PostgreSQL + pgvector.**

One table. Hybrid search. Lifecycle management. Any framework.

---

## Why pgmemory?

Every agent framework re-invents memory. They all need the same thing: store what the agent learned about a user, find it later, let old stuff fade. The database part is always the same — it's Postgres with vectors.

pgmemory is *that database part*, extracted into a standalone library. Use it with ADK, LangChain, CrewAI, Semantic Kernel, or plain Python. The core has zero framework dependencies.

## Features

- **Single table** — one well-indexed PostgreSQL table handles everything: categories, importance, temporal validity, provenance, and hybrid search
- **Hybrid search** — combines semantic similarity (pgvector), keyword matching (tsvector), and recency decay into a single weighted score
- **Memory lifecycle** — importance levels, temporal validity, promote/expire/decay operations, and automatic conflict resolution via `supersede()`
- **Framework-agnostic** — core library has no framework deps. Thin adapters for [Google ADK](adapters/adk.md) and [LangChain](adapters/langchain.md) included
- **Multi-tenant** — scoped by `app_name` + `user_id` out of the box
- **Provenance** — every memory tracks where it came from: session, event, timestamp, role

## Quick install

```bash
pip install pgmemory
```

With an embedding provider:

```bash
pip install pgmemory[vertex]   # Google Vertex AI
pip install pgmemory[ollama]   # Local Ollama
pip install pgmemory[openai]   # OpenAI
```

## Quick example

```python
from pgmemory import MemoryStore, Category, SearchQuery, OllamaEmbeddingProvider

store = MemoryStore(
    "postgresql+asyncpg://user:pass@localhost/mydb",
    OllamaEmbeddingProvider(),
)
await store.init()

# Store a memory
await store.add("my_app", "user_123", "Prefers dark mode and compact layouts",
                category=Category.PREFERENCE, importance=3)

# Search (hybrid: keyword + semantic + recency)
results = await store.search(SearchQuery(
    app_name="my_app",
    user_id="user_123",
    text="UI preferences",
))

for r in results:
    print(f"[{r.memory.category.value}] {r.text} (score={r.combined_score})")
```

## Next steps

- [Getting Started](getting-started.md) — prerequisites, install, and a full working example
- [Concepts](concepts.md) — categories, hybrid search, lifecycle, conflict resolution
- [API Reference](api/store.md) — complete API documentation
