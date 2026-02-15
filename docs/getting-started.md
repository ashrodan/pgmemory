# Getting Started

## Prerequisites

- **Python 3.11+**
- **PostgreSQL** with the [pgvector](https://github.com/pgvector/pgvector) extension

### Start PostgreSQL with pgvector

The easiest way to get a pgvector-enabled Postgres:

```bash
docker run -e POSTGRES_USER=mem_user \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=mem_db \
  --name pgmemory \
  -p 5432:5432 \
  -d ankane/pgvector
```

## Installation

=== "Core only"

    ```bash
    pip install pgmemory
    ```

=== "With Vertex AI"

    ```bash
    pip install pgmemory[vertex]
    ```

=== "With Ollama"

    ```bash
    pip install pgmemory[ollama]
    ```

=== "With OpenAI"

    ```bash
    pip install pgmemory[openai]
    ```

=== "With Google ADK"

    ```bash
    pip install pgmemory[adk]
    ```

=== "Everything"

    ```bash
    pip install pgmemory[all]
    ```

## Full working example

```python
import asyncio
from datetime import datetime, timedelta, timezone
from pgmemory import MemoryStore, Category, SearchQuery, OllamaEmbeddingProvider

async def main():
    # 1. Initialize the store
    store = MemoryStore(
        "postgresql+asyncpg://mem_user:password@localhost/mem_db",
        OllamaEmbeddingProvider(),  # or VertexEmbeddingProvider(), OpenAIEmbeddingProvider()
    )
    await store.init()

    # 2. Add memories
    pref_id = await store.add(
        "my_app", "user_123",
        "Prefers dark mode and compact layouts",
        category=Category.PREFERENCE,
        importance=3,
    )

    fact_id = await store.add(
        "my_app", "user_123",
        "Works at Acme Corp as a data engineer",
        category=Category.FACT,
        importance=2,
        source_session_id="sess_abc",
        metadata={"confidence": 0.95},
    )

    # 3. Search (hybrid: keyword + semantic + recency)
    results = await store.search(SearchQuery(
        app_name="my_app",
        user_id="user_123",
        text="UI preferences",
    ))

    for r in results:
        print(f"[{r.memory.category.value}] {r.text} (score={r.combined_score})")

    # 4. Lifecycle operations
    await store.promote(pref_id)              # bump importance, prevent decay
    await store.expire(fact_id, reason="outdated")  # soft-delete with audit trail
    await store.decay()                       # hard-delete expired memories

    # 5. Conflict resolution
    new_id, superseded = await store.supersede(
        "my_app", "user_123",
        "Now works at Dash Corp",
        Category.FACT,
    )
    print(f"New memory {new_id}, superseded: {superseded}")

    # 6. Admin
    users = await store.list_users("my_app")
    count = await store.count(app_name="my_app")
    print(f"Users: {users}, Total memories: {count}")

    await store.close()

asyncio.run(main())
```

## Project structure

```
pgmemory/
├── src/pgmemory/
│   ├── __init__.py          # public API (MemoryStore, Category, etc.)
│   ├── types.py             # Memory, SearchResult, SearchQuery, Category
│   ├── models.py            # single-table SQLAlchemy ORM
│   ├── embeddings.py        # pluggable providers (Vertex, Ollama, OpenAI)
│   ├── store.py             # MemoryStore — the core
│   └── adapters/
│       ├── adk.py           # Google ADK BaseMemoryService + tools
│       └── langchain.py     # LangChain memory + tools
└── tests/
    ├── helpers.py           # FakeEmbeddingProvider
    ├── test_unit.py         # no DB needed
    └── test_integration.py  # testcontainers pgvector
```
