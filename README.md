# pgmemory

Opinionated multi-user agent memory on PostgreSQL + pgvector.

**One table. Hybrid search. Lifecycle management. Any framework.**

```
pip install pgmemory
```

## The idea

Every agent framework re-invents memory. They all need the same thing: store what the agent learned about a user, find it later, let old stuff fade. The database part is always the same — it's Postgres with vectors.

pgmemory is *that database part*, extracted into a standalone library. Use it with ADK, LangChain, CrewAI, Semantic Kernel, or plain Python. The core has zero framework dependencies.

## Quickstart

```python
from pgmemory import MemoryStore, Category, SearchQuery, OllamaEmbeddingProvider

store = MemoryStore(
    "postgresql+asyncpg://user:pass@localhost/mydb",
    OllamaEmbeddingProvider(),        # or VertexEmbeddingProvider(), OpenAIEmbeddingProvider()
)
await store.init()

# Store
await store.add("my_app", "user_123", "Prefers dark mode and compact layouts",
                category=Category.PREFERENCE, importance=3)

await store.add("my_app", "user_123", "Works at Acme Corp as a data engineer",
                category=Category.FACT, importance=2,
                source_session_id="sess_abc",  # link back to the conversation
                metadata={"confidence": 0.95})

# Search (hybrid: keyword + semantic + recency)
results = await store.search(SearchQuery(
    app_name="my_app",
    user_id="user_123",
    text="UI preferences",
))

for r in results:
    print(f"[{r.memory.category.value}] {r.text} (score={r.combined_score})")

# Lifecycle
await store.promote(memory_id)        # reinforce — bump importance, prevent decay
await store.expire(memory_id)         # soft-delete with reason in metadata
await store.decay()                   # hard-delete everything past valid_until

# Conflict resolution
new_id, superseded = await store.supersede(
    "my_app", "user_123",
    "Now works at Dash Corp",         # new fact
    Category.FACT,                    # same category
)   # → automatically expires "Works at Acme Corp" if similarity > 0.85
```

## Schema (single table)

```
memory
├── id                      SERIAL PK
├── app_name                TEXT           ── multi-app isolation
├── user_id                 TEXT           ── per-user scoping
│
├── content                 TEXT           ── the memory text
├── content_embedding       VECTOR(n)      ── cosine similarity search
├── content_tsv             TSVECTOR       ── generated, for keyword search
│
├── category                TEXT           ── fact/preference/skill/context/rule/event/general
├── importance              INT (1-5)      ── higher = survives decay longer
│
├── created_at              TIMESTAMPTZ
├── valid_from              TIMESTAMPTZ    ── when this became true
├── valid_until             TIMESTAMPTZ    ── NULL = never expires
├── last_accessed           TIMESTAMPTZ    ── updated on search retrieval
│
├── source_session_id       TEXT           ── which conversation
├── source_event_id         TEXT           ── which message
├── source_event_timestamp  TIMESTAMPTZ    ── when that message happened
├── source_role             TEXT           ── user / assistant / system
│
└── metadata                JSONB          ── your extensible data
```

**Indexes:** (app_name, user_id), (app_name, user_id, category), importance, created_at, valid_until, GIN on tsvector, HNSW on embedding.

## Hybrid search

Every search combines three signals:

| Signal | Method | What it catches |
|--------|--------|-----------------|
| **Semantic** | pgvector cosine similarity | "UI preferences" finds "likes dark mode" |
| **Keyword** | PostgreSQL `ts_rank` + `tsvector` | Exact terms, names, codes |
| **Recency** | Time-decay function | Recent memories rank higher |

Combined score: `0.6 × similarity + 0.25 × keyword_rank + 0.15 × recency`

**Embedding enrichment** — by default, pgmemory prepends the memory's category to the text before embedding (e.g. `"rule: Never store passwords in plaintext"`). This improves search quality by giving the embedding model category context. Disable with `enrich_embeddings=False` in `MemoryStore()`.

Weights are configurable per query:

```python
SearchQuery(
    ...,
    weight_similarity=0.8,   # lean into semantic
    weight_keyword=0.15,
    weight_recency=0.05,
)
```

## Memory lifecycle

**Categories** — every memory gets one: `fact`, `preference`, `skill`, `context`, `rule`, `event`, `general`. Filter searches by category to reduce noise.

**Importance** (1–5) — memories start at 1. Call `promote()` when a memory proves useful. High-importance memories survive decay.

**Temporal validity** — set `valid_until` for time-bound facts ("user is on project X this quarter"). Expired memories are excluded from search and cleaned by `decay()`.

**Conflict resolution** — `supersede()` checks if a new memory semantically duplicates an existing one in the same category. If yes, the old one is soft-expired (audit trail preserved) and the new one replaces it.

**Provenance** — every memory records where it came from: `source_session_id`, `source_event_id`, `source_event_timestamp`, `source_role`. You always know *why* something is in memory.

## Framework adapters

The core is framework-agnostic. Adapters are thin wrappers.

### Google ADK

```python
pip install pgmemory[adk]
```

```python
from pgmemory import MemoryStore, VertexEmbeddingProvider
from pgmemory.adapters.adk import ADKMemoryService, build_adk_tools
from google.adk.tools import preload_memory, load_memory

store = MemoryStore("postgresql+asyncpg://...", VertexEmbeddingProvider())
memory_service = ADKMemoryService(store)

agent = LlmAgent(
    ...,
    tools=[preload_memory, load_memory, *build_adk_tools(store)],
)
runner = Runner(..., memory_service=memory_service)
```

ADK tools provided: `commit_session_to_memory`, `remember_fact` (with category + importance + expiry + auto conflict resolution), `forget_memory`, `reinforce_memory`.

### LangChain / LangGraph

```python
from pgmemory import MemoryStore, OpenAIEmbeddingProvider
from pgmemory.adapters.langchain import LangChainMemory, build_langchain_tools

store = MemoryStore("postgresql+asyncpg://...", OpenAIEmbeddingProvider())
memory = LangChainMemory(store, app_name="my_app", user_id="user_1")

# Message history interface
await memory.aadd_message("User mentioned they prefer Python over Java")
results = await memory.asearch("programming language preferences")

# Or as agent tools
tools = build_langchain_tools(store, "my_app", "user_1")
```

### Any framework (direct use)

```python
from pgmemory import MemoryStore, Category, SearchQuery

store = MemoryStore("postgresql+asyncpg://...", my_embedder)
await store.init()

# That's it. Call store.add(), store.search(), store.promote(), etc.
# Wrap in whatever interface your framework needs.
```

### Writing a new adapter

Adapters are ~100 lines. They translate between your framework's interface and `MemoryStore`. See `pgmemory/adapters/adk.py` for the pattern:

1. Implement your framework's memory interface
2. Delegate to `store.add()`, `store.search()`, `store.add_many()`
3. Convert between your framework's types and `Memory` / `SearchResult`

## Embedding providers

```python
# Google Vertex AI (768d)
from pgmemory import VertexEmbeddingProvider
embedder = VertexEmbeddingProvider(model="text-embedding-004")

# Local Ollama (768d)
from pgmemory import OllamaEmbeddingProvider
embedder = OllamaEmbeddingProvider(model="nomic-embed-text")

# OpenAI (1536d)
from pgmemory import OpenAIEmbeddingProvider
embedder = OpenAIEmbeddingProvider(model="text-embedding-3-small")

# Custom
from pgmemory import EmbeddingProvider

class MyEmbedder(EmbeddingProvider):
    @property
    def dimensions(self) -> int:
        return 384
    async def embed(self, texts):
        return [my_model.encode(t) for t in texts]
```

## Admin operations

```python
# GDPR right-to-erasure
await store.delete_user("my_app", "user_123")

# List active users
users = await store.list_users("my_app")

# Count memories
total = await store.count(app_name="my_app")

# Scheduled maintenance
await store.soft_expire_stale(max_age_days=90, min_importance=3)
await store.decay(app_name="my_app")
```

## Docker (pgvector)

```bash
docker run -e POSTGRES_USER=mem_user \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=mem_db \
  --name pgmemory \
  -p 5432:5432 \
  -d ankane/pgvector
```

## Testing

```bash
# Unit tests (no database)
pytest tests/test_unit.py -v

# Integration tests (requires Docker)
pip install pgmemory[dev]
pytest tests/test_integration.py -v
```

## Evals

Search quality evaluation across embedding providers. Requires API keys and a running Postgres instance.

```bash
# Single provider
source ~/.secrets
uv run --extra openai python scripts/eval_search.py --provider openai

# Without embedding enrichment (enrichment is on by default)
uv run --extra openai python scripts/eval_search.py --provider openai --no-enrich

# All providers × enrich on/off
uv run --extra openai --extra vertex --extra voyage python scripts/eval_search.py --matrix

# With percentile filtering (drop bottom 30% of results)
uv run --extra openai python scripts/eval_search.py --provider openai --percentile 0.3
```

Metrics reported: Top-1 Accuracy, MRR, Precision@3, Discrimination, and per-query/seed timing stats.

There's also a GitHub Actions workflow (`.github/workflows/eval.yml`) that can be triggered manually from the Actions tab.

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

## Credits

Built on the research and work of: [Michael Gordon](https://medium.com/@cosmic.mick/developing-a-pgvector-based-memory-service-for-google-adk-e3a5ed5705de) (pgvector ADK reference implementation), [Memori Labs](https://memorilabs.ai/blog/ai-agent-memory-on-postgres-back-to-sql/) (SQL-first memory thesis + category model), and the [HN discussion](https://news.ycombinator.com/item?id=45329322) that confirmed: most agent memory is structured facts — SQL was designed for this.

## License

MIT
