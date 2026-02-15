# Concepts

## Categories

Every memory gets exactly one category. These are based on the Memori model and cover the types of information agents typically need to remember.

| Category | Enum | When to use |
|----------|------|-------------|
| **Fact** | `Category.FACT` | Technical info, data points, definitions — "Works at Acme Corp" |
| **Preference** | `Category.PREFERENCE` | Likes, dislikes, personal choices — "Prefers dark mode" |
| **Skill** | `Category.SKILL` | Competencies, learning progress — "Knows Python and SQL" |
| **Context** | `Category.CONTEXT` | Project details, current situation — "Working on Q4 migration" |
| **Rule** | `Category.RULE` | Constraints, policies, guidelines — "Never deploy on Fridays" |
| **Event** | `Category.EVENT` | Something that happened (episodic) — "Had a meeting with PM" |
| **General** | `Category.GENERAL` | Uncategorised or raw content — the catch-all |

Filter searches by category to reduce noise:

```python
results = await store.search(SearchQuery(
    app_name="my_app",
    user_id="user_1",
    text="what does the user like?",
    categories=[Category.PREFERENCE],
))
```

## Hybrid search

Every search combines three signals into a single score:

| Signal | Method | What it catches |
|--------|--------|-----------------|
| **Semantic** | pgvector cosine similarity | "UI preferences" finds "likes dark mode" |
| **Keyword** | PostgreSQL `ts_rank` + `tsvector` | Exact terms, names, codes |
| **Recency** | Time-decay function | Recent memories rank higher |

### Scoring formula

```
combined = (w_sim × cosine_similarity)
         + (w_kw  × ts_rank)
         + (w_rec × recency_decay)
```

Where `recency_decay = 1 / (1 + days_old / 30)`

### Default weights

| Weight | Default | Description |
|--------|---------|-------------|
| `weight_similarity` | 0.6 | Semantic similarity via pgvector |
| `weight_keyword` | 0.25 | Full-text keyword matching |
| `weight_recency` | 0.15 | Time-decay bonus for recent memories |

### Tuning weights

Adjust per query to match your use case:

```python
# Lean heavily into semantic similarity
SearchQuery(
    ...,
    weight_similarity=0.8,
    weight_keyword=0.15,
    weight_recency=0.05,
)

# Prioritise exact keyword matches
SearchQuery(
    ...,
    weight_similarity=0.4,
    weight_keyword=0.5,
    weight_recency=0.1,
)
```

## Memory lifecycle

### Importance

Memories have an importance level from 1 (low) to 5 (critical). Higher importance means the memory survives decay longer.

- Default importance is 1
- Call `store.promote(memory_id)` to bump importance and clear any expiry
- Filter by minimum importance in searches with `min_importance`

### Temporal validity

Memories can have a time window during which they're valid:

- `valid_from` — when the memory becomes active (defaults to now)
- `valid_until` — when the memory expires (`None` = never)

Expired memories are automatically excluded from search results and cleaned up by `decay()`.

```python
from datetime import datetime, timedelta, timezone

# Memory that expires in 30 days
await store.add("my_app", "user_1", "User is on parental leave",
                category=Category.CONTEXT,
                valid_until=datetime.now(timezone.utc) + timedelta(days=30))
```

### Promote, expire, decay

| Operation | What it does |
|-----------|-------------|
| `store.promote(id)` | Bump importance by 1, clear `valid_until`, update `last_accessed` |
| `store.expire(id)` | Set `valid_until` to now, log reason in metadata |
| `store.decay()` | Hard-delete all memories past their `valid_until` |
| `store.soft_expire_stale()` | Set `valid_until` on old, low-importance memories that never expire |

## Conflict resolution

When you learn something new that contradicts an existing memory, use `supersede()`:

```python
new_id, superseded = await store.supersede(
    "my_app", "user_123",
    "Now works at Dash Corp",       # new fact
    Category.FACT,                  # same category
)
# If "Works at Acme Corp" exists with similarity > 0.85,
# it gets soft-expired and the new memory replaces it.
```

### How it works

1. `find_conflicts()` embeds the new text and searches for active memories in the same category with cosine similarity above the threshold (default: 0.85)
2. Each conflict is soft-expired with reason `"superseded (sim=X.XX)"` — the old memory is preserved for audit
3. The new memory is inserted

### Threshold tuning

- **0.85** (default) — only very similar memories are treated as conflicts
- **0.7** — more aggressive, catches broader semantic overlaps
- **0.95** — conservative, only near-duplicates

```python
new_id, superseded = await store.supersede(
    "my_app", "user_1", "...", Category.FACT,
    threshold=0.7,  # more aggressive conflict detection
)
```

## Multi-tenancy

Every memory is scoped by two dimensions:

- `app_name` — isolates different applications sharing the same database
- `user_id` — isolates different users within an application

All operations (search, add, lifecycle, admin) require these scoping parameters. There is no cross-user or cross-app leakage.

## Provenance

Every memory records where it came from:

| Field | Purpose |
|-------|---------|
| `source_session_id` | Which conversation produced this memory |
| `source_event_id` | Which specific message |
| `source_event_timestamp` | When that message happened |
| `source_role` | Who said it: `user`, `assistant`, `system` |

This lets you trace any memory back to its origin. The ADK adapter populates these automatically from session events.

## Schema

pgmemory uses a single table with well-chosen indexes:

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

**Indexes:** `(app_name, user_id)`, `(app_name, user_id, category)`, `importance`, `created_at`, `valid_until`, GIN on tsvector, HNSW on embedding.
