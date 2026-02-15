# LangChain Adapter

pgmemory provides a LangChain-compatible memory class and agent tools.

## Installation

No extra dependencies needed — just install `langchain-core` in your project alongside pgmemory:

```bash
pip install pgmemory langchain-core
```

## LangChainMemory

`LangChainMemory` wraps pgmemory as a LangChain-compatible message history interface.

```python
from pgmemory import MemoryStore, OpenAIEmbeddingProvider
from pgmemory.adapters.langchain import LangChainMemory

store = MemoryStore(
    "postgresql+asyncpg://user:pass@localhost/mydb",
    OpenAIEmbeddingProvider(),
)
await store.init()

memory = LangChainMemory(
    store,
    app_name="my_app",
    user_id="user_1",
    session_id="sess_abc",   # optional, for provenance tracking
    search_top_k=10,         # default: 10
)
```

### Async usage (recommended)

```python
# Add a message
await memory.aadd_message("User mentioned they prefer Python over Java")

# Search memories
results = await memory.asearch("programming language preferences")
for r in results:
    print(f"[{r.memory.category.value}] {r.text} (sim={r.similarity})")

# Clear all memories for this user
deleted = await memory.aclear()
```

### Sync usage

Sync wrappers are provided for LangChain's sync interface:

```python
memory.add_message("User mentioned they prefer Python over Java")
results = memory.search("programming language preferences")
memory.clear()
```

!!! note
    Sync methods use `asyncio.get_event_loop().run_until_complete()` internally. If you're already in an async context, use the `a`-prefixed methods.

## Agent tools

`build_langchain_tools()` creates LangChain `Tool` objects for agent use:

```python
from pgmemory.adapters.langchain import build_langchain_tools

tools = build_langchain_tools(store, "my_app", "user_1")
```

| Tool | Description |
|------|-------------|
| `search_memory` | Search long-term memory for relevant past information |
| `remember_fact` | Store a fact in long-term memory |
| `forget_memory` | Expire a memory by its ID |

All tools are async-native (`coroutine` parameter).

## Full example

```python
import asyncio
from pgmemory import MemoryStore, OpenAIEmbeddingProvider
from pgmemory.adapters.langchain import LangChainMemory, build_langchain_tools

async def main():
    store = MemoryStore(
        "postgresql+asyncpg://mem_user:password@localhost/mem_db",
        OpenAIEmbeddingProvider(),
    )
    await store.init()

    # As message history
    memory = LangChainMemory(store, app_name="my_app", user_id="user_1")
    await memory.aadd_message("I'm working on a data pipeline in Python")
    await memory.aadd_message("The deadline is end of March")

    results = await memory.asearch("current project")
    for r in results:
        print(f"{r.text} (score={r.combined_score})")

    # As agent tools
    tools = build_langchain_tools(store, "my_app", "user_1")
    # Pass `tools` to your LangChain agent

    await store.close()

asyncio.run(main())
```
