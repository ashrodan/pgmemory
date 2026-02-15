# Google ADK Adapter

pgmemory integrates with [Google's Agent Development Kit (ADK)](https://google.github.io/adk-docs/) as both a memory service and a set of agent tools.

## Installation

```bash
pip install pgmemory[adk]
```

This installs `google-adk` and `google-genai` alongside pgmemory.

## Setup

```python
from pgmemory import MemoryStore, VertexEmbeddingProvider
from pgmemory.adapters.adk import ADKMemoryService, build_adk_tools
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.tools import preload_memory, load_memory

# 1. Create the store
store = MemoryStore(
    "postgresql+asyncpg://user:pass@localhost/mydb",
    VertexEmbeddingProvider(),
)

# 2. Wrap as ADK memory service
memory_service = ADKMemoryService(store)

# 3. Build agent tools
tools = build_adk_tools(store)

# 4. Wire into your agent
agent = LlmAgent(
    name="my_agent",
    model="gemini-2.0-flash",
    tools=[preload_memory, load_memory, *tools],
)

# 5. Create runner with memory service
runner = Runner(
    agent=agent,
    app_name="my_app",
    memory_service=memory_service,
)
```

## ADKMemoryService

`ADKMemoryService` implements ADK's `BaseMemoryService` interface:

- **`add_session_to_memory(session)`** — extracts text from session events, preserves provenance (session ID, event ID, timestamp, role), and batch-inserts into pgmemory
- **`search_memory(app_name, user_id, query)`** — runs hybrid search and returns results as ADK `SearchMemoryResponse`

```python
memory_service = ADKMemoryService(
    store,
    search_top_k=10,        # max results per search (default: 10)
    search_threshold=0.4,   # minimum similarity (default: 0.4)
)
```

## Agent tools

`build_adk_tools(store)` returns four `FunctionTool` instances:

| Tool | Description |
|------|-------------|
| `commit_session_to_memory` | Save the current conversation to long-term memory |
| `remember_fact` | Store a specific fact with category, importance, and optional expiry. Automatically runs conflict resolution via `supersede()` |
| `forget_memory` | Expire a specific memory by ID |
| `reinforce_memory` | Increase importance of a memory to prevent decay |

### remember_fact parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fact` | `str` | required | The information to remember |
| `category` | `str` | `"fact"` | One of: fact, preference, skill, context, rule, event |
| `importance` | `int` | `1` | 1 (low) to 5 (critical) |
| `expires_in_days` | `int \| None` | `None` | Days until expiry. Omit for permanent |

## Full example

```python
import asyncio
from pgmemory import MemoryStore, VertexEmbeddingProvider
from pgmemory.adapters.adk import ADKMemoryService, build_adk_tools
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import preload_memory, load_memory
from google.genai import types

async def main():
    store = MemoryStore(
        "postgresql+asyncpg://mem_user:password@localhost/mem_db",
        VertexEmbeddingProvider(),
    )
    await store.init()

    memory_service = ADKMemoryService(store)
    session_service = InMemorySessionService()

    agent = LlmAgent(
        name="assistant",
        model="gemini-2.0-flash",
        instruction="You are a helpful assistant with long-term memory.",
        tools=[preload_memory, load_memory, *build_adk_tools(store)],
    )

    runner = Runner(
        agent=agent,
        app_name="my_app",
        session_service=session_service,
        memory_service=memory_service,
    )

    session = await session_service.create_session(
        app_name="my_app", user_id="user_123"
    )

    message = types.Content(
        parts=[types.Part(text="Remember that I prefer Python over Java")],
        role="user",
    )

    async for event in runner.run_async(
        session_id=session.id, user_id="user_123", new_message=message
    ):
        if event.content and event.content.parts:
            print(event.content.parts[0].text)

    await store.close()

asyncio.run(main())
```
