import os

from google.adk.agents.llm_agent import Agent

from pgmemory import MemoryStore, OpenAIEmbeddingProvider
from pgmemory import MEMORY_INSTRUCTIONS
from pgmemory.adapters.adk import build_adk_tools, build_context_injection
from pgmemory.types import SearchQuery

store = MemoryStore(
    os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://mem_user:password@localhost/mem_db",
    ),
    OpenAIEmbeddingProvider(),
)

memory_tools = build_adk_tools(store)


def my_queries(text, app_name, user_id):
    """Generate multiple search queries for broader recall.

    Runs a targeted semantic search alongside a general profile load
    (empty text = most recent memories) so the model always has user
    context even when the query doesn't match stored memories well.
    """
    targeted = SearchQuery(app_name=app_name, user_id=user_id, text=text, top_k=5)
    profile = SearchQuery(app_name=app_name, user_id=user_id, text="", top_k=3)
    return [targeted, profile]


inject_memory = build_context_injection(store, queries=my_queries)


async def _init_memory(callback_context):
    """Eagerly create the pgvector extension and memory table on first run."""
    await store.init()


root_agent = Agent(
    model="gemini-3-flash-preview",
    name="root_agent",
    description="A helpful assistant with long-term memory.",
    instruction=(
        "You are a helpful assistant with long-term memory.\n\n"
        + MEMORY_INSTRUCTIONS
    ),
    tools=memory_tools,
    before_agent_callback=_init_memory,
    before_model_callback=inject_memory,
)
