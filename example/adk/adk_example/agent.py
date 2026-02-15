import os

from google.adk.agents.llm_agent import Agent

from pgmemory import MemoryStore, OpenAIEmbeddingProvider
from pgmemory import MEMORY_INSTRUCTIONS
from pgmemory.adapters.adk import build_adk_tools, build_context_injection

store = MemoryStore(
    os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://mem_user:password@localhost/mem_db",
    ),
    OpenAIEmbeddingProvider(),
)

memory_tools = build_adk_tools(store)
inject_memory = build_context_injection(store)


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
