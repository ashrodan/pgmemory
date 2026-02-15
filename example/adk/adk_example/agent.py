import os

from google.adk.agents.llm_agent import Agent

from pgmemory import MemoryStore, OpenAIEmbeddingProvider
from pgmemory.adapters.adk import build_adk_tools

store = MemoryStore(
    os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://mem_user:password@localhost/mem_db",
    ),
    OpenAIEmbeddingProvider(),
)

memory_tools = build_adk_tools(store)

root_agent = Agent(
    model="gemini-2.5-flash",
    name="root_agent",
    description="A helpful assistant with long-term memory.",
    instruction=(
        "You are a helpful assistant with long-term memory. "
        "Use the remember_fact tool to store important information the user tells you "
        "(preferences, facts about them, project details, etc). "
        "Use reinforce_memory to mark a memory as important when the user confirms it. "
        "Use forget_memory to remove outdated information. "
        "When the user asks you to recall something, search your memory first."
    ),
    tools=memory_tools,
)
