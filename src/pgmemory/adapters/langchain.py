"""LangChain / LangGraph adapter.

Wraps pgmemory's MemoryStore as a LangChain-compatible memory class
and provides LangChain Tools for agent-driven operations.

    pip install pgmemory  # no extra needed — langchain is your dep

Usage:
    from pgmemory import MemoryStore, OpenAIEmbeddingProvider
    from pgmemory.adapters.langchain import LangChainMemory

    store = MemoryStore("postgresql+asyncpg://...", OpenAIEmbeddingProvider())
    memory = LangChainMemory(store, app_name="my_app", user_id="user_1")

    # Use in a chain
    chain = ConversationChain(memory=memory, ...)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..store import MemoryStore
from ..types import Category, Memory, SearchQuery, SearchResult

logger = logging.getLogger(__name__)


class LangChainMemory:
    """LangChain BaseChatMessageHistory-compatible wrapper around pgmemory.

    Implements the interface LangChain expects:
    - add_message(message)
    - messages  (property)
    - clear()

    Also exposes search for retrieval-augmented generation.
    """

    def __init__(
        self,
        store: MemoryStore,
        app_name: str,
        user_id: str,
        *,
        session_id: str | None = None,
        search_top_k: int = 10,
    ):
        self._store = store
        self._app_name = app_name
        self._user_id = user_id
        self._session_id = session_id
        self._search_top_k = search_top_k

    async def aadd_message(self, message: Any) -> int:
        """Store a message as a memory. Returns memory ID.

        Accepts a string or any object with a .content attribute
        (e.g. LangChain BaseMessage).
        """
        if isinstance(message, str):
            text = message
            role = "user"
        else:
            text = getattr(message, "content", str(message))
            role = getattr(message, "type", "user")  # human, ai, system

        return await self._store.add(
            self._app_name,
            self._user_id,
            text,
            source_session_id=self._session_id,
            source_role=role,
        )

    async def asearch(self, query: str, **kwargs) -> list[SearchResult]:
        """Semantic + keyword hybrid search."""
        return await self._store.search(
            SearchQuery(
                app_name=self._app_name,
                user_id=self._user_id,
                text=query,
                top_k=kwargs.get("top_k", self._search_top_k),
                **{k: v for k, v in kwargs.items() if k != "top_k"},
            )
        )

    async def aclear(self) -> int:
        """Delete all memories for this user. Returns count."""
        return await self._store.delete_user(self._app_name, self._user_id)

    # Sync wrappers for LangChain's sync interface
    def add_message(self, message: Any) -> int:
        return asyncio.get_event_loop().run_until_complete(self.aadd_message(message))

    def search(self, query: str, **kwargs) -> list[SearchResult]:
        return asyncio.get_event_loop().run_until_complete(self.asearch(query, **kwargs))

    def clear(self) -> int:
        return asyncio.get_event_loop().run_until_complete(self.aclear())


def build_langchain_tools(store: MemoryStore, app_name: str, user_id: str) -> list:
    """Create LangChain Tool objects bound to a MemoryStore.

    Returns tools compatible with LangChain's agent framework.
    """
    try:
        from langchain_core.tools import Tool
    except ImportError:
        raise ImportError(
            "langchain-core is required for LangChain tools. "
            "pip install langchain-core"
        )

    async def _search(query: str) -> str:
        results = await store.search(
            SearchQuery(app_name=app_name, user_id=user_id, text=query)
        )
        if not results:
            return "No relevant memories found."
        return "\n".join(
            f"[id={r.memory.id} cat={r.memory.category.value} sim={r.similarity}] {r.text}"
            for r in results
        )

    async def _remember(fact: str) -> str:
        mid = await store.add(app_name, user_id, fact, category=Category.FACT)
        return f"Remembered (id={mid}): {fact}"

    async def _forget(memory_id_str: str) -> str:
        await store.expire(int(memory_id_str))
        return f"Memory {memory_id_str} expired."

    return [
        Tool(name="search_memory", func=None, coroutine=_search,
             description="Search long-term memory for relevant past information."),
        Tool(name="remember_fact", func=None, coroutine=_remember,
             description="Store a fact in long-term memory."),
        Tool(name="forget_memory", func=None, coroutine=_forget,
             description="Expire a memory by its ID."),
    ]
