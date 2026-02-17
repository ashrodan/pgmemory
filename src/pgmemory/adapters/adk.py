"""Google ADK adapter.

Wraps pgmemory's MemoryStore into ADK's BaseMemoryService interface.
Also provides ADK FunctionTools for agent-driven memory operations.

    pip install pgmemory[adk]

Usage:
    from pgmemory import MemoryStore, VertexEmbeddingProvider
    from pgmemory.adapters.adk import ADKMemoryService, build_adk_tools

    store = MemoryStore("postgresql+asyncpg://...", VertexEmbeddingProvider())
    memory_service = ADKMemoryService(store)

    tools = build_adk_tools(store)
    agent = LlmAgent(
        ...,
        tools=[preload_memory, load_memory, *tools],
    )
    runner = Runner(
        ...,
        memory_service=memory_service,
    )
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

from typing_extensions import override

from .. import MEMORY_INSTRUCTIONS
from ..store import MemoryStore
from ..types import Category, Memory, SearchQuery, SearchResult

logger = logging.getLogger(__name__)


class ADKMemoryService:
    """ADK BaseMemoryService backed by pgmemory's MemoryStore.

    Implements the two required methods:
    - add_session_to_memory(session)
    - search_memory(app_name, user_id, query)

    Inherits from BaseMemoryService at runtime (so ADK recognises it)
    but is defined here to keep the import optional.
    """

    def __init__(
        self,
        store: MemoryStore,
        *,
        search_top_k: int = 10,
        search_threshold: float = 0.4,
    ):
        # Defer the import so pgmemory core doesn't depend on ADK
        from google.adk.memory import BaseMemoryService

        # Dynamically make this instance also a BaseMemoryService
        self.__class__ = type(
            "ADKMemoryService",
            (BaseMemoryService,),
            {
                "add_session_to_memory": self.add_session_to_memory,
                "search_memory": self.search_memory,
            },
        )
        BaseMemoryService.__init__(self)

        self._store = store
        self._search_top_k = search_top_k
        self._search_threshold = search_threshold

    async def add_session_to_memory(self, session) -> None:
        """Ingest ADK Session events into pgmemory.

        Extracts text from each event, preserves provenance (session ID,
        event ID, timestamp, role), and batch-inserts.
        """
        memories: list[Memory] = []

        for event in session.events:
            content = getattr(event, "content", None)
            if not content or not content.parts:
                continue

            texts = [p.text for p in content.parts if getattr(p, "text", None)]
            if not texts:
                continue

            event_text = " ".join(texts).strip()
            if not event_text:
                continue

            role = getattr(content, "role", None) or "user"
            ts = (
                datetime.fromtimestamp(event.timestamp, tz=timezone.utc)
                if hasattr(event, "timestamp") and event.timestamp
                else datetime.now(timezone.utc)
            )

            memories.append(
                Memory(
                    app_name=getattr(session, "app_name", ""),
                    user_id=session.user_id,
                    text=event_text,
                    category=Category.GENERAL,
                    importance=1,
                    created_at=ts,
                    valid_from=ts,
                    source_session_id=session.id,
                    source_event_id=getattr(event, "id", None),
                    source_event_timestamp=ts,
                    source_role=role,
                    metadata={"raw_content": json.dumps(
                        content.model_dump(exclude_none=True, mode="json")
                    )},
                )
            )

        if memories:
            await self._store.add_many(memories)
            logger.info(
                "ADK: ingested %d events from session %s", len(memories), session.id
            )

    async def search_memory(self, *, app_name: str, user_id: str, query: str):
        """Search pgmemory and return ADK SearchMemoryResponse."""
        from google.adk.memory.base_memory_service import SearchMemoryResponse
        from google.adk.memory.memory_entry import MemoryEntry
        from google.genai import types

        results = await self._store.search(
            SearchQuery(
                app_name=app_name,
                user_id=user_id,
                text=query,
                top_k=self._search_top_k,
                similarity_threshold=self._search_threshold,
            )
        )

        entries = []
        for r in results:
            # Try to reconstruct Content from raw_content if available
            raw = r.memory.metadata.get("raw_content")
            if raw:
                try:
                    data = json.loads(raw)
                    content = types.Content(
                        parts=[types.Part(**p) for p in data.get("parts", [])],
                        role=data.get("role", "user"),
                    )
                except Exception:
                    content = types.Content(
                        parts=[types.Part(text=r.memory.text)],
                        role=r.memory.source_role or "user",
                    )
            else:
                content = types.Content(
                    parts=[types.Part(text=r.memory.text)],
                    role=r.memory.source_role or "user",
                )

            entries.append(
                MemoryEntry(
                    content=content,
                    author=r.memory.source_role or "user",
                    timestamp=str(r.memory.created_at.timestamp()),
                    custom_metadata={
                        "memory_id": r.memory.id,
                        "category": r.memory.category.value,
                        "importance": r.memory.importance,
                        "similarity": r.similarity,
                        "keyword_score": r.keyword_score,
                        "combined_score": r.combined_score,
                        "source_session_id": r.memory.source_session_id,
                        "source_event_id": r.memory.source_event_id,
                        **{
                            k: v
                            for k, v in r.memory.metadata.items()
                            if k != "raw_content"
                        },
                    },
                )
            )

        return SearchMemoryResponse(memories=entries)


# ────────────────────────────────────────────────────────────────────────
# ADK Function Tools
# ────────────────────────────────────────────────────────────────────────


def build_adk_tools(store: MemoryStore) -> list:
    """Create ADK FunctionTools bound to a MemoryStore.

    Returns tools for: search_memory, commit_session, remember_fact, forget, reinforce.
    Wire into your agent alongside ADK's preload_memory / load_memory.
    """
    from google.adk.tools import FunctionTool, ToolContext

    async def commit_session_to_memory(tool_context: ToolContext) -> str:
        """Save the current conversation to long-term memory."""
        session = tool_context._invocation_context.session
        svc = tool_context._invocation_context.memory_service
        if svc is None:
            return "Memory service not configured."
        await svc.add_session_to_memory(session)
        return f"Session {session.id} committed to memory."

    async def remember_fact(
        tool_context: ToolContext,
        fact: str,
        category: str = "fact",
        importance: int = 1,
        expires_in_days: Optional[int] = None,
    ) -> str:
        """Store a specific fact with category and importance.

        Args:
            fact: The information to remember.
            category: One of: fact, preference, skill, context, rule, event.
            importance: 1 (low) to 5 (critical).
            expires_in_days: Days until this memory expires. Omit for permanent.
        """
        session = tool_context._invocation_context.session
        try:
            cat = Category(category.lower())
        except ValueError:
            cat = Category.GENERAL

        valid_until = None
        if expires_in_days and expires_in_days > 0:
            valid_until = datetime.now(timezone.utc) + timedelta(days=expires_in_days)

        new_id, superseded = await store.supersede(
            getattr(session, "app_name", ""),
            session.user_id,
            fact,
            cat,
            importance=max(1, min(5, importance)),
            valid_until=valid_until,
            source_session_id=session.id,
            metadata={"extracted_by": "agent_tool"},
        )
        msg = f'Remembered (id={new_id}): "{fact}" [{cat.value}, imp={importance}]'
        if superseded:
            msg += f" Superseded {len(superseded)} older memor{'y' if len(superseded)==1 else 'ies'}."
        return msg

    async def forget_memory(tool_context: ToolContext, memory_id: int) -> str:
        """Expire a specific memory by ID.

        Args:
            memory_id: The ID of the memory to expire.
        """
        await store.expire(memory_id, reason="forgotten by user request")
        return f"Memory {memory_id} expired."

    async def reinforce_memory(tool_context: ToolContext, memory_id: int) -> str:
        """Increase importance of a useful memory to prevent decay.

        Args:
            memory_id: The ID of the memory to reinforce.
        """
        await store.promote(memory_id)
        return f"Memory {memory_id} reinforced."

    async def search_memory(
        tool_context: ToolContext,
        query: str = "",
        category: Optional[str] = None,
    ) -> str:
        """Search long-term memory for relevant information.

        Args:
            query: Natural language search query. Leave blank to browse by category.
            category: Optional category filter (fact, preference, skill, context, rule, event).
        """
        session = tool_context._invocation_context.session
        categories = None
        if category:
            try:
                categories = [Category(category.lower())]
            except ValueError:
                pass

        results = await store.search(
            SearchQuery(
                app_name=getattr(session, "app_name", ""),
                user_id=session.user_id,
                text=query,
                categories=categories,
            )
        )

        return SearchResult.format_results(results)

    return [
        FunctionTool(search_memory),
        FunctionTool(commit_session_to_memory),
        FunctionTool(remember_fact),
        FunctionTool(forget_memory),
        FunctionTool(reinforce_memory),
    ]


def build_context_injection(
    store: MemoryStore,
    *,
    top_k: int = 5,
    similarity_threshold: float = 0.5,
    profile_k: int = 3,
    queries: Callable[[str, str, str], list[SearchQuery]] | None = None,
):
    """Return a before_model_callback that injects relevant memories.

    On each LLM turn, searches the MemoryStore with the user's message
    and appends matching results as instructions so the model sees them
    without needing to call search_memory itself.

    When ``queries`` is provided, it receives ``(query_text, app_name, user_id)``
    and should return a list of ``SearchQuery`` objects. All queries are run in
    parallel via ``store.search_many()`` and the merged, deduplicated results are
    injected. When ``queries`` is not provided, a single search is performed
    (backward compatible).

    When no results match the query, falls back to injecting the top
    ``profile_k`` most recent memories as general user context.
    """

    async def _inject(callback_context, llm_request):
        session = callback_context._invocation_context.session
        user_content = callback_context.user_content
        if not user_content or not user_content.parts:
            return None
        texts = [p.text for p in user_content.parts if getattr(p, "text", None)]
        query_text = " ".join(texts).strip()
        if not query_text:
            return None

        app_name = getattr(session, "app_name", "")

        if queries is not None:
            query_list = queries(query_text, app_name, session.user_id)
            results = await store.search_many(query_list, top_k=top_k)

            # Split results: targeted (had query text) vs profile (empty text)
            targeted = [
                r for r in results
                if r.source_query and r.source_query.text.strip()
            ]
            profile = [
                r for r in results
                if not r.source_query or not r.source_query.text.strip()
            ]

            parts = []
            if targeted:
                parts.append(
                    "Relevant memories:\n"
                    + SearchResult.format_results(targeted)
                )
            if profile:
                parts.append(
                    "General context about this user:\n"
                    + SearchResult.format_results(profile)
                )
            if parts:
                llm_request.append_instructions(parts)
        else:
            results = await store.search(
                SearchQuery(
                    app_name=app_name,
                    user_id=session.user_id,
                    text=query_text,
                    top_k=top_k,
                    similarity_threshold=similarity_threshold,
                )
            )

            if results:
                llm_request.append_instructions([
                    "Relevant memories:\n" + SearchResult.format_results(results)
                ])
            elif profile_k > 0:
                # No direct matches — inject a user profile from recent memories
                profile = await store.search(
                    SearchQuery(
                        app_name=app_name,
                        user_id=session.user_id,
                        text="",
                        top_k=profile_k,
                    )
                )
                if profile:
                    llm_request.append_instructions([
                        "General context about this user:\n"
                        + SearchResult.format_results(profile)
                    ])
        return None

    return _inject
