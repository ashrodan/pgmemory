"""Core domain types for pgmemory.

These are plain Python objects with no framework dependencies.
They are the lingua franca between the store, adapters, and your code.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


class Category(str, enum.Enum):
    """What kind of memory this is.

    Opinionated categories based on the Memori model. Every memory gets one.
    Use GENERAL as the catch-all when you don't know or don't care.
    """

    FACT = "fact"              # technical info, data points, definitions
    PREFERENCE = "preference"  # likes, dislikes, personal choices
    SKILL = "skill"            # competencies, learning progress
    CONTEXT = "context"        # project details, current situation
    RULE = "rule"              # constraints, policies, guidelines
    EVENT = "event"            # something that happened (episodic)
    GENERAL = "general"        # uncategorised / raw content


@dataclass
class Memory:
    """A single unit of remembered information.

    This is what you store, search for, and get back. Every memory belongs
    to one user within one app, has a category, and tracks its own lifecycle.
    """

    # ── identity ────────────────────────────────────────────────────
    id: int | None = None
    app_name: str = ""
    user_id: str = ""

    # ── content ─────────────────────────────────────────────────────
    text: str = ""
    category: Category = Category.GENERAL

    # ── lifecycle ───────────────────────────────────────────────────
    importance: int = 1           # 1 (low) to 5 (critical). higher = survives decay
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    valid_from: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    valid_until: datetime | None = None   # None = never expires
    last_accessed: datetime | None = None

    # ── provenance ──────────────────────────────────────────────────
    source_session_id: str | None = None  # which conversation this came from
    source_event_id: str | None = None    # which specific message
    source_event_timestamp: datetime | None = None
    source_role: str | None = None        # 'user', 'assistant', 'system', etc.

    # ── extensibility ───────────────────────────────────────────────
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        if self.valid_until is None:
            return False
        return datetime.now(timezone.utc) > self.valid_until

    @property
    def is_active(self) -> bool:
        now = datetime.now(timezone.utc)
        if self.valid_from > now:
            return False
        if self.valid_until is not None and self.valid_until < now:
            return False
        return True


@dataclass
class SearchResult:
    """A memory returned from search, enriched with relevance scoring."""

    memory: Memory
    similarity: float = 0.0       # cosine similarity (0–1)
    keyword_score: float = 0.0    # full-text search rank
    recency_score: float = 0.0    # time-decay score
    combined_score: float = 0.0   # weighted composite

    @property
    def id(self) -> int | None:
        return self.memory.id

    @property
    def text(self) -> str:
        return self.memory.text

    @staticmethod
    def format_results(results: list[SearchResult]) -> str:
        """Format search results as an LLM-readable string."""
        if not results:
            return "No relevant memories found."
        return "\n".join(
            f"[id={r.memory.id}] ({r.memory.category.value}, "
            f"score={r.combined_score:.2f}) {r.memory.text}"
            for r in results
        )


@dataclass
class SearchQuery:
    """Parameters for a memory search.

    The store will combine these into a hybrid SQL + semantic query.
    """

    app_name: str
    user_id: str
    text: str                       # natural language query
    categories: list[Category] | None = None  # filter to specific categories
    min_importance: int | None = None
    include_expired: bool = False
    top_k: int = 10
    similarity_threshold: float = 0.4

    # Scoring weights (must sum to ~1.0)
    weight_similarity: float = 0.6
    weight_keyword: float = 0.25
    weight_recency: float = 0.15
