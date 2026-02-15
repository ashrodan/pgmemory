"""Single-table SQLAlchemy model.

One table. All production concerns — category, importance, temporal validity,
decay, provenance, hybrid search — are columns and indexes, not separate tables.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column,
    Computed,
    DateTime,
    Index,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import DeclarativeBase

from .types import Category, Memory


class Base(DeclarativeBase):
    pass


def build_table(table_name: str, embedding_dimensions: int) -> type:
    """Factory: create a MemoryRow ORM class with custom table name + vector size.

    Called once at store init time. The pgvector Vector() type needs the
    dimension at class-definition time, so we can't use a static class.
    """

    class MemoryRow(Base):
        __tablename__ = table_name

        # ── identity ────────────────────────────────────────────────
        id: int = Column(Integer, primary_key=True, autoincrement=True)
        app_name: str = Column(String(256), nullable=False)
        user_id: str = Column(String(256), nullable=False)

        # ── content ─────────────────────────────────────────────────
        content: str = Column(Text, nullable=False)  # the memory text
        content_embedding = Column(Vector(embedding_dimensions))
        content_tsv = Column(
            TSVECTOR,
            Computed("to_tsvector('english', content)", persisted=True),
        )

        # ── classification ──────────────────────────────────────────
        category: str = Column(
            String(32),
            nullable=False,
            default=Category.GENERAL.value,
            server_default=Category.GENERAL.value,
        )

        # ── lifecycle ───────────────────────────────────────────────
        importance: int = Column(
            Integer, nullable=False, default=1, server_default="1"
        )
        created_at: datetime = Column(
            DateTime(timezone=True), nullable=False, server_default=text("now()")
        )
        valid_from: datetime = Column(
            DateTime(timezone=True), nullable=False, server_default=text("now()")
        )
        valid_until: datetime | None = Column(DateTime(timezone=True), nullable=True)
        last_accessed: datetime | None = Column(DateTime(timezone=True), nullable=True)

        # ── provenance ──────────────────────────────────────────────
        source_session_id: str | None = Column(String(256), nullable=True)
        source_event_id: str | None = Column(String(256), nullable=True)
        source_event_timestamp: datetime | None = Column(
            DateTime(timezone=True), nullable=True
        )
        source_role: str | None = Column(String(32), nullable=True)

        # ── extensibility ───────────────────────────────────────────
        metadata_: dict = Column(
            "metadata",
            JSONB,
            nullable=False,
            server_default=text("'{}'::jsonb"),
        )

        # ── indexes ─────────────────────────────────────────────────
        __table_args__ = (
            Index(f"ix_{table_name}_user_app", "app_name", "user_id"),
            Index(f"ix_{table_name}_user_app_cat", "app_name", "user_id", "category"),
            Index(f"ix_{table_name}_importance", "importance"),
            Index(f"ix_{table_name}_created", "created_at"),
            Index(f"ix_{table_name}_valid_until", "valid_until"),
            Index(
                f"ix_{table_name}_tsv",
                "content_tsv",
                postgresql_using="gin",
            ),
            Index(
                f"ix_{table_name}_embedding",
                "content_embedding",
                postgresql_using="hnsw",
                postgresql_with={"m": 16, "ef_construction": 64},
                postgresql_ops={"content_embedding": "vector_cosine_ops"},
            ),
        )

        def to_memory(self) -> Memory:
            """Convert ORM row → domain Memory object."""
            return Memory(
                id=self.id,
                app_name=self.app_name,
                user_id=self.user_id,
                text=self.content,
                category=Category(self.category),
                importance=self.importance,
                created_at=self.created_at,
                valid_from=self.valid_from,
                valid_until=self.valid_until,
                last_accessed=self.last_accessed,
                source_session_id=self.source_session_id,
                source_event_id=self.source_event_id,
                source_event_timestamp=self.source_event_timestamp,
                source_role=self.source_role,
                metadata=self.metadata_ or {},
            )

        @classmethod
        def from_memory(cls, mem: Memory, embedding: list[float]) -> "MemoryRow":
            """Convert domain Memory → ORM row for insert."""
            return cls(
                app_name=mem.app_name,
                user_id=mem.user_id,
                content=mem.text,
                content_embedding=embedding,
                category=mem.category.value if isinstance(mem.category, Category) else mem.category,
                importance=mem.importance,
                created_at=mem.created_at,
                valid_from=mem.valid_from,
                valid_until=mem.valid_until,
                source_session_id=mem.source_session_id,
                source_event_id=mem.source_event_id,
                source_event_timestamp=mem.source_event_timestamp,
                source_role=mem.source_role,
                metadata_=mem.metadata,
            )

        def __repr__(self) -> str:
            return (
                f"<MemoryRow id={self.id} user={self.user_id} "
                f"cat={self.category} imp={self.importance}>"
            )

    return MemoryRow
