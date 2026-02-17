"""The core memory store.

This is pgmemory. No framework dependencies — just PostgreSQL, pgvector,
and opinionated conventions for multi-user agent memory.

Hybrid search: SQL filters → full-text keyword matching → cosine similarity,
combined with a weighted score that includes recency.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Sequence

from sqlalchemy import delete, func, literal_column, select, text, update
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from .embeddings import EmbeddingProvider
from .models import Base, build_table
from .types import Category, Memory, SearchQuery, SearchResult

logger = logging.getLogger(__name__)


class MemoryStore:
    """Multi-user memory store backed by PostgreSQL + pgvector.

    One table. Hybrid search. Lifecycle management. That's it.

    Usage:
        store = MemoryStore(
            connection_string="postgresql+asyncpg://user:pass@localhost/mydb",
            embedding_provider=my_embedder,
        )
        await store.init()

        # Store
        mem_id = await store.add("my_app", "user_1", "User likes dark mode",
                                  category=Category.PREFERENCE, importance=2)

        # Search (hybrid: keyword + semantic + recency)
        results = await store.search(SearchQuery(
            app_name="my_app", user_id="user_1", text="UI preferences"
        ))

        # Lifecycle
        await store.promote(mem_id)
        await store.expire(mem_id)
        await store.decay()
    """

    def __init__(
        self,
        connection_string: str,
        embedding_provider: EmbeddingProvider,
        *,
        table_name: str = "memory",
        pool_size: int = 5,
        pool_recycle: int = 300,
        enrich_embeddings: bool = True,
    ):
        self._connection_string = connection_string
        self._embedder = embedding_provider
        self._table_name = table_name
        self._enrich_embeddings = enrich_embeddings

        self._model = build_table(table_name, embedding_provider.dimensions)

        self._engine: AsyncEngine = create_async_engine(
            connection_string,
            pool_size=pool_size,
            pool_recycle=pool_recycle,
        )
        self._session_factory = async_sessionmaker(
            self._engine, expire_on_commit=False
        )

        self._initialized = False

    def _enriched_text(self, text: str, category: Category) -> str:
        """Prepend category to text for embedding when enrichment is enabled.

        The stored content is always raw text — this only affects the vector
        that gets embedded, so "I love cake" with category PREFERENCE embeds
        as "preference: I love cake", placing it closer to food/preference
        queries in the embedding space.
        """
        if self._enrich_embeddings:
            return f"{category.value}: {text}"
        return text

    # ────────────────────────────────────────────────────────────────
    # Setup
    # ────────────────────────────────────────────────────────────────

    async def __aenter__(self) -> MemoryStore:
        """Eager init: ``async with MemoryStore(...) as store:``."""
        await self.init()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def init(self) -> None:
        """Create the pgvector extension and memory table if needed.

        Call once at app startup, or let it auto-init on first operation.
        """
        if self._initialized:
            return
        async with self._engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.create_all)
        self._initialized = True
        logger.info("pgmemory: table '%s' ready", self._table_name)

    async def _ensure_init(self) -> None:
        if not self._initialized:
            await self.init()

    async def close(self) -> None:
        """Dispose of the connection pool."""
        await self._engine.dispose()

    # ────────────────────────────────────────────────────────────────
    # Write operations
    # ────────────────────────────────────────────────────────────────

    async def add(
        self,
        app_name: str,
        user_id: str,
        text_content: str,
        *,
        category: Category = Category.GENERAL,
        importance: int = 1,
        valid_until: datetime | None = None,
        source_session_id: str | None = None,
        source_event_id: str | None = None,
        source_event_timestamp: datetime | None = None,
        source_role: str | None = None,
        metadata: dict | None = None,
    ) -> int:
        """Store a single memory. Returns its ID."""
        await self._ensure_init()

        embed_text = self._enriched_text(text_content, category)
        embeddings = await self._embedder.embed([embed_text])
        now = datetime.now(timezone.utc)

        mem = Memory(
            app_name=app_name,
            user_id=user_id,
            text=text_content,
            category=category,
            importance=max(1, min(5, importance)),
            created_at=now,
            valid_from=now,
            valid_until=valid_until,
            source_session_id=source_session_id,
            source_event_id=source_event_id,
            source_event_timestamp=source_event_timestamp,
            source_role=source_role,
            metadata=metadata or {},
        )

        async with self._session_factory() as db:
            row = self._model.from_memory(mem, embeddings[0])
            db.add(row)
            await db.commit()
            await db.refresh(row)
            return row.id

    async def add_many(
        self,
        memories: Sequence[Memory],
    ) -> list[int]:
        """Batch-insert multiple memories. Returns their IDs."""
        await self._ensure_init()

        if not memories:
            return []

        texts = [self._enriched_text(m.text, m.category) for m in memories]
        embeddings = await self._embedder.embed(texts)

        ids = []
        async with self._session_factory() as db:
            for mem, emb in zip(memories, embeddings):
                row = self._model.from_memory(mem, emb)
                db.add(row)
                await db.flush()
                ids.append(row.id)
            await db.commit()
        return ids

    # ────────────────────────────────────────────────────────────────
    # Hybrid search
    # ────────────────────────────────────────────────────────────────

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """Hybrid search: SQL filters → full-text + semantic + recency scoring.

        The pipeline:
        1. Filter by app_name, user_id (always)
        2. Filter by categories, min_importance, temporal validity (if set)
        3. Score: weighted combination of cosine similarity, ts_rank, and recency
        4. Sort by combined score, return top_k

        Scoring formula:
            combined = (w_sim * cosine_similarity)
                     + (w_kw  * ts_rank)
                     + (w_rec * recency_decay)

        Where recency_decay = 1 / (1 + days_old / 30)
        """
        await self._ensure_init()

        M = self._model
        now = datetime.now(timezone.utc)
        is_browse = not query.text.strip()

        # Recency: 1 / (1 + age_in_days / 30). Newer = higher.
        age_seconds = func.extract("epoch", now - M.created_at)
        recency_expr = (1.0 / (1.0 + age_seconds / (30.0 * 86400.0))).label(
            "recency_score"
        )

        if is_browse:
            # Browse mode: skip embedding/keyword scoring, order by recency
            similarity_expr = literal_column("0.0").label("similarity")
            keyword_expr = literal_column("0.0").label("keyword_score")
            combined_expr = (
                query.weight_recency * (1.0 / (1.0 + age_seconds / (30.0 * 86400.0)))
            ).label("combined_score")
        else:
            # ── Build score expressions ─────────────────────────────
            query_embeddings = await self._embedder.embed([query.text])
            query_vec = query_embeddings[0]

            similarity_expr = (1 - M.content_embedding.cosine_distance(query_vec)).label(
                "similarity"
            )

            keyword_expr = func.ts_rank(
                M.content_tsv,
                func.plainto_tsquery("english", query.text),
            ).label("keyword_score")

            combined_expr = (
                query.weight_similarity * (1 - M.content_embedding.cosine_distance(query_vec))
                + query.weight_keyword
                * func.ts_rank(M.content_tsv, func.plainto_tsquery("english", query.text))
                + query.weight_recency * (1.0 / (1.0 + age_seconds / (30.0 * 86400.0)))
            ).label("combined_score")

        # ── Base query ──────────────────────────────────────────────
        stmt = select(M, similarity_expr, keyword_expr, recency_expr, combined_expr).where(
            M.app_name == query.app_name,
            M.user_id == query.user_id,
        )

        # ── Optional filters ────────────────────────────────────────
        if not query.include_expired:
            stmt = stmt.where(
                M.valid_from <= now,
                (M.valid_until.is_(None) | (M.valid_until > now)),
            )

        if query.categories:
            cat_values = [
                c.value if isinstance(c, Category) else c for c in query.categories
            ]
            stmt = stmt.where(M.category.in_(cat_values))

        if query.min_importance is not None:
            stmt = stmt.where(M.importance >= query.min_importance)

        # ── Sort + limit ────────────────────────────────────────────
        if is_browse:
            stmt = stmt.order_by(M.created_at.desc()).limit(query.top_k)
        else:
            stmt = stmt.order_by(combined_expr.desc()).limit(query.top_k)

        # ── Execute ─────────────────────────────────────────────────
        async with self._session_factory() as db:
            result = await db.execute(stmt)
            rows = result.all()

            # Touch last_accessed for retrieved rows
            if rows:
                retrieved_ids = [r[0].id for r in rows]
                await db.execute(
                    update(M)
                    .where(M.id.in_(retrieved_ids))
                    .values(last_accessed=now)
                )
                await db.commit()

        # ── Build results ───────────────────────────────────────────
        results = []
        for row, sim, kw, rec, combined in rows:
            logger.debug(
                "  candidate id=%s sim=%.4f kw=%.4f rec=%.4f combined=%.4f "
                "threshold=%.4f text=%r",
                row.id, float(sim), float(kw), float(rec), float(combined),
                query.similarity_threshold, row.content[:80],
            )
            if not is_browse and combined < query.similarity_threshold:
                logger.debug("  → FILTERED (combined %.4f < threshold %.4f)",
                             float(combined), query.similarity_threshold)
                continue
            results.append(
                SearchResult(
                    memory=row.to_memory(),
                    similarity=round(float(sim), 4),
                    keyword_score=round(float(kw), 4),
                    recency_score=round(float(rec), 4),
                    combined_score=round(float(combined), 4),
                    source_query=query,
                )
            )

        # ── Percentile filtering ────────────────────────────────────
        if query.threshold_percentile is not None and results:
            cutoff_idx = int(len(results) * query.threshold_percentile)
            if cutoff_idx > 0:
                cutoff_score = results[-cutoff_idx].combined_score
                results = [r for r in results if r.combined_score > cutoff_score]

        logger.debug(
            "search: user=%s query=%r → %d results (from %d candidates)",
            query.user_id,
            query.text[:60],
            len(results),
            len(rows),
        )
        return results

    async def search_many(
        self,
        queries: Sequence[SearchQuery],
        *,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Run multiple searches in parallel and merge into one ranked list.

        - Fires all queries concurrently via asyncio.gather
        - Deduplicates by memory.id, keeping the result with the highest combined_score
        - Returns up to ``top_k`` results sorted by combined_score descending
        """
        if not queries:
            return []

        all_results = await asyncio.gather(*[self.search(q) for q in queries])

        best: dict[int | None, SearchResult] = {}
        for result_list in all_results:
            for r in result_list:
                key = r.memory.id
                if key not in best or r.combined_score > best[key].combined_score:
                    best[key] = r

        merged = sorted(best.values(), key=lambda r: r.combined_score, reverse=True)
        return merged[:top_k]

    async def get(self, memory_id: int) -> Memory | None:
        """Fetch a single memory by ID."""
        await self._ensure_init()
        async with self._session_factory() as db:
            row = await db.get(self._model, memory_id)
            return row.to_memory() if row else None

    # ────────────────────────────────────────────────────────────────
    # Lifecycle
    # ────────────────────────────────────────────────────────────────

    async def promote(self, memory_id: int, increment: int = 1) -> None:
        """Bump importance. Clears valid_until (reinforced = durable)."""
        await self._ensure_init()
        async with self._session_factory() as db:
            await db.execute(
                update(self._model)
                .where(self._model.id == memory_id)
                .values(
                    importance=self._model.importance + increment,
                    last_accessed=datetime.now(timezone.utc),
                    valid_until=None,
                )
            )
            await db.commit()

    async def expire(self, memory_id: int, *, reason: str = "expired") -> None:
        """Soft-expire a memory (set valid_until = now, log reason in metadata)."""
        await self._ensure_init()
        now = datetime.now(timezone.utc)
        async with self._session_factory() as db:
            row = await db.get(self._model, memory_id)
            if row is None:
                return
            existing = dict(row.metadata_ or {})
            existing["expired_reason"] = reason
            existing["expired_at"] = now.isoformat()
            await db.execute(
                update(self._model)
                .where(self._model.id == memory_id)
                .values(valid_until=now, metadata_=existing)
            )
            await db.commit()

    async def decay(
        self,
        *,
        app_name: str | None = None,
        hard_delete: bool = True,
    ) -> int:
        """Remove memories past their valid_until.

        If hard_delete=False, does nothing (they're already filtered from search).
        Call from a scheduled job or let it run before each search.

        Returns number of rows deleted.
        """
        await self._ensure_init()
        now = datetime.now(timezone.utc)
        stmt = delete(self._model).where(
            self._model.valid_until.isnot(None),
            self._model.valid_until < now,
        )
        if app_name:
            stmt = stmt.where(self._model.app_name == app_name)

        async with self._session_factory() as db:
            result = await db.execute(stmt)
            await db.commit()
            count = result.rowcount
            if count:
                logger.info("decay: deleted %d expired memories", count)
            return count

    async def soft_expire_stale(
        self,
        *,
        max_age_days: int = 90,
        min_importance: int = 3,
        app_name: str | None = None,
    ) -> int:
        """Set valid_until on old, low-importance memories that never expire.

        Memories with importance >= min_importance are left alone.
        Returns number of rows updated.
        """
        await self._ensure_init()
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        now = datetime.now(timezone.utc)

        stmt = (
            update(self._model)
            .where(
                self._model.valid_until.is_(None),
                self._model.created_at < cutoff,
                self._model.importance < min_importance,
            )
            .values(valid_until=now)
        )
        if app_name:
            stmt = stmt.where(self._model.app_name == app_name)

        async with self._session_factory() as db:
            result = await db.execute(stmt)
            await db.commit()
            return result.rowcount

    # ────────────────────────────────────────────────────────────────
    # Conflict resolution
    # ────────────────────────────────────────────────────────────────

    async def find_conflicts(
        self,
        app_name: str,
        user_id: str,
        text_content: str,
        category: Category,
        *,
        threshold: float = 0.85,
        limit: int = 5,
    ) -> list[SearchResult]:
        """Find active memories that are semantically close (potential conflicts).

        High similarity + same category + same user = likely duplicate or
        contradicting memory.
        """
        await self._ensure_init()
        embed_text = self._enriched_text(text_content, category)
        embeddings = await self._embedder.embed([embed_text])
        vec = embeddings[0]
        M = self._model
        cat_val = category.value if isinstance(category, Category) else category

        similarity_expr = (1 - M.content_embedding.cosine_distance(vec)).label(
            "similarity"
        )

        stmt = (
            select(M, similarity_expr)
            .where(
                M.app_name == app_name,
                M.user_id == user_id,
                M.category == cat_val,
                M.valid_until.is_(None),  # only active
            )
            .order_by(M.content_embedding.cosine_distance(vec))
            .limit(limit)
        )

        async with self._session_factory() as db:
            result = await db.execute(stmt)
            rows = result.all()

        return [
            SearchResult(memory=row.to_memory(), similarity=round(float(sim), 4))
            for row, sim in rows
            if sim >= threshold
        ]

    async def supersede(
        self,
        app_name: str,
        user_id: str,
        new_text: str,
        category: Category,
        *,
        threshold: float = 0.85,
        **add_kwargs,
    ) -> tuple[int, list[int]]:
        """Add a new memory and expire any conflicts it supersedes.

        Returns (new_memory_id, list_of_superseded_ids).
        """
        conflicts = await self.find_conflicts(
            app_name, user_id, new_text, category, threshold=threshold
        )
        superseded_ids = []
        for c in conflicts:
            if c.memory.id is not None:
                await self.expire(c.memory.id, reason=f"superseded (sim={c.similarity})")
                superseded_ids.append(c.memory.id)

        new_id = await self.add(
            app_name,
            user_id,
            new_text,
            category=category,
            **add_kwargs,
        )
        return new_id, superseded_ids

    # ────────────────────────────────────────────────────────────────
    # Bulk / admin
    # ────────────────────────────────────────────────────────────────

    async def count(
        self,
        app_name: str | None = None,
        user_id: str | None = None,
        *,
        include_expired: bool = False,
    ) -> int:
        """Count memories, optionally filtered."""
        await self._ensure_init()
        stmt = select(func.count(self._model.id))
        if app_name:
            stmt = stmt.where(self._model.app_name == app_name)
        if user_id:
            stmt = stmt.where(self._model.user_id == user_id)
        if not include_expired:
            now = datetime.now(timezone.utc)
            stmt = stmt.where(
                (self._model.valid_until.is_(None)) | (self._model.valid_until > now)
            )
        async with self._session_factory() as db:
            result = await db.execute(stmt)
            return result.scalar_one()

    async def delete_user(self, app_name: str, user_id: str) -> int:
        """Hard-delete all memories for a user. GDPR right-to-erasure."""
        await self._ensure_init()
        async with self._session_factory() as db:
            result = await db.execute(
                delete(self._model).where(
                    self._model.app_name == app_name,
                    self._model.user_id == user_id,
                )
            )
            await db.commit()
            return result.rowcount

    async def list_users(self, app_name: str) -> list[str]:
        """List distinct user_ids that have active memories in an app."""
        await self._ensure_init()
        now = datetime.now(timezone.utc)
        stmt = (
            select(self._model.user_id)
            .where(
                self._model.app_name == app_name,
                (self._model.valid_until.is_(None)) | (self._model.valid_until > now),
            )
            .distinct()
        )
        async with self._session_factory() as db:
            result = await db.execute(stmt)
            return [row[0] for row in result.all()]
