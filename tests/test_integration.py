"""Integration tests — requires Docker (testcontainers pgvector).

Run: pytest tests/test_integration.py -v
Skip: pytest tests/ -v -k "not integration"

Set PGMEMORY_TEST_URL to skip testcontainers and use an existing database:
    PGMEMORY_TEST_URL=postgresql+asyncpg://user:pass@localhost/db pytest ...
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio

from pgmemory import Category, Memory, MemoryStore, SearchQuery
from tests.helpers import FakeEmbeddingProvider


def _pg_available() -> bool:
    if os.environ.get("PGMEMORY_TEST_URL"):
        return True
    try:
        from testcontainers.postgres import PostgresContainer  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _pg_available(), reason="testcontainers not installed and PGMEMORY_TEST_URL not set"
)


@pytest.fixture(scope="module")
def pg_url():
    env_url = os.environ.get("PGMEMORY_TEST_URL")
    if env_url:
        yield env_url
        return

    from testcontainers.postgres import PostgresContainer

    with PostgresContainer(
        image="ankane/pgvector:latest",
        username="test_user",
        password="test_pass",
        dbname="test_db",
    ) as pg:
        sync_url = pg.get_connection_url()
        yield sync_url.replace("psycopg2", "asyncpg").replace(
            "postgresql://", "postgresql+asyncpg://"
        )


@pytest_asyncio.fixture
async def store(pg_url):
    s = MemoryStore(
        pg_url,
        FakeEmbeddingProvider(dims=16),
        table_name="test_memory",
    )
    await s.init()
    yield s
    await s.close()


# ── Core CRUD ──────────────────────────────────────────────────────────


class TestAdd:
    @pytest.mark.asyncio
    async def test_add_returns_id(self, store):
        mid = await store.add("app", "u1", "The sky is blue")
        assert isinstance(mid, int)
        assert mid > 0

    @pytest.mark.asyncio
    async def test_add_with_all_fields(self, store):
        mid = await store.add(
            "app", "u1", "Full field test",
            category=Category.PREFERENCE,
            importance=3,
            valid_until=datetime.now(timezone.utc) + timedelta(days=30),
            source_session_id="sess_1",
            source_event_id="evt_1",
            source_event_timestamp=datetime.now(timezone.utc),
            source_role="user",
            metadata={"project": "dash", "tags": ["retail"]},
        )
        mem = await store.get(mid)
        assert mem is not None
        assert mem.category == Category.PREFERENCE
        assert mem.importance == 3
        assert mem.source_session_id == "sess_1"
        assert mem.metadata["project"] == "dash"
        assert mem.metadata["tags"] == ["retail"]

    @pytest.mark.asyncio
    async def test_add_many(self, store):
        now = datetime.now(timezone.utc)
        mems = [
            Memory(app_name="app", user_id="u1", text="batch 1", created_at=now, valid_from=now),
            Memory(app_name="app", user_id="u1", text="batch 2", created_at=now, valid_from=now),
            Memory(app_name="app", user_id="u1", text="batch 3", created_at=now, valid_from=now),
        ]
        ids = await store.add_many(mems)
        assert len(ids) == 3
        assert all(isinstance(i, int) for i in ids)


# ── Hybrid search ──────────────────────────────────────────────────────


class TestSearch:
    @pytest.mark.asyncio
    async def test_basic_search(self, store):
        await store.add("search_app", "u1", "User's favourite colour is blue",
                        category=Category.PREFERENCE)
        results = await store.search(
            SearchQuery(app_name="search_app", user_id="u1", text="colour preference")
        )
        assert len(results) >= 1
        assert results[0].similarity > 0
        assert results[0].combined_score > 0

    @pytest.mark.asyncio
    async def test_user_isolation(self, store):
        await store.add("iso_app", "alice", "Alice's secret")
        results = await store.search(
            SearchQuery(app_name="iso_app", user_id="bob", text="secret")
        )
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_app_isolation(self, store):
        await store.add("app_A", "u1", "App A fact")
        results = await store.search(
            SearchQuery(app_name="app_B", user_id="u1", text="fact")
        )
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_category_filter(self, store):
        await store.add("cat_app", "u1", "A fact", category=Category.FACT)
        await store.add("cat_app", "u1", "A preference", category=Category.PREFERENCE)
        results = await store.search(
            SearchQuery(
                app_name="cat_app", user_id="u1", text="something",
                categories=[Category.FACT],
            )
        )
        assert all(r.memory.category == Category.FACT for r in results)

    @pytest.mark.asyncio
    async def test_importance_filter(self, store):
        await store.add("imp_app", "u1", "Low importance", importance=1)
        await store.add("imp_app", "u1", "High importance", importance=4)
        results = await store.search(
            SearchQuery(
                app_name="imp_app", user_id="u1", text="importance",
                min_importance=3,
            )
        )
        assert all(r.memory.importance >= 3 for r in results)

    @pytest.mark.asyncio
    async def test_expired_excluded_by_default(self, store):
        await store.add(
            "exp_app", "u1", "Expired memory",
            valid_until=datetime.now(timezone.utc) - timedelta(days=1),
        )
        results = await store.search(
            SearchQuery(app_name="exp_app", user_id="u1", text="expired")
        )
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_expired_included_when_asked(self, store):
        await store.add(
            "exp_inc_app", "u1", "Expired but searchable",
            valid_until=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        results = await store.search(
            SearchQuery(
                app_name="exp_inc_app", user_id="u1", text="expired",
                include_expired=True,
                similarity_threshold=0.0,
            )
        )
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_browse_mode_empty_query(self, store):
        await store.add("browse_app", "u1", "Browsable fact", category=Category.FACT)
        results = await store.search(
            SearchQuery(app_name="browse_app", user_id="u1", text="")
        )
        assert len(results) >= 1
        assert results[0].similarity == 0.0
        assert results[0].keyword_score == 0.0

    @pytest.mark.asyncio
    async def test_search_updates_last_accessed(self, store):
        mid = await store.add("la_app", "u1", "Track access time")
        mem_before = await store.get(mid)
        assert mem_before.last_accessed is None

        await store.search(
            SearchQuery(app_name="la_app", user_id="u1", text="access time",
                        similarity_threshold=0.0)
        )
        mem_after = await store.get(mid)
        assert mem_after.last_accessed is not None

    @pytest.mark.asyncio
    async def test_search_offset_pagination(self, store):
        # Seed 15 memories with sequential content to ensure stable ordering
        for i in range(15):
            await store.add(
                "page_app", "u1", f"Memory number {i:02d}",
                category=Category.GENERAL, importance=3
            )
        
        # Page 1: offset=0, top_k=5
        page1 = await store.search(
            SearchQuery(app_name="page_app", user_id="u1", text="", top_k=5, offset=0)
        )
        
        # Page 2: offset=5, top_k=5
        page2 = await store.search(
            SearchQuery(app_name="page_app", user_id="u1", text="", top_k=5, offset=5)
        )
        
        # Page 3: offset=10, top_k=5
        page3 = await store.search(
            SearchQuery(app_name="page_app", user_id="u1", text="", top_k=5, offset=10)
        )
        
        # Verify all pages have correct size
        assert len(page1) == 5
        assert len(page2) == 5
        assert len(page3) == 5
        
        # Verify non-overlapping results
        ids1 = {r.memory.id for r in page1}
        ids2 = {r.memory.id for r in page2}
        ids3 = {r.memory.id for r in page3}
        
        assert len(ids1 & ids2) == 0, "Page 1 and 2 should not overlap"
        assert len(ids1 & ids3) == 0, "Page 1 and 3 should not overlap"
        assert len(ids2 & ids3) == 0, "Page 2 and 3 should not overlap"


# ── Lifecycle ──────────────────────────────────────────────────────────


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_promote(self, store):
        mid = await store.add("lc_app", "u1", "Promotable", importance=1)
        await store.promote(mid)
        await store.promote(mid)
        mem = await store.get(mid)
        assert mem.importance == 3
        assert mem.last_accessed is not None

    @pytest.mark.asyncio
    async def test_promote_clears_valid_until(self, store):
        mid = await store.add(
            "lc_app", "u1", "Was expiring",
            valid_until=datetime.now(timezone.utc) + timedelta(days=1),
        )
        await store.promote(mid)
        mem = await store.get(mid)
        assert mem.valid_until is None  # reinforced = durable

    @pytest.mark.asyncio
    async def test_expire(self, store):
        mid = await store.add("lc_app", "u1", "Will be expired")
        await store.expire(mid, reason="no longer relevant")
        mem = await store.get(mid)
        assert mem.is_expired
        assert mem.metadata.get("expired_reason") == "no longer relevant"

    @pytest.mark.asyncio
    async def test_decay_hard_delete(self, store):
        await store.add(
            "decay_app", "u1", "Already expired",
            valid_until=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        deleted = await store.decay(app_name="decay_app")
        assert deleted >= 1

    @pytest.mark.asyncio
    async def test_soft_expire_stale(self, store):
        # Insert something "old" by setting created_at far in the past
        mid = await store.add("stale_app", "u1", "Old low-importance", importance=1)
        # Manually backdate created_at
        async with store._session_factory() as db:
            from sqlalchemy import update
            await db.execute(
                update(store._model)
                .where(store._model.id == mid)
                .values(created_at=datetime.now(timezone.utc) - timedelta(days=120))
            )
            await db.commit()

        updated = await store.soft_expire_stale(
            max_age_days=90, min_importance=3, app_name="stale_app"
        )
        assert updated >= 1
        mem = await store.get(mid)
        assert mem.valid_until is not None


# ── Conflict resolution ────────────────────────────────────────────────


class TestConflicts:
    @pytest.mark.asyncio
    async def test_find_conflicts(self, store):
        await store.add("conf_app", "u1", "User works at Company A",
                        category=Category.FACT)
        conflicts = await store.find_conflicts(
            "conf_app", "u1", "User works at Company A", Category.FACT,
            threshold=0.8,
        )
        assert len(conflicts) >= 1

    @pytest.mark.asyncio
    async def test_supersede(self, store):
        old_id = await store.add("sup_app", "u1", "User lives in Sydney",
                                 category=Category.FACT)
        new_id, superseded = await store.supersede(
            "sup_app", "u1", "User lives in Sydney", Category.FACT,
            threshold=0.8,
        )
        assert new_id != old_id
        assert old_id in superseded
        old_mem = await store.get(old_id)
        assert old_mem.is_expired


# ── Admin ──────────────────────────────────────────────────────────────


class TestAdmin:
    @pytest.mark.asyncio
    async def test_count(self, store):
        await store.add("count_app", "u1", "countable")
        c = await store.count(app_name="count_app")
        assert c >= 1

    @pytest.mark.asyncio
    async def test_count_excludes_expired(self, store):
        await store.add(
            "count_exp_app", "u1", "expired",
            valid_until=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        c = await store.count(app_name="count_exp_app", include_expired=False)
        assert c == 0

    @pytest.mark.asyncio
    async def test_delete_user(self, store):
        await store.add("del_app", "delete_me", "personal data")
        deleted = await store.delete_user("del_app", "delete_me")
        assert deleted >= 1
        c = await store.count(app_name="del_app", user_id="delete_me", include_expired=True)
        assert c == 0

    @pytest.mark.asyncio
    async def test_list_users(self, store):
        await store.add("users_app", "alpha", "alpha data")
        await store.add("users_app", "beta", "beta data")
        users = await store.list_users("users_app")
        assert "alpha" in users
        assert "beta" in users


# ── Concurrency ────────────────────────────────────────────────────────


class TestConcurrency:
    @pytest.mark.asyncio
    async def test_concurrent_promote_same_memory(self, store):
        """Fire 5 concurrent promote(increment=1) calls on the same memory,
        verify final importance is clamped to 5."""
        mid = await store.add("conc_app", "u1", "Concurrently promoted", importance=1)
        
        # Fire 5 concurrent promotions
        await asyncio.gather(
            store.promote(mid, increment=1),
            store.promote(mid, increment=1),
            store.promote(mid, increment=1),
            store.promote(mid, increment=1),
            store.promote(mid, increment=1),
        )
        
        mem = await store.get(mid)
        # Starting importance: 1
        # 5 increments of 1 = 6, but clamped to 5
        assert mem.importance == 5
        assert mem.last_accessed is not None

    @pytest.mark.asyncio
    async def test_concurrent_supersede_same_category(self, store):
        """Fire 3 concurrent supersede() calls with similar texts in the same category.
        Verifies that:
        1. Pre-existing memory is expired by at least one transaction
        2. Concurrent supersedes may create duplicates (READ COMMITTED allows this)
        3. Each new memory has correct content"""
        # Add a pre-existing memory to supersede
        existing_id = await store.add(
            "sup_conc_app", "u1", "User prefers light theme",
            category=Category.PREFERENCE
        )
        
        # Fire 3 concurrent supersedes with similar text
        results = await asyncio.gather(
            store.supersede("sup_conc_app", "u1", "User prefers dark theme",
                           Category.PREFERENCE, threshold=0.7),
            store.supersede("sup_conc_app", "u1", "User prefers dark theme",
                           Category.PREFERENCE, threshold=0.7),
            store.supersede("sup_conc_app", "u1", "User prefers dark theme",
                           Category.PREFERENCE, threshold=0.7),
        )
        
        # Each supersede returns (new_id, list_of_superseded_ids)
        new_ids = [r[0] for r in results]
        all_superseded = [sid for r in results for sid in r[1]]
        
        # Verify all three operations created new memories
        assert len(new_ids) == 3
        assert len(set(new_ids)) == 3, "Each supersede should create a distinct memory"
        
        # The pre-existing memory should be expired by at least one transaction
        existing_mem = await store.get(existing_id)
        assert existing_mem.is_expired, \
            "Pre-existing memory should have been expired"
        
        # Get all memories for this app/user/category
        q = SearchQuery(
            app_name="sup_conc_app",
            user_id="u1",
            text="",
            categories=[Category.PREFERENCE],
            include_expired=True,
            similarity_threshold=0.0,
        )
        all_results = await store.search(q)
        all_memories = [r.memory for r in all_results]
        
        # Count active memories - due to READ COMMITTED isolation, concurrent
        # supersedes racing may create 1-3 active memories depending on timing
        active_memories = [m for m in all_memories if not m.is_expired]
        
        assert 1 <= len(active_memories) <= 3, \
            f"Expected 1-3 active memories (timing dependent), got {len(active_memories)}"
        
        # All active memories should have the new text
        for mem in active_memories:
            assert mem.text == "User prefers dark theme"

    @pytest.mark.asyncio
    async def test_promote_at_max_importance(self, store):
        """Verify promoting a memory already at importance=5 doesn't break (stays at 5)."""
        mid = await store.add("max_app", "u1", "Already maxed", importance=5)
        
        # Promote 3 times
        await store.promote(mid, increment=1)
        await store.promote(mid, increment=2)
        await store.promote(mid, increment=1)
        
        mem = await store.get(mid)
        # Should stay at 5 (clamped)
        assert mem.importance == 5

    @pytest.mark.asyncio
    async def test_supersede_with_no_conflicts(self, store):
        """Verify supersede() with threshold=0.99 (no matches) correctly adds
        the new memory without errors."""
        new_id, superseded = await store.supersede(
            "noconf_app", "u1", "Unique memory with no conflicts",
            Category.FACT, threshold=0.99,
        )
        
        assert new_id > 0
        assert len(superseded) == 0
        
        mem = await store.get(new_id)
        assert mem is not None
        assert mem.text == "Unique memory with no conflicts"
        assert mem.category == Category.FACT
        assert not mem.is_expired


# ── Embedding enrichment ──────────────────────────────────────────────


@pytest_asyncio.fixture
async def enriched_store(pg_url):
    s = MemoryStore(
        pg_url,
        FakeEmbeddingProvider(dims=16),
        table_name="test_enriched",
        enrich_embeddings=True,
    )
    await s.init()
    yield s
    await s.close()


class TestEnrichment:
    @pytest.mark.asyncio
    async def test_content_stores_raw_text(self, enriched_store):
        """Content column should store raw text, not enriched text."""
        mid = await enriched_store.add(
            "enrich_app", "u1", "I love cake",
            category=Category.PREFERENCE,
        )
        mem = await enriched_store.get(mid)
        assert mem.text == "I love cake"
        assert "preference:" not in mem.text

    @pytest.mark.asyncio
    async def test_enriched_produces_different_embeddings(self, pg_url):
        """Same text with enrichment on vs off should produce different embeddings."""
        plain = MemoryStore(
            pg_url,
            FakeEmbeddingProvider(dims=16),
            table_name="test_enrich_plain",
            enrich_embeddings=False,
        )
        enriched = MemoryStore(
            pg_url,
            FakeEmbeddingProvider(dims=16),
            table_name="test_enrich_rich",
            enrich_embeddings=True,
        )
        await plain.init()
        await enriched.init()

        try:
            plain_id = await plain.add("app", "u1", "I love cake",
                                       category=Category.PREFERENCE)
            enriched_id = await enriched.add("app", "u1", "I love cake",
                                             category=Category.PREFERENCE)

            # Retrieve raw embeddings from the database
            async with plain._session_factory() as db:
                row = await db.get(plain._model, plain_id)
                plain_vec = list(row.content_embedding)

            async with enriched._session_factory() as db:
                row = await db.get(enriched._model, enriched_id)
                enriched_vec = list(row.content_embedding)

            # Vectors should differ because enriched embeds "preference: I love cake"
            assert plain_vec != enriched_vec
        finally:
            await plain.close()
            await enriched.close()
