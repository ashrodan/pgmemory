"""Integration tests — requires Docker (testcontainers pgvector).

Run: pytest tests/test_integration.py -v
Skip: pytest tests/ -v -k "not integration"

Set PGMEMORY_TEST_URL to skip testcontainers and use an existing database:
    PGMEMORY_TEST_URL=postgresql+asyncpg://user:pass@localhost/db pytest ...
"""

from __future__ import annotations

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
