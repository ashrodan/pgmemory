"""Unit tests — no database required."""

from datetime import datetime, timedelta, timezone

import pytest

from pgmemory.types import Category, Memory, SearchQuery, SearchResult
from pgmemory.models import build_table
from pgmemory.store import MemoryStore


class TestCategory:
    def test_all_values(self):
        expected = {"fact", "preference", "skill", "context", "rule", "event", "general"}
        assert {c.value for c in Category} == expected

    def test_from_string(self):
        assert Category("fact") == Category.FACT
        assert Category("preference") == Category.PREFERENCE

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            Category("nonexistent")


class TestMemory:
    def test_defaults(self):
        m = Memory(text="hello")
        assert m.category == Category.GENERAL
        assert m.importance == 1
        assert m.is_active
        assert not m.is_expired

    def test_expired(self):
        m = Memory(
            text="old",
            valid_until=datetime.now(timezone.utc) - timedelta(days=1),
        )
        assert m.is_expired
        assert not m.is_active

    def test_future_valid_from(self):
        m = Memory(
            text="future",
            valid_from=datetime.now(timezone.utc) + timedelta(days=1),
        )
        assert not m.is_active

    def test_metadata_default_empty(self):
        m = Memory(text="x")
        assert m.metadata == {}


class TestSearchQuery:
    def test_defaults(self):
        q = SearchQuery(app_name="a", user_id="u", text="q")
        assert q.top_k == 10
        assert q.similarity_threshold == 0.2
        assert not q.include_expired

    def test_weights_customisable(self):
        q = SearchQuery(
            app_name="a", user_id="u", text="q",
            weight_similarity=0.8, weight_keyword=0.1, weight_recency=0.1,
        )
        assert q.weight_similarity == 0.8

    def test_category_filter(self):
        q = SearchQuery(
            app_name="a", user_id="u", text="q",
            categories=[Category.FACT, Category.PREFERENCE],
        )
        assert q.categories is not None and len(q.categories) == 2


class TestSearchResult:
    def test_proxy_properties(self):
        m = Memory(id=42, text="hello world")
        r = SearchResult(memory=m, similarity=0.95)
        assert r.id == 42
        assert r.text == "hello world"


class TestBuildTable:
    def test_creates_class_with_tablename(self):
        Model = build_table("test_tbl", 768)
        assert Model.__tablename__ == "test_tbl"

    def test_all_columns_present(self):
        Model = build_table("test_tbl_cols", 768)
        col_names = {c.name for c in Model.__table__.columns}
        expected = {
            "id", "app_name", "user_id", "content", "content_embedding",
            "content_tsv", "category", "importance", "created_at",
            "valid_from", "valid_until", "last_accessed",
            "source_session_id", "source_event_id",
            "source_event_timestamp", "source_role", "metadata",
        }
        assert expected.issubset(col_names)

    def test_different_dimensions(self):
        M1 = build_table("tbl_768", 768)
        M2 = build_table("tbl_1536", 1536)
        # Both should create without error
        assert M1.__tablename__ == "tbl_768"
        assert M2.__tablename__ == "tbl_1536"

    def test_to_memory_roundtrip(self):
        Model = build_table("test_rt", 16)
        mem = Memory(
            app_name="app", user_id="u1", text="hello",
            category=Category.FACT, importance=3,
            metadata={"key": "value"},
        )
        row = Model.from_memory(mem, [0.1] * 16)
        back = row.to_memory()
        assert back.text == "hello"
        assert back.category == Category.FACT
        assert back.importance == 3
        assert back.metadata == {"key": "value"}


class TestPercentileThreshold:
    def test_defaults_to_none(self):
        q = SearchQuery(app_name="a", user_id="u", text="q")
        assert q.threshold_percentile is None

    def test_setting_value(self):
        q = SearchQuery(app_name="a", user_id="u", text="q", threshold_percentile=0.3)
        assert q.threshold_percentile == 0.3

    def test_coexists_with_similarity_threshold(self):
        q = SearchQuery(
            app_name="a", user_id="u", text="q",
            similarity_threshold=0.1, threshold_percentile=0.5,
        )
        assert q.similarity_threshold == 0.1
        assert q.threshold_percentile == 0.5


class TestEnrichedText:
    """Tests for MemoryStore._enriched_text (no DB needed)."""

    def _make_store(self, enrich: bool):
        from tests.helpers import FakeEmbeddingProvider
        return MemoryStore(
            "postgresql+asyncpg://fake@localhost/fake",
            FakeEmbeddingProvider(dims=16),
            enrich_embeddings=enrich,
        )

    def test_disabled_returns_raw_text(self):
        store = self._make_store(enrich=False)
        assert store._enriched_text("I love cake", Category.PREFERENCE) == "I love cake"

    def test_enabled_prepends_category(self):
        store = self._make_store(enrich=True)
        result = store._enriched_text("I love cake", Category.PREFERENCE)
        assert result == "preference: I love cake"

    def test_enabled_all_categories(self):
        store = self._make_store(enrich=True)
        for cat in Category:
            result = store._enriched_text("test text", cat)
            assert result == f"{cat.value}: test text"

    def test_disabled_ignores_category(self):
        store = self._make_store(enrich=False)
        for cat in Category:
            assert store._enriched_text("hello", cat) == "hello"
