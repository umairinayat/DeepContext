"""
Tests for the vector store (SQLite/Python cosine similarity fallback path).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from sqlalchemy import select

from deepcontext.db.models.memory import Memory
from deepcontext.vectorstore.base import VectorSearchResult
from deepcontext.vectorstore.pgvector_store import PgVectorStore, _cosine_similarity
from tests.conftest import fake_embedding


# ---------------------------------------------------------------------------
# Cosine similarity function tests
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(_cosine_similarity(a, b) - (-1.0)) < 1e-6

    def test_zero_vector(self):
        a = [0.0, 0.0]
        b = [1.0, 2.0]
        assert _cosine_similarity(a, b) == 0.0

    def test_different_lengths(self):
        a = [1.0, 2.0]
        b = [1.0, 2.0, 3.0]
        assert _cosine_similarity(a, b) == 0.0

    def test_known_similarity(self):
        a = [1.0, 0.0, 0.0]
        b = [1.0, 1.0, 0.0]
        expected = 1.0 / math.sqrt(2)
        assert abs(_cosine_similarity(a, b) - expected) < 1e-6


# ---------------------------------------------------------------------------
# PgVectorStore tests (SQLite path)
# ---------------------------------------------------------------------------


class TestPgVectorStore:
    @pytest.mark.asyncio
    async def test_search_empty_db(self, session_factory):
        """Search on empty DB should return empty list."""
        store = PgVectorStore(session_factory)
        results = await store.search(
            query_embedding=fake_embedding(0),
            user_id="u1",
            limit=10,
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_search_returns_results(self, session_factory, sample_memories):
        """Search should return ranked results by cosine similarity."""
        store = PgVectorStore(session_factory)
        # Use the embedding of the first memory as query
        query_emb = fake_embedding(hash("User is a Python developer") % 1000)
        results = await store.search(
            query_embedding=query_emb,
            user_id="test_user",
            limit=5,
        )
        assert len(results) > 0
        # Results should be sorted by score descending
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    @pytest.mark.asyncio
    async def test_search_respects_limit(self, session_factory, sample_memories):
        store = PgVectorStore(session_factory)
        results = await store.search(
            query_embedding=fake_embedding(0),
            user_id="test_user",
            limit=2,
        )
        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_search_filters_by_user(self, session_factory, sample_memories):
        """Search should only return memories for the specified user."""
        store = PgVectorStore(session_factory)
        results = await store.search(
            query_embedding=fake_embedding(0),
            user_id="nonexistent_user",
            limit=10,
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_search_min_score(self, session_factory, sample_memories):
        """Search with high min_score should filter low-similarity results."""
        store = PgVectorStore(session_factory)
        results = await store.search(
            query_embedding=fake_embedding(0),
            user_id="test_user",
            limit=10,
            min_score=0.999,  # Very high threshold
        )
        # All returned results should meet the threshold
        for r in results:
            assert r.score >= 0.999

    @pytest.mark.asyncio
    async def test_search_with_filters(self, session_factory, sample_memories):
        """search_with_filters should apply tier and memory_type."""
        store = PgVectorStore(session_factory)
        results = await store.search_with_filters(
            query_embedding=fake_embedding(0),
            user_id="test_user",
            tier="long_term",
            limit=10,
        )
        # All returned memory IDs should correspond to long_term memories
        long_term_ids = {m.id for m in sample_memories if m.tier == "long_term"}
        for r in results:
            assert r.memory_id in long_term_ids

    @pytest.mark.asyncio
    async def test_search_with_memory_type_filter(self, session_factory, sample_memories):
        store = PgVectorStore(session_factory)
        results = await store.search_with_filters(
            query_embedding=fake_embedding(0),
            user_id="test_user",
            memory_type="episodic",
            limit=10,
        )
        episodic_ids = {m.id for m in sample_memories if m.memory_type == "episodic"}
        for r in results:
            assert r.memory_id in episodic_ids

    @pytest.mark.asyncio
    async def test_add_embedding(self, session_factory):
        """add() should set the embedding on an existing memory."""
        now = datetime.now(timezone.utc)
        async with session_factory() as session:
            mem = Memory(
                user_id="u1",
                text="test",
                memory_type="semantic",
                tier="short_term",
                created_at=now,
                updated_at=now,
            )
            session.add(mem)
            await session.commit()
            await session.refresh(mem)

        store = PgVectorStore(session_factory)
        emb = fake_embedding(99)
        await store.add(mem.id, emb)

        async with session_factory() as session:
            loaded = await session.get(Memory, mem.id)
            recovered = loaded.get_embedding()
            assert recovered is not None
            assert len(recovered) == len(emb)

    @pytest.mark.asyncio
    async def test_remove_embedding(self, session_factory, sample_memories):
        """remove() should set embedding to NULL."""
        store = PgVectorStore(session_factory)
        mem_id = sample_memories[0].id
        await store.remove(mem_id)

        async with session_factory() as session:
            loaded = await session.get(Memory, mem_id)
            assert loaded.get_embedding() is None

    @pytest.mark.asyncio
    async def test_add_batch(self, session_factory):
        """add_batch() should set embeddings on multiple memories."""
        now = datetime.now(timezone.utc)
        ids = []
        async with session_factory() as session:
            for i in range(3):
                mem = Memory(
                    user_id="u1",
                    text=f"batch test {i}",
                    memory_type="semantic",
                    tier="short_term",
                    created_at=now,
                    updated_at=now,
                )
                session.add(mem)
            await session.commit()
            result = await session.execute(select(Memory))
            for m in result.scalars().all():
                ids.append(m.id)

        store = PgVectorStore(session_factory)
        items = [(mid, fake_embedding(mid)) for mid in ids]
        await store.add_batch(items)

        async with session_factory() as session:
            for mid in ids:
                loaded = await session.get(Memory, mid)
                assert loaded.get_embedding() is not None
