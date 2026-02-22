"""
Multi-user isolation tests for DeepContext.

Verifies that the multi-tenant memory system properly isolates data between users
across search, graph, and lifecycle operations.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from deepcontext.core.types import (
    EntityType,
    ExtractedEntity,
    ExtractedRelationship,
    MemoryTier,
    MemoryType,
)
from deepcontext.db.models.memory import Memory
from deepcontext.graph.knowledge_graph import KnowledgeGraph
from deepcontext.lifecycle.manager import LifecycleManager
from deepcontext.vectorstore.pgvector_store import PgVectorStore
from tests.conftest import fake_embedding


# ---------------------------------------------------------------------------
# Memory Isolation
# ---------------------------------------------------------------------------

class TestMemoryIsolation:
    """User A's memories must be completely invisible to User B."""

    @pytest.mark.asyncio
    async def test_search_only_returns_own_user(self, session_factory):
        """Vector search for user_a should never return user_b's memories."""
        now = datetime.now(timezone.utc)

        # Insert memories for two different users
        async with session_factory() as session:
            for user, text in [
                ("user_a", "User A likes Python"),
                ("user_b", "User B likes Rust"),
            ]:
                mem = Memory(
                    user_id=user,
                    conversation_id="c1",
                    text=text,
                    memory_type="semantic",
                    tier="short_term",
                    importance=0.8,
                    confidence=0.9,
                    source_entities=[],
                    created_at=now,
                    updated_at=now,
                )
                mem.set_embedding(fake_embedding(hash(text) % 1000))
                session.add(mem)
            await session.commit()

        store = PgVectorStore(session_factory)
        results_a = await store.search(
            query_embedding=fake_embedding(hash("User A likes Python") % 1000),
            user_id="user_a",
            limit=10,
        )
        results_b = await store.search(
            query_embedding=fake_embedding(hash("User B likes Rust") % 1000),
            user_id="user_b",
            limit=10,
        )

        # User A should only see their own memories
        for r in results_a:
            async with session_factory() as session:
                mem = await session.get(Memory, r.memory_id)
                assert mem.user_id == "user_a", f"User A saw user_b memory: {mem.text}"

        # User B should only see their own memories
        for r in results_b:
            async with session_factory() as session:
                mem = await session.get(Memory, r.memory_id)
                assert mem.user_id == "user_b", f"User B saw user_a memory: {mem.text}"

    @pytest.mark.asyncio
    async def test_user_a_cannot_see_user_b_even_same_content(self, session_factory):
        """Even with identical text, memories should be isolated by user_id."""
        now = datetime.now(timezone.utc)

        # Both users have the exact same memory text
        async with session_factory() as session:
            for user in ["user_a", "user_b"]:
                mem = Memory(
                    user_id=user,
                    conversation_id="c1",
                    text="User is a developer",
                    memory_type="semantic",
                    tier="short_term",
                    importance=0.8,
                    confidence=0.9,
                    source_entities=[],
                    created_at=now,
                    updated_at=now,
                )
                mem.set_embedding(fake_embedding(42))
                session.add(mem)
            await session.commit()

        store = PgVectorStore(session_factory)
        results = await store.search(
            query_embedding=fake_embedding(42),
            user_id="user_a",
            limit=10,
        )

        # Should only get 1 result (user_a's), not 2
        assert len(results) == 1
        async with session_factory() as session:
            mem = await session.get(Memory, results[0].memory_id)
            assert mem.user_id == "user_a"


# ---------------------------------------------------------------------------
# Graph Isolation
# ---------------------------------------------------------------------------

class TestGraphIsolation:
    """Entity and relationship data must be scoped by user_id."""

    @pytest.mark.asyncio
    async def test_entities_isolated_between_users(self, session_factory):
        """Entity created for user_a should not appear in user_b's graph."""
        graph = KnowledgeGraph(session_factory)

        await graph.upsert_entity(
            "user_a",
            ExtractedEntity(name="Python", entity_type=EntityType.TECHNOLOGY),
        )
        await graph.upsert_entity(
            "user_b",
            ExtractedEntity(name="Rust", entity_type=EntityType.TECHNOLOGY),
        )

        # user_a's graph should not know about Rust
        context_a = await graph.get_entity_context("user_a", ["Rust"])
        assert context_a == []

        # user_b's graph should not know about Python
        context_b = await graph.get_entity_context("user_b", ["Python"])
        assert context_b == []

    @pytest.mark.asyncio
    async def test_relationships_isolated_between_users(self, session_factory):
        """Relationships for user_a should be invisible to user_b."""
        graph = KnowledgeGraph(session_factory)

        await graph.add_relationship(
            "user_a",
            ExtractedRelationship(source="Alice", target="Python", relation="uses"),
        )
        await graph.add_relationship(
            "user_b",
            ExtractedRelationship(source="Bob", target="Rust", relation="uses"),
        )

        # user_a should only see Alice → Python
        neighbors_a = await graph.get_neighbors("user_a", "Alice", max_depth=1)
        assert len(neighbors_a) == 1
        assert neighbors_a[0]["entity"] == "Python"

        # user_b should only see Bob → Rust
        neighbors_b = await graph.get_neighbors("user_b", "Bob", max_depth=1)
        assert len(neighbors_b) == 1
        assert neighbors_b[0]["entity"] == "Rust"

        # Cross-user should be empty
        assert await graph.get_neighbors("user_a", "Bob") == []
        assert await graph.get_neighbors("user_b", "Alice") == []

    @pytest.mark.asyncio
    async def test_same_entity_name_different_users_independent(self, session_factory):
        """Same entity 'Python' for two users should be independent records."""
        graph = KnowledgeGraph(session_factory)

        r1 = await graph.upsert_entity(
            "user_a",
            ExtractedEntity(name="Python", entity_type=EntityType.TECHNOLOGY),
        )
        r2 = await graph.upsert_entity(
            "user_b",
            ExtractedEntity(name="Python", entity_type=EntityType.TECHNOLOGY),
        )

        assert r1.id != r2.id
        assert r1.mention_count == 1
        assert r2.mention_count == 1

        # Upsert user_a's Python again
        r1_again = await graph.upsert_entity(
            "user_a",
            ExtractedEntity(name="Python", entity_type=EntityType.TECHNOLOGY),
        )
        assert r1_again.mention_count == 2
        # user_b's Python should still be 1
        r2_check = await graph.upsert_entity(
            "user_b",
            ExtractedEntity(name="Python", entity_type=EntityType.TECHNOLOGY),
        )
        assert r2_check.mention_count == 2  # now 2 because we just upserted it


# ---------------------------------------------------------------------------
# Lifecycle Isolation
# ---------------------------------------------------------------------------

class TestLifecycleIsolation:
    """Decay/cleanup for one user must not affect another user's memories."""

    @pytest.mark.asyncio
    async def test_decay_only_affects_target_user(self, session_factory, mock_clients, settings):
        """Applying decay to user_a should not touch user_b's memories."""
        from deepcontext.extraction.extractor import Extractor

        now = datetime.now(timezone.utc)
        old = now - timedelta(days=30)

        # Create old short-term memories for both users
        async with session_factory() as session:
            for user in ["user_a", "user_b"]:
                mem = Memory(
                    user_id=user,
                    conversation_id="c1",
                    text=f"{user} old memory",
                    memory_type="episodic",
                    tier="short_term",
                    importance=0.8,
                    confidence=0.9,
                    source_entities=[],
                    created_at=old,
                    updated_at=old,
                )
                mem.set_embedding(fake_embedding(hash(user) % 1000))
                session.add(mem)
            await session.commit()

        extractor = Extractor(mock_clients, settings)
        lifecycle = LifecycleManager(mock_clients, settings, extractor, session_factory)

        # Decay only user_a
        decayed = await lifecycle.apply_decay("user_a")
        assert decayed >= 1

        # Check user_b's memory is untouched
        async with session_factory() as session:
            b_mem = (await session.execute(
                select(Memory).where(Memory.user_id == "user_b")
            )).scalars().first()
            assert b_mem.importance == 0.8  # unchanged
            assert b_mem.is_active is True

    @pytest.mark.asyncio
    async def test_cleanup_only_affects_target_user(self, session_factory, mock_clients, settings):
        """Cleanup for user_a should not deactivate user_b's low-importance memories."""
        from deepcontext.extraction.extractor import Extractor

        now = datetime.now(timezone.utc)

        async with session_factory() as session:
            for user in ["user_a", "user_b"]:
                mem = Memory(
                    user_id=user,
                    conversation_id="c1",
                    text=f"{user} low importance memory",
                    memory_type="semantic",
                    tier="short_term",
                    importance=0.01,  # below cleanup threshold
                    confidence=0.5,
                    source_entities=[],
                    created_at=now,
                    updated_at=now,
                )
                mem.set_embedding(fake_embedding(hash(user) % 1000))
                session.add(mem)
            await session.commit()

        extractor = Extractor(mock_clients, settings)
        lifecycle = LifecycleManager(mock_clients, settings, extractor, session_factory)

        # Cleanup only user_a
        cleaned = await lifecycle.cleanup("user_a")
        assert cleaned >= 1

        # user_b's memory should still be active
        async with session_factory() as session:
            b_mem = (await session.execute(
                select(Memory).where(Memory.user_id == "user_b")
            )).scalars().first()
            assert b_mem.is_active is True
