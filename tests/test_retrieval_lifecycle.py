"""
Tests for the hybrid retriever and the lifecycle manager.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy import select

from deepcontext.core.types import MemoryTier, MemoryType
from deepcontext.db.models.memory import Memory
from deepcontext.lifecycle.manager import LifecycleManager
from deepcontext.retrieval.hybrid import HybridRetriever
from tests.conftest import fake_embedding


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion tests (static method, no DB needed)
# ---------------------------------------------------------------------------


class TestReciprocalRankFusion:
    def test_empty_inputs(self):
        result = HybridRetriever._reciprocal_rank_fusion([], [], [])
        assert result == {}

    def test_vector_only(self):
        vector = [(1, 0.95), (2, 0.85), (3, 0.70)]
        result = HybridRetriever._reciprocal_rank_fusion(vector, [], [])
        assert 1 in result
        assert 2 in result
        assert 3 in result
        # Higher ranked should have higher score
        assert result[1] > result[2] > result[3]

    def test_keyword_only(self):
        keyword = [(10, 0.8), (20, 0.5)]
        result = HybridRetriever._reciprocal_rank_fusion([], keyword, [])
        assert 10 in result
        assert result[10] > result[20]

    def test_graph_only(self):
        graph = [100, 200]
        result = HybridRetriever._reciprocal_rank_fusion([], [], graph)
        assert 100 in result
        assert 200 in result

    def test_fusion_combines_scores(self):
        """A memory appearing in multiple sources should have a higher fused score."""
        vector = [(1, 0.9), (2, 0.8)]
        keyword = [(1, 0.7), (3, 0.6)]
        graph = [1]

        result = HybridRetriever._reciprocal_rank_fusion(vector, keyword, graph)

        # Memory 1 appears in all three sources, should have the highest score
        assert result[1] > result[2]
        assert result[1] > result[3]

    def test_custom_weights(self):
        vector = [(1, 0.9)]
        keyword = [(2, 0.9)]

        # Give all weight to keyword
        result = HybridRetriever._reciprocal_rank_fusion(
            vector, keyword, [],
            vector_weight=0.0,
            keyword_weight=1.0,
            graph_weight=0.0,
        )
        assert result.get(1, 0) == 0  # Vector result gets 0
        assert result[2] > 0  # Keyword result gets weight


# ---------------------------------------------------------------------------
# Lifecycle Manager tests
# ---------------------------------------------------------------------------


class TestLifecycleGroupByEntityOverlap:
    """Test the static grouping method."""

    def _make_memory(self, entities: list[str], id_: int = 0) -> Memory:
        now = datetime.now(timezone.utc)
        mem = Memory(
            id=id_,
            user_id="u1",
            text=f"memory {id_}",
            memory_type="semantic",
            tier="short_term",
            source_entities=entities,
            created_at=now,
            updated_at=now,
        )
        return mem

    def test_no_overlap(self):
        m1 = self._make_memory(["A"], 1)
        m2 = self._make_memory(["B"], 2)
        groups = LifecycleManager._group_by_entity_overlap([m1, m2])
        assert len(groups) == 2

    def test_full_overlap(self):
        m1 = self._make_memory(["A", "B"], 1)
        m2 = self._make_memory(["A", "C"], 2)
        groups = LifecycleManager._group_by_entity_overlap([m1, m2])
        # Should be grouped together (share "A")
        assert len(groups) == 1
        assert len(groups[0]) == 2

    def test_transitive_grouping(self):
        """A-B and B-C should be in the same group (via B)."""
        m1 = self._make_memory(["A", "B"], 1)
        m2 = self._make_memory(["B", "C"], 2)
        m3 = self._make_memory(["C", "D"], 3)
        groups = LifecycleManager._group_by_entity_overlap([m1, m2, m3])
        assert len(groups) == 1

    def test_empty_entities(self):
        """Memories with no entities shouldn't form groups."""
        m1 = self._make_memory([], 1)
        m2 = self._make_memory([], 2)
        groups = LifecycleManager._group_by_entity_overlap([m1, m2])
        assert len(groups) == 2  # Each in its own group

    def test_single_memory(self):
        m1 = self._make_memory(["A"], 1)
        groups = LifecycleManager._group_by_entity_overlap([m1])
        assert len(groups) == 1

    def test_mixed_overlap(self):
        m1 = self._make_memory(["A", "B"], 1)
        m2 = self._make_memory(["A"], 2)
        m3 = self._make_memory(["C", "D"], 3)
        m4 = self._make_memory(["D", "E"], 4)
        groups = LifecycleManager._group_by_entity_overlap([m1, m2, m3, m4])
        assert len(groups) == 2  # {m1, m2} and {m3, m4}


class TestLifecycleDecay:
    @pytest.mark.asyncio
    async def test_decay_affects_old_memories(self, session_factory):
        """Old short-term memories should have their importance reduced."""
        # Create a memory from 30 days ago
        old_time = datetime.now(timezone.utc) - timedelta(days=30)
        async with session_factory() as session:
            mem = Memory(
                user_id="u1",
                text="old memory",
                memory_type="semantic",
                tier="short_term",
                importance=0.8,
                confidence=0.9,
                created_at=old_time,
                updated_at=old_time,
            )
            session.add(mem)
            await session.commit()
            await session.refresh(mem)

        from deepcontext.core.settings import DeepContextSettings

        settings = DeepContextSettings(
            openai_api_key="sk-test",
            decay_half_life_days=7.0,
        )

        from unittest.mock import MagicMock
        mock_clients = MagicMock()
        mock_extractor = MagicMock()

        manager = LifecycleManager(mock_clients, settings, mock_extractor, session_factory)
        affected = await manager.apply_decay("u1")

        assert affected > 0

        async with session_factory() as session:
            loaded = await session.get(Memory, mem.id)
            # After 30 days with 7-day half-life, importance should be very low
            # or memory should be deactivated
            assert loaded.importance < 0.8 or loaded.is_active is False

    @pytest.mark.asyncio
    async def test_decay_skips_long_term(self, session_factory):
        """Long-term memories should not be affected by decay."""
        old_time = datetime.now(timezone.utc) - timedelta(days=30)
        async with session_factory() as session:
            mem = Memory(
                user_id="u1",
                text="long term fact",
                memory_type="semantic",
                tier="long_term",
                importance=0.9,
                confidence=0.9,
                created_at=old_time,
                updated_at=old_time,
            )
            session.add(mem)
            await session.commit()
            await session.refresh(mem)

        from deepcontext.core.settings import DeepContextSettings
        from unittest.mock import MagicMock

        settings = DeepContextSettings(openai_api_key="sk-test")
        manager = LifecycleManager(MagicMock(), settings, MagicMock(), session_factory)
        affected = await manager.apply_decay("u1")

        assert affected == 0

        async with session_factory() as session:
            loaded = await session.get(Memory, mem.id)
            assert loaded.importance == 0.9  # Unchanged

    @pytest.mark.asyncio
    async def test_decay_deactivates_fully_decayed(self, session_factory):
        """Very old memories should be soft-deleted."""
        very_old = datetime.now(timezone.utc) - timedelta(days=365)
        async with session_factory() as session:
            mem = Memory(
                user_id="u1",
                text="ancient memory",
                memory_type="episodic",
                tier="short_term",
                importance=0.3,
                confidence=0.7,
                created_at=very_old,
                updated_at=very_old,
            )
            session.add(mem)
            await session.commit()
            await session.refresh(mem)

        from deepcontext.core.settings import DeepContextSettings
        from unittest.mock import MagicMock

        settings = DeepContextSettings(openai_api_key="sk-test", decay_half_life_days=7.0)
        manager = LifecycleManager(MagicMock(), settings, MagicMock(), session_factory)
        await manager.apply_decay("u1")

        async with session_factory() as session:
            loaded = await session.get(Memory, mem.id)
            assert loaded.is_active is False


class TestLifecycleCleanup:
    @pytest.mark.asyncio
    async def test_cleanup_removes_low_importance(self, session_factory):
        """Cleanup should deactivate non-long-term memories below threshold."""
        now = datetime.now(timezone.utc)
        async with session_factory() as session:
            mem = Memory(
                user_id="u1",
                text="low importance",
                memory_type="semantic",
                tier="short_term",
                importance=0.01,
                confidence=0.5,
                created_at=now,
                updated_at=now,
            )
            session.add(mem)
            await session.commit()

        from deepcontext.core.settings import DeepContextSettings
        from unittest.mock import MagicMock

        settings = DeepContextSettings(openai_api_key="sk-test")
        manager = LifecycleManager(MagicMock(), settings, MagicMock(), session_factory)
        cleaned = await manager.cleanup("u1")

        assert cleaned == 1

    @pytest.mark.asyncio
    async def test_cleanup_preserves_long_term(self, session_factory):
        """Cleanup should NOT deactivate long-term memories."""
        now = datetime.now(timezone.utc)
        async with session_factory() as session:
            mem = Memory(
                user_id="u1",
                text="important long term",
                memory_type="semantic",
                tier="long_term",
                importance=0.01,  # Low importance but long-term
                confidence=0.5,
                created_at=now,
                updated_at=now,
            )
            session.add(mem)
            await session.commit()

        from deepcontext.core.settings import DeepContextSettings
        from unittest.mock import MagicMock

        settings = DeepContextSettings(openai_api_key="sk-test")
        manager = LifecycleManager(MagicMock(), settings, MagicMock(), session_factory)
        cleaned = await manager.cleanup("u1")

        assert cleaned == 0
