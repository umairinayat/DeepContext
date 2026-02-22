"""
Tests for the hybrid retriever and the lifecycle manager.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from sqlalchemy import select

from deepcontext.core.settings import DeepContextSettings
from deepcontext.core.types import MemoryTier, MemoryType
from deepcontext.db.models.memory import Memory
from deepcontext.lifecycle.manager import LifecycleManager
from deepcontext.retrieval.hybrid import HybridRetriever
from tests.conftest import fake_embedding


def _mock_llm_response(content: str) -> MagicMock:
    """Create a mock ChatCompletion response."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


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


# ---------------------------------------------------------------------------
# Phase 4: Lifecycle Deep Tests
# ---------------------------------------------------------------------------


class TestDecayMathCorrectness:
    """Verify the Ebbinghaus decay formula produces correct values."""

    @pytest.mark.asyncio
    async def test_decay_formula_7day_halflife(self, session_factory):
        """After exactly 7 days with 7-day half-life, importance ≈ 0.5 × original."""
        seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
        async with session_factory() as session:
            mem = Memory(
                user_id="u1",
                text="exactly 7 days old",
                memory_type="semantic",
                tier="short_term",
                importance=1.0,
                confidence=0.9,
                access_count=0,  # no reinforcement
                created_at=seven_days_ago,
                updated_at=seven_days_ago,
            )
            session.add(mem)
            await session.commit()
            await session.refresh(mem)

        settings = DeepContextSettings(openai_api_key="sk-test", decay_half_life_days=7.0)
        manager = LifecycleManager(MagicMock(), settings, MagicMock(), session_factory)
        await manager.apply_decay("u1")

        async with session_factory() as session:
            loaded = await session.get(Memory, mem.id)
            # decay_factor = exp(-0.693 * 7 / 7) = exp(-0.693) ≈ 0.5
            assert 0.45 < loaded.importance < 0.55, f"Expected ~0.5, got {loaded.importance}"

    @pytest.mark.asyncio
    async def test_access_count_slows_decay(self, session_factory):
        """Higher access_count should result in less decay (reinforcement)."""
        old_time = datetime.now(timezone.utc) - timedelta(days=14)
        async with session_factory() as session:
            mem_no_access = Memory(
                user_id="u1", text="no access", memory_type="semantic",
                tier="short_term", importance=1.0, confidence=0.9,
                access_count=0, created_at=old_time, updated_at=old_time,
            )
            mem_with_access = Memory(
                user_id="u1", text="high access", memory_type="semantic",
                tier="short_term", importance=1.0, confidence=0.9,
                access_count=20, created_at=old_time, updated_at=old_time,
            )
            session.add(mem_no_access)
            session.add(mem_with_access)
            await session.commit()
            await session.refresh(mem_no_access)
            await session.refresh(mem_with_access)

        settings = DeepContextSettings(openai_api_key="sk-test", decay_half_life_days=7.0)
        manager = LifecycleManager(MagicMock(), settings, MagicMock(), session_factory)
        await manager.apply_decay("u1")

        async with session_factory() as session:
            no_acc = await session.get(Memory, mem_no_access.id)
            hi_acc = await session.get(Memory, mem_with_access.id)
            # Frequently accessed memory should retain more importance
            assert hi_acc.importance > no_acc.importance, (
                f"High-access ({hi_acc.importance}) should be > no-access ({no_acc.importance})"
            )


class TestConsolidationFullFlow:
    """Test the complete consolidation pipeline."""

    @pytest.mark.asyncio
    async def test_consolidation_groups_and_merges(self, session_factory, mock_clients, settings):
        """Full flow: group → LLM merge → long-term created → sources deactivated."""
        from deepcontext.extraction.extractor import Extractor
        import json

        now = datetime.now(timezone.utc)

        # Create 4 short-term semantic memories with overlapping entities
        async with session_factory() as session:
            for i, (text, entities) in enumerate([
                ("User knows Python", ["Python"]),
                ("User is good at Python programming", ["Python"]),
                ("User uses FastAPI for web APIs", ["FastAPI"]),
                ("User prefers FastAPI over Flask", ["FastAPI", "Flask"]),
            ]):
                mem = Memory(
                    user_id="u1", conversation_id="c1", text=text,
                    memory_type="semantic", tier="short_term",
                    importance=0.7, confidence=0.8, source_entities=entities,
                    created_at=now, updated_at=now,
                )
                mem.set_embedding(fake_embedding(hash(text) % 1000))
                session.add(mem)
            await session.commit()

        # Mock the LLM consolidation response
        mock_clients.llm.chat.completions.create = AsyncMock(
            return_value=_mock_llm_response(json.dumps({
                "text": "User is a skilled Python developer who uses FastAPI",
                "importance": 0.85,
                "confidence": 0.9,
                "entities": ["Python", "FastAPI"],
            }))
        )

        settings_low = DeepContextSettings(
            openai_api_key="sk-test",
            consolidation_threshold=2,  # very low threshold
        )
        extractor = Extractor(mock_clients, settings_low)
        manager = LifecycleManager(mock_clients, settings_low, extractor, session_factory)

        consolidated = await manager.consolidate("u1")
        assert consolidated >= 1

        # Verify new long-term memory exists
        async with session_factory() as session:
            lt_mems = (await session.execute(
                select(Memory).where(
                    Memory.user_id == "u1",
                    Memory.tier == "long_term",
                    Memory.is_active == True,
                )
            )).scalars().all()
            assert len(lt_mems) >= 1

            # Verify source memories are deactivated
            st_mems = (await session.execute(
                select(Memory).where(
                    Memory.user_id == "u1",
                    Memory.tier == "short_term",
                )
            )).scalars().all()
            deactivated = [m for m in st_mems if not m.is_active]
            assert len(deactivated) >= 2  # at least one group was consolidated

    @pytest.mark.asyncio
    async def test_consolidation_below_threshold_skipped(self, session_factory, mock_clients, settings):
        """Fewer memories than threshold → no consolidation."""
        from deepcontext.extraction.extractor import Extractor

        now = datetime.now(timezone.utc)

        # Only 1 short-term semantic memory
        async with session_factory() as session:
            mem = Memory(
                user_id="u1", text="Only memory", memory_type="semantic",
                tier="short_term", importance=0.7, confidence=0.8,
                source_entities=["X"], created_at=now, updated_at=now,
            )
            mem.set_embedding(fake_embedding(42))
            session.add(mem)
            await session.commit()

        settings_high = DeepContextSettings(
            openai_api_key="sk-test",
            consolidation_threshold=10,  # need 10 memories
        )
        extractor = Extractor(mock_clients, settings_high)
        manager = LifecycleManager(mock_clients, settings_high, extractor, session_factory)

        consolidated = await manager.consolidate("u1")
        assert consolidated == 0

    @pytest.mark.asyncio
    async def test_lifecycle_double_decay_not_double_apply(self, session_factory):
        """Running decay twice should not halve importance twice from original."""
        old_time = datetime.now(timezone.utc) - timedelta(days=7)
        async with session_factory() as session:
            mem = Memory(
                user_id="u1", text="decay test", memory_type="semantic",
                tier="short_term", importance=1.0, confidence=0.9,
                access_count=0, created_at=old_time, updated_at=old_time,
            )
            session.add(mem)
            await session.commit()
            await session.refresh(mem)

        settings = DeepContextSettings(openai_api_key="sk-test", decay_half_life_days=7.0)
        manager = LifecycleManager(MagicMock(), settings, MagicMock(), session_factory)

        # First decay
        await manager.apply_decay("u1")
        async with session_factory() as session:
            loaded = await session.get(Memory, mem.id)
            importance_after_first = loaded.importance

        # Second decay — should decay from current importance, not from original
        await manager.apply_decay("u1")
        async with session_factory() as session:
            loaded = await session.get(Memory, mem.id)
            # After second decay, importance should be lower than after first
            # But the key point is that it decays from the CURRENT value
            assert loaded.importance <= importance_after_first


# ---------------------------------------------------------------------------
# Phase 5: Retrieval Quality Tests
# ---------------------------------------------------------------------------


class TestRRFWeightSensitivity:
    """Test that changing RRF weights changes the ranking."""

    def test_vector_weight_dominates(self):
        """With high vector weight, vector rank 1 should beat keyword rank 1."""
        vector = [(1, 0.99)]
        keyword = [(2, 0.99)]

        result = HybridRetriever._reciprocal_rank_fusion(
            vector, keyword, [],
            vector_weight=0.9, keyword_weight=0.1, graph_weight=0.0,
        )
        assert result[1] > result[2]

    def test_keyword_weight_dominates(self):
        """With high keyword weight, keyword rank 1 should beat vector rank 1."""
        vector = [(1, 0.99)]
        keyword = [(2, 0.99)]

        result = HybridRetriever._reciprocal_rank_fusion(
            vector, keyword, [],
            vector_weight=0.1, keyword_weight=0.9, graph_weight=0.0,
        )
        assert result[2] > result[1]

    def test_graph_boost_helps_ranking(self):
        """A memory with graph boost should rank higher than one without."""
        vector = [(1, 0.9), (2, 0.85)]
        keyword = []
        graph = [2]  # memory 2 gets graph boost

        result = HybridRetriever._reciprocal_rank_fusion(
            vector, keyword, graph,
            vector_weight=0.5, keyword_weight=0.0, graph_weight=0.5,
        )
        # Memory 2 appears in vector AND graph, memory 1 only in vector
        assert result[2] > result[1]

    def test_multi_source_appearance_boosts(self):
        """Memory appearing in all 3 sources should always rank highest."""
        vector = [(1, 0.9), (2, 0.95)]  # memory 2 ranked higher in vector
        keyword = [(1, 0.8)]
        graph = [1]

        result = HybridRetriever._reciprocal_rank_fusion(vector, keyword, graph)
        # Memory 1 appears in all 3 sources, should beat memory 2
        assert result[1] > result[2]


class TestAccessTracking:
    """Verify that search operations update access counts."""

    @pytest.mark.asyncio
    async def test_search_increments_access_count(
        self, session_factory, mock_clients, settings, sample_memories
    ):
        """After searching, accessed memories should have incremented access_count."""
        from deepcontext.vectorstore.pgvector_store import PgVectorStore
        from deepcontext.graph.knowledge_graph import KnowledgeGraph

        store = PgVectorStore(session_factory)
        graph = KnowledgeGraph(session_factory)
        retriever = HybridRetriever(
            mock_clients, settings, store, graph, session_factory
        )

        # Get initial access count
        test_mem = sample_memories[0]
        initial_count = test_mem.access_count

        # Perform a search that should retrieve this memory
        await retriever.search(
            query=test_mem.text,
            user_id="test_user",
            limit=5,
        )

        # Check that access_count was incremented
        async with session_factory() as session:
            reloaded = await session.get(Memory, test_mem.id)
            assert reloaded.access_count >= initial_count + 1

