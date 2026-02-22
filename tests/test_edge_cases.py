"""
Edge case and error recovery tests for DeepContext.

These tests verify that the system handles unexpected inputs, malformed LLM
responses, and boundary conditions gracefully without crashing or corrupting state.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from deepcontext.core.settings import DeepContextSettings
from deepcontext.core.types import (
    EntityType,
    ExtractedEntity,
    ExtractedFact,
    ExtractedRelationship,
    ExtractionResult,
    MemoryTier,
    MemoryType,
    ToolAction,
    ToolDecision,
)
from deepcontext.db.models.memory import Memory
from deepcontext.extraction.extractor import Extractor
from deepcontext.graph.knowledge_graph import KnowledgeGraph
from deepcontext.retrieval.hybrid import HybridRetriever
from deepcontext.vectorstore.pgvector_store import PgVectorStore, _cosine_similarity
from tests.conftest import fake_embedding


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_llm_response(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# Extraction Edge Cases
# ---------------------------------------------------------------------------

class TestExtractionEdgeCases:
    """Tests for LLM returning unexpected/malformed output."""

    @pytest.fixture
    def extractor(self, mock_clients, settings):
        return Extractor(mock_clients, settings)

    @pytest.mark.asyncio
    async def test_extraction_empty_json_object(self, extractor, mock_clients):
        """LLM returns {} — should produce an empty ExtractionResult."""
        mock_clients.llm.chat.completions.create = AsyncMock(
            return_value=_mock_llm_response("{}")
        )
        result = await extractor.extract_memories(
            ["USER: hello"], summary="", recent_messages=[]
        )
        assert result.semantic == []
        assert result.episodic == []
        assert result.entities == []
        assert result.relationships == []

    @pytest.mark.asyncio
    async def test_extraction_empty_string(self, extractor, mock_clients):
        """LLM returns empty string — should produce empty result."""
        mock_clients.llm.chat.completions.create = AsyncMock(
            return_value=_mock_llm_response("")
        )
        result = await extractor.extract_memories(
            ["USER: hi"], summary="", recent_messages=[]
        )
        assert result.semantic == []

    @pytest.mark.asyncio
    async def test_extraction_invalid_json(self, extractor, mock_clients):
        """LLM returns malformed JSON — should return empty result, not crash."""
        mock_clients.llm.chat.completions.create = AsyncMock(
            return_value=_mock_llm_response("this is not valid json {broken}")
        )
        result = await extractor.extract_memories(
            ["USER: test"], summary="", recent_messages=[]
        )
        assert isinstance(result, ExtractionResult)
        assert result.semantic == []

    @pytest.mark.asyncio
    async def test_extraction_partial_json(self, extractor, mock_clients):
        """LLM returns JSON with only some fields — should handle missing keys."""
        mock_clients.llm.chat.completions.create = AsyncMock(
            return_value=_mock_llm_response(json.dumps({
                "semantic": [{"text": "User likes cats"}],
                # missing episodic, entities, relationships
            }))
        )
        result = await extractor.extract_memories(
            ["USER: I like cats"], summary="", recent_messages=[]
        )
        assert len(result.semantic) == 1
        assert result.episodic == []
        assert result.entities == []
        assert result.relationships == []

    @pytest.mark.asyncio
    async def test_extraction_semantic_as_plain_strings(self, extractor, mock_clients):
        """LLM returns semantic facts as plain strings instead of dicts."""
        mock_clients.llm.chat.completions.create = AsyncMock(
            return_value=_mock_llm_response(json.dumps({
                "semantic": ["User is a developer", "User prefers dark mode"],
                "episodic": [],
                "entities": [],
                "relationships": [],
            }))
        )
        result = await extractor.extract_memories(
            ["USER: test"], summary="", recent_messages=[]
        )
        assert len(result.semantic) == 2
        assert result.semantic[0].text == "User is a developer"

    @pytest.mark.asyncio
    async def test_extraction_with_extra_fields(self, extractor, mock_clients):
        """LLM returns JSON with extra unexpected fields — shouldn't crash."""
        mock_clients.llm.chat.completions.create = AsyncMock(
            return_value=_mock_llm_response(json.dumps({
                "semantic": [{"text": "fact1", "importance": 0.8, "confidence": 0.9, "entities": []}],
                "episodic": [],
                "entities": [],
                "relationships": [],
                "extra_field": "should be ignored",
                "notes": ["this is extra"],
            }))
        )
        result = await extractor.extract_memories(
            ["USER: test"], summary="", recent_messages=[]
        )
        assert len(result.semantic) == 1

    @pytest.mark.asyncio
    async def test_extraction_wrapped_in_markdown(self, extractor, mock_clients):
        """LLM wraps JSON in ```json code block."""
        json_content = json.dumps({
            "semantic": [{"text": "User codes in Rust"}],
            "episodic": [],
            "entities": [],
            "relationships": [],
        })
        mock_clients.llm.chat.completions.create = AsyncMock(
            return_value=_mock_llm_response(f"```json\n{json_content}\n```")
        )
        result = await extractor.extract_memories(
            ["USER: Rust is great"], summary="", recent_messages=[]
        )
        assert len(result.semantic) == 1
        assert result.semantic[0].text == "User codes in Rust"


# ---------------------------------------------------------------------------
# Classification Edge Cases
# ---------------------------------------------------------------------------

class TestClassificationEdgeCases:
    """Tests for classify_action handling unexpected LLM output."""

    @pytest.fixture
    def extractor(self, mock_clients, settings):
        return Extractor(mock_clients, settings)

    @pytest.mark.asyncio
    async def test_classify_with_no_similar_memories(self, extractor, mock_clients):
        """Classifying with empty similar_memories should default to ADD."""
        mock_clients.llm.chat.completions.create = AsyncMock(
            return_value=_mock_llm_response(json.dumps({
                "action": "ADD", "memory_id": None, "text": "new fact", "reason": "new",
            }))
        )
        result = await extractor.classify_action("new fact", [])
        assert result.action == ToolAction.ADD

    @pytest.mark.asyncio
    async def test_classify_invalid_json_defaults_to_add(self, extractor, mock_clients):
        """If LLM returns invalid JSON for classification, should default to ADD."""
        mock_clients.llm.chat.completions.create = AsyncMock(
            return_value=_mock_llm_response("not json at all")
        )
        result = await extractor.classify_action("some fact", [])
        assert result.action == ToolAction.ADD

    @pytest.mark.asyncio
    async def test_classify_unknown_action_defaults_to_add(self, extractor, mock_clients):
        """If LLM returns an unknown action string, should default to ADD."""
        mock_clients.llm.chat.completions.create = AsyncMock(
            return_value=_mock_llm_response(json.dumps({
                "action": "YEET", "memory_id": None, "text": "fact", "reason": "??",
            }))
        )
        result = await extractor.classify_action("some fact", [])
        assert result.action == ToolAction.ADD


# ---------------------------------------------------------------------------
# Consolidation Edge Cases
# ---------------------------------------------------------------------------

class TestConsolidationEdgeCases:
    """Tests for consolidate_memories handling unexpected LLM output."""

    @pytest.fixture
    def extractor(self, mock_clients, settings):
        return Extractor(mock_clients, settings)

    @pytest.mark.asyncio
    async def test_consolidate_empty_list(self, extractor, mock_clients):
        """Consolidating an empty memory list should return a reasonable default."""
        mock_clients.llm.chat.completions.create = AsyncMock(
            return_value=_mock_llm_response(json.dumps({
                "text": "", "importance": 0.5, "confidence": 0.5, "entities": [],
            }))
        )
        result = await extractor.consolidate_memories([])
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_consolidate_single_memory(self, extractor, mock_clients):
        """Consolidating a single memory should still work."""
        mock_clients.llm.chat.completions.create = AsyncMock(
            return_value=_mock_llm_response(json.dumps({
                "text": "User is a Python developer",
                "importance": 0.8,
                "confidence": 0.9,
                "entities": ["Python"],
            }))
        )
        result = await extractor.consolidate_memories(["User is a Python developer"])
        assert result["text"] == "User is a Python developer"

    @pytest.mark.asyncio
    async def test_consolidate_malformed_response(self, extractor, mock_clients):
        """Malformed consolidation response should return a default dict."""
        mock_clients.llm.chat.completions.create = AsyncMock(
            return_value=_mock_llm_response("broken json {{{")
        )
        result = await extractor.consolidate_memories(["fact1", "fact2"])
        # Should still return a dict, not crash
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Knowledge Graph Edge Cases
# ---------------------------------------------------------------------------

class TestGraphEdgeCases:
    @pytest.mark.asyncio
    async def test_entity_self_relationship(self, session_factory):
        """An entity related to itself should be handled without error."""
        graph = KnowledgeGraph(session_factory)
        rel = ExtractedRelationship(
            source="Python", target="Python", relation="is_superset_of",
        )
        result = await graph.add_relationship("u1", rel)
        assert result is not None
        assert result.relation == "is_superset_of"

    @pytest.mark.asyncio
    async def test_entity_empty_name(self, session_factory):
        """Entity with empty name should be handled."""
        graph = KnowledgeGraph(session_factory)
        entity = ExtractedEntity(name="", entity_type=EntityType.OTHER)
        # Should either reject or handle gracefully
        result = await graph.upsert_entity("u1", entity)
        # If it accepts empty names, it should at least not crash
        assert result is not None

    @pytest.mark.asyncio
    async def test_relationship_empty_target(self, session_factory):
        """Relationship with empty target should return None."""
        graph = KnowledgeGraph(session_factory)
        rel = ExtractedRelationship(source="User", target="", relation="uses")
        result = await graph.add_relationship("u1", rel)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_neighbors_very_deep(self, session_factory):
        """get_neighbors with large depth on a small graph should not crash."""
        graph = KnowledgeGraph(session_factory)
        await graph.add_relationship(
            "u1", ExtractedRelationship(source="A", target="B", relation="r1")
        )
        result = await graph.get_neighbors("u1", "A", max_depth=100)
        # Should just find B at depth 1, not crash
        assert len(result) >= 1

    @pytest.mark.asyncio
    async def test_duplicate_entity_different_types(self, session_factory):
        """Same entity name with different types should upsert (not duplicate)."""
        graph = KnowledgeGraph(session_factory)
        e1 = ExtractedEntity(name="Python", entity_type=EntityType.TECHNOLOGY)
        e2 = ExtractedEntity(name="Python", entity_type=EntityType.CONCEPT)

        r1 = await graph.upsert_entity("u1", e1)
        r2 = await graph.upsert_entity("u1", e2)

        # Should be the same entity (upserted by name+user), not two separate ones
        assert r1.id == r2.id
        assert r2.mention_count == 2


# ---------------------------------------------------------------------------
# Vector Store Edge Cases
# ---------------------------------------------------------------------------

class TestVectorStoreEdgeCases:
    @pytest.mark.asyncio
    async def test_search_with_zero_vector(self, session_factory, sample_memories):
        """Searching with a zero vector should not crash."""
        store = PgVectorStore(session_factory)
        results = await store.search(
            query_embedding=[0.0] * 8,
            user_id="test_user",
            limit=5,
        )
        # Should return results (cosine sim with zero vector = 0.0)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_with_single_dimension(self, session_factory):
        """Searching with mismatched embedding dimensions should handle gracefully."""
        store = PgVectorStore(session_factory)
        results = await store.search(
            query_embedding=[1.0],
            user_id="test_user",
            limit=5,
        )
        # Mismatched dims: _cosine_similarity returns 0.0, so everything is filtered
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Cosine Similarity Edge Cases
# ---------------------------------------------------------------------------

class TestCosineSimilarityEdgeCases:
    def test_very_large_vectors(self):
        """Cosine similarity with large vectors should not overflow."""
        a = [1e10] * 100
        b = [1e10] * 100
        result = _cosine_similarity(a, b)
        assert abs(result - 1.0) < 1e-6

    def test_very_small_vectors(self):
        """Cosine similarity with very small values should still work."""
        a = [1e-10] * 100
        b = [1e-10] * 100
        result = _cosine_similarity(a, b)
        assert abs(result - 1.0) < 1e-6

    def test_empty_vectors(self):
        """Cosine similarity with empty vectors should return 0.0."""
        result = _cosine_similarity([], [])
        assert result == 0.0

    def test_single_element_vectors(self):
        """Cosine similarity with single-element vectors."""
        result = _cosine_similarity([1.0], [1.0])
        assert abs(result - 1.0) < 1e-6
