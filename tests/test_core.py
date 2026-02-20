"""
Tests for core types and settings.
"""

from __future__ import annotations

import os
import unittest.mock

import pytest

from deepcontext.core.settings import DeepContextSettings
from deepcontext.core.types import (
    AddResponse,
    EntityType,
    ExtractedEntity,
    ExtractedFact,
    ExtractedRelationship,
    ExtractionResult,
    MemorySearchResult,
    MemoryTier,
    MemoryType,
    SearchResponse,
    ToolAction,
    ToolDecision,
)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestEnums:
    def test_memory_tier_values(self):
        assert MemoryTier.WORKING.value == "working"
        assert MemoryTier.SHORT_TERM.value == "short_term"
        assert MemoryTier.LONG_TERM.value == "long_term"

    def test_memory_type_values(self):
        assert MemoryType.SEMANTIC.value == "semantic"
        assert MemoryType.EPISODIC.value == "episodic"
        assert MemoryType.PROCEDURAL.value == "procedural"

    def test_entity_type_values(self):
        assert EntityType.PERSON.value == "person"
        assert EntityType.TECHNOLOGY.value == "technology"
        assert EntityType.OTHER.value == "other"

    def test_tool_action_values(self):
        assert ToolAction.ADD.value == "ADD"
        assert ToolAction.UPDATE.value == "UPDATE"
        assert ToolAction.DELETE.value == "DELETE"
        assert ToolAction.REPLACE.value == "REPLACE"
        assert ToolAction.NOOP.value == "NOOP"

    def test_enum_from_string(self):
        assert MemoryTier("short_term") == MemoryTier.SHORT_TERM
        assert ToolAction("ADD") == ToolAction.ADD


# ---------------------------------------------------------------------------
# Pydantic model tests
# ---------------------------------------------------------------------------


class TestExtractedFact:
    def test_defaults(self):
        fact = ExtractedFact(text="User likes Python")
        assert fact.text == "User likes Python"
        assert fact.memory_type == MemoryType.SEMANTIC
        assert fact.importance == 0.5
        assert fact.confidence == 0.8
        assert fact.entities == []

    def test_full_construction(self):
        fact = ExtractedFact(
            text="User is learning Rust",
            memory_type=MemoryType.EPISODIC,
            importance=0.9,
            confidence=0.95,
            entities=["Rust"],
        )
        assert fact.importance == 0.9
        assert fact.entities == ["Rust"]

    def test_importance_bounds(self):
        with pytest.raises(Exception):
            ExtractedFact(text="bad", importance=1.5)
        with pytest.raises(Exception):
            ExtractedFact(text="bad", importance=-0.1)


class TestExtractionResult:
    def test_empty(self):
        result = ExtractionResult()
        assert result.semantic == []
        assert result.episodic == []
        assert result.entities == []
        assert result.relationships == []

    def test_full(self):
        result = ExtractionResult(
            semantic=[ExtractedFact(text="fact1")],
            episodic=[ExtractedFact(text="event1", memory_type=MemoryType.EPISODIC)],
            entities=[ExtractedEntity(name="Python", entity_type=EntityType.TECHNOLOGY)],
            relationships=[
                ExtractedRelationship(source="User", target="Python", relation="uses")
            ],
        )
        assert len(result.semantic) == 1
        assert len(result.entities) == 1
        assert result.relationships[0].relation == "uses"


class TestToolDecision:
    def test_add_action(self):
        decision = ToolDecision(action=ToolAction.ADD, text="new fact")
        assert decision.action == ToolAction.ADD
        assert decision.memory_id is None

    def test_update_action(self):
        decision = ToolDecision(action=ToolAction.UPDATE, memory_id=42, text="updated")
        assert decision.memory_id == 42


class TestSearchResponse:
    def test_empty_response(self):
        resp = SearchResponse(query="test", user_id="u1", total=0, results=[])
        assert resp.total == 0

    def test_serialization(self):
        from datetime import datetime, timezone

        resp = SearchResponse(
            query="Python",
            user_id="u1",
            total=1,
            results=[
                MemorySearchResult(
                    memory_id=1,
                    text="User is a Python developer",
                    memory_type=MemoryType.SEMANTIC,
                    tier=MemoryTier.SHORT_TERM,
                    score=0.95,
                    created_at=datetime.now(timezone.utc),
                )
            ],
        )
        d = resp.model_dump()
        assert d["total"] == 1
        assert d["results"][0]["score"] == 0.95


class TestAddResponse:
    def test_defaults(self):
        resp = AddResponse()
        assert resp.memories_added == 0
        assert resp.semantic_facts == []

    def test_with_data(self):
        resp = AddResponse(
            semantic_facts=["fact1", "fact2"],
            memories_added=2,
            entities_found=["Python"],
        )
        assert resp.memories_added == 2
        assert len(resp.semantic_facts) == 2


# ---------------------------------------------------------------------------
# Settings tests
# ---------------------------------------------------------------------------


class TestSettings:
    def test_default_settings_with_api_key(self):
        """Settings should work with just an API key."""
        s = DeepContextSettings(openai_api_key="sk-test-key")
        assert s.openai_api_key == "sk-test-key"
        assert s.llm_model == "gpt-4o-mini"
        assert s.embedding_model == "text-embedding-3-small"
        assert s.embedding_dimensions == 1536
        assert s.debug is False

    def test_sqlite_fallback(self):
        """When no database_url is set, should fall back to SQLite."""
        s = DeepContextSettings(openai_api_key="sk-test-key", database_url="")
        assert "sqlite" in s.database_url
        assert ".deepcontext" in s.database_url

    def test_llm_api_key_property(self):
        s = DeepContextSettings(openai_api_key="sk-test-key")
        assert s.llm_api_key == "sk-test-key"

    def test_embedding_api_key_property(self):
        s = DeepContextSettings(openai_api_key="sk-test-key")
        assert s.embedding_api_key == "sk-test-key"

    def test_missing_api_key_raises(self):
        """Should raise if no API key is provided."""
        # Clear env vars AND prevent pydantic-settings from reading .env file
        clean_env = {
            k: v for k, v in os.environ.items()
            if "OPENAI" not in k and "OPENROUTER" not in k and "DEEPCONTEXT" not in k
        }
        with unittest.mock.patch.dict(os.environ, clean_env, clear=True):
            with pytest.raises(ValueError, match="openai_api_key is required"):
                DeepContextSettings(
                    database_url="sqlite+aiosqlite:///:memory:",
                    _env_file=None,  # prevent reading .env file
                )

    def test_custom_models(self):
        s = DeepContextSettings(
            openai_api_key="sk-test",
            llm_model="gpt-4o",
            embedding_model="text-embedding-3-large",
            embedding_dimensions=3072,
        )
        assert s.llm_model == "gpt-4o"
        assert s.embedding_dimensions == 3072

    def test_memory_tuning_defaults(self):
        s = DeepContextSettings(openai_api_key="sk-test")
        assert s.consolidation_threshold == 20
        assert s.decay_half_life_days == 7.0
        assert s.connection_similarity_threshold == 0.6
        assert s.max_connections_per_memory == 5
