"""
Tests for the extraction module.

Uses mock LLM responses to avoid API calls.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepcontext.core.types import (
    ExtractionResult,
    MemoryType,
    ToolAction,
    ToolDecision,
)
from deepcontext.extraction.extractor import Extractor


def _mock_llm_response(content: str):
    """Create a mock OpenAI ChatCompletion response."""
    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message.content = content
    return mock


class TestExtractorParseJson:
    """Test the static _parse_json method."""

    def test_plain_json(self):
        raw = '{"semantic": [{"text": "fact1"}]}'
        result = Extractor._parse_json(raw)
        assert result["semantic"][0]["text"] == "fact1"

    def test_markdown_code_block(self):
        raw = '```json\n{"semantic": [{"text": "fact1"}]}\n```'
        result = Extractor._parse_json(raw)
        assert result["semantic"][0]["text"] == "fact1"

    def test_markdown_code_block_no_lang(self):
        raw = '```\n{"key": "value"}\n```'
        result = Extractor._parse_json(raw)
        assert result["key"] == "value"

    def test_invalid_json(self):
        raw = "this is not json"
        result = Extractor._parse_json(raw)
        assert result == {}

    def test_empty_string(self):
        result = Extractor._parse_json("")
        assert result == {}

    def test_nested_json(self):
        data = {
            "semantic": [{"text": "fact", "importance": 0.9, "entities": ["Python"]}],
            "entities": [{"name": "Python", "entity_type": "technology"}],
        }
        result = Extractor._parse_json(json.dumps(data))
        assert len(result["semantic"]) == 1
        assert result["entities"][0]["name"] == "Python"


class TestBuildExtractionResult:
    """Test the static _build_extraction_result method."""

    def test_empty_input(self):
        result = Extractor._build_extraction_result({})
        assert result.semantic == []
        assert result.episodic == []
        assert result.entities == []
        assert result.relationships == []

    def test_semantic_facts_as_strings(self):
        parsed = {"semantic": ["fact1", "fact2"]}
        result = Extractor._build_extraction_result(parsed)
        assert len(result.semantic) == 2
        assert result.semantic[0].text == "fact1"
        assert result.semantic[0].memory_type == MemoryType.SEMANTIC

    def test_semantic_facts_as_dicts(self):
        parsed = {
            "semantic": [
                {
                    "text": "User likes Python",
                    "importance": 0.9,
                    "confidence": 0.95,
                    "entities": ["Python"],
                }
            ]
        }
        result = Extractor._build_extraction_result(parsed)
        assert len(result.semantic) == 1
        assert result.semantic[0].importance == 0.9
        assert result.semantic[0].entities == ["Python"]

    def test_episodic_facts(self):
        parsed = {"episodic": [{"text": "debugging auth", "importance": 0.6}]}
        result = Extractor._build_extraction_result(parsed)
        assert len(result.episodic) == 1
        assert result.episodic[0].memory_type == MemoryType.EPISODIC

    def test_entities(self):
        parsed = {
            "entities": [
                {"name": "Python", "entity_type": "technology"},
                {"name": "Acme", "entity_type": "organization"},
            ]
        }
        result = Extractor._build_extraction_result(parsed)
        assert len(result.entities) == 2
        assert result.entities[0].name == "Python"

    def test_relationships(self):
        parsed = {
            "relationships": [
                {"source": "User", "target": "Python", "relation": "uses"}
            ]
        }
        result = Extractor._build_extraction_result(parsed)
        assert len(result.relationships) == 1
        assert result.relationships[0].relation == "uses"


class TestExtractMemories:
    @pytest.mark.asyncio
    async def test_extract_memories(self, mock_clients, settings):
        """extract_memories should call LLM and parse the response."""
        llm_response = json.dumps(
            {
                "semantic": [
                    {"text": "User is a developer", "importance": 0.8, "entities": ["developer"]},
                ],
                "episodic": [],
                "entities": [{"name": "developer", "entity_type": "concept"}],
                "relationships": [],
            }
        )

        mock_clients.llm.chat.completions.create = AsyncMock(
            return_value=_mock_llm_response(llm_response)
        )

        extractor = Extractor(mock_clients, settings)
        result = await extractor.extract_memories(
            latest_messages=["USER: I am a developer"],
            summary="",
            recent_messages=["USER: I am a developer"],
        )

        assert len(result.semantic) == 1
        assert result.semantic[0].text == "User is a developer"
        assert len(result.entities) == 1

    @pytest.mark.asyncio
    async def test_extract_handles_empty_response(self, mock_clients, settings):
        """Should handle empty LLM response gracefully."""
        mock_clients.llm.chat.completions.create = AsyncMock(
            return_value=_mock_llm_response("{}")
        )

        extractor = Extractor(mock_clients, settings)
        result = await extractor.extract_memories(
            latest_messages=["USER: hello"],
            summary="",
            recent_messages=[],
        )

        assert result.semantic == []
        assert result.episodic == []


class TestClassifyAction:
    @pytest.mark.asyncio
    async def test_classify_add(self, mock_clients, settings):
        """Should classify new fact as ADD."""
        response = json.dumps(
            {"action": "ADD", "text": "User likes Python", "reason": "New fact"}
        )
        mock_clients.llm.chat.completions.create = AsyncMock(
            return_value=_mock_llm_response(response)
        )

        extractor = Extractor(mock_clients, settings)
        decision = await extractor.classify_action("User likes Python", [])

        assert decision.action == ToolAction.ADD
        assert decision.text == "User likes Python"

    @pytest.mark.asyncio
    async def test_classify_update(self, mock_clients, settings):
        """Should classify as UPDATE with memory_id."""
        response = json.dumps(
            {
                "action": "UPDATE",
                "memory_id": 5,
                "text": "User loves Python",
                "reason": "More specific",
            }
        )
        mock_clients.llm.chat.completions.create = AsyncMock(
            return_value=_mock_llm_response(response)
        )

        extractor = Extractor(mock_clients, settings)
        decision = await extractor.classify_action(
            "User loves Python",
            [{"id": 5, "text": "User likes Python", "sim": 0.9}],
        )

        assert decision.action == ToolAction.UPDATE
        assert decision.memory_id == 5

    @pytest.mark.asyncio
    async def test_classify_invalid_action_defaults_to_add(self, mock_clients, settings):
        """Invalid action string should default to ADD."""
        response = json.dumps({"action": "INVALID_ACTION", "text": "test"})
        mock_clients.llm.chat.completions.create = AsyncMock(
            return_value=_mock_llm_response(response)
        )

        extractor = Extractor(mock_clients, settings)
        decision = await extractor.classify_action("test", [])
        assert decision.action == ToolAction.ADD


class TestConsolidateMemories:
    @pytest.mark.asyncio
    async def test_consolidate(self, mock_clients, settings):
        """Should call LLM and return parsed consolidation result."""
        response = json.dumps(
            {
                "text": "User is an experienced Python developer",
                "importance": 0.9,
                "confidence": 0.95,
            }
        )
        mock_clients.llm.chat.completions.create = AsyncMock(
            return_value=_mock_llm_response(response)
        )

        extractor = Extractor(mock_clients, settings)
        result = await extractor.consolidate_memories(
            ["User likes Python", "User writes Python daily", "User knows Python well"]
        )

        assert result["text"] == "User is an experienced Python developer"
        assert result["importance"] == 0.9
