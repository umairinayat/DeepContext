"""
Tests for the knowledge graph module.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
import pytest_asyncio
from sqlalchemy import select

from deepcontext.core.types import (
    EntityType,
    ExtractedEntity,
    ExtractedRelationship,
)
from deepcontext.db.models.graph import Entity, Relationship
from deepcontext.graph.knowledge_graph import KnowledgeGraph


class TestKnowledgeGraph:
    @pytest.mark.asyncio
    async def test_upsert_entity_creates_new(self, session_factory):
        """upsert_entity should create a new entity if it doesn't exist."""
        graph = KnowledgeGraph(session_factory)
        entity = ExtractedEntity(
            name="Python",
            entity_type=EntityType.TECHNOLOGY,
            attributes={"paradigm": "multi-paradigm"},
        )

        result = await graph.upsert_entity("u1", entity)

        assert result.id is not None
        assert result.name == "Python"
        assert result.entity_type == "technology"
        assert result.mention_count == 1
        assert result.attributes.get("paradigm") == "multi-paradigm"

    @pytest.mark.asyncio
    async def test_upsert_entity_increments_count(self, session_factory):
        """Upserting an existing entity should increment mention_count."""
        graph = KnowledgeGraph(session_factory)
        entity = ExtractedEntity(name="Python", entity_type=EntityType.TECHNOLOGY)

        await graph.upsert_entity("u1", entity)
        result = await graph.upsert_entity("u1", entity)

        assert result.mention_count == 2

    @pytest.mark.asyncio
    async def test_upsert_entity_merges_attributes(self, session_factory):
        """Upserting should merge new attributes with existing ones."""
        graph = KnowledgeGraph(session_factory)

        e1 = ExtractedEntity(
            name="Python",
            entity_type=EntityType.TECHNOLOGY,
            attributes={"version": "3.13"},
        )
        await graph.upsert_entity("u1", e1)

        e2 = ExtractedEntity(
            name="Python",
            entity_type=EntityType.TECHNOLOGY,
            attributes={"paradigm": "multi-paradigm"},
        )
        result = await graph.upsert_entity("u1", e2)

        assert result.attributes.get("version") == "3.13"
        assert result.attributes.get("paradigm") == "multi-paradigm"

    @pytest.mark.asyncio
    async def test_upsert_different_users(self, session_factory):
        """Same entity name for different users should be separate."""
        graph = KnowledgeGraph(session_factory)
        entity = ExtractedEntity(name="Python", entity_type=EntityType.TECHNOLOGY)

        r1 = await graph.upsert_entity("u1", entity)
        r2 = await graph.upsert_entity("u2", entity)

        assert r1.id != r2.id
        assert r1.mention_count == 1
        assert r2.mention_count == 1

    @pytest.mark.asyncio
    async def test_add_relationship(self, session_factory):
        """add_relationship should create entities and relationship."""
        graph = KnowledgeGraph(session_factory)
        rel = ExtractedRelationship(
            source="User",
            target="Python",
            relation="uses",
        )

        result = await graph.add_relationship("u1", rel)

        assert result is not None
        assert result.relation == "uses"
        assert result.strength == 1.0

    @pytest.mark.asyncio
    async def test_add_relationship_strengthens_existing(self, session_factory):
        """Adding the same relationship again should increase strength."""
        graph = KnowledgeGraph(session_factory)
        rel = ExtractedRelationship(source="User", target="Python", relation="uses")

        await graph.add_relationship("u1", rel)
        result = await graph.add_relationship("u1", rel)

        assert result.strength == 1.1  # 1.0 + 0.1

    @pytest.mark.asyncio
    async def test_add_relationship_empty_source_returns_none(self, session_factory):
        """Relationship with empty source should return None."""
        graph = KnowledgeGraph(session_factory)
        rel = ExtractedRelationship(source="", target="Python", relation="uses")
        result = await graph.add_relationship("u1", rel)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_neighbors_empty(self, session_factory):
        """get_neighbors for nonexistent entity should return empty."""
        graph = KnowledgeGraph(session_factory)
        result = await graph.get_neighbors("u1", "nonexistent")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_neighbors(self, session_factory):
        """get_neighbors should return connected entities."""
        graph = KnowledgeGraph(session_factory)

        # Build a small graph: User -> uses -> Python -> is_a -> Language
        await graph.add_relationship(
            "u1", ExtractedRelationship(source="User", target="Python", relation="uses")
        )
        await graph.add_relationship(
            "u1", ExtractedRelationship(source="Python", target="Language", relation="is_a")
        )

        # Depth 1: should find Python
        neighbors = await graph.get_neighbors("u1", "User", max_depth=1)
        assert len(neighbors) == 1
        assert neighbors[0]["entity"] == "Python"
        assert neighbors[0]["relation"] == "uses"
        assert neighbors[0]["depth"] == 1

    @pytest.mark.asyncio
    async def test_get_neighbors_depth_2(self, session_factory):
        """get_neighbors with depth=2 should traverse two hops."""
        graph = KnowledgeGraph(session_factory)

        await graph.add_relationship(
            "u1", ExtractedRelationship(source="User", target="Python", relation="uses")
        )
        await graph.add_relationship(
            "u1", ExtractedRelationship(source="Python", target="Language", relation="is_a")
        )

        neighbors = await graph.get_neighbors("u1", "User", max_depth=2)
        assert len(neighbors) == 2
        names = {n["entity"] for n in neighbors}
        assert "Python" in names
        assert "Language" in names

    @pytest.mark.asyncio
    async def test_get_entity_context(self, session_factory):
        """get_entity_context should return all relationships for given entities."""
        graph = KnowledgeGraph(session_factory)

        await graph.add_relationship(
            "u1", ExtractedRelationship(source="User", target="Python", relation="uses")
        )
        await graph.add_relationship(
            "u1", ExtractedRelationship(source="User", target="FastAPI", relation="prefers")
        )

        context = await graph.get_entity_context("u1", ["User"])
        assert len(context) == 2
        relations = {c["relation"] for c in context}
        assert "uses" in relations
        assert "prefers" in relations

    @pytest.mark.asyncio
    async def test_get_entity_context_empty(self, session_factory):
        """get_entity_context with empty entity list should return empty."""
        graph = KnowledgeGraph(session_factory)
        result = await graph.get_entity_context("u1", [])
        assert result == []
