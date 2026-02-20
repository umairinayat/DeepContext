"""
Type definitions used throughout DeepContext.

These are the canonical data types for the memory system.
All modules should use these types for consistency.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MemoryTier(str, Enum):
    """Hierarchical memory tiers, inspired by human cognition."""

    WORKING = "working"  # Current conversation buffer (ephemeral, in-memory)
    SHORT_TERM = "short_term"  # Recent episodic events (hours/days, decays)
    LONG_TERM = "long_term"  # Consolidated semantic facts (stable)


class MemoryType(str, Enum):
    """The kind of information stored in a memory."""

    SEMANTIC = "semantic"  # Stable facts: "User is a Python developer"
    EPISODIC = "episodic"  # Time-bound events: "User is debugging auth issue"
    PROCEDURAL = "procedural"  # How-to knowledge: "User prefers pytest over unittest"


class EntityType(str, Enum):
    """Types of entities in the knowledge graph."""

    PERSON = "person"
    ORGANIZATION = "organization"
    TECHNOLOGY = "technology"
    CONCEPT = "concept"
    LOCATION = "location"
    EVENT = "event"
    PREFERENCE = "preference"
    OTHER = "other"


class ToolAction(str, Enum):
    """Actions the LLM can decide for a candidate fact."""

    ADD = "ADD"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    REPLACE = "REPLACE"
    NOOP = "NOOP"


# ---------------------------------------------------------------------------
# Memory Data Models
# ---------------------------------------------------------------------------


class ExtractedFact(BaseModel):
    """A fact extracted from a conversation by the LLM."""

    text: str = Field(..., description="The fact statement")
    memory_type: MemoryType = Field(default=MemoryType.SEMANTIC)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    entities: list[str] = Field(default_factory=list, description="Entity names mentioned")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class ExtractionResult(BaseModel):
    """Result of LLM fact extraction from a conversation turn."""

    semantic: list[ExtractedFact] = Field(default_factory=list)
    episodic: list[ExtractedFact] = Field(default_factory=list)
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)


class ExtractedEntity(BaseModel):
    """An entity extracted from conversation."""

    name: str
    entity_type: EntityType = Field(default=EntityType.OTHER)
    attributes: dict[str, Any] = Field(default_factory=dict)


class ExtractedRelationship(BaseModel):
    """A relationship between two entities."""

    source: str = Field(..., description="Source entity name")
    target: str = Field(..., description="Target entity name")
    relation: str = Field(..., description="Relationship type, e.g. 'works_at', 'prefers'")
    properties: dict[str, Any] = Field(default_factory=dict)


# Need to rebuild ExtractionResult since it references forward-declared types
ExtractionResult.model_rebuild()


class ToolDecision(BaseModel):
    """LLM's decision on what to do with a candidate fact."""

    action: ToolAction
    memory_id: Optional[int] = Field(default=None, description="Target memory ID for UPDATE/DELETE/REPLACE")
    text: Optional[str] = Field(default=None, description="Text to store for ADD/UPDATE")
    reason: Optional[str] = Field(default=None, description="Why this action was chosen")


class MemorySearchResult(BaseModel):
    """A single memory search result."""

    memory_id: int
    text: str
    memory_type: MemoryType
    tier: MemoryTier
    score: float = Field(ge=0.0)
    importance: float = Field(default=0.5)
    confidence: float = Field(default=0.8)
    created_at: datetime
    connections: list[int] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Full search response."""

    query: str
    user_id: str
    total: int
    results: list[MemorySearchResult]


class AddResponse(BaseModel):
    """Response from adding memories."""

    semantic_facts: list[str] = Field(default_factory=list)
    episodic_facts: list[str] = Field(default_factory=list)
    entities_found: list[str] = Field(default_factory=list)
    relationships_found: int = 0
    memories_added: int = 0
    memories_updated: int = 0
    memories_replaced: int = 0
