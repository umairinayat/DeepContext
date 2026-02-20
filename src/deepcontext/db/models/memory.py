"""
Memory ORM model with support for both pgvector (PostgreSQL) and SQLite.

This is the core table - stores all memory tiers (working, short-term, long-term)
with vector embeddings. On PostgreSQL, embeddings use native VECTOR columns for
efficient similarity search. On SQLite, embeddings are stored as JSON text
and similarity is computed in Python.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    text as sa_text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from deepcontext.db.models.base import Base

# Detect database backend to choose the right column types
_DB_URL = os.environ.get("DEEPCONTEXT_DATABASE_URL", "")
_USE_PG = "postgresql" in _DB_URL

if _USE_PG:
    from pgvector.sqlalchemy import Vector
    from sqlalchemy.dialects.postgresql import JSONB as _JSONB_TYPE
else:
    _JSONB_TYPE = JSON  # type: ignore[assignment,misc]


class Memory(Base):
    """
    Core memory table.

    Stores all types of memories across all tiers:
    - working: ephemeral, current conversation
    - short_term: recent episodic events, decays
    - long_term: consolidated semantic facts, stable

    When running on PostgreSQL with pgvector, the embedding column uses the
    native VECTOR type. On SQLite, embeddings are stored as JSON arrays.
    """

    __tablename__ = "memories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    conversation_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)

    # Content
    text: Mapped[str] = mapped_column(Text, nullable=False)
    memory_type: Mapped[str] = mapped_column(
        String(32), nullable=False, default="semantic"
    )  # semantic, episodic, procedural
    tier: Mapped[str] = mapped_column(
        String(32), nullable=False, default="short_term"
    )  # working, short_term, long_term

    # Vector embedding
    # PostgreSQL: native pgvector VECTOR column
    # SQLite: JSON text column storing the embedding array
    if _USE_PG:
        embedding = mapped_column(Vector(1536), nullable=True)
    else:
        embedding = mapped_column(Text, nullable=True)  # JSON-encoded list[float]

    # Scoring
    importance: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.8)
    access_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_accessed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Lifecycle
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    decay_rate: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.05
    )  # Ebbinghaus decay factor
    consolidated_from: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # JSON list of source memory IDs

    # Metadata
    metadata_: Mapped[Optional[dict[str, Any]]] = mapped_column(
        "metadata", _JSONB_TYPE, nullable=True, default=dict
    )
    connections: Mapped[Optional[dict[str, Any]]] = mapped_column(
        _JSONB_TYPE, nullable=True, default=dict
    )  # {"memory_ids": [...], "scores": {...}}
    source_entities: Mapped[Optional[list[str]]] = mapped_column(
        _JSONB_TYPE, nullable=True, default=list
    )  # Entity names referenced

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=sa_text("CURRENT_TIMESTAMP"),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=sa_text("CURRENT_TIMESTAMP"),
    )
    occurred_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )  # For episodic memories

    # Indexes for fast retrieval
    __table_args__ = (
        Index("ix_memories_user_tier", "user_id", "tier"),
        Index("ix_memories_user_type", "user_id", "memory_type"),
        Index("ix_memories_user_active", "user_id", "is_active"),
        Index("ix_memories_created", "created_at"),
    )

    def set_embedding(self, embedding: list[float] | None) -> None:
        """Set the embedding, encoding as JSON for SQLite."""
        if _USE_PG:
            self.embedding = embedding
        else:
            self.embedding = json.dumps(embedding) if embedding is not None else None

    def get_embedding(self) -> list[float] | None:
        """Get the embedding, decoding from JSON for SQLite."""
        if self.embedding is None:
            return None
        if _USE_PG:
            return list(self.embedding)
        if isinstance(self.embedding, str):
            return json.loads(self.embedding)
        return list(self.embedding)

    def __repr__(self) -> str:
        return (
            f"<Memory(id={self.id}, tier={self.tier}, type={self.memory_type}, "
            f"text='{self.text[:50]}...')>"
        )
