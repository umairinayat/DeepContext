"""
Entity and Relationship models for the knowledge graph.

These live alongside memories, avoiding the need for a separate graph database.
Supports both PostgreSQL (JSONB) and SQLite (JSON) backends.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text, text as sa_text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from deepcontext.db.models.base import Base

# Detect database backend
_DB_URL = os.environ.get("DEEPCONTEXT_DATABASE_URL", "")
_USE_PG = "postgresql" in _DB_URL

if _USE_PG:
    from sqlalchemy.dialects.postgresql import JSONB as _JSONB_TYPE
else:
    _JSONB_TYPE = JSON  # type: ignore[assignment,misc]


class Entity(Base):
    """
    A named entity extracted from conversations.

    Entities are the nodes in our knowledge graph.
    Examples: "Python", "FastAPI", "John's company", "machine learning"
    """

    __tablename__ = "entities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    entity_type: Mapped[str] = mapped_column(
        String(64), nullable=False, default="other"
    )  # person, technology, concept, etc.
    attributes: Mapped[Optional[dict[str, Any]]] = mapped_column(
        _JSONB_TYPE, nullable=True, default=dict
    )
    mention_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
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

    __table_args__ = (
        Index("ix_entities_user_name", "user_id", "name", unique=True),
        Index("ix_entities_user_type", "user_id", "entity_type"),
    )

    def __repr__(self) -> str:
        return f"<Entity(id={self.id}, name='{self.name}', type={self.entity_type})>"


class Relationship(Base):
    """
    A directed relationship between two entities.

    Relationships are the edges in our knowledge graph.
    Examples: ("User", "works_with", "Python"), ("User", "prefers", "FastAPI")
    """

    __tablename__ = "relationships"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    source_entity_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("entities.id", ondelete="CASCADE"), nullable=False
    )
    target_entity_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("entities.id", ondelete="CASCADE"), nullable=False
    )
    relation: Mapped[str] = mapped_column(
        String(128), nullable=False
    )  # "works_with", "prefers", "is_a", etc.
    strength: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    properties: Mapped[Optional[dict[str, Any]]] = mapped_column(
        _JSONB_TYPE, nullable=True, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=sa_text("CURRENT_TIMESTAMP"),
    )

    __table_args__ = (
        Index("ix_rel_source", "source_entity_id"),
        Index("ix_rel_target", "target_entity_id"),
        Index("ix_rel_user_relation", "user_id", "relation"),
    )

    def __repr__(self) -> str:
        return (
            f"<Relationship(id={self.id}, "
            f"source={self.source_entity_id} --[{self.relation}]--> "
            f"target={self.target_entity_id})>"
        )


class ConversationSummary(Base):
    """Rolling summary of a conversation, updated periodically."""

    __tablename__ = "conversation_summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    conversation_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    summary_text: Mapped[str] = mapped_column(Text, nullable=False)
    message_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
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

    __table_args__ = (
        Index("ix_summary_user_conv", "user_id", "conversation_id", unique=True),
    )
