"""initial schema - memories entities relationships summaries

Revision ID: 001
Revises: None
Create Date: 2026-02-21
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- Enable pgvector extension (PostgreSQL only) ---
    # This is a no-op if already enabled; safe to run multiple times.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # --- memories table ---
    op.create_table(
        "memories",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.String(128), nullable=False, index=True),
        sa.Column("conversation_id", sa.String(128), nullable=True, index=True),
        # Content
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("memory_type", sa.String(32), nullable=False, server_default="semantic"),
        sa.Column("tier", sa.String(32), nullable=False, server_default="short_term"),
        # Vector embedding (pgvector)
        sa.Column("embedding", sa.Column("embedding", sa.Text(), nullable=True).type
                   if False else sa.Text(), nullable=True),
        # Scoring
        sa.Column("importance", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0.8"),
        sa.Column("access_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_accessed_at", sa.DateTime(timezone=True), nullable=True),
        # Lifecycle
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("decay_rate", sa.Float(), nullable=False, server_default="0.05"),
        sa.Column("consolidated_from", sa.Text(), nullable=True),
        # Metadata (JSONB on PG)
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("connections", sa.JSON(), nullable=True),
        sa.Column("source_entities", sa.JSON(), nullable=True),
        # Timestamps
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=True),
    )

    # Replace the plain Text embedding column with a pgvector VECTOR column
    op.execute("ALTER TABLE memories DROP COLUMN embedding")
    op.execute("ALTER TABLE memories ADD COLUMN embedding vector(1536)")

    # Composite indexes
    op.create_index("ix_memories_user_tier", "memories", ["user_id", "tier"])
    op.create_index("ix_memories_user_type", "memories", ["user_id", "memory_type"])
    op.create_index("ix_memories_user_active", "memories", ["user_id", "is_active"])
    op.create_index("ix_memories_created", "memories", ["created_at"])

    # pgvector index for fast cosine similarity (IVFFlat)
    op.execute(
        "CREATE INDEX ix_memories_embedding ON memories "
        "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
    )

    # Full-text search index
    op.execute(
        "CREATE INDEX ix_memories_text_search ON memories "
        "USING gin (to_tsvector('english', text))"
    )

    # --- entities table ---
    op.create_table(
        "entities",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.String(128), nullable=False, index=True),
        sa.Column("name", sa.String(256), nullable=False),
        sa.Column("entity_type", sa.String(64), nullable=False, server_default="other"),
        sa.Column("attributes", sa.JSON(), nullable=True),
        sa.Column("mention_count", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("CURRENT_TIMESTAMP")),
    )
    op.create_index("ix_entities_user_name", "entities", ["user_id", "name"], unique=True)
    op.create_index("ix_entities_user_type", "entities", ["user_id", "entity_type"])

    # --- relationships table ---
    op.create_table(
        "relationships",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.String(128), nullable=False, index=True),
        sa.Column("source_entity_id", sa.Integer(),
                  sa.ForeignKey("entities.id", ondelete="CASCADE"), nullable=False),
        sa.Column("target_entity_id", sa.Integer(),
                  sa.ForeignKey("entities.id", ondelete="CASCADE"), nullable=False),
        sa.Column("relation", sa.String(128), nullable=False),
        sa.Column("strength", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("properties", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("CURRENT_TIMESTAMP")),
    )
    op.create_index("ix_rel_source", "relationships", ["source_entity_id"])
    op.create_index("ix_rel_target", "relationships", ["target_entity_id"])
    op.create_index("ix_rel_user_relation", "relationships", ["user_id", "relation"])

    # --- conversation_summaries table ---
    op.create_table(
        "conversation_summaries",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.String(128), nullable=False, index=True),
        sa.Column("conversation_id", sa.String(128), nullable=False, index=True),
        sa.Column("summary_text", sa.Text(), nullable=False),
        sa.Column("message_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("CURRENT_TIMESTAMP")),
    )
    op.create_index("ix_summary_user_conv", "conversation_summaries",
                    ["user_id", "conversation_id"], unique=True)


def downgrade() -> None:
    op.drop_table("conversation_summaries")
    op.drop_table("relationships")
    op.drop_table("entities")
    op.drop_table("memories")
    op.execute("DROP EXTENSION IF EXISTS vector")
