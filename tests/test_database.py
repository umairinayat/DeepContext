"""
Tests for database models and the Database manager.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from sqlalchemy import select

from deepcontext.db.database import Database
from deepcontext.db.models.memory import Memory
from deepcontext.db.models.graph import Entity, Relationship, ConversationSummary
from tests.conftest import fake_embedding


# ---------------------------------------------------------------------------
# Memory model tests
# ---------------------------------------------------------------------------


class TestMemoryModel:
    @pytest.mark.asyncio
    async def test_create_memory(self, session_factory):
        """Should create and persist a memory."""
        now = datetime.now(timezone.utc)
        async with session_factory() as session:
            mem = Memory(
                user_id="u1",
                conversation_id="c1",
                text="User likes Python",
                memory_type="semantic",
                tier="short_term",
                importance=0.8,
                confidence=0.9,
                created_at=now,
                updated_at=now,
            )
            session.add(mem)
            await session.commit()
            await session.refresh(mem)

        assert mem.id is not None
        assert mem.text == "User likes Python"
        assert mem.is_active is True

    @pytest.mark.asyncio
    async def test_embedding_set_get_sqlite(self, session_factory):
        """set_embedding/get_embedding should roundtrip correctly on SQLite."""
        emb = fake_embedding(42, dims=8)
        now = datetime.now(timezone.utc)

        async with session_factory() as session:
            mem = Memory(
                user_id="u1",
                text="test embedding",
                memory_type="semantic",
                tier="short_term",
                created_at=now,
                updated_at=now,
            )
            mem.set_embedding(emb)
            session.add(mem)
            await session.commit()
            await session.refresh(mem)

        # Read back
        async with session_factory() as session:
            loaded = await session.get(Memory, mem.id)
            assert loaded is not None
            recovered = loaded.get_embedding()
            assert recovered is not None
            assert len(recovered) == 8
            for a, b in zip(emb, recovered):
                assert abs(a - b) < 1e-6

    @pytest.mark.asyncio
    async def test_embedding_none(self, session_factory):
        """get_embedding should return None if not set."""
        now = datetime.now(timezone.utc)
        async with session_factory() as session:
            mem = Memory(
                user_id="u1",
                text="no embedding",
                memory_type="semantic",
                tier="short_term",
                created_at=now,
                updated_at=now,
            )
            session.add(mem)
            await session.commit()
            await session.refresh(mem)

        async with session_factory() as session:
            loaded = await session.get(Memory, mem.id)
            assert loaded.get_embedding() is None

    @pytest.mark.asyncio
    async def test_soft_delete(self, session_factory):
        """Soft-deleting a memory should set is_active=False."""
        now = datetime.now(timezone.utc)
        async with session_factory() as session:
            mem = Memory(
                user_id="u1",
                text="to delete",
                memory_type="semantic",
                tier="short_term",
                created_at=now,
                updated_at=now,
            )
            session.add(mem)
            await session.commit()
            await session.refresh(mem)

            mem.is_active = False
            await session.commit()

        async with session_factory() as session:
            loaded = await session.get(Memory, mem.id)
            assert loaded.is_active is False

    @pytest.mark.asyncio
    async def test_source_entities_json(self, session_factory):
        """source_entities should roundtrip as JSON list."""
        now = datetime.now(timezone.utc)
        entities = ["Python", "FastAPI"]
        async with session_factory() as session:
            mem = Memory(
                user_id="u1",
                text="test entities",
                memory_type="semantic",
                tier="short_term",
                source_entities=entities,
                created_at=now,
                updated_at=now,
            )
            session.add(mem)
            await session.commit()
            await session.refresh(mem)

        async with session_factory() as session:
            loaded = await session.get(Memory, mem.id)
            assert loaded.source_entities == ["Python", "FastAPI"]

    @pytest.mark.asyncio
    async def test_repr(self, session_factory):
        now = datetime.now(timezone.utc)
        mem = Memory(
            id=1,
            user_id="u1",
            text="A very long text that should be truncated in the repr output for readability purposes",
            memory_type="semantic",
            tier="short_term",
            created_at=now,
            updated_at=now,
        )
        r = repr(mem)
        assert "Memory" in r
        assert "..." in r


# ---------------------------------------------------------------------------
# Entity model tests
# ---------------------------------------------------------------------------


class TestEntityModel:
    @pytest.mark.asyncio
    async def test_create_entity(self, session_factory):
        async with session_factory() as session:
            entity = Entity(
                user_id="u1",
                name="Python",
                entity_type="technology",
            )
            session.add(entity)
            await session.commit()
            await session.refresh(entity)

        assert entity.id is not None
        assert entity.mention_count == 1

    @pytest.mark.asyncio
    async def test_unique_user_name_index(self, session_factory):
        """Same user+name should fail on unique index."""
        async with session_factory() as session:
            e1 = Entity(user_id="u1", name="Python", entity_type="technology")
            session.add(e1)
            await session.commit()

            e2 = Entity(user_id="u1", name="Python", entity_type="technology")
            session.add(e2)
            with pytest.raises(Exception):  # IntegrityError
                await session.commit()


# ---------------------------------------------------------------------------
# Relationship model tests
# ---------------------------------------------------------------------------


class TestRelationshipModel:
    @pytest.mark.asyncio
    async def test_create_relationship(self, session_factory):
        async with session_factory() as session:
            src = Entity(user_id="u1", name="User", entity_type="person")
            tgt = Entity(user_id="u1", name="Python", entity_type="technology")
            session.add_all([src, tgt])
            await session.commit()
            await session.refresh(src)
            await session.refresh(tgt)

            rel = Relationship(
                user_id="u1",
                source_entity_id=src.id,
                target_entity_id=tgt.id,
                relation="uses",
            )
            session.add(rel)
            await session.commit()
            await session.refresh(rel)

        assert rel.id is not None
        assert rel.strength == 1.0
        assert rel.relation == "uses"


# ---------------------------------------------------------------------------
# ConversationSummary tests
# ---------------------------------------------------------------------------


class TestConversationSummary:
    @pytest.mark.asyncio
    async def test_create_summary(self, session_factory):
        async with session_factory() as session:
            summary = ConversationSummary(
                user_id="u1",
                conversation_id="c1",
                summary_text="User discussed Python development.",
                message_count=5,
            )
            session.add(summary)
            await session.commit()
            await session.refresh(summary)

        assert summary.id is not None
        assert summary.message_count == 5


# ---------------------------------------------------------------------------
# Database manager tests
# ---------------------------------------------------------------------------


class TestDatabase:
    @pytest.mark.asyncio
    async def test_init_creates_tables(self):
        """Database.init() should create all tables."""
        db = Database("sqlite+aiosqlite:///:memory:", echo=False)
        await db.init()

        async with db.session() as session:
            # Verify tables exist by querying them
            result = await session.execute(select(Memory))
            assert result.scalars().all() == []

            result = await session.execute(select(Entity))
            assert result.scalars().all() == []

        await db.close()

    @pytest.mark.asyncio
    async def test_session_before_init_raises(self):
        """Calling session() before init() should raise."""
        db = Database("sqlite+aiosqlite:///:memory:")
        with pytest.raises(RuntimeError, match="not initialized"):
            db.session()

    @pytest.mark.asyncio
    async def test_engine_before_init_raises(self):
        db = Database("sqlite+aiosqlite:///:memory:")
        with pytest.raises(RuntimeError, match="not initialized"):
            _ = db.engine

    @pytest.mark.asyncio
    async def test_close_disposes_engine(self):
        db = Database("sqlite+aiosqlite:///:memory:")
        await db.init()
        assert db._engine is not None
        await db.close()
        assert db._engine is None
        assert db._session_factory is None
