"""
Shared test fixtures for DeepContext tests.

All tests use an in-memory SQLite database to avoid side effects.
LLM/embedding calls are mocked to avoid API costs and ensure determinism.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from deepcontext.core.settings import DeepContextSettings
from deepcontext.core.types import MemoryTier, MemoryType
from deepcontext.db.models.base import Base
from deepcontext.db.models.memory import Memory


# ---------------------------------------------------------------------------
# Fake embedding helper
# ---------------------------------------------------------------------------

def fake_embedding(seed: int = 0, dims: int = 8) -> list[float]:
    """Generate a deterministic fake embedding vector (small dims for tests)."""
    import math

    return [math.sin(seed + i) * 0.5 + 0.5 for i in range(dims)]


# ---------------------------------------------------------------------------
# Database fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def async_engine():
    """Create an in-memory SQLite engine for testing."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def session_factory(async_engine):
    """Create a session factory bound to the test engine."""
    factory = async_sessionmaker(
        bind=async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    return factory


@pytest_asyncio.fixture
async def session(session_factory):
    """Create a single session for tests that need direct DB access."""
    async with session_factory() as sess:
        yield sess


# ---------------------------------------------------------------------------
# Settings fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def settings():
    """Create test settings that don't require real API keys."""
    return DeepContextSettings(
        database_url="sqlite+aiosqlite:///:memory:",
        openai_api_key="sk-test-fake-key-for-unit-tests",
        debug=False,
    )


# ---------------------------------------------------------------------------
# Mock LLM clients
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_clients(settings):
    """Create a mock LLMClients that returns deterministic results."""
    from deepcontext.core.clients import LLMClients

    clients = MagicMock(spec=LLMClients)
    clients._settings = settings

    # Mock embed_text to return a fake embedding
    async def mock_embed(text: str) -> list[float]:
        # Use hash of text as seed for deterministic but text-dependent embeddings
        seed = hash(text) % 1000
        return fake_embedding(seed)

    clients.embed_text = AsyncMock(side_effect=mock_embed)
    clients.embed_batch = AsyncMock(
        side_effect=lambda texts: [fake_embedding(hash(t) % 1000) for t in texts]
    )
    clients.close = AsyncMock()

    # Mock LLM client
    clients.llm = MagicMock()
    clients.embedding = MagicMock()

    return clients


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def sample_memories(session_factory):
    """Insert sample memories for testing retrieval."""
    now = datetime.now(timezone.utc)
    memories = []

    test_data = [
        ("User is a Python developer", "semantic", "short_term", 0.8, 0.9, ["Python"]),
        ("User loves working with FastAPI", "semantic", "short_term", 0.7, 0.85, ["FastAPI"]),
        ("User is debugging an auth issue", "episodic", "short_term", 0.5, 0.7, ["auth"]),
        ("User prefers pytest over unittest", "procedural", "long_term", 0.9, 0.95, ["pytest", "unittest"]),
        ("User works at Acme Corp", "semantic", "long_term", 0.6, 0.8, ["Acme Corp"]),
    ]

    async with session_factory() as session:
        for text, mtype, tier, importance, confidence, entities in test_data:
            emb = fake_embedding(hash(text) % 1000)
            mem = Memory(
                user_id="test_user",
                conversation_id="test_conv",
                text=text,
                memory_type=mtype,
                tier=tier,
                importance=importance,
                confidence=confidence,
                source_entities=entities,
                created_at=now,
                updated_at=now,
            )
            mem.set_embedding(emb)
            session.add(mem)
            memories.append(mem)
        await session.commit()
        # Refresh to get IDs
        for m in memories:
            await session.refresh(m)

    return memories
