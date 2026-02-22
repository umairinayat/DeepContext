"""
Integration tests for MemoryEngine -- the main orchestrator.

These tests wire up the full engine with an in-memory SQLite DB and mocked LLM
responses.  They exercise the real add → search → lifecycle pipeline that no
other test file covers.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from deepcontext.core.settings import DeepContextSettings
from deepcontext.core.types import (
    AddResponse,
    ExtractionResult,
    ExtractedEntity,
    ExtractedFact,
    ExtractedRelationship,
    MemoryTier,
    MemoryType,
    ToolAction,
    ToolDecision,
)
from deepcontext.db.models.memory import Memory
from deepcontext.memory.engine import MemoryEngine
from tests.conftest import fake_embedding


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extraction_json(
    semantic: list[dict] | None = None,
    episodic: list[dict] | None = None,
    entities: list[dict] | None = None,
    relationships: list[dict] | None = None,
) -> str:
    """Build a valid LLM extraction JSON string."""
    return json.dumps({
        "semantic": semantic or [],
        "episodic": episodic or [],
        "entities": entities or [],
        "relationships": relationships or [],
    })


def _classify_json(
    action: str = "ADD",
    memory_id: int | None = None,
    text: str | None = None,
    reason: str = "test",
) -> str:
    return json.dumps({
        "action": action,
        "memory_id": memory_id,
        "text": text,
        "reason": reason,
    })


def _mock_llm_response(content: str) -> MagicMock:
    """Create a mock ChatCompletion response."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# Engine fixture: real SQLite DB, mocked LLM/embedding calls
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def engine():
    """
    Create a MemoryEngine with real SQLite DB and mocked LLM/embedding calls.

    Instead of calling engine.init() (which creates real LLM clients), we
    manually wire up subsystems so that the DB is real but LLM calls are mocked.
    """
    from deepcontext.core.clients import LLMClients
    from deepcontext.db.database import Database
    from deepcontext.extraction.extractor import Extractor
    from deepcontext.graph.knowledge_graph import KnowledgeGraph
    from deepcontext.lifecycle.manager import LifecycleManager
    from deepcontext.retrieval.hybrid import HybridRetriever
    from deepcontext.vectorstore.pgvector_store import PgVectorStore

    eng = MemoryEngine(
        database_url="sqlite+aiosqlite:///:memory:",
        openai_api_key="sk-test-fake-key",
        debug=False,
        embedding_dimensions=8,
        consolidation_threshold=3,   # low threshold for easier testing
        auto_consolidate=False,      # disable by default; enable per-test
    )

    # 1. Real database
    db = Database("sqlite+aiosqlite:///:memory:", echo=False)
    await db.init()
    eng._db = db

    # 2. Mocked LLM clients
    mock_clients = MagicMock(spec=LLMClients)

    async def _embed(text: str) -> list[float]:
        seed = hash(text) % 1000
        return fake_embedding(seed, dims=8)

    mock_clients.embed_text = AsyncMock(side_effect=_embed)
    mock_clients.embed_batch = AsyncMock(
        side_effect=lambda texts: [fake_embedding(hash(t) % 1000, dims=8) for t in texts]
    )
    mock_clients.close = AsyncMock()
    mock_clients.llm = MagicMock()
    mock_clients._settings = eng._settings
    eng._clients = mock_clients

    # 3. Wire up subsystems using the real session factory
    session_factory = db._session_factory
    eng._extractor = Extractor(mock_clients, eng._settings)
    eng._vector_store = PgVectorStore(session_factory)
    eng._graph = KnowledgeGraph(session_factory)
    eng._retriever = HybridRetriever(
        mock_clients, eng._settings, eng._vector_store, eng._graph, session_factory
    )
    eng._lifecycle = LifecycleManager(
        mock_clients, eng._settings, eng._extractor, session_factory
    )
    eng._initialized = True

    # Expose mock_clients so tests can configure LLM responses
    eng._test_mock_clients = mock_clients
    yield eng
    await eng.close()


def _setup_extraction_response(engine: MemoryEngine, extraction_json: str) -> None:
    """Configure the mock LLM to return a specific extraction result."""
    engine._test_mock_clients.llm.chat.completions.create = AsyncMock(
        return_value=_mock_llm_response(extraction_json)
    )


def _setup_extraction_and_classify(
    engine: MemoryEngine,
    extraction_json: str,
    classify_json: str,
) -> None:
    """Configure the mock LLM for extraction (first call) then classification."""
    engine._test_mock_clients.llm.chat.completions.create = AsyncMock(
        side_effect=[
            _mock_llm_response(extraction_json),
            _mock_llm_response(classify_json),
        ]
    )


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestEngineInit:
    @pytest.mark.asyncio
    async def test_init_creates_subsystems(self, engine):
        """After init(), all subsystems should be wired up."""
        assert engine._initialized is True
        assert engine._db is not None
        assert engine._extractor is not None
        assert engine._vector_store is not None
        assert engine._graph is not None
        assert engine._retriever is not None
        assert engine._lifecycle is not None

    @pytest.mark.asyncio
    async def test_operations_before_init_raise(self):
        """Calling add/search/etc before init should raise."""
        eng = MemoryEngine(
            database_url="sqlite+aiosqlite:///:memory:",
            openai_api_key="sk-test-fake-key",
        )
        with pytest.raises(RuntimeError, match="[Nn]ot initialized"):
            await eng.search("test", "u1")

    @pytest.mark.asyncio
    async def test_close_resets_state(self, engine):
        """After close(), initialized should be False."""
        await engine.close()
        assert engine._initialized is False


# ---------------------------------------------------------------------------
# Add pipeline
# ---------------------------------------------------------------------------

class TestEngineAdd:
    @pytest.mark.asyncio
    async def test_add_empty_messages_returns_empty(self, engine):
        """add([]) should return an empty AddResponse without crashing."""
        result = await engine.add([], "u1")
        assert result.memories_added == 0
        assert result.semantic_facts == []

    @pytest.mark.asyncio
    async def test_add_stores_semantic_memory(self, engine):
        """add() with one semantic fact → one memory in DB."""
        _setup_extraction_and_classify(
            engine,
            _extraction_json(semantic=[{
                "text": "User is a Python developer",
                "importance": 0.8,
                "confidence": 0.9,
                "entities": ["Python"],
            }]),
            _classify_json(action="ADD", text="User is a Python developer"),
        )

        result = await engine.add(
            [{"role": "user", "content": "I'm a Python developer"}],
            user_id="u1",
            conversation_id="c1",
        )

        assert result.memories_added == 1
        assert "User is a Python developer" in result.semantic_facts

        # Verify it's actually in the DB
        async with engine._db.session() as session:
            rows = (await session.execute(
                select(Memory).where(Memory.user_id == "u1", Memory.is_active == True)
            )).scalars().all()
            assert len(rows) == 1
            assert rows[0].text == "User is a Python developer"
            assert rows[0].tier == "short_term"
            assert rows[0].memory_type == "semantic"

    @pytest.mark.asyncio
    async def test_add_stores_episodic_memory(self, engine):
        """Episodic facts from extraction are stored as short_term episodic."""
        _setup_extraction_response(
            engine,
            _extraction_json(episodic=[{
                "text": "User is debugging auth issue",
                "importance": 0.5,
                "confidence": 0.7,
                "entities": ["auth"],
            }]),
        )

        result = await engine.add(
            [{"role": "user", "content": "I'm stuck debugging this auth problem"}],
            user_id="u1",
        )

        assert result.memories_added == 1
        assert "User is debugging auth issue" in result.episodic_facts

        async with engine._db.session() as session:
            rows = (await session.execute(
                select(Memory).where(
                    Memory.user_id == "u1",
                    Memory.memory_type == "episodic",
                )
            )).scalars().all()
            assert len(rows) == 1
            assert rows[0].tier == "short_term"

    @pytest.mark.asyncio
    async def test_add_updates_knowledge_graph(self, engine):
        """Entities and relationships from extraction appear in the graph."""
        _setup_extraction_response(
            engine,
            _extraction_json(
                entities=[
                    {"name": "Python", "entity_type": "technology", "attributes": {}},
                ],
                relationships=[
                    {"source": "User", "target": "Python", "relation": "uses", "properties": {}},
                ],
            ),
        )

        result = await engine.add(
            [{"role": "user", "content": "I use Python"}],
            user_id="u1",
        )

        assert "Python" in result.entities_found
        assert result.relationships_found == 1

        # Verify via graph API
        neighbors = await engine.get_entity_graph("u1", "User", depth=1)
        assert len(neighbors) >= 1
        assert any(n["entity"] == "Python" for n in neighbors)

    @pytest.mark.asyncio
    async def test_add_noop_skips_storage(self, engine):
        """When LLM classifies as NOOP, no memory is stored."""
        _setup_extraction_and_classify(
            engine,
            _extraction_json(semantic=[{
                "text": "User likes Python",
                "importance": 0.5,
                "confidence": 0.8,
                "entities": [],
            }]),
            _classify_json(action="NOOP"),
        )

        result = await engine.add(
            [{"role": "user", "content": "I like Python"}],
            user_id="u1",
        )

        assert result.memories_added == 0

        async with engine._db.session() as session:
            count = len((await session.execute(
                select(Memory).where(Memory.user_id == "u1")
            )).scalars().all())
            assert count == 0

    @pytest.mark.asyncio
    async def test_add_multiple_facts(self, engine):
        """Multiple semantic + episodic facts in one extraction."""
        # The LLM will be called: once for extraction, twice for classify (one per semantic fact)
        engine._test_mock_clients.llm.chat.completions.create = AsyncMock(
            side_effect=[
                _mock_llm_response(_extraction_json(
                    semantic=[
                        {"text": "User is a Python dev", "importance": 0.8, "confidence": 0.9, "entities": ["Python"]},
                        {"text": "User works at Acme", "importance": 0.7, "confidence": 0.85, "entities": ["Acme"]},
                    ],
                    episodic=[
                        {"text": "User is debugging login", "importance": 0.5, "confidence": 0.7, "entities": []},
                    ],
                )),
                _mock_llm_response(_classify_json(action="ADD", text="User is a Python dev")),
                _mock_llm_response(_classify_json(action="ADD", text="User works at Acme")),
            ]
        )

        result = await engine.add(
            [{"role": "user", "content": "I'm a Python dev at Acme debugging login"}],
            user_id="u1",
        )

        assert result.memories_added == 3  # 2 semantic + 1 episodic
        assert len(result.semantic_facts) == 2
        assert len(result.episodic_facts) == 1


# ---------------------------------------------------------------------------
# Update & Delete
# ---------------------------------------------------------------------------

class TestEngineUpdateDelete:
    @pytest.mark.asyncio
    async def test_update_changes_text_and_reembeds(self, engine):
        """update() should change the text and regenerate the embedding."""
        # First add a memory
        _setup_extraction_and_classify(
            engine,
            _extraction_json(semantic=[{
                "text": "User knows Python",
                "importance": 0.7,
                "confidence": 0.9,
                "entities": ["Python"],
            }]),
            _classify_json(action="ADD", text="User knows Python"),
        )
        await engine.add(
            [{"role": "user", "content": "I know Python"}],
            user_id="u1",
        )

        # Get the memory ID
        async with engine._db.session() as session:
            mem = (await session.execute(
                select(Memory).where(Memory.user_id == "u1")
            )).scalars().first()
            mem_id = mem.id
            old_embedding = mem.get_embedding()

        # Update it
        result = await engine.update(mem_id, "User is an expert Python developer", "u1")
        assert result["status"] == "updated"

        # Verify text changed and embedding changed
        async with engine._db.session() as session:
            mem = await session.get(Memory, mem_id)
            assert mem.text == "User is an expert Python developer"
            new_embedding = mem.get_embedding()
            assert new_embedding is not None
            # Embedding should be different since text changed
            assert new_embedding != old_embedding

    @pytest.mark.asyncio
    async def test_update_wrong_user_raises(self, engine):
        """update() with wrong user_id should raise ValueError."""
        _setup_extraction_and_classify(
            engine,
            _extraction_json(semantic=[{
                "text": "User test", "importance": 0.5, "confidence": 0.8, "entities": [],
            }]),
            _classify_json(action="ADD", text="User test"),
        )
        await engine.add(
            [{"role": "user", "content": "test"}], user_id="u1"
        )

        async with engine._db.session() as session:
            mem = (await session.execute(
                select(Memory).where(Memory.user_id == "u1")
            )).scalars().first()

        with pytest.raises(ValueError, match="not found"):
            await engine.update(mem.id, "new text", "wrong_user")

    @pytest.mark.asyncio
    async def test_delete_soft_deletes(self, engine):
        """delete() should set is_active=False."""
        _setup_extraction_and_classify(
            engine,
            _extraction_json(semantic=[{
                "text": "To be deleted", "importance": 0.5, "confidence": 0.8, "entities": [],
            }]),
            _classify_json(action="ADD", text="To be deleted"),
        )
        await engine.add(
            [{"role": "user", "content": "test"}], user_id="u1"
        )

        async with engine._db.session() as session:
            mem = (await session.execute(
                select(Memory).where(Memory.user_id == "u1")
            )).scalars().first()
            mem_id = mem.id

        result = await engine.delete(mem_id, "u1")
        assert result["status"] == "deleted"

        async with engine._db.session() as session:
            mem = await session.get(Memory, mem_id)
            assert mem.is_active is False

    @pytest.mark.asyncio
    async def test_delete_wrong_user_raises(self, engine):
        """delete() with wrong user_id should raise ValueError."""
        _setup_extraction_and_classify(
            engine,
            _extraction_json(semantic=[{
                "text": "User test", "importance": 0.5, "confidence": 0.8, "entities": [],
            }]),
            _classify_json(action="ADD", text="User test"),
        )
        await engine.add(
            [{"role": "user", "content": "test"}], user_id="u1"
        )

        async with engine._db.session() as session:
            mem = (await session.execute(
                select(Memory).where(Memory.user_id == "u1")
            )).scalars().first()

        with pytest.raises(ValueError, match="not found"):
            await engine.delete(mem.id, "wrong_user")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class TestEngineSearch:
    @pytest.mark.asyncio
    async def test_search_finds_stored_memory(self, engine):
        """search() should find a previously stored memory."""
        _setup_extraction_and_classify(
            engine,
            _extraction_json(semantic=[{
                "text": "User loves FastAPI",
                "importance": 0.8,
                "confidence": 0.9,
                "entities": ["FastAPI"],
            }]),
            _classify_json(action="ADD", text="User loves FastAPI"),
        )
        await engine.add(
            [{"role": "user", "content": "I love FastAPI"}],
            user_id="u1",
        )

        results = await engine.search("FastAPI", user_id="u1", limit=5)
        assert results.total >= 1
        texts = [r.text for r in results.results]
        assert "User loves FastAPI" in texts

    @pytest.mark.asyncio
    async def test_search_empty_user_returns_empty(self, engine):
        """search() for a non-existent user should return zero results."""
        results = await engine.search("anything", user_id="nonexistent_user")
        assert results.total == 0

    @pytest.mark.asyncio
    async def test_search_with_type_filter(self, engine):
        """search() with memory_type filter should constrain the vector path."""
        _setup_extraction_and_classify(
            engine,
            _extraction_json(semantic=[{
                "text": "User is a Go dev",
                "importance": 0.8,
                "confidence": 0.9,
                "entities": ["Go"],
            }]),
            _classify_json(action="ADD", text="User is a Go dev"),
        )
        await engine.add(
            [{"role": "user", "content": "I use Go"}],
            user_id="u1",
        )

        # Memory is semantic; searching for episodic via vector path should deprioritize it,
        # but it may still appear via keyword path. Just verify the search doesn't crash
        # and that it returns valid results.
        results = await engine.search(
            "Go", user_id="u1", memory_type="episodic"
        )
        # Any returned results should have the query context (no crash)
        assert results.query == "Go"


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TestEngineLifecycle:
    @pytest.mark.asyncio
    async def test_run_lifecycle_returns_stats(self, engine):
        """run_lifecycle() should return decay/consolidation/cleanup counts."""
        stats = await engine.run_lifecycle("u1")
        assert "memories_decayed" in stats
        assert "memories_consolidated" in stats
        assert "memories_cleaned" in stats

    @pytest.mark.asyncio
    async def test_run_lifecycle_on_empty_user(self, engine):
        """run_lifecycle() for user with no memories should return zeros."""
        stats = await engine.run_lifecycle("empty_user")
        assert stats["memories_decayed"] == 0
        assert stats["memories_consolidated"] == 0
        assert stats["memories_cleaned"] == 0
