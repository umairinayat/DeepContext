"""
Tests for the FastAPI REST API server.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from deepcontext.api.server import app, get_engine
from deepcontext.core.types import AddResponse, MemorySearchResult, MemoryTier, MemoryType, SearchResponse


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def mock_engine():
    """Create a mock MemoryEngine for API tests."""
    engine = MagicMock()
    engine.init = AsyncMock()
    engine.close = AsyncMock()
    engine.add = AsyncMock(return_value=AddResponse(
        semantic_facts=["User likes Python"],
        memories_added=1,
        entities_found=["Python"],
    ))
    engine.search = AsyncMock(return_value=SearchResponse(
        query="test",
        user_id="u1",
        total=1,
        results=[
            MemorySearchResult(
                memory_id=1,
                text="User likes Python",
                memory_type=MemoryType.SEMANTIC,
                tier=MemoryTier.SHORT_TERM,
                score=0.9,
                created_at="2026-01-01T00:00:00Z",
            )
        ],
    ))
    engine.update = AsyncMock(return_value={"memory_id": 1, "text": "updated", "status": "updated"})
    engine.delete = AsyncMock(return_value={"memory_id": 1, "status": "deleted"})
    engine.get_entity_graph = AsyncMock(return_value=[
        {"entity": "Python", "entity_type": "technology", "relation": "uses", "depth": 1, "strength": 1.0}
    ])
    engine.run_lifecycle = AsyncMock(return_value={
        "memories_decayed": 2,
        "memories_consolidated": 1,
        "memories_cleaned": 0,
    })
    return engine


@pytest_asyncio.fixture
async def client(mock_engine):
    """Create an httpx AsyncClient with the mock engine injected."""
    import deepcontext.api.server as server_module
    original_engine = server_module._engine
    server_module._engine = mock_engine

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    server_module._engine = original_engine


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestAddEndpoint:
    @pytest.mark.asyncio
    async def test_add_memory(self, client, mock_engine):
        resp = await client.post("/memory/add", json={
            "messages": [{"role": "user", "content": "I like Python"}],
            "user_id": "u1",
            "conversation_id": "c1",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["memories_added"] == 1
        assert "Python" in data["entities_found"]

    @pytest.mark.asyncio
    async def test_add_missing_fields(self, client):
        resp = await client.post("/memory/add", json={
            "messages": [{"role": "user", "content": "hi"}],
            # missing user_id
        })
        assert resp.status_code == 422  # Validation error


class TestSearchEndpoint:
    @pytest.mark.asyncio
    async def test_search(self, client, mock_engine):
        resp = await client.post("/memory/search", json={
            "query": "Python",
            "user_id": "u1",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["results"][0]["text"] == "User likes Python"

    @pytest.mark.asyncio
    async def test_search_with_options(self, client, mock_engine):
        resp = await client.post("/memory/search", json={
            "query": "Python",
            "user_id": "u1",
            "limit": 5,
            "tier": "short_term",
            "memory_type": "semantic",
            "include_graph": False,
        })
        assert resp.status_code == 200


class TestUpdateEndpoint:
    @pytest.mark.asyncio
    async def test_update(self, client, mock_engine):
        resp = await client.put("/memory/update", json={
            "memory_id": 1,
            "text": "User loves Python",
            "user_id": "u1",
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "updated"

    @pytest.mark.asyncio
    async def test_update_not_found(self, client, mock_engine):
        mock_engine.update = AsyncMock(side_effect=ValueError("Not found"))
        resp = await client.put("/memory/update", json={
            "memory_id": 999,
            "text": "nope",
            "user_id": "u1",
        })
        assert resp.status_code == 404


class TestDeleteEndpoint:
    @pytest.mark.asyncio
    async def test_delete(self, client, mock_engine):
        resp = await client.request("DELETE", "/memory/delete", json={
            "memory_id": 1,
            "user_id": "u1",
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"


class TestGraphEndpoint:
    @pytest.mark.asyncio
    async def test_graph_neighbors(self, client, mock_engine):
        resp = await client.post("/graph/neighbors", json={
            "user_id": "u1",
            "entity_name": "Python",
            "depth": 2,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["entity"] == "Python"


class TestLifecycleEndpoint:
    @pytest.mark.asyncio
    async def test_lifecycle(self, client, mock_engine):
        resp = await client.post("/lifecycle/run", json={
            "user_id": "u1",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["memories_decayed"] == 2
        assert data["memories_consolidated"] == 1
