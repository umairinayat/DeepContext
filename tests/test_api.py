"""
Tests for the FastAPI REST API server.

All protected endpoints require JWT auth. In tests we override the
`get_current_user` dependency to return a fake AuthenticatedUser, and
patch `_get_user_api_key` so endpoints that need an LLM API key don't
hit the database.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from deepcontext.api.auth import AuthenticatedUser, get_current_user
from deepcontext.api.server import app, get_engine
from deepcontext.core.types import (
    AddResponse,
    MemorySearchResult,
    MemoryTier,
    MemoryType,
    SearchResponse,
)


# ---------------------------------------------------------------------------
# Fake authenticated user for dependency override
# ---------------------------------------------------------------------------

_FAKE_USER = AuthenticatedUser(
    user_id=1,
    username="testuser",
    deep_context_user_id="user_1",
)


async def _fake_get_current_user() -> AuthenticatedUser:
    """Dependency override that bypasses JWT verification."""
    return _FAKE_USER


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def mock_engine():
    """Create a mock MemoryEngine for API tests."""
    engine = MagicMock()
    engine.init = AsyncMock()
    engine.close = AsyncMock()
    engine.set_user_api_key = MagicMock()
    engine.clear_user_api_key = MagicMock()
    engine.add = AsyncMock(
        return_value=AddResponse(
            semantic_facts=["User likes Python"],
            memories_added=1,
            entities_found=["Python"],
        )
    )
    engine.search = AsyncMock(
        return_value=SearchResponse(
            query="test",
            user_id="user_1",
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
        )
    )
    engine.update = AsyncMock(return_value={"memory_id": 1, "text": "updated", "status": "updated"})
    engine.delete = AsyncMock(return_value={"memory_id": 1, "status": "deleted"})
    engine.get_entity_graph = AsyncMock(
        return_value=[
            {
                "entity": "Python",
                "entity_type": "technology",
                "relation": "uses",
                "depth": 1,
                "strength": 1.0,
            }
        ]
    )
    engine.run_lifecycle = AsyncMock(
        return_value={
            "memories_decayed": 2,
            "memories_consolidated": 1,
            "memories_cleaned": 0,
        }
    )
    engine.list_memories = AsyncMock(
        return_value={
            "total": 2,
            "limit": 50,
            "offset": 0,
            "memories": [
                {
                    "id": 1,
                    "text": "User likes Python",
                    "memory_type": "semantic",
                    "tier": "short_term",
                    "importance": 0.8,
                    "confidence": 0.9,
                    "access_count": 3,
                    "source_entities": ["Python"],
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                },
                {
                    "id": 2,
                    "text": "User discussed FastAPI",
                    "memory_type": "episodic",
                    "tier": "short_term",
                    "importance": 0.5,
                    "confidence": 0.8,
                    "access_count": 1,
                    "source_entities": ["FastAPI"],
                    "created_at": "2026-01-02T00:00:00+00:00",
                    "updated_at": "2026-01-02T00:00:00+00:00",
                },
            ],
        }
    )
    engine.get_full_graph = AsyncMock(
        return_value={
            "nodes": [
                {
                    "id": "Python",
                    "type": "technology",
                    "mentionCount": 5,
                    "val": 5,
                    "attributes": {},
                },
                {
                    "id": "FastAPI",
                    "type": "technology",
                    "mentionCount": 3,
                    "val": 3,
                    "attributes": {},
                },
            ],
            "links": [
                {
                    "source": "Python",
                    "target": "FastAPI",
                    "relation": "built_with",
                    "strength": 0.9,
                },
            ],
        }
    )
    engine.list_entities = AsyncMock(
        return_value=[
            {
                "id": 1,
                "name": "Python",
                "entity_type": "technology",
                "mention_count": 5,
                "attributes": {},
                "created_at": "2026-01-01T00:00:00+00:00",
            },
            {
                "id": 2,
                "name": "FastAPI",
                "entity_type": "technology",
                "mention_count": 3,
                "attributes": {},
                "created_at": "2026-01-01T00:00:00+00:00",
            },
        ]
    )
    engine.get_stats = AsyncMock(
        return_value={
            "total_memories": 10,
            "by_tier": {"working": 0, "short_term": 7, "long_term": 3},
            "by_type": {"semantic": 6, "episodic": 3, "procedural": 1},
            "total_entities": 5,
            "total_relationships": 4,
        }
    )
    return engine


@pytest_asyncio.fixture
async def client(mock_engine):
    """Create an httpx AsyncClient with the mock engine and auth override."""
    import deepcontext.api.server as server_module

    original_engine = server_module._engine
    server_module._engine = mock_engine

    # Override auth dependency so all endpoints see a fake authenticated user
    app.dependency_overrides[get_current_user] = _fake_get_current_user

    # Patch _get_user_api_key to return a fake API key (for LLM-dependent endpoints)
    with patch.object(
        server_module, "_get_user_api_key", new=AsyncMock(return_value="sk-fake-test-key")
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c

    # Cleanup
    app.dependency_overrides.pop(get_current_user, None)
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
        resp = await client.post(
            "/memory/add",
            json={
                "messages": [{"role": "user", "content": "I like Python"}],
                "conversation_id": "c1",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["memories_added"] == 1
        assert "Python" in data["entities_found"]

    @pytest.mark.asyncio
    async def test_add_missing_messages(self, client):
        """Missing messages field -> 422."""
        resp = await client.post(
            "/memory/add",
            json={
                "conversation_id": "c1",
                # missing 'messages'
            },
        )
        assert resp.status_code == 422


class TestSearchEndpoint:
    @pytest.mark.asyncio
    async def test_search(self, client, mock_engine):
        resp = await client.post(
            "/memory/search",
            json={
                "query": "Python",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["results"][0]["text"] == "User likes Python"

    @pytest.mark.asyncio
    async def test_search_with_options(self, client, mock_engine):
        resp = await client.post(
            "/memory/search",
            json={
                "query": "Python",
                "limit": 5,
                "tier": "short_term",
                "memory_type": "semantic",
                "include_graph": False,
            },
        )
        assert resp.status_code == 200


class TestUpdateEndpoint:
    @pytest.mark.asyncio
    async def test_update(self, client, mock_engine):
        resp = await client.put(
            "/memory/update",
            json={
                "memory_id": 1,
                "text": "User loves Python",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "updated"

    @pytest.mark.asyncio
    async def test_update_not_found(self, client, mock_engine):
        mock_engine.update = AsyncMock(side_effect=ValueError("Not found"))
        resp = await client.put(
            "/memory/update",
            json={
                "memory_id": 999,
                "text": "nope",
            },
        )
        assert resp.status_code == 404


class TestDeleteEndpoint:
    @pytest.mark.asyncio
    async def test_delete(self, client, mock_engine):
        resp = await client.request(
            "DELETE",
            "/memory/delete",
            json={
                "memory_id": 1,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"


class TestGraphEndpoint:
    @pytest.mark.asyncio
    async def test_graph_neighbors(self, client, mock_engine):
        resp = await client.post(
            "/graph/neighbors",
            json={
                "entity_name": "Python",
                "depth": 2,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["entity"] == "Python"


class TestLifecycleEndpoint:
    @pytest.mark.asyncio
    async def test_lifecycle(self, client, mock_engine):
        resp = await client.post("/lifecycle/run")
        assert resp.status_code == 200
        data = resp.json()
        assert data["memories_decayed"] == 2
        assert data["memories_consolidated"] == 1


# ---------------------------------------------------------------------------
# Phase 6: API Hardening Tests
# ---------------------------------------------------------------------------


class TestAPIHardening:
    """Edge cases and error paths for the API."""

    @pytest.mark.asyncio
    async def test_add_invalid_json_body(self, client):
        """Posting malformed JSON should return 422."""
        resp = await client.post(
            "/memory/add",
            content="this is not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_add_empty_messages(self, client, mock_engine):
        """Empty messages list should still work (handled by engine)."""
        resp = await client.post(
            "/memory/add",
            json={
                "messages": [],
            },
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_search_missing_query(self, client):
        """Missing query field -> 422."""
        resp = await client.post(
            "/memory/search",
            json={
                # missing 'query'
            },
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_update_missing_text(self, client):
        """Missing text field -> 422."""
        resp = await client.put(
            "/memory/update",
            json={
                "memory_id": 1,
                # missing 'text'
            },
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_delete_not_found_returns_404(self, client, mock_engine):
        """delete() raising ValueError -> 404."""
        mock_engine.delete = AsyncMock(
            side_effect=ValueError("Memory 999 not found for user user_1")
        )
        resp = await client.request(
            "DELETE",
            "/memory/delete",
            json={
                "memory_id": 999,
            },
        )
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_add_engine_internal_error(self, client, mock_engine):
        """Engine raising an unexpected exception -> 500."""
        mock_engine.add = AsyncMock(side_effect=RuntimeError("Database connection lost"))
        resp = await client.post(
            "/memory/add",
            json={
                "messages": [{"role": "user", "content": "test"}],
            },
        )
        assert resp.status_code == 500
        assert "Database connection lost" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_graph_empty_entity_name(self, client, mock_engine):
        """Empty entity name should still return a valid response."""
        mock_engine.get_entity_graph = AsyncMock(return_value=[])
        resp = await client.post(
            "/graph/neighbors",
            json={
                "entity_name": "",
                "depth": 1,
            },
        )
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_lifecycle_nonexistent_user(self, client, mock_engine):
        """Lifecycle for user with zero results should return zeros."""
        mock_engine.run_lifecycle = AsyncMock(
            return_value={
                "memories_decayed": 0,
                "memories_consolidated": 0,
                "memories_cleaned": 0,
            }
        )
        resp = await client.post("/lifecycle/run")
        assert resp.status_code == 200
        data = resp.json()
        assert data["memories_decayed"] == 0

    @pytest.mark.asyncio
    async def test_dashboard_endpoint(self, client):
        """The root / endpoint should serve the dashboard HTML."""
        resp = await client.get("/")
        assert resp.status_code == 200
        assert "DeepContext" in resp.text


# ---------------------------------------------------------------------------
# Phase 7: New Dashboard Endpoints Tests
# ---------------------------------------------------------------------------


class TestMemoryListEndpoint:
    @pytest.mark.asyncio
    async def test_list_memories(self, client, mock_engine):
        resp = await client.post(
            "/memory/list",
            json={},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["memories"]) == 2
        assert data["memories"][0]["text"] == "User likes Python"

    @pytest.mark.asyncio
    async def test_list_memories_with_filters(self, client, mock_engine):
        resp = await client.post(
            "/memory/list",
            json={
                "tier": "short_term",
                "memory_type": "semantic",
                "limit": 10,
                "offset": 0,
            },
        )
        assert resp.status_code == 200


class TestFullGraphEndpoint:
    @pytest.mark.asyncio
    async def test_full_graph(self, client, mock_engine):
        resp = await client.post("/graph/full")
        assert resp.status_code == 200
        data = resp.json()
        assert "nodes" in data
        assert "links" in data
        assert len(data["nodes"]) == 2
        assert len(data["links"]) == 1
        assert data["nodes"][0]["id"] == "Python"
        assert data["links"][0]["relation"] == "built_with"

    @pytest.mark.asyncio
    async def test_full_graph_empty(self, client, mock_engine):
        mock_engine.get_full_graph = AsyncMock(return_value={"nodes": [], "links": []})
        resp = await client.post("/graph/full")
        assert resp.status_code == 200
        data = resp.json()
        assert data["nodes"] == []
        assert data["links"] == []


class TestEntitiesEndpoint:
    @pytest.mark.asyncio
    async def test_list_entities(self, client, mock_engine):
        resp = await client.post("/graph/entities")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]["name"] == "Python"
        assert data[1]["name"] == "FastAPI"

    @pytest.mark.asyncio
    async def test_list_entities_empty(self, client, mock_engine):
        mock_engine.list_entities = AsyncMock(return_value=[])
        resp = await client.post("/graph/entities")
        assert resp.status_code == 200
        assert resp.json() == []


class TestStatsEndpoint:
    @pytest.mark.asyncio
    async def test_stats(self, client, mock_engine):
        resp = await client.post("/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_memories"] == 10
        assert data["total_entities"] == 5
        assert data["total_relationships"] == 4
        assert data["by_tier"]["short_term"] == 7
        assert data["by_type"]["semantic"] == 6

    @pytest.mark.asyncio
    async def test_stats_empty_user(self, client, mock_engine):
        mock_engine.get_stats = AsyncMock(
            return_value={
                "total_memories": 0,
                "by_tier": {"working": 0, "short_term": 0, "long_term": 0},
                "by_type": {"semantic": 0, "episodic": 0, "procedural": 0},
                "total_entities": 0,
                "total_relationships": 0,
            }
        )
        resp = await client.post("/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_memories"] == 0
