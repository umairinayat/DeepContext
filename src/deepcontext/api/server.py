"""
FastAPI REST API for DeepContext.

Provides HTTP endpoints for all memory operations.
Run with: uvicorn deepcontext.api.server:app --reload

Environment variables:
    DEEPCONTEXT_DATABASE_URL: PostgreSQL connection string
    DEEPCONTEXT_OPENAI_API_KEY: OpenAI API key
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from deepcontext.memory.engine import MemoryEngine


# ---------------------------------------------------------------------------
# Request/Response schemas
# ---------------------------------------------------------------------------


class AddRequest(BaseModel):
    messages: list[dict[str, str]]
    user_id: str
    conversation_id: str = "default"


class SearchRequest(BaseModel):
    query: str
    user_id: str
    limit: int = Field(default=10, le=100)
    tier: Optional[str] = None
    memory_type: Optional[str] = None
    include_graph: bool = True


class UpdateRequest(BaseModel):
    memory_id: int
    text: str
    user_id: str


class DeleteRequest(BaseModel):
    memory_id: int
    user_id: str


class GraphRequest(BaseModel):
    user_id: str
    entity_name: str
    depth: int = Field(default=2, le=5)


class LifecycleRequest(BaseModel):
    user_id: str


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

_engine: Optional[MemoryEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    """Initialize and cleanup MemoryEngine."""
    global _engine
    _engine = MemoryEngine()
    await _engine.init()
    yield
    if _engine:
        await _engine.close()


def get_engine() -> MemoryEngine:
    """Get the initialized engine."""
    if _engine is None:
        raise RuntimeError("Engine not initialized")
    return _engine


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DeepContext API",
    description="Hierarchical memory system for AI agents",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "version": "0.1.0"}


@app.post("/memory/add")
async def add_memory(req: AddRequest) -> dict[str, Any]:
    """
    Extract and store memories from a conversation turn.

    The LLM extracts semantic facts and episodic events,
    classifies actions (ADD/UPDATE/REPLACE/NOOP),
    and updates the knowledge graph.
    """
    engine = get_engine()
    try:
        result = await engine.add(
            messages=req.messages,
            user_id=req.user_id,
            conversation_id=req.conversation_id,
        )
        return result.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/search")
async def search_memory(req: SearchRequest) -> dict[str, Any]:
    """
    Search memories using hybrid retrieval.

    Combines vector similarity, keyword search, and knowledge graph
    traversal using Reciprocal Rank Fusion.
    """
    engine = get_engine()
    try:
        result = await engine.search(
            query=req.query,
            user_id=req.user_id,
            limit=req.limit,
            tier=req.tier,
            memory_type=req.memory_type,
            include_graph=req.include_graph,
        )
        return result.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/memory/update")
async def update_memory(req: UpdateRequest) -> dict[str, Any]:
    """Update a memory's text and re-generate its embedding."""
    engine = get_engine()
    try:
        return await engine.update(
            memory_id=req.memory_id,
            text=req.text,
            user_id=req.user_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memory/delete")
async def delete_memory(req: DeleteRequest) -> dict[str, Any]:
    """Soft-delete a memory."""
    engine = get_engine()
    try:
        return await engine.delete(
            memory_id=req.memory_id,
            user_id=req.user_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/graph/neighbors")
async def graph_neighbors(req: GraphRequest) -> list[dict[str, Any]]:
    """Get knowledge graph neighbors for an entity."""
    engine = get_engine()
    try:
        return await engine.get_entity_graph(
            user_id=req.user_id,
            entity_name=req.entity_name,
            depth=req.depth,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/lifecycle/run")
async def run_lifecycle(req: LifecycleRequest) -> dict[str, int]:
    """
    Run lifecycle maintenance: decay + consolidation + cleanup.

    Should be called periodically (e.g., daily or per-session).
    """
    engine = get_engine()
    try:
        return await engine.run_lifecycle(user_id=req.user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
