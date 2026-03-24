"""
FastAPI REST API for DeepContext.

Provides HTTP endpoints for all memory operations with JWT authentication.
Run with: uvicorn deepcontext.api.server:app --reload

Environment variables:
    DEEPCONTEXT_DATABASE_URL: PostgreSQL connection string
    DEEPCONTEXT_JWT_SECRET: Secret key for JWT tokens (REQUIRED in production)
    DEEPCONTEXT_OPENAI_API_KEY: OpenAI API key (optional server-level fallback)
"""

from __future__ import annotations

import pathlib
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import select

from deepcontext.api.auth import (
    AuthenticatedUser,
    create_access_token,
    decrypt_api_key,
    encrypt_api_key,
    get_current_user,
    hash_password,
    verify_password,
)
from deepcontext.db.database import Database
from deepcontext.db.models.user import User
from deepcontext.memory.engine import MemoryEngine


# ---------------------------------------------------------------------------
# Request/Response schemas
# ---------------------------------------------------------------------------


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=128)
    password: str = Field(min_length=6, max_length=256)
    email: Optional[str] = None


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    username: str


class SetApiKeyRequest(BaseModel):
    api_key: str = Field(min_length=1, description="OpenRouter API key")


class UserProfileResponse(BaseModel):
    user_id: str
    username: str
    email: Optional[str] = None
    has_api_key: bool
    created_at: Optional[str] = None


class AddRequest(BaseModel):
    messages: list[dict[str, str]]
    conversation_id: str = "default"


class SearchRequest(BaseModel):
    query: str
    limit: int = Field(default=10, le=100)
    tier: Optional[str] = None
    memory_type: Optional[str] = None
    include_graph: bool = True


class UpdateRequest(BaseModel):
    memory_id: int
    text: str


class DeleteRequest(BaseModel):
    memory_id: int


class GraphRequest(BaseModel):
    entity_name: str
    depth: int = Field(default=2, le=5)


class MemoryListRequest(BaseModel):
    tier: Optional[str] = None
    memory_type: Optional[str] = None
    limit: int = Field(default=50, le=200)
    offset: int = Field(default=0, ge=0)


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

_engine: Optional[MemoryEngine] = None
_db: Optional[Database] = None


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    """Initialize and cleanup MemoryEngine and auth database."""
    global _engine, _db

    # Initialize MemoryEngine (for memory operations)
    _engine = MemoryEngine()
    await _engine.init()

    # Get a reference to the database for auth operations
    # We reuse the engine's database to avoid a second connection pool
    _db = _engine._db

    # Create users table if it doesn't exist
    if _db and _db._engine:
        from deepcontext.db.models.user import User  # noqa: F811

        async with _db._engine.begin() as conn:
            await conn.run_sync(User.metadata.create_all)

    yield

    if _engine:
        await _engine.close()


def get_engine() -> MemoryEngine:
    """Get the initialized engine."""
    if _engine is None:
        raise RuntimeError("Engine not initialized")
    return _engine


def get_db() -> Database:
    """Get the database for auth operations."""
    if _db is None:
        raise RuntimeError("Database not initialized")
    return _db


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DeepContext API",
    description="Hierarchical memory system for AI agents with authentication",
    version="0.2.0",
    lifespan=lifespan,
)

# CORS -- allow frontend origins (GitHub Pages + localhost dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static dashboard
_static_dir = pathlib.Path(__file__).parent / "static"
if _static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/", include_in_schema=False)
async def dashboard():
    """Serve the testing dashboard."""
    index = _static_dir / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "Dashboard not found. Place index.html in api/static/"}


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint (unauthenticated)."""
    return {"status": "ok", "version": "0.2.0"}


# ===========================================================================
# AUTH ENDPOINTS (unauthenticated)
# ===========================================================================


@app.post("/auth/register", response_model=TokenResponse)
async def register(req: RegisterRequest) -> TokenResponse:
    """Register a new user account."""
    db = get_db()
    async with db.session() as session:
        # Check if username exists
        existing = await session.execute(select(User).where(User.username == req.username))
        if existing.scalar_one_or_none():
            raise HTTPException(status_code=409, detail="Username already taken")

        # Check if email exists (if provided)
        if req.email:
            existing_email = await session.execute(select(User).where(User.email == req.email))
            if existing_email.scalar_one_or_none():
                raise HTTPException(status_code=409, detail="Email already registered")

        # Create user
        user = User(
            username=req.username,
            email=req.email,
            hashed_password=hash_password(req.password),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)

        # Generate JWT token
        token = create_access_token(user.id, user.username)
        return TokenResponse(
            access_token=token,
            user_id=user.deep_context_user_id,
            username=user.username,
        )


@app.post("/auth/login", response_model=TokenResponse)
async def login(req: LoginRequest) -> TokenResponse:
    """Authenticate and get a JWT access token."""
    db = get_db()
    async with db.session() as session:
        result = await session.execute(select(User).where(User.username == req.username))
        user = result.scalar_one_or_none()

        if not user or not verify_password(req.password, user.hashed_password):
            raise HTTPException(status_code=401, detail="Invalid username or password")

        if not user.is_active:
            raise HTTPException(status_code=403, detail="Account is disabled")

        token = create_access_token(user.id, user.username)
        return TokenResponse(
            access_token=token,
            user_id=user.deep_context_user_id,
            username=user.username,
        )


# ===========================================================================
# AUTH ENDPOINTS (authenticated)
# ===========================================================================


@app.get("/auth/me", response_model=UserProfileResponse)
async def get_profile(
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> UserProfileResponse:
    """Get the current user's profile."""
    db = get_db()
    async with db.session() as session:
        user = await session.get(User, current_user.user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        return UserProfileResponse(
            user_id=user.deep_context_user_id,
            username=user.username,
            email=user.email,
            has_api_key=user.encrypted_api_key is not None,
            created_at=user.created_at.isoformat() if user.created_at else None,
        )


@app.post("/auth/apikey")
async def set_api_key(
    req: SetApiKeyRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> dict[str, Any]:
    """Store the user's OpenRouter API key (encrypted at rest)."""
    db = get_db()
    async with db.session() as session:
        user = await session.get(User, current_user.user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        user.encrypted_api_key = encrypt_api_key(req.api_key)
        user.updated_at = datetime.now(timezone.utc)
        await session.commit()

        return {"status": "ok", "message": "API key saved successfully"}


@app.delete("/auth/apikey")
async def delete_api_key(
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> dict[str, Any]:
    """Remove the user's stored API key."""
    db = get_db()
    async with db.session() as session:
        user = await session.get(User, current_user.user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        user.encrypted_api_key = None
        user.updated_at = datetime.now(timezone.utc)
        await session.commit()

        return {"status": "ok", "message": "API key removed"}


# ===========================================================================
# Helper: get user's API key for LLM operations
# ===========================================================================


async def _get_user_api_key(user_id: int) -> Optional[str]:
    """Retrieve and decrypt a user's OpenRouter API key."""
    db = get_db()
    async with db.session() as session:
        user = await session.get(User, user_id)
        if user and user.encrypted_api_key:
            return decrypt_api_key(user.encrypted_api_key)
    return None


# ===========================================================================
# MEMORY ENDPOINTS (authenticated -- user_id derived from JWT)
# ===========================================================================


@app.post("/memory/add")
async def add_memory(
    req: AddRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> dict[str, Any]:
    """Extract and store memories from a conversation turn."""
    engine = get_engine()

    # Get the user's API key for LLM calls
    api_key = await _get_user_api_key(current_user.user_id)
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="No API key configured. Set your OpenRouter API key in Settings.",
        )

    try:
        # Override the engine's LLM clients with user's API key
        engine.set_user_api_key(api_key)
        result = await engine.add(
            messages=req.messages,
            user_id=current_user.deep_context_user_id,
            conversation_id=req.conversation_id,
        )
        return result.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        engine.clear_user_api_key()


@app.post("/memory/search")
async def search_memory(
    req: SearchRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> dict[str, Any]:
    """Search memories using hybrid retrieval."""
    engine = get_engine()

    # Search requires embeddings, so we need the user's API key
    api_key = await _get_user_api_key(current_user.user_id)
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="No API key configured. Set your OpenRouter API key in Settings.",
        )

    try:
        engine.set_user_api_key(api_key)
        result = await engine.search(
            query=req.query,
            user_id=current_user.deep_context_user_id,
            limit=req.limit,
            tier=req.tier,
            memory_type=req.memory_type,
            include_graph=req.include_graph,
        )
        return result.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        engine.clear_user_api_key()


@app.put("/memory/update")
async def update_memory(
    req: UpdateRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> dict[str, Any]:
    """Update a memory's text and re-generate its embedding."""
    engine = get_engine()
    api_key = await _get_user_api_key(current_user.user_id)
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="No API key configured. Set your OpenRouter API key in Settings.",
        )

    try:
        engine.set_user_api_key(api_key)
        return await engine.update(
            memory_id=req.memory_id,
            text=req.text,
            user_id=current_user.deep_context_user_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        engine.clear_user_api_key()


@app.delete("/memory/delete")
async def delete_memory(
    req: DeleteRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> dict[str, Any]:
    """Soft-delete a memory."""
    engine = get_engine()
    try:
        return await engine.delete(
            memory_id=req.memory_id,
            user_id=current_user.deep_context_user_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/list")
async def list_memories(
    req: MemoryListRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> dict[str, Any]:
    """List all memories for the current user with optional filtering and pagination."""
    engine = get_engine()
    try:
        return await engine.list_memories(
            user_id=current_user.deep_context_user_id,
            tier=req.tier,
            memory_type=req.memory_type,
            limit=req.limit,
            offset=req.offset,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/graph/neighbors")
async def graph_neighbors(
    req: GraphRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> list[dict[str, Any]]:
    """Get knowledge graph neighbors for an entity."""
    engine = get_engine()
    try:
        return await engine.get_entity_graph(
            user_id=current_user.deep_context_user_id,
            entity_name=req.entity_name,
            depth=req.depth,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/graph/full")
async def full_graph(
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> dict[str, Any]:
    """Get the complete knowledge graph for the current user."""
    engine = get_engine()
    try:
        return await engine.get_full_graph(user_id=current_user.deep_context_user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/graph/entities")
async def list_entities(
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> list[dict[str, Any]]:
    """List all entities for the current user."""
    engine = get_engine()
    try:
        return await engine.list_entities(user_id=current_user.deep_context_user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/lifecycle/run")
async def run_lifecycle(
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> dict[str, int]:
    """Run lifecycle maintenance: decay + consolidation + cleanup."""
    engine = get_engine()
    api_key = await _get_user_api_key(current_user.user_id)
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="No API key configured. Set your OpenRouter API key in Settings.",
        )

    try:
        engine.set_user_api_key(api_key)
        return await engine.run_lifecycle(user_id=current_user.deep_context_user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        engine.clear_user_api_key()


@app.post("/stats")
async def get_stats(
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> dict[str, Any]:
    """Get summary statistics for the current user."""
    engine = get_engine()
    try:
        return await engine.get_stats(user_id=current_user.deep_context_user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
