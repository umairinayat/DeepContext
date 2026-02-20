"""
Async database engine and session management.

Uses SQLAlchemy 2.0 async API with asyncpg for PostgreSQL
or aiosqlite for local development.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from deepcontext.db.models.base import Base


class Database:
    """
    Async database manager.

    Usage:
        db = Database("postgresql+asyncpg://user:pass@host/db")
        await db.init()
        async with db.session() as session:
            ...
        await db.close()
    """

    def __init__(self, url: str, echo: bool = False) -> None:
        self._url = url
        self._echo = echo
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker[AsyncSession]] = None

    async def init(self) -> None:
        """Create engine, session factory, and all tables."""
        connect_args: dict = {}
        if self._url.startswith("sqlite"):
            connect_args["check_same_thread"] = False

        self._engine = create_async_engine(
            self._url,
            echo=self._echo,
            connect_args=connect_args,
        )
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        # Create tables (safe to call multiple times)
        async with self._engine.begin() as conn:
            # Enable pgvector extension if using PostgreSQL
            if "postgresql" in self._url:
                await conn.execute(
                    __import__("sqlalchemy").text("CREATE EXTENSION IF NOT EXISTS vector")
                )
            await conn.run_sync(Base.metadata.create_all)

    def session(self) -> AsyncSession:
        """Create a new async session."""
        if self._session_factory is None:
            raise RuntimeError("Database not initialized. Call await db.init() first.")
        return self._session_factory()

    @property
    def engine(self) -> AsyncEngine:
        """Get the async engine."""
        if self._engine is None:
            raise RuntimeError("Database not initialized. Call await db.init() first.")
        return self._engine

    async def close(self) -> None:
        """Dispose of the engine and all connections."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
