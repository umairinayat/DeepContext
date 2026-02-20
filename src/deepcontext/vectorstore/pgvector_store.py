"""
Vector store with support for both pgvector (PostgreSQL) and SQLite.

On PostgreSQL: Uses pgvector's native cosine distance operator for fast search.
On SQLite: Stores embeddings as JSON and computes cosine similarity in Python.

Supports:
- Cosine similarity (default)
- Filtered search by tier/type
- Batch operations
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from deepcontext.db.models.memory import Memory
from deepcontext.vectorstore.base import BaseVectorStore, VectorSearchResult

# Detect database backend
_DB_URL = os.environ.get("DEEPCONTEXT_DATABASE_URL", "")
_USE_PG = "postgresql" in _DB_URL


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class PgVectorStore(BaseVectorStore):
    """
    Vector store backed by PostgreSQL pgvector or SQLite with Python cosine sim.

    On PostgreSQL: Vectors are stored in the Memory table's `embedding` VECTOR column.
    Search uses pgvector's cosine distance operator `<=>`.

    On SQLite: Vectors are stored as JSON text. Cosine similarity is computed in Python
    by loading all active embeddings and ranking them. This is fine for development
    but not suitable for production with large numbers of memories.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def add(self, memory_id: int, embedding: list[float]) -> None:
        """Update the embedding column for an existing memory."""
        async with self._session_factory() as session:
            memory = await session.get(Memory, memory_id)
            if memory:
                memory.set_embedding(embedding)
                await session.commit()

    async def add_batch(self, items: list[tuple[int, list[float]]]) -> None:
        """Update embeddings for multiple memories."""
        async with self._session_factory() as session:
            for memory_id, embedding in items:
                memory = await session.get(Memory, memory_id)
                if memory:
                    memory.set_embedding(embedding)
            await session.commit()

    async def search(
        self,
        query_embedding: list[float],
        user_id: str,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[VectorSearchResult]:
        """Search for similar memories using cosine similarity."""
        if _USE_PG:
            return await self._search_pgvector(query_embedding, user_id, limit, min_score)
        return await self._search_sqlite(query_embedding, user_id, limit, min_score)

    async def _search_pgvector(
        self,
        query_embedding: list[float],
        user_id: str,
        limit: int,
        min_score: float,
    ) -> list[VectorSearchResult]:
        """Search using pgvector's native cosine distance operator."""
        async with self._session_factory() as session:
            vector_str = f"[{','.join(str(v) for v in query_embedding)}]"

            query = text("""
                SELECT 
                    id,
                    1 - (embedding <=> :query_vec::vector) AS similarity
                FROM memories
                WHERE user_id = :user_id
                  AND is_active = true
                  AND embedding IS NOT NULL
                ORDER BY embedding <=> :query_vec::vector
                LIMIT :limit
            """)

            result = await session.execute(
                query,
                {
                    "query_vec": vector_str,
                    "user_id": user_id,
                    "limit": limit,
                },
            )

            results = []
            for row in result:
                similarity = float(row.similarity)
                if similarity >= min_score:
                    results.append(
                        VectorSearchResult(memory_id=row.id, score=similarity)
                    )
            return results

    async def _search_sqlite(
        self,
        query_embedding: list[float],
        user_id: str,
        limit: int,
        min_score: float,
    ) -> list[VectorSearchResult]:
        """Search using Python cosine similarity (SQLite fallback)."""
        async with self._session_factory() as session:
            stmt = select(Memory).where(
                Memory.user_id == user_id,
                Memory.is_active == True,  # noqa: E712
                Memory.embedding.isnot(None),
            )
            result = await session.execute(stmt)
            memories = result.scalars().all()

        scored: list[VectorSearchResult] = []
        for mem in memories:
            emb = mem.get_embedding()
            if emb is None:
                continue
            sim = _cosine_similarity(query_embedding, emb)
            if sim >= min_score:
                scored.append(VectorSearchResult(memory_id=mem.id, score=sim))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:limit]

    async def remove(self, memory_id: int) -> None:
        """Remove embedding from a memory (set to NULL)."""
        async with self._session_factory() as session:
            memory = await session.get(Memory, memory_id)
            if memory:
                memory.embedding = None
                await session.commit()

    async def update(self, memory_id: int, embedding: list[float]) -> None:
        """Update an existing memory's embedding."""
        await self.add(memory_id, embedding)

    async def search_with_filters(
        self,
        query_embedding: list[float],
        user_id: str,
        tier: str | None = None,
        memory_type: str | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[VectorSearchResult]:
        """Search with additional filters on tier and memory type."""
        if _USE_PG:
            return await self._search_pgvector_filtered(
                query_embedding, user_id, tier, memory_type, limit, min_score
            )
        return await self._search_sqlite_filtered(
            query_embedding, user_id, tier, memory_type, limit, min_score
        )

    async def _search_pgvector_filtered(
        self,
        query_embedding: list[float],
        user_id: str,
        tier: str | None,
        memory_type: str | None,
        limit: int,
        min_score: float,
    ) -> list[VectorSearchResult]:
        """Filtered search using pgvector."""
        async with self._session_factory() as session:
            vector_str = f"[{','.join(str(v) for v in query_embedding)}]"

            where_clauses = [
                "user_id = :user_id",
                "is_active = true",
                "embedding IS NOT NULL",
            ]
            params: dict[str, Any] = {
                "query_vec": vector_str,
                "user_id": user_id,
                "limit": limit,
            }

            if tier:
                where_clauses.append("tier = :tier")
                params["tier"] = tier
            if memory_type:
                where_clauses.append("memory_type = :memory_type")
                params["memory_type"] = memory_type

            where_sql = " AND ".join(where_clauses)

            query = text(f"""
                SELECT 
                    id,
                    1 - (embedding <=> :query_vec::vector) AS similarity
                FROM memories
                WHERE {where_sql}
                ORDER BY embedding <=> :query_vec::vector
                LIMIT :limit
            """)

            result = await session.execute(query, params)

            results = []
            for row in result:
                similarity = float(row.similarity)
                if similarity >= min_score:
                    results.append(
                        VectorSearchResult(memory_id=row.id, score=similarity)
                    )
            return results

    async def _search_sqlite_filtered(
        self,
        query_embedding: list[float],
        user_id: str,
        tier: str | None,
        memory_type: str | None,
        limit: int,
        min_score: float,
    ) -> list[VectorSearchResult]:
        """Filtered search using Python cosine similarity (SQLite fallback)."""
        async with self._session_factory() as session:
            conditions = [
                Memory.user_id == user_id,
                Memory.is_active == True,  # noqa: E712
                Memory.embedding.isnot(None),
            ]
            if tier:
                conditions.append(Memory.tier == tier)
            if memory_type:
                conditions.append(Memory.memory_type == memory_type)

            stmt = select(Memory).where(*conditions)
            result = await session.execute(stmt)
            memories = result.scalars().all()

        scored: list[VectorSearchResult] = []
        for mem in memories:
            emb = mem.get_embedding()
            if emb is None:
                continue
            sim = _cosine_similarity(query_embedding, emb)
            if sim >= min_score:
                scored.append(VectorSearchResult(memory_id=mem.id, score=sim))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:limit]
