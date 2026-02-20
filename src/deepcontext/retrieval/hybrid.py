"""
Hybrid retrieval engine.

Combines three retrieval strategies:
1. Vector similarity search (pgvector)
2. Full-text keyword search (PostgreSQL tsvector)
3. Graph traversal (entity relationships)

Results are fused using Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from deepcontext.core.clients import LLMClients
from deepcontext.core.settings import DeepContextSettings
from deepcontext.core.types import MemorySearchResult, MemoryTier, MemoryType, SearchResponse
from deepcontext.db.models.memory import Memory
from deepcontext.graph.knowledge_graph import KnowledgeGraph
from deepcontext.vectorstore.pgvector_store import PgVectorStore


class HybridRetriever:
    """
    Hybrid retrieval engine that fuses vector, keyword, and graph results.

    The retrieval pipeline:
    1. Generate query embedding
    2. Run vector search (pgvector cosine similarity)
    3. Run keyword search (PostgreSQL full-text search)
    4. Run graph expansion (find related entities)
    5. Fuse results using Reciprocal Rank Fusion (RRF)
    6. Apply recency decay and importance weighting
    7. Return top-k results
    """

    def __init__(
        self,
        clients: LLMClients,
        settings: DeepContextSettings,
        vector_store: PgVectorStore,
        graph: KnowledgeGraph,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._clients = clients
        self._settings = settings
        self._vector_store = vector_store
        self._graph = graph
        self._session_factory = session_factory

    async def search(
        self,
        query: str,
        user_id: str,
        limit: int = 10,
        tier: Optional[str] = None,
        memory_type: Optional[str] = None,
        include_graph_context: bool = True,
    ) -> SearchResponse:
        """
        Execute hybrid search across all retrieval strategies.

        Args:
            query: Natural language search query
            user_id: User to search memories for
            limit: Maximum results to return
            tier: Filter by memory tier (working/short_term/long_term)
            memory_type: Filter by type (semantic/episodic/procedural)
            include_graph_context: Whether to include graph-based expansion

        Returns:
            SearchResponse with ranked, scored results
        """
        # Step 1: Generate query embedding
        query_embedding = await self._clients.embed_text(query)

        # Step 2-4: Run retrieval strategies in parallel
        # (In a production system, these would be truly concurrent with asyncio.gather)
        vector_results = await self._vector_search(
            query_embedding, user_id, limit * 3, tier, memory_type
        )
        keyword_results = await self._keyword_search(query, user_id, limit * 2)

        graph_memory_ids: list[int] = []
        if include_graph_context:
            graph_memory_ids = await self._graph_expansion(query, user_id, limit)

        # Step 5: Fuse results using RRF
        fused = self._reciprocal_rank_fusion(
            vector_results=vector_results,
            keyword_results=keyword_results,
            graph_ids=graph_memory_ids,
        )

        # Step 6: Fetch full memory objects and apply scoring
        memory_ids = list(fused.keys())[:limit * 2]
        if not memory_ids:
            return SearchResponse(query=query, user_id=user_id, total=0, results=[])

        async with self._session_factory() as session:
            stmt = select(Memory).where(
                Memory.id.in_(memory_ids),
                Memory.is_active == True,
            )
            result = await session.execute(stmt)
            memories = result.scalars().all()

        # Apply recency decay and importance weighting
        now = datetime.now(timezone.utc)
        scored_results: list[MemorySearchResult] = []

        for mem in memories:
            base_score = fused.get(mem.id, 0.0)

            # Recency decay (Ebbinghaus-inspired)
            recency = 1.0
            if mem.tier == MemoryTier.SHORT_TERM.value and mem.created_at:
                created = mem.created_at
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                days_ago = (now - created).total_seconds() / 86400
                half_life = self._settings.decay_half_life_days
                recency = math.exp(-0.693 * days_ago / half_life)

            # Access frequency boost (memories accessed more often are more relevant)
            access_boost = 1.0 + 0.01 * min(mem.access_count, 50)

            # Final score
            final_score = base_score * mem.importance * recency * mem.confidence * access_boost

            scored_results.append(
                MemorySearchResult(
                    memory_id=mem.id,
                    text=mem.text,
                    memory_type=MemoryType(mem.memory_type),
                    tier=MemoryTier(mem.tier),
                    score=round(final_score, 4),
                    importance=mem.importance,
                    confidence=mem.confidence,
                    created_at=mem.created_at,
                    connections=(mem.connections or {}).get("memory_ids", []),
                    entities=mem.source_entities or [],
                    metadata=mem.metadata_ or {},
                )
            )

            # Update access tracking
            async with self._session_factory() as session:
                update_mem = await session.get(Memory, mem.id)
                if update_mem:
                    update_mem.access_count += 1
                    update_mem.last_accessed_at = now
                    await session.commit()

        # Sort by final score and limit
        scored_results.sort(key=lambda x: x.score, reverse=True)
        top = scored_results[:limit]

        return SearchResponse(
            query=query,
            user_id=user_id,
            total=len(top),
            results=top,
        )

    async def _vector_search(
        self,
        embedding: list[float],
        user_id: str,
        limit: int,
        tier: Optional[str] = None,
        memory_type: Optional[str] = None,
    ) -> list[tuple[int, float]]:
        """Run vector similarity search."""
        if tier or memory_type:
            results = await self._vector_store.search_with_filters(
                query_embedding=embedding,
                user_id=user_id,
                tier=tier,
                memory_type=memory_type,
                limit=limit,
            )
        else:
            results = await self._vector_store.search(
                query_embedding=embedding,
                user_id=user_id,
                limit=limit,
            )

        return [(r.memory_id, r.score) for r in results]

    async def _keyword_search(
        self,
        query: str,
        user_id: str,
        limit: int,
    ) -> list[tuple[int, float]]:
        """
        Run full-text keyword search using PostgreSQL's tsvector.

        Falls back to ILIKE if tsvector is not available.
        """
        async with self._session_factory() as session:
            try:
                # Try PostgreSQL full-text search first
                stmt = text("""
                    SELECT id, ts_rank(
                        to_tsvector('english', text),
                        plainto_tsquery('english', :query)
                    ) AS rank
                    FROM memories
                    WHERE user_id = :user_id
                      AND is_active = true
                      AND to_tsvector('english', text) @@ plainto_tsquery('english', :query)
                    ORDER BY rank DESC
                    LIMIT :limit
                """)
                result = await session.execute(
                    stmt, {"query": query, "user_id": user_id, "limit": limit}
                )
                return [(row.id, float(row.rank)) for row in result]
            except Exception:
                # Fallback to ILIKE for SQLite or other DBs
                stmt = text("""
                    SELECT id FROM memories
                    WHERE user_id = :user_id
                      AND is_active = true
                      AND LOWER(text) LIKE LOWER(:pattern)
                    LIMIT :limit
                """)
                pattern = f"%{query}%"
                result = await session.execute(
                    stmt, {"pattern": pattern, "user_id": user_id, "limit": limit}
                )
                return [(row.id, 0.5) for row in result]

    async def _graph_expansion(
        self,
        query: str,
        user_id: str,
        limit: int,
    ) -> list[int]:
        """
        Find memories connected to entities mentioned in the query.

        Extracts entity names from the query, finds their graph neighbors,
        and returns memory IDs that reference those entities.
        """
        # Simple entity extraction from query (just word overlap with known entities)
        async with self._session_factory() as session:
            # Get all entities for this user
            from deepcontext.db.models.graph import Entity

            stmt = select(Entity).where(Entity.user_id == user_id)
            result = await session.execute(stmt)
            entities = result.scalars().all()

        # Find entities mentioned in query
        query_lower = query.lower()
        mentioned = [e.name for e in entities if e.name.lower() in query_lower]

        if not mentioned:
            return []

        # Get graph context for mentioned entities
        context = await self._graph.get_entity_context(user_id, mentioned)
        related_entities = set()
        for c in context:
            related_entities.add(c["source"])
            related_entities.add(c["target"])

        # Find memories that reference these entities
        async with self._session_factory() as session:
            stmt = select(Memory.id).where(
                Memory.user_id == user_id,
                Memory.is_active == True,
            )
            result = await session.execute(stmt)
            all_memory_ids = [row[0] for row in result]

            # Filter memories that mention related entities
            graph_ids: list[int] = []
            for mid in all_memory_ids:
                mem = await session.get(Memory, mid)
                if mem and mem.source_entities:
                    if any(e in related_entities for e in mem.source_entities):
                        graph_ids.append(mid)
                        if len(graph_ids) >= limit:
                            break

        return graph_ids

    @staticmethod
    def _reciprocal_rank_fusion(
        vector_results: list[tuple[int, float]],
        keyword_results: list[tuple[int, float]],
        graph_ids: list[int],
        k: int = 60,
        vector_weight: float = 0.6,
        keyword_weight: float = 0.25,
        graph_weight: float = 0.15,
    ) -> dict[int, float]:
        """
        Fuse results from multiple retrieval strategies using Reciprocal Rank Fusion.

        RRF score = sum(weight / (k + rank)) for each strategy.

        Args:
            k: RRF constant (higher = more uniform fusion)
            vector_weight: Weight for vector similarity results
            keyword_weight: Weight for keyword search results
            graph_weight: Weight for graph expansion results

        Returns:
            Dict of memory_id -> fused score
        """
        scores: dict[int, float] = defaultdict(float)

        # Vector results (already sorted by similarity)
        for rank, (memory_id, _sim) in enumerate(vector_results):
            scores[memory_id] += vector_weight / (k + rank + 1)

        # Keyword results
        for rank, (memory_id, _rank_score) in enumerate(keyword_results):
            scores[memory_id] += keyword_weight / (k + rank + 1)

        # Graph-expanded results (no ranking, flat boost)
        for rank, memory_id in enumerate(graph_ids):
            scores[memory_id] += graph_weight / (k + rank + 1)

        return dict(scores)
