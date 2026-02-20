"""
MemoryEngine - The main orchestrator for DeepContext.

This is the primary interface. It coordinates:
- Extraction (LLM-based fact/entity extraction)
- Storage (pgvector + knowledge graph)
- Retrieval (hybrid: vector + keyword + graph)
- Lifecycle (decay, consolidation, cleanup)

Usage:
    ctx = MemoryEngine(
        database_url="postgresql+asyncpg://user:pass@localhost/deepcontext",
        openai_api_key="sk-...",
    )
    await ctx.init()

    # Add memories
    await ctx.add(
        messages=[{"role": "user", "content": "I love Python"}],
        user_id="user_1",
    )

    # Search
    results = await ctx.search("programming", user_id="user_1")

    # Cleanup
    await ctx.close()
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from deepcontext.core.clients import LLMClients
from deepcontext.core.settings import DeepContextSettings, LLMProvider
from deepcontext.core.types import (
    AddResponse,
    ExtractedFact,
    MemoryTier,
    MemoryType,
    SearchResponse,
    ToolAction,
)
from deepcontext.db.database import Database
from deepcontext.db.models.memory import Memory
from deepcontext.extraction.extractor import Extractor
from deepcontext.graph.knowledge_graph import KnowledgeGraph
from deepcontext.lifecycle.manager import LifecycleManager
from deepcontext.retrieval.hybrid import HybridRetriever
from deepcontext.vectorstore.pgvector_store import PgVectorStore


class MemoryEngine:
    """
    The main DeepContext interface.

    Orchestrates all subsystems: extraction, storage, retrieval, lifecycle.
    All operations are async.
    """

    def __init__(
        self,
        database_url: str = "",
        openai_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        llm_provider: str | LLMProvider = LLMProvider.OPENAI,
        llm_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        debug: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize MemoryEngine.

        Args:
            database_url: PostgreSQL connection URL (asyncpg)
            openai_api_key: OpenAI API key
            openrouter_api_key: OpenRouter API key
            llm_provider: "openai" or "openrouter"
            llm_model: Model for extraction/classification
            embedding_model: Model for embeddings
            debug: Enable debug logging
            **kwargs: Additional settings passed to DeepContextSettings
        """
        # Build settings kwargs, omitting None values so .env can fill them
        settings_kwargs: dict[str, Any] = {
            "llm_model": llm_model,
            "embedding_model": embedding_model,
            "debug": debug,
            **kwargs,
        }
        if database_url:
            settings_kwargs["database_url"] = database_url
        if openai_api_key is not None:
            settings_kwargs["openai_api_key"] = openai_api_key
        if openrouter_api_key is not None:
            settings_kwargs["openrouter_api_key"] = openrouter_api_key

        # Coerce string to enum if needed
        provider = LLMProvider(llm_provider) if isinstance(llm_provider, str) else llm_provider
        settings_kwargs["llm_provider"] = provider

        self._settings = DeepContextSettings(**settings_kwargs)
        self._db: Optional[Database] = None
        self._clients: Optional[LLMClients] = None
        self._extractor: Optional[Extractor] = None
        self._vector_store: Optional[PgVectorStore] = None
        self._graph: Optional[KnowledgeGraph] = None
        self._retriever: Optional[HybridRetriever] = None
        self._lifecycle: Optional[LifecycleManager] = None
        self._initialized = False

    async def init(self) -> None:
        """
        Initialize all subsystems.

        Must be called before any other method. Creates database tables,
        initializes clients, and sets up indexes.
        """
        # Database
        self._db = Database(self._settings.database_url, echo=self._settings.debug)
        await self._db.init()

        # LLM clients
        self._clients = LLMClients(self._settings)

        # Subsystems
        session_factory = self._db._session_factory
        assert session_factory is not None, "Database init failed: no session factory"

        self._extractor = Extractor(self._clients, self._settings)
        self._vector_store = PgVectorStore(session_factory)
        self._graph = KnowledgeGraph(session_factory)
        self._retriever = HybridRetriever(
            self._clients, self._settings, self._vector_store, self._graph, session_factory
        )
        self._lifecycle = LifecycleManager(
            self._clients, self._settings, self._extractor, session_factory
        )

        self._initialized = True

        if self._settings.debug:
            print("[DeepContext] Initialized successfully")

    def _check_init(self) -> None:
        """Raise if not initialized."""
        if not self._initialized:
            raise RuntimeError("MemoryEngine not initialized. Call `await engine.init()` first.")

    # -----------------------------------------------------------------------
    # ADD - Extract and store memories from conversation
    # -----------------------------------------------------------------------

    async def add(
        self,
        messages: list[dict[str, str]],
        user_id: str,
        conversation_id: str = "default",
    ) -> AddResponse:
        """
        Extract and store memories from a conversation turn.

        Pipeline:
        1. Extract facts, entities, relationships from latest messages
        2. For each fact, classify action (ADD/UPDATE/REPLACE/NOOP)
        3. Store in DB with embeddings
        4. Update knowledge graph
        5. Optionally trigger consolidation

        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."}
            user_id: User identifier for memory isolation
            conversation_id: Conversation identifier

        Returns:
            AddResponse with details of what was stored
        """
        self._check_init()
        assert self._extractor is not None
        assert self._clients is not None
        assert self._graph is not None
        assert self._lifecycle is not None
        assert self._db is not None

        if len(messages) < 1:
            return AddResponse()

        # Format latest messages
        latest = []
        for msg in messages[-2:]:
            latest.append(f"{msg['role'].upper()}: {msg['content']}")

        # Get existing summary (if any)
        summary = await self._get_summary(user_id, conversation_id)

        # Recent messages for context
        recent = [f"{m['role'].upper()}: {m['content']}" for m in messages[-10:]]

        # Step 1: Extract
        extraction = await self._extractor.extract_memories(
            latest_messages=latest,
            summary=summary,
            recent_messages=recent,
        )

        if self._settings.debug:
            print(
                f"[DeepContext] Extracted: "
                f"{len(extraction.semantic)} semantic, "
                f"{len(extraction.episodic)} episodic, "
                f"{len(extraction.entities)} entities, "
                f"{len(extraction.relationships)} relationships"
            )

        response = AddResponse()

        # Step 2: Process semantic facts
        for fact in extraction.semantic:
            action_result = await self._process_fact(fact, user_id, conversation_id)
            response.semantic_facts.append(fact.text)
            if action_result == ToolAction.ADD:
                response.memories_added += 1
            elif action_result == ToolAction.UPDATE:
                response.memories_updated += 1
            elif action_result == ToolAction.REPLACE:
                response.memories_replaced += 1

        # Step 3: Process episodic facts
        for fact in extraction.episodic:
            await self._store_memory(
                text=fact.text,
                user_id=user_id,
                conversation_id=conversation_id,
                memory_type=MemoryType.EPISODIC,
                tier=MemoryTier.SHORT_TERM,
                importance=fact.importance,
                confidence=fact.confidence,
                entities=fact.entities,
            )
            response.episodic_facts.append(fact.text)
            response.memories_added += 1

        # Step 4: Update knowledge graph
        for entity in extraction.entities:
            await self._graph.upsert_entity(user_id, entity)
            response.entities_found.append(entity.name)

        for rel in extraction.relationships:
            await self._graph.add_relationship(user_id, rel)
            response.relationships_found += 1

        # Step 5: Auto-consolidation check
        if self._settings.auto_consolidate:
            consolidated = await self._lifecycle.consolidate(user_id)
            if consolidated > 0 and self._settings.debug:
                print(f"[DeepContext] Consolidated {consolidated} memories")

        return response

    # -----------------------------------------------------------------------
    # SEARCH - Hybrid retrieval
    # -----------------------------------------------------------------------

    async def search(
        self,
        query: str,
        user_id: str,
        limit: int = 10,
        tier: Optional[str] = None,
        memory_type: Optional[str] = None,
        include_graph: bool = True,
    ) -> SearchResponse:
        """
        Search for relevant memories using hybrid retrieval.

        Combines vector similarity, keyword search, and graph traversal.

        Args:
            query: Natural language search query
            user_id: User whose memories to search
            limit: Maximum results
            tier: Filter by tier (working/short_term/long_term)
            memory_type: Filter by type (semantic/episodic/procedural)
            include_graph: Include graph-based expansion

        Returns:
            SearchResponse with ranked results
        """
        self._check_init()
        assert self._retriever is not None

        return await self._retriever.search(
            query=query,
            user_id=user_id,
            limit=limit,
            tier=tier,
            memory_type=memory_type,
            include_graph_context=include_graph,
        )

    # -----------------------------------------------------------------------
    # UPDATE / DELETE
    # -----------------------------------------------------------------------

    async def update(self, memory_id: int, text: str, user_id: str) -> dict[str, Any]:
        """Update a memory's text and re-embed."""
        self._check_init()
        assert self._db is not None
        assert self._clients is not None

        async with self._db.session() as session:
            memory = await session.get(Memory, memory_id)
            if not memory or memory.user_id != user_id:
                raise ValueError(f"Memory {memory_id} not found for user {user_id}")

            memory.text = text
            new_embedding = await self._clients.embed_text(text)
            memory.set_embedding(new_embedding)
            memory.updated_at = datetime.now(timezone.utc)
            await session.commit()

        return {"memory_id": memory_id, "text": text, "status": "updated"}

    async def delete(self, memory_id: int, user_id: str) -> dict[str, Any]:
        """Soft-delete a memory."""
        self._check_init()
        assert self._db is not None

        async with self._db.session() as session:
            memory = await session.get(Memory, memory_id)
            if not memory or memory.user_id != user_id:
                raise ValueError(f"Memory {memory_id} not found for user {user_id}")

            memory.is_active = False
            await session.commit()

        return {"memory_id": memory_id, "status": "deleted"}

    # -----------------------------------------------------------------------
    # LIFECYCLE
    # -----------------------------------------------------------------------

    async def run_lifecycle(self, user_id: str) -> dict[str, int]:
        """
        Run full lifecycle maintenance: decay + consolidation + cleanup.

        This should be called periodically (e.g., daily or per-session).
        """
        self._check_init()
        assert self._lifecycle is not None

        decayed = await self._lifecycle.apply_decay(user_id)
        consolidated = await self._lifecycle.consolidate(user_id)
        cleaned = await self._lifecycle.cleanup(user_id)

        result = {
            "memories_decayed": decayed,
            "memories_consolidated": consolidated,
            "memories_cleaned": cleaned,
        }

        if self._settings.debug:
            print(f"[DeepContext] Lifecycle: {result}")

        return result

    # -----------------------------------------------------------------------
    # GRAPH
    # -----------------------------------------------------------------------

    async def get_entity_graph(
        self,
        user_id: str,
        entity_name: str,
        depth: int = 2,
    ) -> list[dict[str, Any]]:
        """Get the knowledge graph neighborhood for an entity."""
        self._check_init()
        assert self._graph is not None

        return await self._graph.get_neighbors(user_id, entity_name, max_depth=depth)

    # -----------------------------------------------------------------------
    # CLEANUP
    # -----------------------------------------------------------------------

    async def close(self) -> None:
        """Close all connections and clients."""
        if self._clients:
            await self._clients.close()
        if self._db:
            await self._db.close()
        self._initialized = False

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    async def _process_fact(
        self,
        fact: ExtractedFact,
        user_id: str,
        conversation_id: str,
    ) -> ToolAction:
        """Process a single semantic fact: classify action and execute."""
        assert self._extractor is not None
        assert self._clients is not None
        assert self._db is not None

        # Find similar existing memories
        embedding = await self._clients.embed_text(fact.text)

        # Use the vector store for similarity search (works on both PG and SQLite)
        assert self._vector_store is not None
        try:
            vector_results = await self._vector_store.search(
                query_embedding=embedding,
                user_id=user_id,
                limit=10,
            )
            # Fetch the text for each similar memory
            similar: list[dict[str, Any]] = []
            if vector_results:
                async with self._db.session() as session:
                    for vr in vector_results:
                        mem = await session.get(Memory, vr.memory_id)
                        if mem:
                            similar.append({"id": mem.id, "text": mem.text, "sim": vr.score})
        except Exception:
            similar = []

        # Classify action
        decision = await self._extractor.classify_action(fact.text, similar)

        if self._settings.debug:
            print(f"[DeepContext] Fact: '{fact.text[:50]}...' -> {decision.action.value}")

        # Execute action
        if decision.action == ToolAction.ADD:
            await self._store_memory(
                text=decision.text or fact.text,
                user_id=user_id,
                conversation_id=conversation_id,
                memory_type=MemoryType.SEMANTIC,
                tier=MemoryTier.SHORT_TERM,
                importance=fact.importance,
                confidence=fact.confidence,
                entities=fact.entities,
                embedding=embedding,
            )

        elif decision.action == ToolAction.UPDATE and decision.memory_id:
            await self.update(decision.memory_id, decision.text or fact.text, user_id)

        elif decision.action == ToolAction.REPLACE and decision.memory_id:
            await self.delete(decision.memory_id, user_id)
            await self._store_memory(
                text=decision.text or fact.text,
                user_id=user_id,
                conversation_id=conversation_id,
                memory_type=MemoryType.SEMANTIC,
                tier=MemoryTier.SHORT_TERM,
                importance=fact.importance,
                confidence=fact.confidence,
                entities=fact.entities,
                embedding=embedding,
            )

        elif decision.action == ToolAction.DELETE and decision.memory_id:
            await self.delete(decision.memory_id, user_id)

        return decision.action

    async def _store_memory(
        self,
        text: str,
        user_id: str,
        conversation_id: str,
        memory_type: MemoryType,
        tier: MemoryTier,
        importance: float = 0.5,
        confidence: float = 0.8,
        entities: Optional[list[str]] = None,
        embedding: Optional[list[float]] = None,
    ) -> Memory:
        """Create and store a new memory with embedding."""
        assert self._db is not None
        assert self._clients is not None

        if embedding is None:
            embedding = await self._clients.embed_text(text)

        now = datetime.now(timezone.utc)
        memory = Memory(
            user_id=user_id,
            conversation_id=conversation_id,
            text=text,
            memory_type=memory_type.value,
            tier=tier.value,
            importance=importance,
            confidence=confidence,
            source_entities=entities or [],
            created_at=now,
            updated_at=now,
            occurred_at=now if memory_type == MemoryType.EPISODIC else None,
        )
        memory.set_embedding(embedding)

        async with self._db.session() as session:
            session.add(memory)
            await session.commit()
            await session.refresh(memory)

        return memory

    async def _get_summary(self, user_id: str, conversation_id: str) -> str:
        """Get the existing conversation summary, if any."""
        assert self._db is not None

        from deepcontext.db.models.graph import ConversationSummary

        async with self._db.session() as session:
            stmt = select(ConversationSummary).where(
                ConversationSummary.user_id == user_id,
                ConversationSummary.conversation_id == conversation_id,
            )
            result = await session.execute(stmt)
            summary = result.scalar_one_or_none()
            return summary.summary_text if summary else ""
