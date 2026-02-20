"""
Memory lifecycle management.

Handles:
1. Decay: Episodic memories fade over time (Ebbinghaus forgetting curve)
2. Consolidation: Short-term memories compress into long-term facts
3. Cleanup: Remove very low-confidence or fully decayed memories
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from deepcontext.core.clients import LLMClients
from deepcontext.core.settings import DeepContextSettings
from deepcontext.core.types import MemoryTier, MemoryType
from deepcontext.db.models.memory import Memory
from deepcontext.extraction.extractor import Extractor


class LifecycleManager:
    """
    Manages the lifecycle of memories.

    Memory lifecycle:
    1. New facts enter as SHORT_TERM memories
    2. Short-term memories decay over time (importance reduces)
    3. When enough related short-term memories accumulate, they consolidate
       into a single LONG_TERM memory
    4. Memories that decay below a threshold are soft-deleted
    5. Long-term memories are stable but can still be updated/replaced
    """

    def __init__(
        self,
        clients: LLMClients,
        settings: DeepContextSettings,
        extractor: Extractor,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._clients = clients
        self._settings = settings
        self._extractor = extractor
        self._session_factory = session_factory

    async def apply_decay(self, user_id: str) -> int:
        """
        Apply Ebbinghaus forgetting curve decay to short-term memories.

        The forgetting curve: R = e^(-t/S)
        Where R = retention, t = time since creation, S = stability

        Memories that decay below 0.05 importance are soft-deleted.

        Returns:
            Number of memories affected
        """
        now = datetime.now(timezone.utc)
        half_life = self._settings.decay_half_life_days

        async with self._session_factory() as session:
            # Get all active short-term memories
            stmt = select(Memory).where(
                Memory.user_id == user_id,
                Memory.tier == MemoryTier.SHORT_TERM.value,
                Memory.is_active == True,
            )
            result = await session.execute(stmt)
            memories = result.scalars().all()

            affected = 0
            for mem in memories:
                created = mem.created_at
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)

                days_ago = (now - created).total_seconds() / 86400

                # Ebbinghaus decay
                # Access count slows decay (reinforcement)
                effective_half_life = half_life * (1 + 0.1 * mem.access_count)
                decay_factor = math.exp(-0.693 * days_ago / effective_half_life)

                new_importance = mem.importance * decay_factor

                if new_importance < 0.05:
                    # Memory has fully decayed - soft delete
                    mem.is_active = False
                    affected += 1
                elif abs(new_importance - mem.importance) > 0.01:
                    mem.importance = round(new_importance, 4)
                    affected += 1

            await session.commit()

        return affected

    async def consolidate(self, user_id: str) -> int:
        """
        Consolidate short-term memories into long-term facts.

        Groups related short-term memories (by entity overlap and semantic similarity),
        then uses the LLM to merge them into concise long-term facts.

        Returns:
            Number of new long-term memories created
        """
        async with self._session_factory() as session:
            # Get short-term memories
            stmt = select(Memory).where(
                Memory.user_id == user_id,
                Memory.tier == MemoryTier.SHORT_TERM.value,
                Memory.memory_type == MemoryType.SEMANTIC.value,
                Memory.is_active == True,
            )
            result = await session.execute(stmt)
            st_memories = result.scalars().all()

            if len(st_memories) < self._settings.consolidation_threshold:
                return 0

            # Group by entity overlap
            groups = self._group_by_entity_overlap(st_memories)

            consolidated_count = 0
            for group in groups:
                if len(group) < 2:
                    continue

                # Use LLM to consolidate
                texts = [m.text for m in group]
                consolidated = await self._extractor.consolidate_memories(texts)

                if not consolidated.get("text"):
                    continue

                # Gather all entities from the group
                all_entities: list[str] = []
                for m in group:
                    all_entities.extend(m.source_entities or [])
                unique_entities = list(set(all_entities))

                # Create new long-term memory
                embedding = await self._clients.embed_text(consolidated["text"])

                source_ids = [m.id for m in group]
                new_memory = Memory(
                    user_id=user_id,
                    text=consolidated["text"],
                    memory_type=MemoryType.SEMANTIC.value,
                    tier=MemoryTier.LONG_TERM.value,
                    importance=float(consolidated.get("importance", 0.8)),
                    confidence=float(consolidated.get("confidence", 0.9)),
                    source_entities=unique_entities,
                    consolidated_from=str(source_ids),
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                )
                new_memory.set_embedding(embedding)
                session.add(new_memory)

                # Deactivate source short-term memories
                for m in group:
                    m.is_active = False

                consolidated_count += 1

            await session.commit()

        return consolidated_count

    async def cleanup(self, user_id: str, min_importance: float = 0.05) -> int:
        """
        Remove decayed and low-value memories.

        Returns:
            Number of memories deactivated
        """
        async with self._session_factory() as session:
            stmt = (
                update(Memory)
                .where(
                    Memory.user_id == user_id,
                    Memory.is_active == True,
                    Memory.importance < min_importance,
                    Memory.tier != MemoryTier.LONG_TERM.value,
                )
                .values(is_active=False)
            )
            result = await session.execute(stmt)
            await session.commit()
            return result.rowcount  # type: ignore[return-value]

    @staticmethod
    def _group_by_entity_overlap(
        memories: list[Memory],
        min_overlap: int = 1,
    ) -> list[list[Memory]]:
        """
        Group memories by entity overlap.

        Two memories are in the same group if they share at least
        min_overlap entities.
        """
        # Union-Find for grouping
        parent: dict[int, int] = {}

        def find(x: int) -> int:
            if x not in parent:
                parent[x] = x
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Build entity index
        for i, m1 in enumerate(memories):
            entities_1 = set(m1.source_entities or [])
            if not entities_1:
                continue
            for j in range(i + 1, len(memories)):
                entities_2 = set(memories[j].source_entities or [])
                overlap = len(entities_1 & entities_2)
                if overlap >= min_overlap:
                    union(i, j)

        # Collect groups
        groups_map: dict[int, list[Memory]] = {}
        for i, mem in enumerate(memories):
            root = find(i)
            if root not in groups_map:
                groups_map[root] = []
            groups_map[root].append(mem)

        return list(groups_map.values())
