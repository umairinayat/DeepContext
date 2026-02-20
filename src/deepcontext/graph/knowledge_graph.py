"""
Knowledge graph manager.

Manages entities and relationships in PostgreSQL,
providing graph traversal capabilities without Neo4j.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from deepcontext.core.types import ExtractedEntity, ExtractedRelationship
from deepcontext.db.models.graph import Entity, Relationship


class KnowledgeGraph:
    """
    Graph manager for entities and relationships.

    Provides:
    - Entity upsert (create or update mention count)
    - Relationship creation
    - Graph traversal (neighbors, paths)
    - Entity search by type
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def upsert_entity(
        self,
        user_id: str,
        entity: ExtractedEntity,
    ) -> Entity:
        """
        Create a new entity or increment its mention count.

        Returns the Entity ORM object.
        """
        async with self._session_factory() as session:
            # Check if entity already exists for this user
            stmt = select(Entity).where(
                Entity.user_id == user_id,
                Entity.name == entity.name,
            )
            result = await session.execute(stmt)
            existing = result.scalar_one_or_none()

            if existing:
                existing.mention_count += 1
                existing.updated_at = datetime.now(timezone.utc)
                # Merge attributes
                if entity.attributes:
                    attrs = existing.attributes or {}
                    attrs.update(entity.attributes)
                    existing.attributes = attrs
                await session.commit()
                return existing
            else:
                new_entity = Entity(
                    user_id=user_id,
                    name=entity.name,
                    entity_type=entity.entity_type,
                    attributes=entity.attributes or {},
                    mention_count=1,
                )
                session.add(new_entity)
                await session.commit()
                await session.refresh(new_entity)
                return new_entity

    async def add_relationship(
        self,
        user_id: str,
        rel: ExtractedRelationship,
    ) -> Relationship | None:
        """
        Add a relationship between two entities.

        Creates entities if they don't exist.
        Returns None if source or target entity names are empty.
        """
        if not rel.source or not rel.target:
            return None

        async with self._session_factory() as session:
            # Find or create source entity
            source = await self._get_or_create_entity(session, user_id, rel.source)
            target = await self._get_or_create_entity(session, user_id, rel.target)

            # Check if relationship already exists
            stmt = select(Relationship).where(
                Relationship.user_id == user_id,
                Relationship.source_entity_id == source.id,
                Relationship.target_entity_id == target.id,
                Relationship.relation == rel.relation,
            )
            result = await session.execute(stmt)
            existing = result.scalar_one_or_none()

            if existing:
                # Strengthen existing relationship
                existing.strength = min(existing.strength + 0.1, 2.0)
                if rel.properties:
                    props = existing.properties or {}
                    props.update(rel.properties)
                    existing.properties = props
                await session.commit()
                return existing

            new_rel = Relationship(
                user_id=user_id,
                source_entity_id=source.id,
                target_entity_id=target.id,
                relation=rel.relation,
                properties=rel.properties or {},
            )
            session.add(new_rel)
            await session.commit()
            await session.refresh(new_rel)
            return new_rel

    async def get_neighbors(
        self,
        user_id: str,
        entity_name: str,
        max_depth: int = 2,
    ) -> list[dict[str, Any]]:
        """
        Get neighboring entities up to max_depth hops away.

        Returns a list of dicts: [{"entity": name, "relation": rel, "depth": d}]
        """
        async with self._session_factory() as session:
            # Get the root entity
            stmt = select(Entity).where(
                Entity.user_id == user_id,
                Entity.name == entity_name,
            )
            result = await session.execute(stmt)
            root = result.scalar_one_or_none()

            if not root:
                return []

            visited: set[int] = {root.id}
            neighbors: list[dict[str, Any]] = []
            current_ids = [root.id]

            for depth in range(1, max_depth + 1):
                if not current_ids:
                    break

                # Find all relationships from/to current entities
                stmt = select(Relationship).where(
                    Relationship.user_id == user_id,
                    (
                        Relationship.source_entity_id.in_(current_ids)
                        | Relationship.target_entity_id.in_(current_ids)
                    ),
                )
                result = await session.execute(stmt)
                rels = result.scalars().all()

                next_ids: list[int] = []
                for r in rels:
                    # Find the neighbor (other end of the relationship)
                    neighbor_id = (
                        r.target_entity_id
                        if r.source_entity_id in current_ids
                        else r.source_entity_id
                    )

                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        next_ids.append(neighbor_id)

                        # Get entity name
                        entity = await session.get(Entity, neighbor_id)
                        if entity:
                            neighbors.append(
                                {
                                    "entity": entity.name,
                                    "entity_type": entity.entity_type,
                                    "relation": r.relation,
                                    "depth": depth,
                                    "strength": r.strength,
                                }
                            )

                current_ids = next_ids

            return neighbors

    async def get_entity_context(
        self,
        user_id: str,
        entity_names: list[str],
    ) -> list[dict[str, Any]]:
        """
        Get all relationships involving the given entities.

        Useful for building context for retrieval augmentation.
        """
        if not entity_names:
            return []

        async with self._session_factory() as session:
            # Get entity IDs
            stmt = select(Entity).where(
                Entity.user_id == user_id,
                Entity.name.in_(entity_names),
            )
            result = await session.execute(stmt)
            entities = result.scalars().all()

            if not entities:
                return []

            entity_ids = [e.id for e in entities]
            id_to_name = {e.id: e.name for e in entities}

            # Get relationships
            stmt = select(Relationship).where(
                Relationship.user_id == user_id,
                (
                    Relationship.source_entity_id.in_(entity_ids)
                    | Relationship.target_entity_id.in_(entity_ids)
                ),
            )
            result = await session.execute(stmt)
            rels = result.scalars().all()

            context: list[dict[str, Any]] = []
            for r in rels:
                # Resolve names (might need to fetch unknown entities)
                source_name = id_to_name.get(r.source_entity_id)
                if not source_name:
                    src_entity = await session.get(Entity, r.source_entity_id)
                    source_name = src_entity.name if src_entity else "Unknown"

                target_name = id_to_name.get(r.target_entity_id)
                if not target_name:
                    tgt_entity = await session.get(Entity, r.target_entity_id)
                    target_name = tgt_entity.name if tgt_entity else "Unknown"

                context.append(
                    {
                        "source": source_name,
                        "relation": r.relation,
                        "target": target_name,
                        "strength": r.strength,
                    }
                )

            return context

    @staticmethod
    async def _get_or_create_entity(
        session: AsyncSession,
        user_id: str,
        name: str,
    ) -> Entity:
        """Get existing entity or create a minimal one."""
        stmt = select(Entity).where(
            Entity.user_id == user_id,
            Entity.name == name,
        )
        result = await session.execute(stmt)
        entity = result.scalar_one_or_none()

        if entity:
            return entity

        new_entity = Entity(
            user_id=user_id,
            name=name,
            entity_type="other",
        )
        session.add(new_entity)
        await session.flush()
        return new_entity
