"""
Abstract base class for vector store backends.

All vector store implementations must implement this interface.
This allows swapping between pgvector, FAISS, Qdrant, etc.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class VectorSearchResult:
    """A single vector search result."""

    memory_id: int
    score: float  # Cosine similarity (0-1)


class BaseVectorStore(ABC):
    """
    Abstract vector store interface.

    Implementations:
    - PgVectorStore: Uses PostgreSQL pgvector extension (recommended)
    - FAISSVectorStore: In-process FAISS index (future)
    - QdrantVectorStore: External Qdrant server (future)
    """

    @abstractmethod
    async def add(self, memory_id: int, embedding: list[float]) -> None:
        """Add a vector to the store."""
        ...

    @abstractmethod
    async def add_batch(self, items: list[tuple[int, list[float]]]) -> None:
        """Add multiple vectors at once."""
        ...

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        user_id: str,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors."""
        ...

    @abstractmethod
    async def remove(self, memory_id: int) -> None:
        """Remove a vector from the store."""
        ...

    @abstractmethod
    async def update(self, memory_id: int, embedding: list[float]) -> None:
        """Update an existing vector."""
        ...
