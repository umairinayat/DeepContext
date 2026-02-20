"""
DeepContext - Hierarchical Memory System for AI Agents.

A production-grade memory layer that gives AI agents persistent context
through hierarchical memory (working/short-term/long-term), knowledge graphs,
hybrid retrieval, and memory consolidation.

Quick Start:
    >>> from deepcontext import DeepContext
    >>>
    >>> ctx = DeepContext(database_url="postgresql+asyncpg://...", openai_api_key="sk-...")
    >>> await ctx.init()
    >>>
    >>> # Add memories from a conversation
    >>> await ctx.add(
    ...     messages=[{"role": "user", "content": "I'm a Python developer"}],
    ...     user_id="user_1",
    ...     conversation_id="conv_1",
    ... )
    >>>
    >>> # Search memories
    >>> results = await ctx.search("What does the user work with?", user_id="user_1")
"""

from deepcontext.core.settings import DeepContextSettings
from deepcontext.memory.engine import MemoryEngine

# Primary interface
DeepContext = MemoryEngine

__all__ = [
    "DeepContext",
    "DeepContextSettings",
    "MemoryEngine",
]

__version__ = "0.1.0"
