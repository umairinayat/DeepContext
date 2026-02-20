"""
Centralized configuration for DeepContext.

Supports two configuration methods:
1. Programmatic: Pass settings directly to DeepContext()
2. Environment variables: Loaded automatically via pydantic-settings

Supports multiple LLM providers via OpenAI-compatible interface.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    OPENROUTER = "openrouter"


class DeepContextSettings(BaseSettings):
    """
    Configuration settings for DeepContext.

    Settings are loaded in this priority order:
    1. Explicitly passed values
    2. Environment variables (prefixed with DEEPCONTEXT_)
    3. .env file
    4. Defaults
    """

    model_config = SettingsConfigDict(
        env_prefix="DEEPCONTEXT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Database ---
    database_url: str = Field(
        default="",
        description="PostgreSQL connection URL (asyncpg). Example: postgresql+asyncpg://user:pass@host/db",
    )

    # --- LLM Provider ---
    llm_provider: LLMProvider = Field(default=LLMProvider.OPENAI)
    openai_api_key: Optional[str] = Field(default=None)
    openrouter_api_key: Optional[str] = Field(default=None)

    # --- Models ---
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="Model for fact extraction and classification",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Model for generating embeddings",
    )
    embedding_dimensions: int = Field(
        default=1536,
        description="Dimensionality of embedding vectors",
    )

    # --- Memory Tuning ---
    consolidation_threshold: int = Field(
        default=20,
        description="Number of short-term memories before triggering consolidation",
    )
    decay_half_life_days: float = Field(
        default=7.0,
        description="Half-life in days for episodic memory decay (Ebbinghaus curve)",
    )
    connection_similarity_threshold: float = Field(
        default=0.6,
        description="Minimum cosine similarity to create a memory connection",
    )
    max_connections_per_memory: int = Field(
        default=5,
        description="Maximum number of connections per memory node",
    )

    # --- Behavior ---
    debug: bool = Field(default=False)
    auto_consolidate: bool = Field(
        default=True,
        description="Automatically consolidate short-term to long-term memories",
    )

    @model_validator(mode="after")
    def _validate_api_keys(self) -> "DeepContextSettings":
        """Ensure at least one API key is configured."""
        if self.llm_provider == LLMProvider.OPENROUTER:
            if not self.openrouter_api_key:
                # Try bare env var as fallback
                self.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
                if not self.openrouter_api_key:
                    raise ValueError(
                        "openrouter_api_key is required when llm_provider='openrouter'. "
                        "Set DEEPCONTEXT_OPENROUTER_API_KEY or OPENROUTER_API_KEY."
                    )
        else:
            if not self.openai_api_key:
                self.openai_api_key = os.environ.get("OPENAI_API_KEY")
                if not self.openai_api_key:
                    raise ValueError(
                        "openai_api_key is required. "
                        "Set DEEPCONTEXT_OPENAI_API_KEY or OPENAI_API_KEY."
                    )
        return self

    @model_validator(mode="after")
    def _validate_database_url(self) -> "DeepContextSettings":
        """Ensure database URL is set, with fallback to env."""
        if not self.database_url:
            self.database_url = os.environ.get("DATABASE_URL", "")
        if not self.database_url:
            # Default to local SQLite for development (sync only)
            db_dir = os.path.expanduser("~/.deepcontext")
            os.makedirs(db_dir, exist_ok=True)
            self.database_url = f"sqlite+aiosqlite:///{db_dir}/memory.db"
        return self

    @property
    def llm_api_key(self) -> str:
        """Get the API key for the configured LLM provider."""
        if self.llm_provider == LLMProvider.OPENROUTER:
            return self.openrouter_api_key or ""
        return self.openai_api_key or ""

    @property
    def llm_base_url(self) -> Optional[str]:
        """Get the base URL for the configured LLM provider."""
        if self.llm_provider == LLMProvider.OPENROUTER:
            return "https://openrouter.ai/api/v1"
        return None

    @property
    def embedding_api_key(self) -> str:
        """Embedding API key. Uses OpenAI key even with OpenRouter for embeddings."""
        return self.openai_api_key or self.openrouter_api_key or ""
