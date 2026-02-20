"""
LLM and Embedding client management.

Async-first, with lazy initialization and provider abstraction.
"""

from __future__ import annotations

from typing import Optional

from openai import AsyncOpenAI

from deepcontext.core.settings import DeepContextSettings, LLMProvider


class LLMClients:
    """
    Manages async OpenAI-compatible clients for LLM and embedding calls.

    Usage:
        clients = LLMClients(settings)
        response = await clients.llm.chat.completions.create(...)
        embedding = await clients.embedding.embeddings.create(...)
    """

    def __init__(self, settings: DeepContextSettings) -> None:
        self._settings = settings
        self._llm_client: Optional[AsyncOpenAI] = None
        self._embedding_client: Optional[AsyncOpenAI] = None

    @property
    def llm(self) -> AsyncOpenAI:
        """Get or create the LLM client."""
        if self._llm_client is None:
            if self._settings.llm_provider == LLMProvider.OPENROUTER:
                self._llm_client = AsyncOpenAI(
                    api_key=self._settings.llm_api_key,
                    base_url="https://openrouter.ai/api/v1",
                    default_headers={
                        "HTTP-Referer": "https://github.com/deepcontext",
                        "X-Title": "DeepContext",
                    },
                )
            else:
                self._llm_client = AsyncOpenAI(api_key=self._settings.llm_api_key)
        return self._llm_client

    @property
    def embedding(self) -> AsyncOpenAI:
        """
        Get or create the embedding client.

        Embeddings always use OpenAI's API (even when LLM uses OpenRouter),
        unless only an OpenRouter key is available.
        """
        if self._embedding_client is None:
            if self._settings.openai_api_key:
                self._embedding_client = AsyncOpenAI(
                    api_key=self._settings.openai_api_key,
                )
            elif self._settings.openrouter_api_key:
                self._embedding_client = AsyncOpenAI(
                    api_key=self._settings.openrouter_api_key,
                    base_url="https://openrouter.ai/api/v1",
                    default_headers={
                        "HTTP-Referer": "https://github.com/deepcontext",
                        "X-Title": "DeepContext",
                    },
                )
            else:
                raise RuntimeError("No API key available for embeddings.")
        return self._embedding_client

    async def embed_text(self, text: str) -> list[float]:
        """Generate an embedding vector for the given text."""
        model = self._settings.embedding_model
        # OpenRouter needs provider prefix
        if self._settings.llm_provider == LLMProvider.OPENROUTER and not model.startswith("openai/"):
            if not self._settings.openai_api_key:
                model = f"openai/{model}"

        response = await self.embedding.embeddings.create(
            model=model,
            input=text,
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts in one API call."""
        if not texts:
            return []

        model = self._settings.embedding_model
        if self._settings.llm_provider == LLMProvider.OPENROUTER and not model.startswith("openai/"):
            if not self._settings.openai_api_key:
                model = f"openai/{model}"

        response = await self.embedding.embeddings.create(
            model=model,
            input=texts,
        )
        # Sort by index to maintain order
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]

    async def close(self) -> None:
        """Close all clients."""
        if self._llm_client:
            await self._llm_client.close()
            self._llm_client = None
        if self._embedding_client:
            await self._embedding_client.close()
            self._embedding_client = None
