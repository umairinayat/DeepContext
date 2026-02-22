"""
LLM-based extraction engine.

Uses structured output (instructor + Pydantic) for reliable JSON parsing.
Falls back to manual JSON parsing for providers that don't support tool calls.
"""

from __future__ import annotations

import json
import re
from typing import Any

from deepcontext.core.clients import LLMClients
from deepcontext.core.settings import DeepContextSettings
from deepcontext.core.types import (
    EntityType,
    ExtractionResult,
    ExtractedEntity,
    ExtractedFact,
    ExtractedRelationship,
    MemoryType,
    ToolAction,
    ToolDecision,
)
from deepcontext.extraction.prompts import (
    CONSOLIDATION_PROMPT,
    EXTRACTION_SYSTEM_PROMPT,
    SUMMARY_PROMPT,
    TOOL_CLASSIFIER_PROMPT,
)


class Extractor:
    """
    LLM-powered extraction engine for facts, entities, and relationships.

    Uses the configured LLM provider to analyze conversations and extract
    structured memory data.
    """

    def __init__(self, clients: LLMClients, settings: DeepContextSettings) -> None:
        self._clients = clients
        self._settings = settings

    async def extract_memories(
        self,
        latest_messages: list[str],
        summary: str,
        recent_messages: list[str],
    ) -> ExtractionResult:
        """
        Extract facts, entities, and relationships from the latest conversation turn.

        Args:
            latest_messages: The latest user-assistant message pair
            summary: Rolling conversation summary
            recent_messages: Recent message history for context

        Returns:
            ExtractionResult with semantic facts, episodic events, entities, relationships
        """
        recent_text = "\n".join(recent_messages)
        latest_text = "\n".join(latest_messages)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Conversation Summary:\n{summary}\n\n"
                    f"Recent Messages:\n{recent_text}\n\n"
                    f"Latest Interaction:\n{latest_text}\n\n"
                    "Extract memory facts, entities, and relationships."
                ),
            },
        ]

        response = await self._clients.llm.chat.completions.create(
            model=self._settings.llm_model,
            messages=messages,  # type: ignore[arg-type]
            temperature=0.1,
        )

        raw = response.choices[0].message.content or "{}"
        parsed = self._parse_json(raw)

        if self._settings.debug:
            print(f"[Extractor] Raw output: {raw[:300]}...")
            print(f"[Extractor] Parsed: {json.dumps(parsed, indent=2)[:300]}...")

        return self._build_extraction_result(parsed)

    async def classify_action(
        self,
        candidate_fact: str,
        similar_memories: list[dict[str, Any]],
    ) -> ToolDecision:
        """
        Have the LLM decide what to do with a candidate fact.

        Args:
            candidate_fact: The new fact to evaluate
            similar_memories: Existing memories that are similar

        Returns:
            ToolDecision with action, memory_id, and text
        """
        if similar_memories:
            memory_context = "\n".join(
                f"- ID {m['id']}: {m['text']}" for m in similar_memories
            )
        else:
            memory_context = "No existing memories found."

        messages = [
            {"role": "system", "content": TOOL_CLASSIFIER_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Candidate fact:\n{candidate_fact}\n\n"
                    f"Existing similar memories:\n{memory_context}\n\n"
                    "Decide the action."
                ),
            },
        ]

        response = await self._clients.llm.chat.completions.create(
            model=self._settings.llm_model,
            messages=messages,
            temperature=0,
        )

        raw = response.choices[0].message.content or "{}"
        parsed = self._parse_json(raw)

        action_str = parsed.get("action", "ADD").upper()
        try:
            action = ToolAction(action_str)
        except ValueError:
            action = ToolAction.ADD

        return ToolDecision(
            action=action,
            memory_id=parsed.get("memory_id"),
            text=parsed.get("text", candidate_fact),
            reason=parsed.get("reason"),
        )

    async def consolidate_memories(
        self,
        memory_texts: list[str],
    ) -> dict[str, Any]:
        """
        Consolidate multiple short-term memories into a single long-term fact.

        Args:
            memory_texts: List of memory texts to consolidate

        Returns:
            Dict with consolidated text, importance, confidence, entities
        """
        memories_str = "\n".join(f"- {t}" for t in memory_texts)

        messages = [
            {"role": "system", "content": CONSOLIDATION_PROMPT},
            {
                "role": "user",
                "content": f"Short-term memories to consolidate:\n{memories_str}",
            },
        ]

        response = await self._clients.llm.chat.completions.create(
            model=self._settings.llm_model,
            messages=messages,
            temperature=0.1,
        )

        raw = response.choices[0].message.content or "{}"
        return self._parse_json(raw)

    async def generate_summary(
        self,
        messages: list[str],
    ) -> str:
        """Generate a conversation summary."""
        conversation = "\n".join(messages)

        prompt_messages = [
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user", "content": f"Conversation:\n{conversation}"},
        ]

        response = await self._clients.llm.chat.completions.create(
            model=self._settings.llm_model,
            messages=prompt_messages,
            temperature=0.2,
        )

        return (response.choices[0].message.content or "").strip()

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        """
        Parse JSON from LLM output, handling markdown code blocks.

        LLMs often wrap JSON in ```json ... ``` blocks.
        """
        text = raw.strip()

        # Extract from markdown code blocks
        if "```" in text:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
            if match:
                text = match.group(1).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _build_extraction_result(parsed: dict[str, Any]) -> ExtractionResult:
        """Convert raw parsed dict to typed ExtractionResult."""
        semantic = []
        for item in parsed.get("semantic", []):
            if isinstance(item, str):
                semantic.append(ExtractedFact(text=item))
            elif isinstance(item, dict):
                semantic.append(
                    ExtractedFact(
                        text=item.get("text", ""),
                        memory_type=MemoryType.SEMANTIC,
                        importance=float(item.get("importance", 0.5)),
                        confidence=float(item.get("confidence", 0.8)),
                        entities=item.get("entities", []),
                    )
                )

        episodic = []
        for item in parsed.get("episodic", parsed.get("bubbles", [])):
            if isinstance(item, str):
                episodic.append(
                    ExtractedFact(text=item, memory_type=MemoryType.EPISODIC)
                )
            elif isinstance(item, dict):
                episodic.append(
                    ExtractedFact(
                        text=item.get("text", ""),
                        memory_type=MemoryType.EPISODIC,
                        importance=float(item.get("importance", 0.5)),
                        confidence=float(item.get("confidence", 0.8)),
                        entities=item.get("entities", []),
                    )
                )

        entities = []
        for item in parsed.get("entities", []):
            if isinstance(item, dict):
                # Validate entity_type against enum; fall back to "other"
                raw_type = item.get("entity_type", "other")
                try:
                    entity_type = EntityType(raw_type)
                except ValueError:
                    entity_type = EntityType.OTHER
                entities.append(
                    ExtractedEntity(
                        name=item.get("name", ""),
                        entity_type=entity_type,
                        attributes=item.get("attributes", {}),
                    )
                )

        relationships = []
        for item in parsed.get("relationships", []):
            if isinstance(item, dict):
                relationships.append(
                    ExtractedRelationship(
                        source=item.get("source", ""),
                        target=item.get("target", ""),
                        relation=item.get("relation", "related_to"),
                        properties=item.get("properties", {}),
                    )
                )

        return ExtractionResult(
            semantic=semantic,
            episodic=episodic,
            entities=entities,
            relationships=relationships,
        )
