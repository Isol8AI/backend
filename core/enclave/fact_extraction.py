"""Enclave-side fact extraction using pattern matching.

This module extracts temporal facts from conversation turns.
Uses regex patterns rather than LLM to avoid latency.
Runs inside the enclave where plaintext is available.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class ExtractedFact:
    """A fact extracted from conversation."""
    subject: str
    predicate: str
    object: str
    confidence: float
    type: str
    source: str = "system"
    entities: List[str] = field(default_factory=list)


# Predicate to fact type mapping
PREDICATE_TO_TYPE = {
    "prefers": "preference",
    "works_at": "identity",
    "located_in": "identity",
    "interested_in": "preference",
    "has_skill": "identity",
    "dislikes": "preference",
    "plans_to": "plan",
    "uses": "preference",
    "knows": "observation",
    "mentioned": "observation",
}


class FactExtractor:
    """Extract facts from conversation using pattern matching.

    Uses regex patterns to identify common fact patterns.
    Fast and reliable, no ML model required.
    """

    def __init__(self):
        """Initialize the fact extractor with patterns."""
        self.patterns = self._build_patterns()

    def _build_patterns(self) -> List[tuple]:
        """Build regex patterns for fact extraction."""
        return [
            # Preferences: "I prefer/like/love/enjoy X"
            (
                r"\bi\s+(?:prefer|like|love|enjoy|use)\s+(\w+(?:\s+\w+)?)",
                "prefers",
                0.7,
            ),
            # Favorites: "my favorite X is Y"
            (
                r"\bmy\s+favorite\s+(?:\w+\s+)?is\s+(\w+(?:\s+\w+)?)",
                "prefers",
                0.8,
            ),
            # Work: "I work at/for X"
            (
                r"\bi\s+work\s+(?:at|for)\s+(\w+(?:\s+\w+)?(?:\s+\w+)?)",
                "works_at",
                0.8,
            ),
            # Job title: "I am a/an X at Y" or "I'm a/an X"
            (
                r"\bi(?:'m|\s+am)\s+(?:a|an)\s+(\w+(?:\s+\w+)?)",
                "works_at",
                0.6,
            ),
            # Location: "I live in X" or "I'm from X" or "I'm in X"
            (
                r"\bi\s+(?:live|am|come)\s+(?:in|from|at)\s+(\w+(?:\s+\w+)?)",
                "located_in",
                0.7,
            ),
            (
                r"\bi'm\s+(?:in|from|at)\s+(\w+(?:\s+\w+)?)",
                "located_in",
                0.6,
            ),
            # Interests: "I'm interested in X"
            (
                r"\bi(?:'m|\s+am)\s+interested\s+in\s+(\w+(?:\s+\w+)?)",
                "interested_in",
                0.7,
            ),
            # Skills: "I know X" or "I can X"
            (
                r"\bi\s+(?:know|can)\s+(\w+(?:\s+\w+)?)",
                "has_skill",
                0.5,
            ),
            # Dislikes: "I don't like X" or "I hate X"
            (
                r"\bi\s+(?:don't\s+like|hate|dislike)\s+(\w+(?:\s+\w+)?)",
                "dislikes",
                0.7,
            ),
            # Plans: "I plan to X" or "I'm going to X" or "I want to X"
            (
                r"\bi\s+(?:plan\s+to|'m\s+going\s+to|want\s+to)\s+(\w+(?:\s+\w+)?(?:\s+\w+)?)",
                "plans_to",
                0.6,
            ),
        ]

    def _extract_entities(self, object_text: str, predicate: str) -> List[str]:
        """Extract entity tags from the object and predicate."""
        entities = []
        normalized = object_text.lower().strip()

        if len(normalized) > 2:
            entities.append(normalized)

        entities.append(predicate)

        # Split multi-word objects
        words = normalized.split()
        for word in words:
            if len(word) > 3 and word not in entities:
                entities.append(word)

        return entities

    def extract(
        self,
        user_message: str,
        assistant_response: str,
    ) -> List[ExtractedFact]:
        """Extract facts from a conversation turn.

        Args:
            user_message: The user's message
            assistant_response: The assistant's response

        Returns:
            List of extracted facts
        """
        facts = []
        text = f"{user_message} {assistant_response}".lower()
        seen = set()

        for pattern, predicate, base_confidence in self.patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                object_text = match.group(1).strip()

                # Skip common words that aren't real objects
                skip_words = {"a", "the", "an", "am", "is", "are", "to", "it"}
                if object_text.lower() in skip_words:
                    continue

                # Skip if we've seen this exact fact
                key = f"user:{predicate}:{object_text.lower()}"
                if key in seen:
                    continue
                seen.add(key)

                fact_type = PREDICATE_TO_TYPE.get(predicate, "observation")
                entities = self._extract_entities(object_text, predicate)

                facts.append(ExtractedFact(
                    subject="user",
                    predicate=predicate,
                    object=object_text,
                    confidence=base_confidence,
                    type=fact_type,
                    source="system",
                    entities=entities,
                ))

        logger.debug(f"[FactExtraction] Extracted {len(facts)} facts")
        return facts
