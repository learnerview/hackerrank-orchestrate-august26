"""Interfaces for the LLM routing stage.

The LLM is the third cascade stage: it is only called when the rule engine
returned no hit and the preference model was low-confidence, i.e. when the
message genuinely needs semantic reasoning over content + context. A
judgment carries the same fields as a RoutingDecision (validated against the
same allowed values) plus no message_id — the orchestrator binds it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from models.decision import ALLOWED_ACTIONS, ALLOWED_MESSAGE_TYPES, RoutingDecision
from models.message import IncomingMessage


@dataclass(frozen=True, slots=True)
class LLMJudgment:
    """A parsed, validated routing judgment returned by the LLM.

    ``confidence`` is coerced to [0, 1] and ``evidence_message_ids`` may be
    the literal string ``none`` or a semicolon-separated id list. Validation
    mirrors RoutingDecision so no invalid value can reach the orchestrator.
    """

    action: str
    message_type: str
    reason: str
    confidence: float
    evidence_message_ids: str = "none"

    def __post_init__(self) -> None:
        if self.action not in ALLOWED_ACTIONS:
            raise ValueError(f"Invalid action: {self.action!r}")
        if self.message_type not in ALLOWED_MESSAGE_TYPES:
            raise ValueError(f"Invalid message_type: {self.message_type!r}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
        if not self.reason or not isinstance(self.reason, str):
            raise ValueError("reason must be a non-empty string")

    def as_routing_decision(self, message_id: str) -> RoutingDecision:
        """Bind this judgment to a message for the orchestration layer."""
        return RoutingDecision(
            message_id=message_id,
            action=self.action,
            message_type=self.message_type,
            reason=self.reason,
            confidence=self.confidence,
            evidence_message_ids=self.evidence_message_ids,
            resolved_by="llm",
        )


class RoutingLLM(ABC):
    """Structured semantic router backed by an LLM (Groq chat completions)."""

    @abstractmethod
    def decide(self, message: IncomingMessage, context: dict[str, object]) -> Optional[LLMJudgment]:
        """Return a structured routing judgment, or None if the LLM cannot
        resolve (API failure or invalid output after retries).

        ``context`` may carry: ``message_text`` (media-merged effective text),
        ``media_summary``, ``context_summary`` (rendered user/group/business
        summary string), ``evidence_snippets`` (list of rendered historical
        evidence strings). Missing keys fall back to the message itself.
        """
        ...
