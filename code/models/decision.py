"""Pure data models for routing decisions.

A RoutingDecision is the single output object produced by the orchestration
layer and written by output.csv_writer. No behavior lives here beyond
validation of allowed values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

ALLOWED_ACTIONS = frozenset({"notify", "digest", "mute"})

ALLOWED_MESSAGE_TYPES = frozenset(
    {
        "personal",
        "urgent",
        "event",
        "payment",
        "business_update",
        "promotion",
        "greeting",
        "forward",
        "spam",
        "scam",
        "unknown",
    }
)


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """The final routing decision for one incoming message.

    ``evidence_message_ids`` is a semicolon-separated string of historical
    message IDs, or the literal string ``none``.
    ``resolved_by`` records which cascade stage produced the decision
    (``rules`` | ``preference_model`` | ``llm`` | ``personalization``) for
    cost/latency reporting; it is never written to output.csv.
    ``personalized`` records whether the personalization stage modified the
    decision; it is never written to output.csv either.
    """

    message_id: str
    action: str
    message_type: str
    reason: str
    confidence: float
    evidence_message_ids: str = "none"
    resolved_by: str = "unknown"
    personalized: bool = False

    def __post_init__(self) -> None:
        if self.action not in ALLOWED_ACTIONS:
            raise ValueError(f"Invalid action: {self.action!r}")
        if self.message_type not in ALLOWED_MESSAGE_TYPES:
            raise ValueError(f"Invalid message_type: {self.message_type!r}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")


@dataclass(frozen=True, slots=True)
class RuleHit:
    """One deterministic safety-rule firing.

    Safety rules always force ``mute`` with message_type ``scam`` or ``spam``
    before any probabilistic stage runs, so action is fixed at mute by the
    engine. confidence is a fixed per-rule constant (rules are deterministic;
    it expresses how sure the rule is by design, not a model estimate).
    """

    action: str
    message_type: str
    reason: str
    confidence: float

    def __post_init__(self) -> None:
        if self.action != "mute":
            raise ValueError(f"Safety rules can only force mute, got {self.action!r}")
        if self.message_type not in ALLOWED_MESSAGE_TYPES:
            raise ValueError(f"Invalid message_type: {self.message_type!r}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
