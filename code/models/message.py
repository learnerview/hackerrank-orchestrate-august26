"""Pure data models for messages.

No behavior lives here: these dataclasses only carry field values that were
explicitly parsed by ingestion.loaders. Optional fields are ``None`` when the
source CSV had an empty value; nothing is silently coerced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True, slots=True)
class IncomingMessage:
    """One row from dataset/messages.csv — a message to be routed."""

    message_id: str
    user_id: str
    conversation_type: str
    group_id: Optional[str]
    business_id: Optional[str]
    sender_user_id: Optional[str]
    created_at: str
    message_text: str
    media_type: str
    media_id: Optional[str]
    forwarded_count: int


@dataclass(frozen=True, slots=True)
class HistoricalMessage(IncomingMessage):
    """One row from dataset/message_history.csv — past context, not routed."""

    pass


@dataclass(frozen=True, slots=True)
class SampleMessage(IncomingMessage):
    """One row from dataset/sample_messages.csv — solved example for few-shot
    learning and offline evaluation only. Never used as inference input."""

    action: str
    message_type: str
    reason: str
    confidence: float
    evidence_message_ids: str


@dataclass(frozen=True, slots=True)
class RoutedMessage:
    """An incoming message with media-derived content merged in."""

    message: IncomingMessage
    effective_text: str
    media_summary: Optional[str]
    #: vision classification of image posters (promotional_poster,
    #: greeting_card, ...) when media analysis succeeded; None otherwise.
    media_intent_hint: Optional[str] = None


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    """A reference to one historical evidence message with its relevance score."""

    message_id: str
    score: float
