"""Pure data models for historical engagement events."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True, slots=True)
class MessageEvent:
    """One row from dataset/message_events.csv — how a user reacted to a
    historical message. Used as the reward proxy for preference modeling."""

    user_id: str
    message_id: str
    message_opened: bool
    message_replied: bool
    reaction_time_minutes: Optional[int]
    notification_dismissed: bool
    muted_after_message: bool
    message_reported: bool
