"""Pure data models for users and their notification behavior."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True, slots=True)
class UserProfile:
    """One row from dataset/users.csv."""

    user_id: str
    do_not_disturb_window: Optional[str]
    messages_opened_30d: int
    messages_replied_30d: int
    notifications_dismissed_30d: int
    messages_reported_30d: int


@dataclass(frozen=True, slots=True)
class DailyNotificationSummary:
    """One row from dataset/daily_notification_summary.csv."""

    user_id: str
    date: str
    notifications_sent: int
    notifications_dismissed: int


@dataclass(frozen=True, slots=True)
class UserPreferenceState:
    """Derived engagement state for a user, computed by scoring/features.

    Not loaded from any single CSV — populated downstream from message_events
    and history aggregates.
    """

    user_id: str
    open_rate: float = 0.0
    reply_rate: float = 0.0
    dismiss_rate: float = 0.0
    report_count: int = 0
    typical_engagement_window_start: Optional[str] = None
    typical_engagement_window_end: Optional[str] = None
    sensitivity: float = 1.0
