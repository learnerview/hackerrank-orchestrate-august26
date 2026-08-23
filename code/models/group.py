"""Pure data models for groups and user-group membership."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True, slots=True)
class GroupContext:
    """One row from dataset/groups.csv."""

    group_id: str
    group_name: str
    group_type: str
    member_count: int
    admin_count: int
    created_at: str
    messages_30d: int


@dataclass(frozen=True, slots=True)
class GroupMemberContext:
    """One row from dataset/group_members.csv — a user's relationship to a group."""

    group_id: str
    user_id: str
    role: Optional[str]
    joined_at: str
    messages_sent_30d: int
    messages_read_30d: int
    replies_sent_30d: int
    notifications_dismissed_30d: int
    group_muted_by_user: bool
