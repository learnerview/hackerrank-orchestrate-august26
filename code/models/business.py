"""Pure data models for business accounts and user-business relationships."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True, slots=True)
class BusinessContext:
    """One row from dataset/business_accounts.csv."""

    business_id: str
    display_name: str
    brand_name: str
    category: str
    verified: bool
    official_domain: Optional[str]
    domain_used_by_sender: Optional[str]
    account_age_days: int
    messages_sent_30d: int
    user_reports_30d: int
    domain_used_by_sender_age_days: int


@dataclass(frozen=True, slots=True)
class UserBusinessHistory:
    """One row from dataset/user_business_history.csv."""

    user_id: str
    business_id: str
    why_user_knows_account: str
    last_activity_at: Optional[str]
    allows_promotions: bool
    promotions_opted_out_at: Optional[str]
    activity_count_180d: int
    messages_opened_30d: int
    messages_dismissed_30d: int
    messages_replied_30d: int
    last_reply_at: Optional[str]
