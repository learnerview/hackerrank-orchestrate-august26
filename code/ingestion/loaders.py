"""Typed CSV loaders for every participant-facing dataset file.

One loader function per CSV, each returning a list of typed models from
``models``. Schema (header) validation and cell coercion live in
``ingestion.parsing``; these loaders only map rows onto models. Empty fields
are handled explicitly:

* optional ID/string fields (``group_id``, ``business_id``, ``sender_user_id``,
  ``media_id``, ...) become ``None`` when the CSV cell is empty;
* ``message_text`` is kept as an empty string for voice notes (that is the
  documented dataset contract, not a missing value);
* numeric and boolean fields raise ``ValueError`` when they are empty or
  unparseable rather than silently coercing to 0/False.

Header names are validated against the expected schema and raise on mismatch
so schema drift is caught loudly instead of misaligning columns silently.
"""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from models import (
    BusinessContext,
    DailyNotificationSummary,
    GroupContext,
    GroupMemberContext,
    HistoricalMessage,
    ImageRecord,
    IncomingMessage,
    MessageEvent,
    SampleMessage,
    UserBusinessHistory,
    UserProfile,
    VoiceNoteRecord,
)
from .parsing import bool_field, float_field, int_field, normalize_domain, opt_str, req_str

MESSAGES_HEADERS = [
    "message_id",
    "user_id",
    "conversation_type",
    "group_id",
    "business_id",
    "sender_user_id",
    "created_at",
    "message_text",
    "media_type",
    "media_id",
    "forwarded_count",
]
USERS_HEADERS = [
    "user_id",
    "do_not_disturb_window",
    "messages_opened_30d",
    "messages_replied_30d",
    "notifications_dismissed_30d",
    "messages_reported_30d",
]
GROUPS_HEADERS = [
    "group_id",
    "group_name",
    "group_type",
    "member_count",
    "admin_count",
    "created_at",
    "messages_30d",
]
GROUP_MEMBERS_HEADERS = [
    "group_id",
    "user_id",
    "role",
    "joined_at",
    "messages_sent_30d",
    "messages_read_30d",
    "replies_sent_30d",
    "notifications_dismissed_30d",
    "group_muted_by_user",
]
BUSINESS_ACCOUNTS_HEADERS = [
    "business_id",
    "display_name",
    "brand_name",
    "category",
    "verified",
    "official_domain",
    "domain_used_by_sender",
    "account_age_days",
    "messages_sent_30d",
    "user_reports_30d",
    "domain_used_by_sender_age_days",
]
USER_BUSINESS_HISTORY_HEADERS = [
    "user_id",
    "business_id",
    "why_user_knows_account",
    "last_activity_at",
    "allows_promotions",
    "promotions_opted_out_at",
    "activity_count_180d",
    "messages_opened_30d",
    "messages_dismissed_30d",
    "messages_replied_30d",
    "last_reply_at",
]
MESSAGE_EVENTS_HEADERS = [
    "user_id",
    "message_id",
    "message_opened",
    "message_replied",
    "reaction_time_minutes",
    "notification_dismissed",
    "muted_after_message",
    "message_reported",
]
IMAGES_HEADERS = ["image_id", "file_path"]
VOICE_NOTES_HEADERS = ["voice_note_id", "file_path"]
DAILY_SUMMARY_HEADERS = [
    "user_id",
    "date",
    "notifications_sent",
    "notifications_dismissed",
]
SAMPLE_MESSAGES_HEADERS = MESSAGES_HEADERS + [
    "action",
    "message_type",
    "reason",
    "confidence",
    "evidence_message_ids",
]


def _read_rows(path: str, expected_headers: list[str]) -> list[dict[str, str]]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        actual_headers = [h.strip() for h in (reader.fieldnames or [])]
        if actual_headers != expected_headers:
            raise ValueError(
                f"Schema mismatch in {path}: expected {expected_headers}, got {actual_headers}"
            )
        return [dict(row) for row in reader]


def _parse_message_row(row: dict[str, str], path: str, row_index: int) -> IncomingMessage:
    return IncomingMessage(
        message_id=req_str(row["message_id"], "message_id", path, row_index),
        user_id=req_str(row["user_id"], "user_id", path, row_index),
        conversation_type=req_str(row["conversation_type"], "conversation_type", path, row_index),
        group_id=opt_str(row.get("group_id", "")),
        business_id=opt_str(row.get("business_id", "")),
        sender_user_id=opt_str(row.get("sender_user_id", "")),
        created_at=req_str(row["created_at"], "created_at", path, row_index),
        message_text=row.get("message_text", ""),
        media_type=row.get("media_type", ""),
        media_id=opt_str(row.get("media_id", "")),
        forwarded_count=int_field(row.get("forwarded_count", ""), "forwarded_count", path, row_index),
    )


def load_messages(path: str) -> list[IncomingMessage]:
    """Load dataset/messages.csv — the messages that must be routed."""
    return [
        _parse_message_row(row, path, i) for i, row in enumerate(_read_rows(path, MESSAGES_HEADERS), start=2)
    ]


def load_message_history(path: str) -> list[HistoricalMessage]:
    """Load dataset/message_history.csv as historical context (never routed)."""
    return [
        HistoricalMessage(**asdict(_parse_message_row(row, path, i)))
        for i, row in enumerate(_read_rows(path, MESSAGES_HEADERS), start=2)
    ]


def load_sample_messages(path: str) -> list[SampleMessage]:
    """Load dataset/sample_messages.csv for few-shot examples and offline eval only."""
    rows = _read_rows(path, SAMPLE_MESSAGES_HEADERS)
    result: list[SampleMessage] = []
    for i, row in enumerate(rows, start=2):
        base = _parse_message_row(row, path, i)
        result.append(
            SampleMessage(
                **asdict(base),
                action=req_str(row["action"], "action", path, i),
                message_type=req_str(row["message_type"], "message_type", path, i),
                reason=req_str(row["reason"], "reason", path, i),
                confidence=float_field(row["confidence"], "confidence", path, i),
                evidence_message_ids=row.get("evidence_message_ids", "none") or "none",
            )
        )
    return result


def load_users(path: str) -> list[UserProfile]:
    """Load dataset/users.csv."""
    return [
        UserProfile(
            user_id=req_str(row["user_id"], "user_id", path, i),
            do_not_disturb_window=opt_str(row.get("do_not_disturb_window", "")),
            messages_opened_30d=int_field(row.get("messages_opened_30d", ""), "messages_opened_30d", path, i),
            messages_replied_30d=int_field(row.get("messages_replied_30d", ""), "messages_replied_30d", path, i),
            notifications_dismissed_30d=int_field(
                row.get("notifications_dismissed_30d", ""), "notifications_dismissed_30d", path, i
            ),
            messages_reported_30d=int_field(row.get("messages_reported_30d", ""), "messages_reported_30d", path, i),
        )
        for i, row in enumerate(_read_rows(path, USERS_HEADERS), start=2)
    ]


def load_groups(path: str) -> list[GroupContext]:
    """Load dataset/groups.csv."""
    return [
        GroupContext(
            group_id=req_str(row["group_id"], "group_id", path, i),
            group_name=req_str(row["group_name"], "group_name", path, i),
            group_type=req_str(row["group_type"], "group_type", path, i),
            member_count=int_field(row.get("member_count", ""), "member_count", path, i),
            admin_count=int_field(row.get("admin_count", ""), "admin_count", path, i),
            created_at=req_str(row["created_at"], "created_at", path, i),
            messages_30d=int_field(row.get("messages_30d", ""), "messages_30d", path, i),
        )
        for i, row in enumerate(_read_rows(path, GROUPS_HEADERS), start=2)
    ]


def load_group_members(path: str) -> list[GroupMemberContext]:
    """Load dataset/group_members.csv."""
    return [
        GroupMemberContext(
            group_id=req_str(row["group_id"], "group_id", path, i),
            user_id=req_str(row["user_id"], "user_id", path, i),
            role=opt_str(row.get("role", "")),
            joined_at=req_str(row["joined_at"], "joined_at", path, i),
            messages_sent_30d=int_field(row.get("messages_sent_30d", ""), "messages_sent_30d", path, i),
            messages_read_30d=int_field(row.get("messages_read_30d", ""), "messages_read_30d", path, i),
            replies_sent_30d=int_field(row.get("replies_sent_30d", ""), "replies_sent_30d", path, i),
            notifications_dismissed_30d=int_field(
                row.get("notifications_dismissed_30d", ""), "notifications_dismissed_30d", path, i
            ),
            group_muted_by_user=bool_field(row.get("group_muted_by_user", ""), "group_muted_by_user", path, i),
        )
        for i, row in enumerate(_read_rows(path, GROUP_MEMBERS_HEADERS), start=2)
    ]


def load_business_accounts(path: str) -> list[BusinessContext]:
    """Load dataset/business_accounts.csv."""
    return [
        BusinessContext(
            business_id=req_str(row["business_id"], "business_id", path, i),
            display_name=req_str(row["display_name"], "display_name", path, i),
            brand_name=req_str(row["brand_name"], "brand_name", path, i),
            category=req_str(row["category"], "category", path, i),
            verified=bool_field(row.get("verified", ""), "verified", path, i),
            official_domain=normalize_domain(row.get("official_domain", "")),
            domain_used_by_sender=normalize_domain(row.get("domain_used_by_sender", "")),
            account_age_days=int_field(row.get("account_age_days", ""), "account_age_days", path, i),
            messages_sent_30d=int_field(row.get("messages_sent_30d", ""), "messages_sent_30d", path, i),
            user_reports_30d=int_field(row.get("user_reports_30d", ""), "user_reports_30d", path, i),
            domain_used_by_sender_age_days=int_field(
                row.get("domain_used_by_sender_age_days", ""), "domain_used_by_sender_age_days", path, i
            ),
        )
        for i, row in enumerate(_read_rows(path, BUSINESS_ACCOUNTS_HEADERS), start=2)
    ]


def load_user_business_history(path: str) -> list[UserBusinessHistory]:
    """Load dataset/user_business_history.csv."""
    return [
        UserBusinessHistory(
            user_id=req_str(row["user_id"], "user_id", path, i),
            business_id=req_str(row["business_id"], "business_id", path, i),
            why_user_knows_account=req_str(row["why_user_knows_account"], "why_user_knows_account", path, i),
            last_activity_at=opt_str(row.get("last_activity_at", "")),
            allows_promotions=bool_field(row.get("allows_promotions", ""), "allows_promotions", path, i),
            promotions_opted_out_at=opt_str(row.get("promotions_opted_out_at", "")),
            activity_count_180d=int_field(row.get("activity_count_180d", ""), "activity_count_180d", path, i),
            messages_opened_30d=int_field(row.get("messages_opened_30d", ""), "messages_opened_30d", path, i),
            messages_dismissed_30d=int_field(
                row.get("messages_dismissed_30d", ""), "messages_dismissed_30d", path, i
            ),
            messages_replied_30d=int_field(row.get("messages_replied_30d", ""), "messages_replied_30d", path, i),
            last_reply_at=opt_str(row.get("last_reply_at", "")),
        )
        for i, row in enumerate(_read_rows(path, USER_BUSINESS_HISTORY_HEADERS), start=2)
    ]


def load_message_events(path: str) -> list[MessageEvent]:
    """Load dataset/message_events.csv — the engagement reward proxy."""
    result: list[MessageEvent] = []
    for i, row in enumerate(_read_rows(path, MESSAGE_EVENTS_HEADERS), start=2):
        reaction = row.get("reaction_time_minutes", "")
        result.append(
            MessageEvent(
                user_id=req_str(row["user_id"], "user_id", path, i),
                message_id=req_str(row["message_id"], "message_id", path, i),
                message_opened=bool_field(row.get("message_opened", ""), "message_opened", path, i),
                message_replied=bool_field(row.get("message_replied", ""), "message_replied", path, i),
                reaction_time_minutes=int_field(reaction, "reaction_time_minutes", path, i, default=None),
                notification_dismissed=bool_field(
                    row.get("notification_dismissed", ""), "notification_dismissed", path, i
                ),
                muted_after_message=bool_field(row.get("muted_after_message", ""), "muted_after_message", path, i),
                message_reported=bool_field(row.get("message_reported", ""), "message_reported", path, i),
            )
        )
    return result


def load_images(path: str) -> list[ImageRecord]:
    """Load dataset/images.csv."""
    return [
        ImageRecord(
            image_id=req_str(row["image_id"], "image_id", path, i),
            file_path=req_str(row["file_path"], "file_path", path, i),
        )
        for i, row in enumerate(_read_rows(path, IMAGES_HEADERS), start=2)
    ]


def load_voice_notes(path: str) -> list[VoiceNoteRecord]:
    """Load dataset/voice_notes.csv."""
    return [
        VoiceNoteRecord(
            voice_note_id=req_str(row["voice_note_id"], "voice_note_id", path, i),
            file_path=req_str(row["file_path"], "file_path", path, i),
        )
        for i, row in enumerate(_read_rows(path, VOICE_NOTES_HEADERS), start=2)
    ]


def load_daily_notification_summary(path: str) -> list[DailyNotificationSummary]:
    """Load dataset/daily_notification_summary.csv."""
    return [
        DailyNotificationSummary(
            user_id=req_str(row["user_id"], "user_id", path, i),
            date=req_str(row["date"], "date", path, i),
            notifications_sent=int_field(row.get("notifications_sent", ""), "notifications_sent", path, i),
            notifications_dismissed=int_field(
                row.get("notifications_dismissed", ""), "notifications_dismissed", path, i
            ),
        )
        for i, row in enumerate(_read_rows(path, DAILY_SUMMARY_HEADERS), start=2)
    ]
