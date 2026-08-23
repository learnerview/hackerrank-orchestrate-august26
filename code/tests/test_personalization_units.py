"""Unit tests for personalization overrides: quiet hours, muted-group mention
escalation, and the safety-rule-mute invariant.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.decision import RoutingDecision  # noqa: E402
from models.group import GroupMemberContext  # noqa: E402
from models.message import IncomingMessage  # noqa: E402
from models.user import UserProfile  # noqa: E402
from personalization.overrides import PersonalizationOverrides  # noqa: E402


def _decision(action="digest", message_type="personal", resolved_by="llm", reason="base") -> RoutingDecision:
    return RoutingDecision(
        message_id="msg_x",
        action=action,
        message_type=message_type,
        reason=reason,
        confidence=0.8,
        evidence_message_ids="none",
        resolved_by=resolved_by,
    )


def _msg(text="hi", hour="2026-07-20T23:00:00Z", user_id="u_001", group_id="g_1") -> IncomingMessage:
    return IncomingMessage(
        message_id="msg_x",
        user_id=user_id,
        conversation_type="group",
        group_id=group_id,
        business_id=None,
        sender_user_id="u_050",
        created_at=hour,
        message_text=text,
        media_type="none",
        media_id=None,
        forwarded_count=0,
    )


def _user(dnd="22:00-07:00") -> UserProfile:
    return UserProfile(
        user_id="u_001",
        do_not_disturb_window=dnd,
        messages_opened_30d=10,
        messages_replied_30d=4,
        notifications_dismissed_30d=2,
        messages_reported_30d=0,
    )


def _member(role="member", muted=True) -> GroupMemberContext:
    return GroupMemberContext(
        group_id="g_1",
        user_id="u_001",
        role=role,
        joined_at="2026-01-01",
        messages_sent_30d=3,
        messages_read_30d=20,
        replies_sent_30d=2,
        notifications_dismissed_30d=1,
        group_muted_by_user=muted,
    )


P = PersonalizationOverrides()


class TestQuietHours:
    def test_notify_downgraded_to_digest_inside_window(self) -> None:
        result = P.apply(_decision("notify"), _user(), None, _msg(hour="2026-07-20T23:30:00Z"))
        assert result.action == "digest"
        assert result.personalized is True
        assert "quiet-hours" in result.reason

    def test_notify_untouched_outside_window(self) -> None:
        result = P.apply(_decision("notify"), _user(), None, _msg(hour="2026-07-20T14:00:00Z"))
        assert result.action == "notify"
        assert result.personalized is False

    def test_wrap_around_window(self) -> None:
        result = P.apply(_decision("notify"), _user(), None, _msg(hour="2026-07-20T02:00:00Z"))
        assert result.action == "digest"

    def test_no_user_profile_means_no_downgrade(self) -> None:
        result = P.apply(_decision("notify"), None, None, _msg(hour="2026-07-20T23:00:00Z"))
        assert result.action == "notify"

    def test_non_notify_untouched(self) -> None:
        result = P.apply(_decision("digest"), _user(), None, _msg(hour="2026-07-20T23:00:00Z"))
        assert result.action == "digest"
        assert result.personalized is False


class TestMutedGroupMention:
    def test_muted_group_mention_raises_mute_to_digest(self) -> None:
        result = P.apply(_decision("mute", "spam"), _user(), _member(), _msg(text="@u_001 check this"))
        assert result.action == "digest"
        assert result.personalized is True
        assert "mentions user" in result.reason

    def test_muted_group_mention_urgency_keyword_raises_to_notify(self) -> None:
        result = P.apply(_decision("mute", "spam"), _user(), _member(), _msg(text="@u_001 deadline tonight"))
        assert result.action == "notify"
        assert result.message_type == "urgent"

    def test_muted_group_mention_admin_role_raises_to_notify(self) -> None:
        result = P.apply(
            _decision("mute", "spam"), _user(), _member(role="admin"), _msg(text="@u_001 gate closed")
        )
        assert result.action == "notify"

    def test_digest_with_urgency_escalates_to_notify(self) -> None:
        result = P.apply(_decision("digest"), _user(), _member(), _msg(text="@u_001 exam tomorrow"))
        assert result.action == "notify"
        assert result.message_type == "urgent"

    def test_digest_without_urgency_stays_digest(self) -> None:
        result = P.apply(_decision("digest"), _user(), _member(), _msg(text="@u_001 lunch plan"))
        assert result.action == "digest"
        assert result.personalized is False

    def test_notify_untouched(self) -> None:
        result = P.apply(
            _decision("notify"), _user(), _member(), _msg(text="@u_001 urgent thing", hour="2026-07-20T14:00:00Z")
        )
        assert result.action == "notify"
        assert result.personalized is False

    def test_no_mention_means_stays_muted(self) -> None:
        result = P.apply(_decision("mute", "spam"), _user(), _member(), _msg(text="forwarded chain"))
        assert result.action == "mute"
        assert result.personalized is False

    def test_unmuted_group_no_escalation(self) -> None:
        result = P.apply(
            _decision("mute", "spam"), _user(), _member(muted=False), _msg(text="@u_001 hi")
        )
        assert result.action == "mute"

    def test_mention_of_other_user_is_not_escalation(self) -> None:
        result = P.apply(_decision("mute", "spam"), _user(), _member(), _msg(text="@u_099 hello"))
        assert result.action == "mute"


class TestSafetyInvariant:
    def test_rule_scam_mute_never_touched(self) -> None:
        d = _decision("mute", "scam", resolved_by="rules")
        result = P.apply(d, _user(), _member(), _msg(text="@u_001 scam deadline"))
        assert result is d
        assert result.personalized is False

    def test_rule_spam_mute_never_touched_even_in_quiet_hours(self) -> None:
        d = _decision("mute", "spam", resolved_by="rules")
        result = P.apply(d, _user(), _member(), _msg(text="@u_001 hey"))
        assert result is d

    def test_quiet_hours_then_mention_escalation_ordering(self) -> None:
        result = P.apply(
            _decision("notify"), _user(), _member(), _msg(text="@u_001 exam at 8am", hour="2026-07-20T23:00:00Z")
        )
        assert result.action == "notify"  # quiet-hours downgrade then re-escalated by urgency
        assert result.personalized is True
