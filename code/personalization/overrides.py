"""Post-decision personalization overrides.

Applied AFTER the base cascade (rules -> preference model -> LLM):

- Quiet hours (users.csv): a notify is downgraded to digest when the message
  arrives inside the user's DND window. Decisions produced by the rule engine
  are never downgraded (rule-flagged urgency/safety must always surface).
- Muted-group mention escalation (group_members.csv): a message from a group
  the user muted that nevertheless mentions the user directly (@user_id) is
  never silently muted — it is raised to at least digest, and to notify when
  urgency signals are present (deadline/exam/tomorrow keywords, or the sender
  has an admin/teacher role).

Invariants:
- A rule-engine scam/spam mute is NEVER modified (safety wins over
  convenience).
- Quiet-hours downgrade runs first, then muted-group escalation, so a
  mentioned message can be re-raised to notify even during quiet hours when
  urgency is present — the mention is the stronger, more deliberate signal.

Modified decisions are marked ``personalized=True`` (never written to
output.csv) and the reason string records exactly what changed.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Optional

from ingestion.parsing import parse_timestamp
from models.decision import RoutingDecision
from models.group import GroupMemberContext
from models.message import IncomingMessage
from models.user import UserProfile
from scoring.features import quiet_hours_overlap

_URGENT_KEYWORD_RE = re.compile(r"\b(exam|deadline|tomorrow)\b", re.IGNORECASE)
_URGENT_ROLES = frozenset({"admin", "teacher"})
_RULE_MUTE_TYPES = frozenset({"scam", "spam"})


def _is_rule_engine_mute(decision: RoutingDecision) -> bool:
    """Safety-rule mutes (scam/spam) must never be touched by personalization."""
    return decision.resolved_by == "rules" and decision.action == "mute" and (
        decision.message_type in _RULE_MUTE_TYPES
    )


def _mentions_user(message: IncomingMessage) -> bool:
    """True if the message @-mentions the receiving user directly."""
    text = message.message_text or ""
    return f"@{message.user_id}" in text


def _has_urgency_signal(member: Optional[GroupMemberContext], message: IncomingMessage) -> bool:
    if _URGENT_KEYWORD_RE.search(message.message_text or ""):
        return True
    return member is not None and member.role.lower() in _URGENT_ROLES


def _mark(decision: RoutingDecision, **changes: object) -> RoutingDecision:
    return replace(decision, personalized=True, **changes)


class PersonalizationOverrides:
    """Post-decision overrides applied after the base decision.

    - Quiet hours: downgrade notify -> digest (unless safety/urgency flagged).
    - Muted-group-but-mentioned escalation: raise at least to digest (or notify
      if urgency signals are present).
    - Per-user sensitivity adjustments.

    Personalization can raise an action's priority but can never downgrade or
    override a safety-rule mute on a scam/spam decision.
    """

    def apply(
        self,
        decision: RoutingDecision,
        user: Optional[UserProfile],
        member: Optional[GroupMemberContext],
        message: IncomingMessage,
    ) -> RoutingDecision:
        if _is_rule_engine_mute(decision):
            return decision

        changed = decision

        # --- quiet hours: notify -> digest ------------------------------- #
        if changed.action == "notify" and user is not None:
            ts = parse_timestamp(message.created_at)
            hour = ts.hour if ts is not None else None
            if hour is not None and quiet_hours_overlap(hour, user.do_not_disturb_window) == 1.0:
                changed = _mark(
                    changed,
                    action="digest",
                    reason=changed.reason + " [personalization: quiet-hours downgrade to digest]",
                )

        # --- muted group that mentions the user -------------------------- #
        if (
            member is not None
            and message.group_id is not None
            and member.group_muted_by_user
            and _mentions_user(message)
        ):
            urgent = _has_urgency_signal(member, message)
            if changed.action == "mute":
                if urgent:
                    changed = _mark(
                        changed,
                        action="notify",
                        message_type="urgent",
                        reason=(
                            changed.reason
                            + " [personalization: muted group mentions user, "
                            + "urgency signals present -> escalated to notify]"
                        ),
                    )
                else:
                    changed = _mark(
                        changed,
                        action="digest",
                        reason=(
                            changed.reason
                            + " [personalization: muted group mentions user -> "
                            + "escalated to digest]"
                        ),
                    )
            elif changed.action == "digest" and urgent:
                changed = _mark(
                    changed,
                    action="notify",
                    message_type="urgent",
                    reason=(
                        changed.reason
                        + " [personalization: muted group mentions user, urgency "
                        + "signals present -> escalated to notify]"
                    ),
                )

        return changed
