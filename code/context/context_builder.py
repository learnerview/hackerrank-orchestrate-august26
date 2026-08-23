"""Context assembly for a single routing decision.

MessageContext gathers everything a decision needs into one object:
the message with media-derived content merged, user/group/business metadata,
retrieved evidence, engagement features, and the routing-visible flags
(user profile + group membership for personalization). ``to_dict()`` is the
dict contract consumed by RoutingOrchestrator.route() and RoutingLLM.decide().
"""

from __future__ import annotations

from typing import Optional

from models.business import BusinessContext
from models.group import GroupContext, GroupMemberContext
from models.message import EvidenceRef, IncomingMessage, RoutedMessage
from models.user import UserProfile


class MessageContext:
    """Assembles the full context for a single routing decision.

    Combines the incoming message, user/group/business metadata,
    media-derived text, retrieved evidence, and engagement features
    into a single object consumed by the decision cascade.
    """

    def __init__(
        self,
        message: IncomingMessage,
        routed: RoutedMessage,
        user: Optional[UserProfile],
        group: Optional[GroupContext],
        member: Optional[GroupMemberContext],
        business: Optional[BusinessContext],
        evidence: list[EvidenceRef],
        engagement_features: dict[str, float],
        evidence_snippets: Optional[list[str]] = None,
        history_ids: Optional[set[str]] = None,
    ) -> None:
        self.message = message
        self.routed = routed
        self.user = user
        self.group = group
        self.member = member
        self.business = business
        self.evidence = list(evidence)
        self.engagement_features = dict(engagement_features)
        self.evidence_snippets = evidence_snippets
        self.history_ids = frozenset(history_ids) if history_ids else frozenset()

    # ------------------------------------------------------------------ #
    def _context_summary(self) -> str:
        lines: list[str] = []
        if self.user is not None:
            lines.append(
                f"user: {self.user.user_id}, dnd_window={self.user.do_not_disturb_window!r}, "
                f"opened_30d={self.user.messages_opened_30d}, "
                f"reported_30d={self.user.messages_reported_30d}"
            )
        if self.group is not None:
            lines.append(
                f"group: {self.group.group_name} ({self.group.group_type}, "
                f"{self.group.member_count} members)"
            )
        if self.member is not None:
            lines.append(
                f"sender role: {self.member.role}, muted_by_user={self.member.group_muted_by_user}, "
                f"msgs_30d={self.member.messages_sent_30d}"
            )
        if self.business is not None:
            lines.append(
                f"business: {self.business.display_name} [{self.business.category}] "
                f"verified={self.business.verified}, user_reports_30d={self.business.user_reports_30d}, "
                f"msgs_30d={self.business.messages_sent_30d}"
            )
        if self.engagement_features:
            f = self.engagement_features
            lines.append(
                f"engagement: open_rate={f.get('open_rate', 0):.2f} "
                f"reply_rate={f.get('reply_rate', 0):.2f} "
                f"dismiss_rate={f.get('dismiss_rate', 0):.2f} "
                f"volume={f.get('volume', 0):.0f} noise_ratio={f.get('noise_ratio', 0):.2f} "
                f"forwarded={f.get('forwarded_count', 0):.0f}"
            )
        return "\n".join(lines)

    def _snippets(self) -> list[str]:
        if self.evidence_snippets is not None:
            return list(self.evidence_snippets)
        return [f"[{ref.message_id}] (score {ref.score:.2f})" for ref in self.evidence]

    @staticmethod
    def source_id_of(message: IncomingMessage) -> Optional[str]:
        """The message's engagement source: sender, else group, else business."""
        return message.sender_user_id or message.group_id or message.business_id

    def to_dict(self) -> dict[str, object]:
        return {
            "message_text": self.routed.effective_text,
            "media_summary": self.routed.media_summary,
            "context_summary": self._context_summary(),
            "evidence_snippets": self._snippets(),
            "evidence_refs": list(self.evidence),
            "history_ids": self.history_ids,
            "features": self.engagement_features,
            "source_id": self.source_id_of(self.message),
            "user": self.user,
            "member": self.member,
        }
