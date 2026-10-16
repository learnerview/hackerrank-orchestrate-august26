"""Unit test for the batch pipeline's per-message context assembly.

Regression: the group membership attached to a message context must be the
RECEIVING user's membership row for the message's group
(``get_group_member(message.user_id, message.group_id)``). That is what the
muted-group-mention personalization override relies on. A previous bug passed
the sender id where the group id was expected, so the lookup always missed and
muted-group escalation could never fire.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Settings  # noqa: E402
from ingestion.repository import InMemoryDataRepository  # noqa: E402
from models.group import GroupMemberContext  # noqa: E402
from models.message import IncomingMessage, RoutedMessage  # noqa: E402
from orchestration.pipeline import PipelineRunner  # noqa: E402


class _Retriever:
    def retrieve(self, message: IncomingMessage, text: str) -> list:
        return []


class _Features:
    def extract(self, message: IncomingMessage, text_hint=None, intent_hint=None) -> dict:
        return {}


def _repo_with_membership() -> InMemoryDataRepository:
    return InMemoryDataRepository.from_loaded(
        messages=[],
        history=[],
        samples=[],
        users=[],
        groups=[],
        group_members=[
            GroupMemberContext(
                group_id="g1",
                user_id="u_receiver",
                role="admin",
                joined_at="2026-01-01",
                messages_sent_30d=1,
                messages_read_30d=1,
                replies_sent_30d=0,
                notifications_dismissed_30d=0,
                group_muted_by_user=True,
            )
        ],
        businesses=[],
        user_business_history=[],
        events=[],
        images=[],
        voice_notes=[],
        daily_summaries=[],
    )


def _runner(repo: InMemoryDataRepository) -> PipelineRunner:
    return PipelineRunner(
        settings=Settings(),
        repo=repo,
        image_analyzer=None,
        audio_transcriber=None,
        retriever=_Retriever(),
        features=_Features(),
        orchestrator=None,
    )


def test_member_is_receivers_membership_for_the_group():
    repo = _repo_with_membership()
    runner = _runner(repo)
    message = IncomingMessage(
        message_id="m1",
        user_id="u_receiver",
        conversation_type="group",
        group_id="g1",
        business_id=None,
        sender_user_id="u_sender",
        created_at="2026-07-20T10:00:00Z",
        message_text="@u_receiver hello",
        media_type="none",
        media_id=None,
        forwarded_count=0,
    )
    routed = RoutedMessage(
        message=message, effective_text=message.message_text, media_summary=None
    )

    ctx = runner._build_context(message, routed)

    assert ctx.member is not None, "receiver's membership row must be found"
    assert ctx.member.user_id == "u_receiver"
    assert ctx.member.group_id == "g1"
    assert ctx.member.group_muted_by_user is True
    assert ctx.member.role == "admin"
