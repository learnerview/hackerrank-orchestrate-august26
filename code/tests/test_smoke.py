"""Smoke test: load every dataset file through the typed loaders, build the
repository, and verify row counts plus a few real joins (message -> user,
message -> group/member, message -> business). Run with: pytest -s tests/
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion import (  # noqa: E402
    load_business_accounts,
    load_daily_notification_summary,
    load_group_members,
    load_groups,
    load_images,
    load_message_events,
    load_message_history,
    load_messages,
    load_sample_messages,
    load_user_business_history,
    load_users,
    load_voice_notes,
)
from ingestion.repository import InMemoryDataRepository  # noqa: E402

DATASET = Path(__file__).resolve().parent.parent.parent / "dataset"


def load_all():
    return {
        "messages": load_messages(DATASET / "messages.csv"),
        "history": load_message_history(DATASET / "message_history.csv"),
        "samples": load_sample_messages(DATASET / "sample_messages.csv"),
        "users": load_users(DATASET / "users.csv"),
        "groups": load_groups(DATASET / "groups.csv"),
        "group_members": load_group_members(DATASET / "group_members.csv"),
        "businesses": load_business_accounts(DATASET / "business_accounts.csv"),
        "user_business_history": load_user_business_history(DATASET / "user_business_history.csv"),
        "events": load_message_events(DATASET / "message_events.csv"),
        "images": load_images(DATASET / "images.csv"),
        "voice_notes": load_voice_notes(DATASET / "voice_notes.csv"),
        "daily_summaries": load_daily_notification_summary(DATASET / "daily_notification_summary.csv"),
    }


def test_smoke_row_counts_and_joins(capsys=None):
    data = load_all()
    repo = InMemoryDataRepository.from_loaded(**data)

    print("\n=== ROW COUNTS ===")
    for name, rows in data.items():
        print(f"  {name}: {len(rows)}")

    expected = {
        "messages": 110,
        "history": 412,
        "samples": 30,
        "users": 54,
        "groups": 23,
        "group_members": 401,
        "businesses": 110,
        "user_business_history": 106,
        "events": 412,
        "images": 20,
        "voice_notes": 13,
        "daily_summaries": 756,
    }
    for name, count in expected.items():
        assert len(data[name]) == count, f"{name}: expected {count}, got {len(data[name])}"

    # every message must resolve to its user (and business/group where set)
    unresolved_user = [m.message_id for m in data["messages"] if repo.get_user(m.user_id) is None]
    unresolved_group = [
        m.message_id for m in data["messages"] if m.group_id and repo.get_group(m.group_id) is None
    ]
    unresolved_biz = [
        m.message_id for m in data["messages"] if m.business_id and repo.get_business(m.business_id) is None
    ]
    assert not unresolved_user, unresolved_user
    assert not unresolved_group, unresolved_group
    assert not unresolved_biz, unresolved_biz

    # every history message must have an event row (engagement proxy joins)
    events_by_msg = {e.message_id for e in data["events"]}
    missing_events = [h.message_id for h in data["history"] if h.message_id not in events_by_msg]
    assert not missing_events, missing_events[:5]

    # every media_id on a message must resolve to an image or voice record
    img_ids = {i.image_id for i in data["images"]}
    vn_ids = {v.voice_note_id for v in data["voice_notes"]}
    for m in data["messages"]:
        if m.media_type == "image":
            assert m.media_id in img_ids, f"image {m.media_id} not found"
        elif m.media_type == "voice":
            assert m.media_id in vn_ids, f"voice {m.media_id} not found"

    print("\n=== JOIN EXAMPLES ===")
    msgs = data["messages"]

    # 1) business message -> user + business
    b_msg = next(m for m in msgs if m.conversation_type == "business")
    user = repo.get_user(b_msg.user_id)
    biz = repo.get_business(b_msg.business_id)
    bh = repo.get_user_business_history(b_msg.user_id, b_msg.business_id)
    print(f"\n[1] business msg {b_msg.message_id}")
    print(f"    text: {b_msg.message_text[:80]!r}...")
    print(f"    -> user {user.user_id} dnd={user.do_not_disturb_window} "
          f"opened30d={user.messages_opened_30d}")
    print(f"    -> business {biz.business_id} {biz.brand_name} verified={biz.verified} "
          f"reports30d={biz.user_reports_30d}")
    print(f"    -> user-business history: {bh.why_user_knows_account} "
          f"allows_promotions={bh.allows_promotions}")
    assert user and biz and bh

    # 2) group message -> user + group + membership
    g_msg = next(m for m in msgs if m.conversation_type == "group")
    g = repo.get_group(g_msg.group_id)
    mem = repo.get_group_member(g_msg.user_id, g_msg.group_id)
    print(f"\n[2] group msg {g_msg.message_id} (sender={g_msg.sender_user_id})")
    print(f"    text: {g_msg.message_text[:80]!r}...")
    print(f"    -> group {g.group_id} {g.group_name} type={g.group_type} "
          f"members={g.member_count} msgs30d={g.messages_30d}")
    print(f"    -> membership role={mem.role} muted={mem.group_muted_by_user} "
          f"read30d={mem.messages_read_30d}")
    print(f"    -> group messages in dataset: {len(repo.messages_for_group(g.group_id))}, "
          f"history: {len(repo.history_for_group(g.group_id))}")
    assert g and mem

    # 3) personal message -> user + sender's history evidence
    p_msg = next(m for m in msgs if m.conversation_type == "personal")
    sender_history = repo.history_for_sender(p_msg.sender_user_id)
    sender_events = [repo.event_for_message(h.message_id) for h in sender_history]
    opened = sum(1 for e in sender_events if e and e.message_opened)
    print(f"\n[3] personal msg {p_msg.message_id} from sender={p_msg.sender_user_id}")
    print(f"    text: {p_msg.message_text[:80]!r}...")
    print(f"    -> sender history: {len(sender_history)} msgs, {opened} opened by receiver")
    assert repo.get_user(p_msg.user_id)
    if sender_history:
        ev = repo.event_for_message(sender_history[0].message_id)
        assert ev is not None

    # 4) voice message -> voice note record path
    v_msg = next(m for m in msgs if m.media_type == "voice")
    vn = repo.get_voice_note(v_msg.media_id)
    print(f"\n[4] voice msg {v_msg.message_id} -> {vn.voice_note_id} @ {vn.file_path}")
    assert vn

    print("\nSMOKE TEST OK")
