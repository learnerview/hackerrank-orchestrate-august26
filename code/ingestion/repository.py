"""In-memory DataRepository and its swappable Protocol.

The DataRepository Protocol is the single query surface every downstream layer
uses. It is defined as a structural Protocol so a database-backed repository
can replace the in-memory implementation without changing any caller.

This module owns indexing only: all data is loaded by ``ingestion.loaders``
and handed in via ``from_loaded``. No file I/O happens here.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional, Protocol, runtime_checkable

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


@runtime_checkable
class DataRepository(Protocol):
    """Query surface for all dataset context. Implementations may be in-memory
    (this package) or database-backed; callers only depend on this Protocol."""

    # --- routed messages (messages.csv) ---
    def get_message(self, message_id: str) -> Optional[IncomingMessage]: ...
    def all_messages(self) -> list[IncomingMessage]: ...
    def messages_for_user(self, user_id: str) -> list[IncomingMessage]: ...
    def messages_for_group(self, group_id: str) -> list[IncomingMessage]: ...
    def messages_for_business(self, business_id: str) -> list[IncomingMessage]: ...
    def messages_for_sender(self, sender_user_id: str) -> list[IncomingMessage]: ...

    # --- historical context (message_history.csv) ---
    def all_history(self) -> list[HistoricalMessage]: ...
    def history_for_user(self, user_id: str) -> list[HistoricalMessage]: ...
    def history_for_group(self, group_id: str) -> list[HistoricalMessage]: ...
    def history_for_business(self, business_id: str) -> list[HistoricalMessage]: ...
    def history_for_sender(self, sender_user_id: str) -> list[HistoricalMessage]: ...
    def history_by_id(self, message_id: str) -> Optional[HistoricalMessage]: ...

    # --- solved examples (sample_messages.csv) ---
    def all_sample_messages(self) -> list[SampleMessage]: ...
    def sample_message_by_id(self, message_id: str) -> Optional[SampleMessage]: ...

    # --- users ---
    def get_user(self, user_id: str) -> Optional[UserProfile]: ...
    def all_users(self) -> list[UserProfile]: ...

    # --- groups ---
    def get_group(self, group_id: str) -> Optional[GroupContext]: ...
    def all_groups(self) -> list[GroupContext]: ...
    def get_group_member(self, user_id: str, group_id: str) -> Optional[GroupMemberContext]: ...
    def members_of_group(self, group_id: str) -> list[GroupMemberContext]: ...
    def groups_for_user(self, user_id: str) -> list[GroupMemberContext]: ...

    # --- businesses ---
    def get_business(self, business_id: str) -> Optional[BusinessContext]: ...
    def all_businesses(self) -> list[BusinessContext]: ...
    def get_user_business_history(self, user_id: str, business_id: str) -> Optional[UserBusinessHistory]: ...
    def businesses_for_user(self, user_id: str) -> list[UserBusinessHistory]: ...

    # --- events ---
    def event_for_message(self, message_id: str) -> Optional[MessageEvent]: ...
    def events_for_user(self, user_id: str) -> list[MessageEvent]: ...
    def all_events(self) -> list[MessageEvent]: ...

    # --- media ---
    def get_image(self, image_id: str) -> Optional[ImageRecord]: ...
    def all_images(self) -> list[ImageRecord]: ...
    def get_voice_note(self, voice_note_id: str) -> Optional[VoiceNoteRecord]: ...
    def all_voice_notes(self) -> list[VoiceNoteRecord]: ...

    # --- daily notification summaries ---
    def summaries_for_user(self, user_id: str) -> list[DailyNotificationSummary]: ...


class InMemoryDataRepository:
    """Indexes every loaded dataset row in memory for O(1)-ish lookups.

    All indexes are built once in ``from_loaded``. Dict keys are the primary
    identifiers; multi-value relations use defaultdict(list) buckets keyed by
    user_id / group_id / business_id / sender_user_id.
    """

    def __init__(self) -> None:
        self._messages_by_id: dict[str, IncomingMessage] = {}
        self._messages_by_user: dict[str, list[IncomingMessage]] = defaultdict(list)
        self._messages_by_group: dict[str, list[IncomingMessage]] = defaultdict(list)
        self._messages_by_business: dict[str, list[IncomingMessage]] = defaultdict(list)
        self._messages_by_sender: dict[str, list[IncomingMessage]] = defaultdict(list)

        self._history_by_id: dict[str, HistoricalMessage] = {}
        self._history_by_user: dict[str, list[HistoricalMessage]] = defaultdict(list)
        self._history_by_group: dict[str, list[HistoricalMessage]] = defaultdict(list)
        self._history_by_business: dict[str, list[HistoricalMessage]] = defaultdict(list)
        self._history_by_sender: dict[str, list[HistoricalMessage]] = defaultdict(list)

        self._samples_by_id: dict[str, SampleMessage] = {}

        self._users_by_id: dict[str, UserProfile] = {}
        self._groups_by_id: dict[str, GroupContext] = {}
        self._members_by_pair: dict[tuple[str, str], GroupMemberContext] = {}
        self._members_by_group: dict[str, list[GroupMemberContext]] = defaultdict(list)
        self._members_by_user: dict[str, list[GroupMemberContext]] = defaultdict(list)
        self._businesses_by_id: dict[str, BusinessContext] = {}
        self._biz_history_by_pair: dict[tuple[str, str], UserBusinessHistory] = {}
        self._biz_history_by_user: dict[str, list[UserBusinessHistory]] = defaultdict(list)
        self._events_by_message: dict[str, MessageEvent] = {}
        self._events_by_user: dict[str, list[MessageEvent]] = defaultdict(list)
        self._images_by_id: dict[str, ImageRecord] = {}
        self._voices_by_id: dict[str, VoiceNoteRecord] = {}
        self._summaries_by_user: dict[str, list[DailyNotificationSummary]] = defaultdict(list)

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_loaded(
        cls,
        messages: list[IncomingMessage],
        history: list[HistoricalMessage],
        samples: list[SampleMessage],
        users: list[UserProfile],
        groups: list[GroupContext],
        group_members: list[GroupMemberContext],
        businesses: list[BusinessContext],
        user_business_history: list[UserBusinessHistory],
        events: list[MessageEvent],
        images: list[ImageRecord],
        voice_notes: list[VoiceNoteRecord],
        daily_summaries: list[DailyNotificationSummary],
    ) -> "InMemoryDataRepository":
        """Build a fully-indexed repository from already-loaded model lists."""
        repo = cls()
        for m in messages:
            repo._index_message(m)
        for h in history:
            repo._index_history(h)
        for s in samples:
            repo._samples_by_id[s.message_id] = s
        for u in users:
            repo._users_by_id[u.user_id] = u
        for g in groups:
            repo._groups_by_id[g.group_id] = g
        for gm in group_members:
            key = (gm.user_id, gm.group_id)
            repo._members_by_pair[key] = gm
            repo._members_by_group[gm.group_id].append(gm)
            repo._members_by_user[gm.user_id].append(gm)
        for b in businesses:
            repo._businesses_by_id[b.business_id] = b
        for h in user_business_history:
            key = (h.user_id, h.business_id)
            repo._biz_history_by_pair[key] = h
            repo._biz_history_by_user[h.user_id].append(h)
        for e in events:
            repo._events_by_message[e.message_id] = e
            repo._events_by_user[e.user_id].append(e)
        for img in images:
            repo._images_by_id[img.image_id] = img
        for vn in voice_notes:
            repo._voices_by_id[vn.voice_note_id] = vn
        for d in daily_summaries:
            repo._summaries_by_user[d.user_id].append(d)
        return repo

    # ------------------------------------------------------------------ #
    # indexing helpers
    # ------------------------------------------------------------------ #
    def _index_message(self, m: IncomingMessage) -> None:
        self._messages_by_id[m.message_id] = m
        self._messages_by_user[m.user_id].append(m)
        if m.group_id:
            self._messages_by_group[m.group_id].append(m)
        if m.business_id:
            self._messages_by_business[m.business_id].append(m)
        if m.sender_user_id:
            self._messages_by_sender[m.sender_user_id].append(m)

    def _index_history(self, h: HistoricalMessage) -> None:
        self._history_by_id[h.message_id] = h
        self._history_by_user[h.user_id].append(h)
        if h.group_id:
            self._history_by_group[h.group_id].append(h)
        if h.business_id:
            self._history_by_business[h.business_id].append(h)
        if h.sender_user_id:
            self._history_by_sender[h.sender_user_id].append(h)

    # ------------------------------------------------------------------ #
    # messages
    # ------------------------------------------------------------------ #
    def get_message(self, message_id: str) -> Optional[IncomingMessage]:
        return self._messages_by_id.get(message_id)

    def all_messages(self) -> list[IncomingMessage]:
        return list(self._messages_by_id.values())

    def messages_for_user(self, user_id: str) -> list[IncomingMessage]:
        return list(self._messages_by_user.get(user_id, []))

    def messages_for_group(self, group_id: str) -> list[IncomingMessage]:
        return list(self._messages_by_group.get(group_id, []))

    def messages_for_business(self, business_id: str) -> list[IncomingMessage]:
        return list(self._messages_by_business.get(business_id, []))

    def messages_for_sender(self, sender_user_id: str) -> list[IncomingMessage]:
        return list(self._messages_by_sender.get(sender_user_id, []))

    # ------------------------------------------------------------------ #
    # history
    # ------------------------------------------------------------------ #
    def all_history(self) -> list[HistoricalMessage]:
        return list(self._history_by_id.values())

    def history_for_user(self, user_id: str) -> list[HistoricalMessage]:
        return list(self._history_by_user.get(user_id, []))

    def history_for_group(self, group_id: str) -> list[HistoricalMessage]:
        return list(self._history_by_group.get(group_id, []))

    def history_for_business(self, business_id: str) -> list[HistoricalMessage]:
        return list(self._history_by_business.get(business_id, []))

    def history_for_sender(self, sender_user_id: str) -> list[HistoricalMessage]:
        return list(self._history_by_sender.get(sender_user_id, []))

    def history_by_id(self, message_id: str) -> Optional[HistoricalMessage]:
        return self._history_by_id.get(message_id)

    # ------------------------------------------------------------------ #
    # samples
    # ------------------------------------------------------------------ #
    def all_sample_messages(self) -> list[SampleMessage]:
        return list(self._samples_by_id.values())

    def sample_message_by_id(self, message_id: str) -> Optional[SampleMessage]:
        return self._samples_by_id.get(message_id)

    # ------------------------------------------------------------------ #
    # users
    # ------------------------------------------------------------------ #
    def get_user(self, user_id: str) -> Optional[UserProfile]:
        return self._users_by_id.get(user_id)

    def all_users(self) -> list[UserProfile]:
        return list(self._users_by_id.values())

    # ------------------------------------------------------------------ #
    # groups
    # ------------------------------------------------------------------ #
    def get_group(self, group_id: str) -> Optional[GroupContext]:
        return self._groups_by_id.get(group_id)

    def all_groups(self) -> list[GroupContext]:
        return list(self._groups_by_id.values())

    def get_group_member(self, user_id: str, group_id: str) -> Optional[GroupMemberContext]:
        return self._members_by_pair.get((user_id, group_id))

    def members_of_group(self, group_id: str) -> list[GroupMemberContext]:
        return list(self._members_by_group.get(group_id, []))

    def groups_for_user(self, user_id: str) -> list[GroupMemberContext]:
        return list(self._members_by_user.get(user_id, []))

    # ------------------------------------------------------------------ #
    # businesses
    # ------------------------------------------------------------------ #
    def get_business(self, business_id: str) -> Optional[BusinessContext]:
        return self._businesses_by_id.get(business_id)

    def all_businesses(self) -> list[BusinessContext]:
        return list(self._businesses_by_id.values())

    def get_user_business_history(self, user_id: str, business_id: str) -> Optional[UserBusinessHistory]:
        return self._biz_history_by_pair.get((user_id, business_id))

    def businesses_for_user(self, user_id: str) -> list[UserBusinessHistory]:
        return list(self._biz_history_by_user.get(user_id, []))

    # ------------------------------------------------------------------ #
    # events
    # ------------------------------------------------------------------ #
    def event_for_message(self, message_id: str) -> Optional[MessageEvent]:
        return self._events_by_message.get(message_id)

    def events_for_user(self, user_id: str) -> list[MessageEvent]:
        return list(self._events_by_user.get(user_id, []))

    def all_events(self) -> list[MessageEvent]:
        return list(self._events_by_message.values())

    # ------------------------------------------------------------------ #
    # media
    # ------------------------------------------------------------------ #
    def get_image(self, image_id: str) -> Optional[ImageRecord]:
        return self._images_by_id.get(image_id)

    def all_images(self) -> list[ImageRecord]:
        return list(self._images_by_id.values())

    def get_voice_note(self, voice_note_id: str) -> Optional[VoiceNoteRecord]:
        return self._voices_by_id.get(voice_note_id)

    def all_voice_notes(self) -> list[VoiceNoteRecord]:
        return list(self._voices_by_id.values())

    # ------------------------------------------------------------------ #
    # daily notification summaries
    # ------------------------------------------------------------------ #
    def summaries_for_user(self, user_id: str) -> list[DailyNotificationSummary]:
        return list(self._summaries_by_user.get(user_id, []))
