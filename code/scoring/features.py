"""Feature engineering for the preference model.

All features are computed causally: for a historical message ``m`` only
events strictly BEFORE ``m.created_at`` feed the (user, source) aggregates,
so training never sees the label of the very message it predicts. Incoming
messages are always after the whole history, so inference features = full
history aggregates.

Source of a message = sender_user_id > group_id > business_id (the narrowest
available scope, mirroring retrieval).
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

from ingestion.parsing import mentions_user, parse_timestamp
from ingestion.repository import DataRepository
from models import IncomingMessage

FEATURE_NAMES: tuple[str, ...] = (
    # per (user, source) pair aggregates (causal)
    "open_rate",          # fraction of prior messages opened
    "reply_rate",         # fraction of prior messages replied to
    "dismiss_rate",       # fraction of prior messages dismissed
    "report_count",       # number of prior messages reported
    "volume",             # number of prior messages from this source
    "noise_ratio",        # (dismissed + reported) / volume
    "weighted_engagement",  # (open + 2*reply - dismiss - 3*report) / max(1, volume)
    # user-wide tendencies (stable even for sparse pairs)
    "user_open_rate",
    "user_reply_rate",
    # message-level signals
    "hour",               # 0-23 hour of day of the message
    "is_night",           # 22:00-06:00
    "is_morning",         # 6-12
    "is_evening",         # 18-24
    "at_mention",         # text mentions @user_id
    "forwarded_count",
    "is_group",
    "is_business",
    "has_image",
    "has_voice",
    "sender_is_admin",    # from group_members.csv role
    "group_muted",        # user muted this group
    "quiet_hours_overlap",  # message hour inside user's DND window
    "promo_signal",       # 0-1: how promotional/greeting the MESSAGE content is
)

_DND_RE = re.compile(r"(\d{1,2}):\d{2}\s*-\s*(\d{1,2}):\d{2}")

#: Keyword/pattern families that mark a message as promotional, greeting
#: boilerplate, or broadcast content. Deliberately rule-like and cheap (no
#: extra model call); each family counts once so "50% off + hurry" and
#: "50% off + hurry + welcome" saturate at the same strong signal.
_PROMO_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("greeting", re.compile(r"\bgood (morning|afternoon|evening|night)\b")),
    (
        "festive",
        re.compile(
            r"\b(happy birthday|happy new year|merry christmas|happy diwali|"
            r"happy holi|happy holidays|festival|greetings)\b"
        ),
    ),
    (
        "chain",
        re.compile(
            r"\b(stay positive|keep smiling|share blessings|no need to respond|"
            r"forwarding because)\b"
        ),
    ),
    ("sale", re.compile(r"\bsale(s)?\b")),
    ("discount", re.compile(r"\bdiscoun(t|ts|ted)\b")),
    ("offer", re.compile(r"\boffer(s|ed)?\b")),
    ("deals", re.compile(r"\bdeal(s)?\b")),
    ("percent_off", re.compile(r"\d+\s*%?\s*off\b|\bflat\s+\d+")),
    ("coupon", re.compile(r"\bcashback\b|\bcoupon(s)?\b|\bpromo ?code\b")),
    (
        "urgency",
        re.compile(
            r"\bhurry\b|\bexpires?\b|\bexpiring\b|\blimited (time|period)\b|"
            r"\blast chance\b|\bact now\b|\bdon'?t miss\b"
        ),
    ),
    (
        "cta",
        re.compile(r"\b(order|book|buy|shop|tap) now\b|\btap below\b|\bclick here\b"),
    ),
    ("welcome", re.compile(r"\bwelcome\b|\bnew here\b")),
    ("marketing", re.compile(r"\b(subscribe|unsubscribe|marketing)\b")),
    ("travel_package", re.compile(r"\bper person\b")),
    ("marketplace", re.compile(r"\bselling\b|\bfor sale\b|\b(dm|pm) if interested\b")),
    ("pickup", re.compile(r"\b(pickup|picked up)\b")),
    # photo/picture handoff specifically (marketplace): "photos for the X are
    # attached"; NOT any attachment (school circulars/documents stay safe)
    ("attached_media", re.compile(r"\b(photos?|pictures?)\b[^.]{0,40}\battached\b")),
)

#: Media vision analysis already classifies poster intent; these hints are
#: treated as the strongest promotional signal (no text patterns needed).
_PROMOTIONAL_INTENT_HINTS = frozenset({"promotional_poster", "greeting_card"})


def source_id_of(message: IncomingMessage) -> Optional[str]:
    """The message's source for engagement aggregation."""
    return message.sender_user_id or message.group_id or message.business_id


def quiet_hours_overlap(hour: int, dnd_window: Optional[str]) -> float:
    """1.0 if ``hour`` falls inside the user's DND window (e.g. "22:00-07:00")."""
    if not dnd_window:
        return 0.0
    m = _DND_RE.search(dnd_window)
    if not m:
        return 0.0
    start, end = int(m.group(1)), int(m.group(2))
    if start <= end:
        return 1.0 if start <= hour < end else 0.0
    return 1.0 if (hour >= start or hour < end) else 0.0


def promo_signal(text: str, intent_hint: Optional[str] = None) -> float:
    """0-1 signal that THIS message's content is promotional/greeting.

    Counts distinct pattern families matched in the (lowercased) text and
    saturates at 1.0 after two families — "50% off" alone is 0.5 (ambiguous,
    mild), "50% off + hurry" is 1.0 (clearly promotional). A media vision
    ``detected_intent_hint`` of promotional_poster/greeting_card is treated
    as a direct 1.0 (the poster was already classified as promotional).

    This is deliberately message-level: the SAME sender's open-rate history
    is the same for two messages, but today's message may read promotional
    while yesterday's was a personal note. That is the exact asymmetry the
    stale-engagement override in scoring.preference_model needs.
    """
    if intent_hint in _PROMOTIONAL_INTENT_HINTS:
        return 1.0
    if not text:
        return 0.0
    lowered = text.lower()
    hits = sum(1 for _, pattern in _PROMO_PATTERNS if pattern.search(lowered))
    if hits <= 0:
        return 0.0
    return min(1.0, hits / 2.0)


class FeatureExtractor:
    """Builds the preference-model feature vector for one message.

    Causal by construction: when ``as_of`` is given, only events with
    created_at < as_of are counted. If as_of is None (inference), every
    event before the message's own timestamp counts (messages.csv is after
    all of message_history.csv).
    """

    def __init__(self, repository: DataRepository) -> None:
        self._repo = repository

    @staticmethod
    def feature_names() -> list[str]:
        return list(FEATURE_NAMES)

    # ------------------------------------------------------------------ #
    # causal pair aggregates
    # ------------------------------------------------------------------ #
    def _pair_events(self, message: IncomingMessage, as_of: Optional[datetime]):
        """(user, source) events strictly before as_of (or before the message)."""
        source = source_id_of(message)
        history = []
        if message.sender_user_id:
            history = self._repo.history_for_sender(message.sender_user_id)
        elif message.group_id:
            history = self._repo.history_for_group(message.group_id)
        elif message.business_id:
            history = self._repo.history_for_business(message.business_id)

        cutoff = as_of or parse_timestamp(message.created_at)
        result = []
        for h in history:
            ts = parse_timestamp(h.created_at)
            if ts is not None and (cutoff is None or ts < cutoff):
                ev = self._repo.event_for_message(h.message_id)
                if ev is not None:
                    result.append(ev)
        return result

    def _pair_stats(self, message: IncomingMessage, as_of: Optional[datetime]) -> dict[str, float]:
        events = self._pair_events(message, as_of)
        n = len(events)
        if n == 0:
            return {
                "open_rate": 0.0,
                "reply_rate": 0.0,
                "dismiss_rate": 0.0,
                "report_count": 0.0,
                "volume": 0.0,
                "noise_ratio": 0.0,
                "weighted_engagement": 0.0,
            }
        opens = sum(1 for e in events if e.message_opened)
        replies = sum(1 for e in events if e.message_replied)
        dismisses = sum(1 for e in events if e.notification_dismissed)
        reports = sum(1 for e in events if e.message_reported)
        noise = dismisses + reports
        return {
            "open_rate": opens / n,
            "reply_rate": replies / n,
            "dismiss_rate": dismisses / n,
            "report_count": float(reports),
            "volume": float(n),
            "noise_ratio": noise / n,
            "weighted_engagement": (opens + 2 * replies - dismisses - 3 * reports) / max(1, n),
        }

    def _user_stats(self, message: IncomingMessage, as_of: Optional[datetime]) -> dict[str, float]:
        events = self._repo.events_for_user(message.user_id)
        cutoff = as_of or parse_timestamp(message.created_at)
        prior = []
        for ev in events:
            h = self._repo.history_by_id(ev.message_id)
            if h is None:
                continue
            ts = parse_timestamp(h.created_at)
            if ts is not None and (cutoff is None or ts < cutoff):
                prior.append(ev)
        n = len(prior)
        if n == 0:
            return {"user_open_rate": 0.0, "user_reply_rate": 0.0}
        opens = sum(1 for e in prior if e.message_opened)
        replies = sum(1 for e in prior if e.message_replied)
        return {"user_open_rate": opens / n, "user_reply_rate": replies / n}

    # ------------------------------------------------------------------ #
    # message-level signals
    # ------------------------------------------------------------------ #
    def _sender_role(self, message: IncomingMessage) -> float:
        if not (message.group_id and message.sender_user_id):
            return 0.0
        member = self._repo.get_group_member(message.sender_user_id, message.group_id)
        if member is None:
            return 0.0
        return 1.0 if member.role == "admin" else 0.0

    def _group_muted(self, message: IncomingMessage) -> float:
        if not message.group_id:
            return 0.0
        member = self._repo.get_group_member(message.user_id, message.group_id)
        return 1.0 if member is not None and member.group_muted_by_user else 0.0

    def _at_mention(self, message: IncomingMessage) -> float:
        return 1.0 if mentions_user(message.message_text or "", message.user_id) else 0.0

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def extract(
        self,
        message: IncomingMessage,
        as_of: Optional[datetime] = None,
        text_hint: Optional[str] = None,
        intent_hint: Optional[str] = None,
    ) -> dict[str, float]:
        """Full feature dict for one message (keys = FEATURE_NAMES).

        ``text_hint`` (OCR-merged effective text) and ``intent_hint``
        (media vision classification) are inference-only enrichments: they
        add content signal that plain ``message.message_text`` cannot carry
        (e.g. a poster whose only text is in the image). Training never
        passes them, so the model sees the same pure text features it will
        always have; the hints only ever ADD signal at inference.
        """
        pair = self._pair_stats(message, as_of)
        user = self._user_stats(message, as_of)
        ts = as_of or parse_timestamp(message.created_at)
        hour = float(ts.hour) if ts else 0.0
        user_profile = self._repo.get_user(message.user_id)

        features: dict[str, float] = {
            **pair,
            **user,
            "hour": hour,
            "is_night": 1.0 if (hour >= 22 or hour < 6) else 0.0,
            "is_morning": 1.0 if 6 <= hour < 12 else 0.0,
            "is_evening": 1.0 if 18 <= hour < 24 else 0.0,
            "at_mention": self._at_mention(message),
            "forwarded_count": float(message.forwarded_count or 0),
            "is_group": 1.0 if message.group_id else 0.0,
            "is_business": 1.0 if message.business_id else 0.0,
            "has_image": 1.0 if message.media_type == "image" else 0.0,
            "has_voice": 1.0 if message.media_type == "voice" else 0.0,
            "sender_is_admin": self._sender_role(message),
            "group_muted": self._group_muted(message),
            "quiet_hours_overlap": quiet_hours_overlap(
                int(hour), user_profile.do_not_disturb_window if user_profile else None
            ),
            "promo_signal": promo_signal(
                text_hint or message.message_text, intent_hint
            ),
        }
        return features

    def to_vector(self, features: dict[str, float]) -> list[float]:
        """Order features into the training schema vector."""
        return [features[name] for name in FEATURE_NAMES]


__all__ = [
    "FeatureExtractor",
    "FEATURE_NAMES",
    "source_id_of",
    "quiet_hours_overlap",
    "promo_signal",
]
