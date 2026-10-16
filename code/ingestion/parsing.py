"""Primitive field-parsing helpers shared by the ingestion layer.

These functions own the strict "empty cell means None / malformed means raise"
policy. They are deliberately dumb and pure: no I/O, no models — just string
to typed value coercion with explicit error messages.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

TimestampFormats = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ")

#: A mention is ``@`` preceded by a delimiter (start of string or any
#: non-word character, so an email local part like ``foo@bar.com`` never
#: counts) followed by an id of word characters. The id ends at the first
#: non-word character, so ``@sam123`` and ``@sam`` are distinct mentions.
_MENTION_RE = re.compile(r"(?<![A-Za-z0-9_])@([A-Za-z0-9_]+)")


def mentioned_user_ids(text: str) -> list[str]:
    """The user ids @-mentioned in ``text`` as complete tokens.

    Each ``@`` must sit on a delimiter (start of string or a non-word
    character) and the id runs until the next non-word character — so
    ``@sam123`` does not mention ``sam``, and ``@sam,``/``@sam.``/``@sam``
    all mention ``sam``. Returns ids in order of appearance (duplicates
    preserved so callers can count them).
    """
    if not text:
        return []
    return _MENTION_RE.findall(text)


def mentions_user(text: str, user_id: str) -> bool:
    """True if ``text`` @-mentions ``user_id`` as a complete token."""
    if not text or not user_id:
        return False
    return user_id in mentioned_user_ids(text)


def parse_timestamp(value: str) -> Optional[datetime]:
    """Parse a CSV timestamp into a datetime, or None if it does not match.

    The dataset uses non-ISO strings like ``2026-07-30 22:19``, so several
    common formats are tried. Returning None (never raising) keeps parsing
    explicit without crashing the whole run on an unexpected format.
    """
    if not value or not value.strip():
        return None
    for fmt in TimestampFormats:
        try:
            return datetime.strptime(value.strip(), fmt)
        except ValueError:
            continue
    return None


def opt_str(value: str) -> Optional[str]:
    """Optional string field: empty cell -> None."""
    return None if value == "" else value


def normalize_domain(value: str) -> Optional[str]:
    """Optional domain field: empty cell -> None, else lowercased.

    DNS names are case-insensitive, so domains must be normalized before they
    are compared. Strips surrounding whitespace and any trailing root dot
    (``Example.COM.`` -> ``example.com``). Returns None when nothing remains.
    """
    domain = value.strip().lower().rstrip(".")
    return domain or None


def req_str(value: str, field: str, path: str, row: int) -> str:
    """Required string field: empty cell raises ValueError."""
    if value == "":
        raise ValueError(f"Required field {field!r} empty in {path} row {row}")
    return value


def int_field(value: str, field: str, path: str, row: int, default: Optional[int] = 0) -> Optional[int]:
    """Numeric field: empty cell -> default; malformed value raises."""
    if value == "":
        return default
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"Field {field!r} not an integer in {path} row {row}: {value!r}") from None


def float_field(value: str, field: str, path: str, row: int) -> float:
    """Floating-point field: malformed value raises."""
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"Field {field!r} not a float in {path} row {row}: {value!r}") from None


def bool_field(value: str, field: str, path: str, row: int) -> bool:
    """Boolean field: accepts 1/0/true/false case-insensitively; anything
    else raises so a schema drift is never silently coerced."""
    if value in ("1", "true", "True", "TRUE"):
        return True
    if value in ("0", "false", "False", "FALSE"):
        return False
    raise ValueError(f"Field {field!r} not a boolean in {path} row {row}: {value!r}")
