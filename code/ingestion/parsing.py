"""Primitive field-parsing helpers shared by the ingestion layer.

These functions own the strict "empty cell means None / malformed means raise"
policy. They are deliberately dumb and pure: no I/O, no models — just string
to typed value coercion with explicit error messages.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

TimestampFormats = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ")


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
