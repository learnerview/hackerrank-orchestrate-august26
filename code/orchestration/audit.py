"""Per-message decision audit log.

Every routing decision is recorded here with the producing stage
(rule/preference/llm), whether personalization touched it, the final action
and the evidence used. The log is the basis for SRS metrics (stage usage,
override rates) and for explaining decisions to the AI Judge. Entries are
kept in memory and optionally appended as JSON lines to a file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

REQUIRED_KEYS = frozenset({"message_id", "stage", "action", "message_type", "confidence"})


class DecisionAuditLog:
    """Append-only structured log of routing decisions."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = Path(path) if path else None
        self._entries: list[dict[str, Any]] = []

    def record(self, entry: dict[str, Any]) -> None:
        """Validate and store one decision entry (raises on missing keys)."""
        missing = REQUIRED_KEYS - set(entry)
        if missing:
            raise ValueError(f"Audit entry missing required keys: {sorted(missing)}")
        self._entries.append(dict(entry))
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    @property
    def records(self) -> list[dict[str, Any]]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)
