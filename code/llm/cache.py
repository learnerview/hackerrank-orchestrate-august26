"""Disk cache for LLM routing judgments.

Keyed by SHA-256 of the rendered prompt content (message text + media
summary + context summary + evidence snippets), so re-runs of the pipeline
never re-call the Groq API for an identical message/context pair. The cache
knows nothing about Groq, JSON parsing or validation — it stores the raw
judgment payload exactly as returned, one JSON file per key.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

VERSION = 1


class LLMJsonCache:
    """Content-addressed disk cache storing one JSON file per judgment key."""

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = Path(cache_dir)

    @staticmethod
    def key_for(content: str) -> str:
        """Stable key for a prompt-content string (utf-8, no BOM surprises)."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def get(self, content: str) -> Optional[dict[str, Any]]:
        entry = self._cache_dir / f"{self.key_for(content)}.json"
        if not entry.exists():
            return None
        try:
            data = json.loads(entry.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if data.get("version") != VERSION:
            return None
        return data.get("judgment")

    def put(self, content: str, judgment: dict[str, Any]) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        entry = self._cache_dir / f"{self.key_for(content)}.json"
        entry.write_text(
            json.dumps({"version": VERSION, "judgment": judgment}, ensure_ascii=False),
            encoding="utf-8",
        )
