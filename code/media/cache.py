"""Disk-based cache for media analysis results, keyed by file content hash.

Single responsibility: persist and retrieve analyzer results on disk so a
re-run of the pipeline never re-calls Groq for a file already processed.
The key is the SHA-256 of the file bytes, so identical content always hits
regardless of path or renames. The cache knows nothing about Groq, rate
limits or fallbacks.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from models import MediaAnalysisResult, TranscriptionResult


class DiskMediaCache:
    """Content-addressed disk cache storing one JSON file per analyzed file."""

    VERSION = 1

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = Path(cache_dir)
        self._image_dir = self._cache_dir / "images"
        self._audio_dir = self._cache_dir / "audio"

    def _key_for(self, file_path: str) -> str:
        return hashlib.sha256(Path(file_path).read_bytes()).hexdigest()

    def _read(self, directory: Path, file_path: str) -> Optional[dict[str, Any]]:
        entry = directory / f"{self._key_for(file_path)}.json"
        if not entry.exists():
            return None
        try:
            data = json.loads(entry.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if data.get("version") != self.VERSION:
            return None
        return data

    def _write(self, directory: Path, file_path: str, payload: dict[str, Any]) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        entry = directory / f"{self._key_for(file_path)}.json"
        entry.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------------ #
    # images
    # ------------------------------------------------------------------ #
    def get_image_analysis(self, image_path: str) -> Optional[MediaAnalysisResult]:
        data = self._read(self._image_dir, image_path)
        if data is None:
            return None
        return MediaAnalysisResult(
            extracted_text=data.get("extracted_text", ""),
            visual_summary=data.get("visual_summary", ""),
            detected_intent_hint=data.get("detected_intent_hint", "other"),
        )

    def put_image_analysis(self, image_path: str, result: MediaAnalysisResult) -> None:
        self._write(
            self._image_dir,
            image_path,
            {
                "version": self.VERSION,
                "extracted_text": result.extracted_text,
                "visual_summary": result.visual_summary,
                "detected_intent_hint": result.detected_intent_hint,
            },
        )

    # ------------------------------------------------------------------ #
    # audio
    # ------------------------------------------------------------------ #
    def get_transcription(self, audio_path: str) -> Optional[TranscriptionResult]:
        data = self._read(self._audio_dir, audio_path)
        if data is None:
            return None
        return TranscriptionResult(transcript=data.get("transcript", ""), confidence=data.get("confidence"))

    def put_transcription(self, audio_path: str, result: TranscriptionResult) -> None:
        self._write(
            self._audio_dir,
            audio_path,
            {"version": self.VERSION, "transcript": result.transcript, "confidence": result.confidence},
        )
