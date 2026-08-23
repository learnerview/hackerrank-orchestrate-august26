"""Pure data models for media analysis results.

These dataclasses are the structured return values of the media layer's
ImageAnalyzer and AudioTranscriber Protocols. Like every model in this
package they carry data only — no behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True, slots=True)
class ImageRecord:
    """One row from dataset/images.csv."""

    image_id: str
    file_path: str


@dataclass(frozen=True, slots=True)
class VoiceNoteRecord:
    """One row from dataset/voice_notes.csv."""

    voice_note_id: str
    file_path: str


@dataclass(frozen=True, slots=True)
class MediaAnalysisResult:
    """Structured output of an ImageAnalyzer: OCR text plus visual intent."""

    extracted_text: str
    visual_summary: str
    detected_intent_hint: str


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """Structured output of an AudioTranscriber: transcript plus confidence."""

    transcript: str
    confidence: Optional[float] = None
