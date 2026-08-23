"""Protocols for the media understanding layer.

Both analyzers are defined as structural Protocols (typing.Protocol) so the
orchestration layer depends on an abstraction, not on Groq or on local
fallbacks. Any implementation — Groq vision, pytesseract, a future
Cloud-Vision adapter — is interchangeable without touching callers.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from models import MediaAnalysisResult, TranscriptionResult


@runtime_checkable
class ImageAnalyzer(Protocol):
    """Something that extracts text and visual intent from an image file."""

    def analyze(self, image_path: str) -> MediaAnalysisResult:
        """Analyze an image file and return OCR text + visual summary.

        Args:
            image_path: absolute path to the image file on disk.

        Returns:
            MediaAnalysisResult with extracted_text, visual_summary and
            detected_intent_hint. Raises MediaAnalysisError on failure.
        """
        ...


@runtime_checkable
class AudioTranscriber(Protocol):
    """Something that transcribes a voice-note audio file."""

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """Transcribe an audio file and return the transcript.

        Args:
            audio_path: absolute path to the audio file on disk.

        Returns:
            TranscriptionResult with transcript text and an optional
            confidence in [0, 1]. Raises MediaAnalysisError on failure.
        """
        ...


class MediaAnalysisError(Exception):
    """Raised when an analyzer cannot process a media file for any reason
    (missing file, unreadable content, provider failure, missing local deps).
    Callers are expected to degrade gracefully, never to crash the pipeline."""

    pass
