"""Groq Whisper-backed AudioTranscriber.

Single responsibility: send the audio file to Groq's Whisper Large v3 Turbo
endpoint and parse the response into a TranscriptionResult. Caching, rate
limiting and local fallback live in the resilient wrapper, not here.
The Groq client is injected (DIP) so tests can pass a mock.

Error contract: provider errors that signal rate limiting or auth failures
(groq.RateLimitError, groq.AuthenticationError, groq.APIConnectionError)
propagate unchanged so the resilient wrapper can switch to the local
fallback. All other failures are wrapped in MediaAnalysisError.
"""

from __future__ import annotations

from typing import Any, Optional

from groq import Groq
from groq import APIConnectionError, AuthenticationError, InternalServerError, RateLimitError

from models import TranscriptionResult
from .interfaces import AudioTranscriber, MediaAnalysisError


class GroqWhisperTranscriber(AudioTranscriber):
    """Primary audio transcriber using Groq's Whisper Large v3 Turbo."""

    def __init__(
        self,
        client: Groq,
        model: str = "whisper-large-v3-turbo",
        temperature: float = 0.0,
    ) -> None:
        if client is None:
            raise MediaAnalysisError("GroqWhisperTranscriber requires a Groq client")
        self._client = client
        self._model = model
        self._temperature = temperature

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        try:
            with open(audio_path, "rb") as handle:
                audio_file = handle.read()
        except OSError as exc:
            raise MediaAnalysisError(f"cannot read audio {audio_path}: {exc}") from exc

        if not audio_file:
            raise MediaAnalysisError(f"audio file is empty: {audio_path}")

        try:
            response = self._client.audio.transcriptions.create(
                model=self._model,
                file=(audio_path, audio_file, "audio/mpeg"),
                temperature=self._temperature,
            )
        except (RateLimitError, AuthenticationError, APIConnectionError, InternalServerError):
            raise
        except Exception as exc:
            raise MediaAnalysisError(f"Groq ASR call failed for {audio_path}: {exc}") from exc

        transcript = getattr(response, "text", None) or getattr(response, "transcription_text", None) or ""
        return TranscriptionResult(transcript=transcript.strip(), confidence=None)
