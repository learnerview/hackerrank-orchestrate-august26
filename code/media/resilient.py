"""Resilient wrappers: cache + rate limit + primary/fallback orchestration.

Single responsibility: compose an analyzer with caching, rate limiting,
retry-with-backoff and automatic fallback so the rest of the pipeline gets
one simple call that "just works". The wrapper depends only on the
ImageAnalyzer/AudioTranscriber Protocols (DIP) — it never imports Groq or
pytesseract directly.

Fallback policy: the local fallback is used only when the primary (Groq)
raises a provider-availability error (rate limit, auth, connection, or a
transient 5xx "over capacity" error). Transient errors are retried with
exponential backoff first. Persistent non-availability errors fall back to
local; any other failure (e.g. a genuinely bad image) propagates so real
bugs are not silently masked.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, TypeVar

from groq import APIConnectionError, AuthenticationError, InternalServerError, RateLimitError

from models import MediaAnalysisResult, TranscriptionResult
from .cache import DiskMediaCache
from .interfaces import AudioTranscriber, ImageAnalyzer, MediaAnalysisError
from .rate_limiter import TokenBucketRateLimiter

logger = logging.getLogger(__name__)

T = TypeVar("T")

_AVAILABILITY_ERRORS = (
    RateLimitError,
    AuthenticationError,
    APIConnectionError,
    InternalServerError,
)


def _call_with_retries(fn: Callable[[], T], retries: int, base_delay: float = 2.0) -> T:
    """Call fn, retrying on availability errors with exponential backoff.

    Returns the first successful result. Raises the last availability error
    once ``retries`` attempts are exhausted.
    """
    attempt = 0
    while True:
        try:
            return fn()
        except _AVAILABILITY_ERRORS as exc:
            attempt += 1
            if attempt > retries:
                raise
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning("provider unavailable (%s); retry %d/%d in %.1fs", exc, attempt, retries, delay)
            time.sleep(delay)


def _primary_or_fallback(primary: Callable[[], T], fallback: Callable[[], T], label: str, retries: int) -> T:
    """Run primary with retries; on exhaustion of provider retries, run fallback.

    If the fallback also fails, raise MediaAnalysisError explaining both
    paths failed, chaining the original provider error.
    """
    try:
        return _call_with_retries(primary, retries)
    except _AVAILABILITY_ERRORS as provider_error:
        pass
    try:
        return fallback()
    except MediaAnalysisError as fallback_error:
        raise MediaAnalysisError(f"both {label} and local fallback failed") from fallback_error


class ResilientImageAnalyzer(ImageAnalyzer):
    """ImageAnalyzer with cache-first, rate-limited, fallback-aware behavior."""

    def __init__(
        self,
        primary: ImageAnalyzer,
        fallback: ImageAnalyzer,
        cache: DiskMediaCache,
        limiter: TokenBucketRateLimiter,
        retries: int = 2,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._cache = cache
        self._limiter = limiter
        self._retries = retries
        self.cache_hits = 0
        self.groq_calls = 0
        self.fallback_calls = 0

    def analyze(self, image_path: str) -> MediaAnalysisResult:
        cached = self._cache.get_image_analysis(image_path)
        if cached is not None:
            self.cache_hits += 1
            return cached

        self._limiter.acquire()

        def primary() -> MediaAnalysisResult:
            self.groq_calls += 1
            return self._primary.analyze(image_path)

        def fallback() -> MediaAnalysisResult:
            self.fallback_calls += 1
            return self._fallback.analyze(image_path)

        result = _primary_or_fallback(primary, fallback, "Groq vision", self._retries)
        self._cache.put_image_analysis(image_path, result)
        return result


class ResilientAudioTranscriber(AudioTranscriber):
    """AudioTranscriber with cache-first, rate-limited, fallback-aware behavior."""

    def __init__(
        self,
        primary: AudioTranscriber,
        fallback: AudioTranscriber,
        cache: DiskMediaCache,
        limiter: TokenBucketRateLimiter,
        retries: int = 2,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._cache = cache
        self._limiter = limiter
        self._retries = retries
        self.cache_hits = 0
        self.groq_calls = 0
        self.fallback_calls = 0

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        cached = self._cache.get_transcription(audio_path)
        if cached is not None:
            self.cache_hits += 1
            return cached

        self._limiter.acquire()

        def primary() -> TranscriptionResult:
            self.groq_calls += 1
            return self._primary.transcribe(audio_path)

        def fallback() -> TranscriptionResult:
            self.fallback_calls += 1
            return self._fallback.transcribe(audio_path)

        result = _primary_or_fallback(primary, fallback, "Groq ASR", self._retries)
        self._cache.put_transcription(audio_path, result)
        return result