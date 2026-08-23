"""Composition root for the media layer.

Single responsibility: build fully-wired analyzers from Settings. This is the
only place that knows how to construct a Groq client from config, which keeps
every other media module free of construction logic (DIP at the composition
root). Callers receive ImageAnalyzer/AudioTranscriber Protocols.
"""

from __future__ import annotations

from groq import Groq

from config import Settings
from .cache import DiskMediaCache
from .groq_vision_analyzer import GroqVisionAnalyzer
from .groq_whisper_transcriber import GroqWhisperTranscriber
from .interfaces import AudioTranscriber, ImageAnalyzer
from .local_fallback import LocalAudioTranscriber, LocalImageAnalyzer
from .rate_limiter import TokenBucketRateLimiter
from .resilient import ResilientAudioTranscriber, ResilientImageAnalyzer


def build_groq_client(settings: Settings) -> Groq:
    """Create a Groq client from settings; caller decides how to use it."""
    return Groq(api_key=settings.groq_api_key, base_url=settings.groq_base_url)


def build_image_analyzer(settings: Settings) -> ImageAnalyzer:
    """Wire the full image pipeline: cache + limiter + Groq primary + local fallback."""
    client = build_groq_client(settings)
    return ResilientImageAnalyzer(
        primary=GroqVisionAnalyzer(client=client, model=settings.groq_model_vision),
        fallback=LocalImageAnalyzer(),
        cache=DiskMediaCache(settings.media_cache_dir),
        limiter=TokenBucketRateLimiter(settings.media_rpm_vision),
    )


def build_audio_transcriber(settings: Settings) -> AudioTranscriber:
    """Wire the full audio pipeline: cache + limiter + Groq primary + local fallback."""
    client = build_groq_client(settings)
    return ResilientAudioTranscriber(
        primary=GroqWhisperTranscriber(client=client, model=settings.groq_model_asr),
        fallback=LocalAudioTranscriber(),
        cache=DiskMediaCache(settings.media_cache_dir),
        limiter=TokenBucketRateLimiter(settings.media_rpm_asr),
    )
