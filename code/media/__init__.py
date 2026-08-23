"""media: multimodal understanding — image OCR/intent and audio transcription.

Every analyzer is exposed behind the ImageAnalyzer / AudioTranscriber
Protocols; the factory is the composition root. Caching, rate limiting and
fallback orchestration live in separate single-responsibility modules.
"""

from .interfaces import ImageAnalyzer, AudioTranscriber, MediaAnalysisError
from .cache import DiskMediaCache
from .rate_limiter import TokenBucketRateLimiter
from .resilient import ResilientImageAnalyzer, ResilientAudioTranscriber
from .local_fallback import LocalImageAnalyzer, LocalAudioTranscriber
from .groq_vision_analyzer import GroqVisionAnalyzer
from .groq_whisper_transcriber import GroqWhisperTranscriber
from .factory import build_image_analyzer, build_audio_transcriber
from .paths import resolve_media_path

__all__ = [
    "ImageAnalyzer",
    "AudioTranscriber",
    "MediaAnalysisError",
    "DiskMediaCache",
    "TokenBucketRateLimiter",
    "ResilientImageAnalyzer",
    "ResilientAudioTranscriber",
    "LocalImageAnalyzer",
    "LocalAudioTranscriber",
    "GroqVisionAnalyzer",
    "GroqWhisperTranscriber",
    "build_image_analyzer",
    "build_audio_transcriber",
    "resolve_media_path",
]
