"""Unit tests for the media layer: cache, rate limiter and fallback
orchestration, using fakes so no network/local deps are required.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

import httpx  # noqa: E402
from groq import RateLimitError  # noqa: E402

from models import MediaAnalysisResult, TranscriptionResult  # noqa: E402
from media.cache import DiskMediaCache  # noqa: E402
from media.rate_limiter import TokenBucketRateLimiter  # noqa: E402
from media.resilient import ResilientAudioTranscriber, ResilientImageAnalyzer  # noqa: E402


class FakeImageAnalyzer:
    """Simulates an analyzer; can be told to raise a fallback-triggering error."""

    def __init__(self, label="fake", fail_with=None):
        self.label = label
        self.calls = 0
        self._fail_with = fail_with

    def analyze(self, image_path: str) -> MediaAnalysisResult:
        self.calls += 1
        if self._fail_with is not None:
            raise self._fail_with
        return MediaAnalysisResult(
            extracted_text=f"{self.label}:{image_path}",
            visual_summary=f"summary from {self.label}",
            detected_intent_hint="notice",
        )


class FakeAudioTranscriber:
    def __init__(self, label="fake", fail_on=None):
        self.label = label
        self.calls = 0
        self._fail_with = fail_on

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        self.calls += 1
        if self._fail_with is not None:
            raise self._fail_with
        return TranscriptionResult(transcript=f"{self.label}:{audio_path}", confidence=0.9)


def _noop_cache(tmp_path):
    return DiskMediaCache(tmp_path / "cache")


def test_disk_cache_roundtrip(tmp_path):
    cache = DiskMediaCache(tmp_path / ".cache")
    src = tmp_path / "img.jpg"
    src.write_bytes(b"same-bytes-123")

    result = MediaAnalysisResult(extracted_text="hello", visual_summary="v", detected_intent_hint="notice")
    cache.put_image_analysis(str(src), result)
    got = cache.get_image_analysis(str(src))
    assert got == result

    # cached by content hash: a file with identical bytes hits, renamed path too
    src2 = tmp_path / "copy.jpg"
    src2.write_bytes(b"same-bytes-123")
    assert cache.get_image_analysis(str(src2)) == result


def test_audio_cache_roundtrip(tmp_path):
    cache = DiskMediaCache(tmp_path / "cache")
    src = tmp_path / "vn.mp3"
    src.write_bytes(b"audio-bytes")
    cache.put_transcription(str(src), TranscriptionResult("hello world", 0.95))
    got = cache.get_transcription(str(src))
    assert got.transcript == "hello world"
    assert got.confidence == 0.95


def test_resilient_image_uses_cache_second_time(tmp_path):
    primary = FakeImageAnalyzer("groq")
    cache = DiskMediaCache(tmp_path / "cache")
    limiter = TokenBucketRateLimiter(1000)
    path = tmp_path / "img.jpg"
    path.write_bytes(b"img")

    wrapper = ResilientImageAnalyzer(primary, FakeImageAnalyzer("local"), cache, limiter)
    first = wrapper.analyze(str(path))
    assert first.extracted_text == "groq:{path}" or first.extracted_text.startswith("groq:")
    assert primary.calls == 1

    second = wrapper.analyze(str(path))
    assert second == first
    assert primary.calls == 1, "second call must hit cache, not the primary"
    assert wrapper.cache_hits == 1


def _rate_limit_error() -> RateLimitError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(429, request=request)
    return RateLimitError("rate limited", response=response, body={})


def test_resilient_image_falls_back_on_rate_limit(tmp_path):
    primary = FakeImageAnalyzer("groq", fail_with=_rate_limit_error())
    fallback = FakeImageAnalyzer("local")
    cache = DiskMediaCache(tmp_path / "cache")
    path = tmp_path / "img.jpg"
    path.write_bytes(b"img")

    wrapper = ResilientImageAnalyzer(
        primary, fallback, cache, TokenBucketRateLimiter(1000), retries=0
    )
    result = wrapper.analyze(str(path))
    assert result.extracted_text.startswith("local:")
    assert wrapper.fallback_calls == 1
    assert wrapper.groq_calls == 1
    # fallback result is cached, so rerun does not re-invoke fallback
    wrapper.analyze(str(path))
    assert fallback.calls == 1
    assert wrapper.cache_hits == 1


def test_resilient_audio_uses_cache_second(tmp_path):
    cache = DiskMediaCache(tmp_path / "cache")
    primary = FakeAudioTranscriber("groq")
    wrapper = ResilientAudioTranscriber(
        primary, FakeAudioTranscriber("local"), cache, TokenBucketRateLimiter(1000)
    )
    path = tmp_path / "a.mp3"
    path.write_bytes(b"audio")
    first = wrapper.transcribe(str(path))
    assert first.transcript.startswith("groq:")
    wrapper.transcribe(str(path))
    assert primary.calls == 1
    assert wrapper.cache_hits == 1


def test_resilient_image_retries_internal_server_error_then_falls_back(tmp_path):
    from groq import InternalServerError

    def _server_error():
        request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        response = httpx.Response(503, request=request)
        return InternalServerError("over capacity", response=response, body={})

    primary = FakeImageAnalyzer("groq", fail_with=_server_error())
    fallback = FakeImageAnalyzer("local")
    cache = DiskMediaCache(tmp_path / "cache")
    path = tmp_path / "img.jpg"
    path.write_bytes(b"img")

    wrapper = ResilientImageAnalyzer(
        primary, fallback, cache, TokenBucketRateLimiter(1000), retries=0
    )
    result = wrapper.analyze(str(path))
    assert result.extracted_text.startswith("local:")
    assert wrapper.groq_calls == 1  # one attempt (retries=0), then fallback
    assert wrapper.fallback_calls == 1


def _internal_server_error():
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(503, request=request)
    from groq import InternalServerError

    return InternalServerError("over capacity", response=response, body={})


def test_groq_vision_reraises_internal_server_error(tmp_path):
    # The analyzer must propagate provider availability errors (incl. 5xx)
    # UNCHANGED so the resilient wrapper can retry/fall back. It must NOT wrap
    # them into MediaAnalysisError, which would defeat the fallback.
    from groq import InternalServerError
    from media.groq_vision_analyzer import GroqVisionAnalyzer

    class _Completions:
        def create(self, **_kwargs):
            raise _internal_server_error()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    path = tmp_path / "img.jpg"
    path.write_bytes(b"img")
    with pytest.raises(InternalServerError):
        GroqVisionAnalyzer(_Client()).analyze(str(path))


def test_groq_whisper_reraises_internal_server_error(tmp_path):
    from groq import InternalServerError
    from media.groq_whisper_transcriber import GroqWhisperTranscriber

    class _Transcriptions:
        def create(self, **_kwargs):
            raise _internal_server_error()

    class _Audio:
        transcriptions = _Transcriptions()

    class _Client:
        audio = _Audio()

    path = tmp_path / "a.mp3"
    path.write_bytes(b"audio")
    with pytest.raises(InternalServerError):
        GroqWhisperTranscriber(_Client()).transcribe(str(path))


def test_rate_limiter_enforces_rpm():
    # capacity starts full (60 tokens for 60 rpm); drain it, then the next
    # token must wait ~1s for the refill rate of 1 token/sec.
    limiter = TokenBucketRateLimiter(60)
    for _ in range(60):
        limiter.acquire()  # burst, all instant
        assert limiter.available_tokens < 60.0
    t0 = time.monotonic()
    limiter.acquire()
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.8, f"token beyond capacity should wait ~1s, elapsed={elapsed:.3f}"


def test_factory_builds_protocols():
    from config import Settings
    from media import build_image_analyzer, build_audio_transcriber
    from media.interfaces import ImageAnalyzer, AudioTranscriber

    settings = Settings(groq_api_key="", media_cache_dir=Path(".") / "tmp_cache_xyz")
    img = build_image_analyzer(settings)
    aud = build_audio_transcriber(settings)
    assert isinstance(img, ImageAnalyzer)
    assert isinstance(aud, AudioTranscriber)
    assert isinstance(img, ResilientImageAnalyzer)
    assert isinstance(aud, ResilientAudioTranscriber)