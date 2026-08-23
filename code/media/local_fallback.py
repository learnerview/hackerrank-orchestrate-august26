"""Local offline fallbacks for media understanding.

Single responsibility: provide working OCR (pytesseract) and transcription
(openai-whisper) when Groq is unavailable. These implementations are lazy:
third-party packages are imported on first call so a missing local
dependency degrades to a clear MediaAnalysisError instead of a broken import.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

from models import MediaAnalysisResult, TranscriptionResult
from .interfaces import AudioTranscriber, ImageAnalyzer, MediaAnalysisError

_WINDOWS_TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]


def _locate_tesseract() -> Optional[str]:
    """Find the tesseract binary: PATH first, then standard Windows installs."""
    on_path = shutil.which("tesseract")
    if on_path:
        return on_path
    for candidate in _WINDOWS_TESSERACT_PATHS:
        if Path(candidate).exists():
            return candidate
    return None


def _locate_ffmpeg() -> Optional[str]:
    """Find ffmpeg: PATH first, then winget's Gyan.FFmpeg install location."""
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path
    packages_root = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    if packages_root.exists():
        for candidate in packages_root.glob("*/ffmpeg-*-full_build/bin/ffmpeg.exe"):
            if candidate.exists():
                return str(candidate)
    return None


class LocalImageAnalyzer(ImageAnalyzer):
    """pytesseract-based fallback: OCR text, no visual intent model.

    detected_intent_hint is always "other" — intent classification is a
    Groq capability; the fallback only preserves the OCR'd text.
    """

    def __init__(self, lang: str = "eng") -> None:
        self._lang = lang

    def analyze(self, image_path: str) -> MediaAnalysisResult:
        try:
            import pytesseract
            from PIL import Image
        except ImportError as exc:
            raise MediaAnalysisError(
                "local image fallback unavailable: install pytesseract and pillow"
            ) from exc

        binary = _locate_tesseract()
        if binary is None:
            raise MediaAnalysisError(
                "local image fallback unavailable: tesseract binary not found on PATH "
                "or in standard install locations"
            )
        pytesseract.pytesseract.tesseract_cmd = binary
        try:
            extracted = pytesseract.image_to_string(Image.open(image_path), lang=self._lang)
        except Exception as exc:
            raise MediaAnalysisError(f"local OCR failed for {image_path}: {exc}") from exc

        return MediaAnalysisResult(
            extracted_text=extracted.strip(),
            visual_summary="",
            detected_intent_hint="other",
        )


class LocalAudioTranscriber(AudioTranscriber):
    """openai-whisper fallback using the smallest viable local model."""

    def __init__(self, model_name: str = "tiny") -> None:
        self._model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                import whisper
            except ImportError as exc:
                raise MediaAnalysisError(
                    "local audio fallback unavailable: install openai-whisper"
                ) from exc
            self._model = whisper.load_model(self._model_name)
        return self._model

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        model = self._load_model()
        ffmpeg = _locate_ffmpeg()
        if ffmpeg is None:
            raise MediaAnalysisError(
                "local audio fallback unavailable: ffmpeg binary not found on PATH "
                "or in the winget install location"
            )
        bin_dir = str(Path(ffmpeg).parent)
        if bin_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        try:
            result = model.transcribe(audio_path)
        except Exception as exc:
            raise MediaAnalysisError(f"local transcription failed for {audio_path}: {exc}") from exc

        transcript = (result.get("text") or "").strip()
        segments = result.get("segments") or []
        confidence = None
        if segments:
            avg_logprob = sum(float(s.get("avg_logprob", 0.0)) for s in segments) / len(segments)
            confidence = max(0.0, min(1.0, 2 ** avg_logprob))
        return TranscriptionResult(transcript=transcript, confidence=confidence)
