"""Groq vision-backed ImageAnalyzer: OCR + visual intent in one call.

Single responsibility: build the vision request, call Groq, parse the JSON
response into a MediaAnalysisResult. It does NOT know about caching, rate
limits or local fallbacks — the resilient wrapper owns those concerns.
The Groq client is injected (DIP); callers may pass a mock for tests.

Error contract: provider errors that signal rate limiting or auth failures
(groq.RateLimitError, groq.AuthenticationError, groq.APIConnectionError)
propagate unchanged so the resilient wrapper can switch to the local
fallback. All other failures are wrapped in MediaAnalysisError.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Optional

from groq import Groq
from groq import APIConnectionError, AuthenticationError, RateLimitError

from models import MediaAnalysisResult
from .interfaces import ImageAnalyzer, MediaAnalysisError

VISION_PROMPT = """Analyze this image (poster, screenshot, notice, flyer, or similar) and return a JSON object with exactly these three keys:
1. "extracted_text": the literal text visible in the image, transcribed verbatim. If no text is present, use an empty string.
2. "visual_summary": a concise 1-2 sentence description of what the image shows.
3. "detected_intent_hint": one of: "promotional_poster", "payment_qr", "notice", "screenshot", "invoice", "greeting_card", "id_document", "other".
Output ONLY the JSON object, no markdown, no commentary."""


class GroqVisionAnalyzer(ImageAnalyzer):
    """Primary vision analyzer using a Groq vision-capable model."""

    def __init__(
        self,
        client: Groq,
        model: str = "llama-4-scout-17b-16e-instruct",
        temperature: float = 0.0,
    ) -> None:
        if client is None:
            raise MediaAnalysisError("GroqVisionAnalyzer requires a Groq client")
        self._client = client
        self._model = model
        self._temperature = temperature

    def analyze(self, image_path: str) -> MediaAnalysisResult:
        try:
            with open(image_path, "rb") as handle:
                image_bytes = handle.read()
        except OSError as exc:
            raise MediaAnalysisError(f"cannot read image {image_path}: {exc}") from exc

        if not image_bytes:
            raise MediaAnalysisError(f"image file is empty: {image_path}")

        data_uri = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("ascii")

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": VISION_PROMPT},
                            {"type": "image_url", "image_url": {"url": data_uri}},
                        ],
                    }
                ],
            )
        except (RateLimitError, AuthenticationError, APIConnectionError):
            raise
        except Exception as exc:
            raise MediaAnalysisError(f"Groq vision call failed for {image_path}: {exc}") from exc

        content = response.choices[0].message.content or ""
        return self._parse_response(content, image_path)

    def _parse_response(self, content: str, image_path: str) -> MediaAnalysisResult:
        text = content.strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start == -1 or end <= start:
                raise MediaAnalysisError(f"Groq vision returned non-JSON for {image_path}") from None
            try:
                payload = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                raise MediaAnalysisError(f"Groq vision returned malformed JSON for {image_path}") from None

        return MediaAnalysisResult(
            extracted_text=str(payload.get("extracted_text", "")),
            visual_summary=str(payload.get("visual_summary", "")),
            detected_intent_hint=str(payload.get("detected_intent_hint", "other")),
        )
