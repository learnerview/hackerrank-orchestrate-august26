"""Batch pipeline runner.

Single responsibility: turn a sequence of incoming messages into routing
decisions by producing each message's context and driving the orchestrator.
No construction happens here — main.py (the composition root) injects every
dependency through its Protocol/interface, so the runner is testable with
fakes and reusable from any entrypoint (CLI, tests, notebooks).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from config import Settings
from context.context_builder import MessageContext
from ingestion.repository import InMemoryDataRepository
from llm.interfaces import RoutingLLM
from media.interfaces import AudioTranscriber, ImageAnalyzer, MediaAnalysisError
from models.decision import RoutingDecision
from models.message import IncomingMessage, RoutedMessage
from orchestration.router import RoutingOrchestrator
from personalization.overrides import PersonalizationOverrides
from retrieval.interfaces import EvidenceRetriever
from safety.rule_engine import SafetyRuleEngine
from scoring.features import FeatureExtractor
from scoring.interfaces import PreferenceModel


class PipelineRunner:
    """Runs the routing pipeline over a sequence of incoming messages."""

    def __init__(
        self,
        *,
        settings: Settings,
        repo: InMemoryDataRepository,
        image_analyzer: ImageAnalyzer,
        audio_transcriber: AudioTranscriber,
        retriever: EvidenceRetriever,
        features: FeatureExtractor,
        orchestrator: RoutingOrchestrator,
    ) -> None:
        self._settings = settings
        self._repo = repo
        self._image_analyzer = image_analyzer
        self._audio_transcriber = audio_transcriber
        self._retriever = retriever
        self._features = features
        self._orchestrator = orchestrator
        self._history_ids = frozenset(h.message_id for h in repo.all_history())
        self._by_history = {h.message_id: h for h in repo.all_history()}
        self.media_failures: list[str] = []

    # ------------------------------------------------------------------ #
    # per-message context
    # ------------------------------------------------------------------ #
    def _resolve_media_path(self, file_path: str) -> str:
        path = Path(file_path)
        if path.is_absolute():
            return str(path)
        return str(self._settings.dataset_dir / path)

    def merge_media(self, message: IncomingMessage) -> RoutedMessage:
        """Merge OCR/transcript text and a media summary into the effective text.

        Best-effort by design: media enrichment is auxiliary to routing, so
        a failure (transient API outage, missing/corrupt file) degrades to
        the raw message text instead of dropping the row. Failures are
        recorded in ``media_failures`` for reporting.
        """
        effective = message.message_text or ""
        summary: Optional[str] = None
        intent_hint: Optional[str] = None
        try:
            if message.media_type == "image" and message.media_id:
                record = self._repo.get_image(message.media_id)
                if record is not None:
                    result = self._image_analyzer.analyze(self._resolve_media_path(record.file_path))
                    if result.extracted_text:
                        effective = f"{effective}\n{result.extracted_text}".strip()
                    summary = result.visual_summary or f"image {message.media_id}"
                    intent_hint = result.detected_intent_hint or None
            elif message.media_type == "voice" and message.media_id:
                record = self._repo.get_voice_note(message.media_id)
                if record is not None:
                    result = self._audio_transcriber.transcribe(self._resolve_media_path(record.file_path))
                    effective = f"{effective}\n{result.transcript}".strip()
                    summary = f"voice note transcript available ({len(result.transcript)} chars)"
        except (MediaAnalysisError, OSError) as exc:
            self.media_failures.append(message.message_id)
            summary = f"media analysis failed for {message.media_id} ({type(exc).__name__})"
        return RoutedMessage(
            message=message,
            effective_text=effective,
            media_summary=summary,
            media_intent_hint=intent_hint,
        )

    def _build_context(self, message: IncomingMessage, routed: RoutedMessage) -> MessageContext:
        evidence = self._retriever.retrieve(message, routed.effective_text)
        snippets = []
        for ref in evidence:
            history = self._by_history.get(ref.message_id)
            text = history.message_text if history else "?"
            snippets.append(f"[{ref.message_id}] {text[:100]!r} (score {ref.score:.2f})")
        return MessageContext(
            message=message,
            routed=routed,
            user=self._repo.get_user(message.user_id),
            group=self._repo.get_group(message.group_id) if message.group_id else None,
            member=(
                self._repo.get_group_member(message.user_id, message.sender_user_id)
                if message.sender_user_id
                else None
            ),
            business=self._repo.get_business(message.business_id) if message.business_id else None,
            evidence=evidence,
            engagement_features=self._features.extract(
                message,
                text_hint=routed.effective_text,
                intent_hint=routed.media_intent_hint,
            ),
            evidence_snippets=snippets,
            history_ids=self._history_ids,
        )

    # ------------------------------------------------------------------ #
    # batch run
    # ------------------------------------------------------------------ #
    def run(self, messages: list[IncomingMessage]) -> list[RoutingDecision]:
        """Route every message in order, fail-fast on any error (no skips)."""
        decisions: list[RoutingDecision] = []
        for message in messages:
            routed = self.merge_media(message)
            context = self._build_context(message, routed)
            decisions.append(self._orchestrator.route(message, context.to_dict()))
        return decisions
