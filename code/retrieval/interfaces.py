"""Retrieval contracts: what the retrieval layer promises the rest of the
pipeline (context building, orchestration) — never how it is implemented.

Swappable via structural typing: any object exposing ``retrieve`` satisfying
the EvidenceRetriever Protocol can replace the TF-IDF implementation.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from models import EvidenceRef, IncomingMessage


@runtime_checkable
class EvidenceRetriever(Protocol):
    """Finds historical messages (message_history.csv) relevant to a routing
    decision for one incoming message.

    The scope (sender / group / business / user) is taken from the incoming
    message itself. Implementations return a ranked list of EvidenceRef sorted
    by descending relevance score.
    """

    def retrieve(self, message: IncomingMessage, effective_text: str) -> list[EvidenceRef]:
        """Return top-k evidence refs for ``message``.

        Args:
            message: the incoming message being routed; its ``sender_user_id``,
                ``group_id``, ``business_id`` and ``user_id`` define the
                candidate pool.
            effective_text: the message's effective text (original text plus
                OCR'd/transcribed media content merged by the context layer).

        Returns:
            Ranked EvidenceRef list, highest score first. MUST be an empty
            list (never None) when no useful history exists or nothing clears
            the minimum similarity threshold — the output layer renders that
            as ``none`` per the submission schema.
        """
        ...
