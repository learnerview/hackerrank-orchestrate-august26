"""TF-IDF + recency evidence retriever for routing decisions.

Candidate scoping (tiered, first non-empty wins):

1. same sender **in the same group** (strongest signal)
2. same sender anywhere
3. same group
4. same business
5. same user (broadest fallback)

This implements the task's core assumption: *who* the message is from and
*where* it arrives matter more for a routing decision than raw text
similarity, so the semantic ranking is applied inside the narrowest
matching scope, never to the whole history.

Ranking: cosine similarity (TF-IDF vectors over candidate pool + query)
multiplied by an exponential recency decay; only results that clear
``min_similarity`` on the combined score are returned.

Explicit empty-handling contract: when there is no query text, no candidate
scope, or nothing above threshold, ``retrieve`` returns ``[]`` (never None)
so the output layer can render ``none`` per the submission schema.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ingestion.parsing import parse_timestamp
from ingestion.repository import DataRepository
from models import EvidenceRef, HistoricalMessage, IncomingMessage
from .interfaces import EvidenceRetriever


class TfidfRetriever(EvidenceRetriever):
    """Scoped TF-IDF/recency retriever over message_history.csv."""

    def __init__(
        self,
        repository: DataRepository,
        top_k: int = 3,
        min_similarity: float = 0.1,
        recency_decay_days: float = 60.0,
    ) -> None:
        self._repo = repository
        self._top_k = max(1, int(top_k))
        self._min_similarity = float(min_similarity)
        self._recency_decay_days = float(recency_decay_days)

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def retrieve(self, message: IncomingMessage, effective_text: str) -> list[EvidenceRef]:
        """Rank historical evidence for ``message``, best first, or []."""
        query = self._query_text(message, effective_text)
        if not query:
            return []

        candidates = self._candidate_pool(message)
        if not candidates:
            return []

        similarities = self._cosine_similarities(query, candidates)
        reference = parse_timestamp(message.created_at)

        results: list[EvidenceRef] = []
        for candidate, similarity in zip(candidates, similarities):
            score = similarity * self._recency_decay(candidate, reference)
            if score >= self._min_similarity:
                results.append(EvidenceRef(message_id=candidate.message_id, score=round(score, 6)))

        results.sort(key=lambda r: (-r.score, r.message_id))
        return results[: self._top_k]

    # ------------------------------------------------------------------ #
    # query / scoping
    # ------------------------------------------------------------------ #
    @staticmethod
    def _query_text(message: IncomingMessage, effective_text: str) -> str:
        """Effective text wins (it includes OCR/transcript); falls back to the
        raw message text; empty -> no retrieval possible."""
        candidate = (effective_text or "").strip()
        if candidate:
            return candidate
        return (message.message_text or "").strip()

    def _candidate_pool(self, message: IncomingMessage) -> list[HistoricalMessage]:
        """Narrowest non-empty scope, in priority order (see module docstring)."""
        sender, group_id = message.sender_user_id, message.group_id
        if sender and group_id:
            tier = [
                h
                for h in self._repo.history_for_sender(sender)
                if h.group_id == group_id
            ]
            if tier:
                return tier
        if sender:
            tier = self._repo.history_for_sender(sender)
            if tier:
                return tier
        if group_id:
            tier = self._repo.history_for_group(group_id)
            if tier:
                return tier
        if message.business_id:
            tier = self._repo.history_for_business(message.business_id)
            if tier:
                return tier
        return self._repo.history_for_user(message.user_id)

    # ------------------------------------------------------------------ #
    # scoring
    # ------------------------------------------------------------------ #
    @staticmethod
    def _cosine_similarities(query: str, candidates: list[HistoricalMessage]) -> list[float]:
        """TF-IDF cosine between the query and each candidate document."""
        texts = [c.message_text or "" for c in candidates]
        vectorizer = TfidfVectorizer(lowercase=True)
        try:
            matrix = vectorizer.fit_transform([query, *texts])
        except ValueError:
            # empty vocabulary (e.g. query with no tokenizable content)
            return [0.0] * len(candidates)
        query_vec = matrix[0]
        doc_vecs = matrix[1:]
        return cosine_similarity(query_vec, doc_vecs).ravel().tolist()

    def _recency_decay(
        self, candidate: HistoricalMessage, reference: Optional[datetime]
    ) -> float:
        """Exponential decay: exp(-age_days / decay_days).

        An unparseable candidate timestamp yields decay 0 (no evidence value
        without a date); a missing reference time yields decay 1 (best case).
        """
        created = parse_timestamp(candidate.created_at)
        if created is None:
            return 0.0
        if reference is None:
            return 1.0
        age_days = max(0.0, (reference - created).total_seconds() / 86400.0)
        return math.exp(-age_days / self._recency_decay_days)


__all__ = ["TfidfRetriever"]
