"""Unit tests for the retrieval layer: scoping, TF-IDF ranking, recency
decay, and the explicit empty-history contract (returns [] so the output
layer renders "none"). Synthetic data only — no network, no dataset files.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from models import EvidenceRef, HistoricalMessage, IncomingMessage  # noqa: E402
from ingestion.repository import InMemoryDataRepository  # noqa: E402
from retrieval.interfaces import EvidenceRetriever  # noqa: E402
from retrieval.tfidf_retriever import TfidfRetriever  # noqa: E402


def _hist(
    message_id: str,
    sender: str,
    text: str,
    created_at: str,
    group_id: str | None = None,
    business_id: str | None = None,
) -> HistoricalMessage:
    return HistoricalMessage(
        message_id=message_id,
        user_id="u1",
        conversation_type="personal",
        group_id=group_id,
        business_id=business_id,
        sender_user_id=sender,
        created_at=created_at,
        message_text=text,
        media_type="text",
        media_id=None,
        forwarded_count=0,
    )


def _query(
    message_id: str = "m1",
    sender: str = "s1",
    group_id: str | None = None,
    text: str = "lunch at the cafe today?",
) -> IncomingMessage:
    return IncomingMessage(
        message_id=message_id,
        user_id="u1",
        conversation_type="personal",
        group_id=group_id,
        business_id=None,
        sender_user_id=sender,
        created_at="2026-07-25 12:00:00",
        message_text=text,
        media_type="text",
        media_id=None,
        forwarded_count=0,
    )


def _repo(history: list[HistoricalMessage]) -> InMemoryDataRepository:
    return InMemoryDataRepository.from_loaded(
        messages=[],
        history=history,
        samples=[],
        users=[],
        groups=[],
        group_members=[],
        businesses=[],
        user_business_history=[],
        events=[],
        images=[],
        voice_notes=[],
        daily_summaries=[],
    )


def _history_with_sender_evidence() -> list[HistoricalMessage]:
    return [
        _hist("h_old_unrelated", "s1", "your order #482 has been dispatched", "2026-03-05 09:00:00"),
        _hist("h_recent_similar", "s1", "want to grab lunch at the cafe tomorrow?", "2026-07-20 18:00:00"),
        # same group but DIFFERENT sender — must be excluded by scoping
        _hist(
            "h_other_sender",
            "s2",
            "want to grab lunch at the cafe tomorrow?",
            "2026-07-22 18:00:00",
            group_id="g1",
        ),
    ]


def test_ranks_recent_similar_above_old_unrelated_from_same_sender():
    repo = _repo(_history_with_sender_evidence())
    retriever = TfidfRetriever(repo, top_k=3, min_similarity=0.01)

    results = retriever.retrieve(_query(), "lunch at the cafe today?")

    assert results, "expected at least one evidence result"
    assert isinstance(results[0], EvidenceRef)
    assert results[0].message_id == "h_recent_similar"
    ids = [r.message_id for r in results]
    assert "h_old_unrelated" not in ids or results[0].score > next(
        r.score for r in results if r.message_id == "h_old_unrelated"
    )
    # scores strictly descending
    assert all(a.score >= b.score for a, b in zip(results, results[1:]))
    # scoping: other sender's textually identical message never enters
    assert "h_other_sender" not in ids


def test_empty_history_returns_empty_list_not_none():
    repo = _repo([])
    retriever = TfidfRetriever(repo)

    results = retriever.retrieve(_query(), "lunch at the cafe today?")

    assert results == []


def test_no_query_text_returns_empty_list():
    repo = _repo(_history_with_sender_evidence())
    retriever = TfidfRetriever(repo)

    # a voice note with no transcript and no raw text -> no retrieval
    voice = _query("m_voice", text="")
    assert retriever.retrieve(voice, "") == []
    assert retriever.retrieve(voice, "   ") == []


def test_nothing_above_threshold_returns_empty_list():
    repo = _repo([_hist("h1", "s1", "buy one get one free sale!!!", "2026-07-20 10:00:00")])
    retriever = TfidfRetriever(repo, min_similarity=0.5)

    assert retriever.retrieve(_query(), "lunch at the cafe today?") == []


def test_top_k_is_respected():
    repo = _repo(
        [
            _hist(f"h{i}", "s1", f"lunch at cafe day {i}?", f"2026-07-{10 + i} 12:00:00")
            for i in range(1, 8)
        ]
    )
    retriever = TfidfRetriever(repo, top_k=3)

    results = retriever.retrieve(_query(), "lunch at the cafe today?")

    assert len(results) <= 3


def test_group_fallback_when_sender_has_no_history():
    repo = _repo(
        [
            _hist("h_group", "s9", "meeting at 3pm in the board room", "2026-07-20 10:00:00", group_id="g1"),
        ]
    )
    retriever = TfidfRetriever(repo, min_similarity=0.0)
    query = _query(sender="s1", group_id="g1")  # sender has no history -> group scope used

    results = retriever.retrieve(query, "lunch at the cafe today?")

    assert [r.message_id for r in results] == ["h_group"]


def test_satisfies_evidence_retriever_protocol():
    repo = _repo(_history_with_sender_evidence())
    assert isinstance(TfidfRetriever(repo), EvidenceRetriever)


def test_recency_ranks_newer_similar_above_older_similar():
    repo = _repo(
        [
            _hist("h_old_similar", "s1", "lunch at the cafe tomorrow?", "2026-03-01 12:00:00"),
            _hist("h_new_similar", "s1", "lunch at the cafe tomorrow?", "2026-07-20 12:00:00"),
        ]
    )
    retriever = TfidfRetriever(repo, min_similarity=0.0)

    results = retriever.retrieve(_query(), "lunch at the cafe today?")

    assert [r.message_id for r in results] == ["h_new_similar", "h_old_similar"]
    assert results[0].score > results[1].score
