"""Retrieval composition root: wires a retriever from Settings + repository."""

from __future__ import annotations

from config import Settings
from ingestion.repository import DataRepository
from .tfidf_retriever import TfidfRetriever


def build_retriever(settings: Settings, repository: DataRepository) -> TfidfRetriever:
    """Construct the production evidence retriever."""
    return TfidfRetriever(
        repository,
        top_k=settings.retrieval_top_k,
        min_similarity=settings.retrieval_min_similarity,
        recency_decay_days=settings.retrieval_recency_decay_days,
    )
