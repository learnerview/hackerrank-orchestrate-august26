"""retrieval: ranked historical evidence for routing decisions."""

from .interfaces import EvidenceRetriever
from .tfidf_retriever import TfidfRetriever
from .factory import build_retriever

__all__ = ["EvidenceRetriever", "TfidfRetriever", "build_retriever"]
