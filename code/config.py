"""Configuration: dataclass-based Settings read from environment variables.

All paths in Settings are anchored to the code directory (the folder
containing this file), never to the process working directory, so the
pipeline behaves identically regardless of where ``python main.py`` is
invoked from. Secrets come from environment variables / .env only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

CODE_DIR = Path(__file__).resolve().parent
REPO_DIR = CODE_DIR.parent


def load_env(dotenv_path: Path | None = None) -> None:
    """Load env vars from a .env file (only fills missing values) so secrets
    stay out of the repository. Called once at process start, before any
    Settings instance is constructed."""
    load_dotenv(dotenv_path=dotenv_path, override=False)


def _repo_path(relative: str) -> Path:
    """Resolve a repo-relative path string to an absolute Path anchored at the
    repository root (code/..). Absolute or already-anchored values pass through."""
    p = Path(relative)
    if p.is_absolute():
        return p
    return (REPO_DIR / p).resolve()


def _code_path(relative: str) -> Path:
    """Resolve a code-relative path string to an absolute Path anchored at the
    code directory. Absolute or already-anchored values pass through."""
    p = Path(relative)
    if p.is_absolute():
        return p
    return (CODE_DIR / p).resolve()


def _env_path(env_key: str, default: str, anchor: str = "code") -> Path:
    raw = os.environ.get(env_key, default)
    if anchor == "repo":
        return _repo_path(raw)
    return _code_path(raw)


@dataclass(frozen=True, slots=True)
class Settings:
    groq_api_key: str = field(default_factory=lambda: os.environ.get("GROQ_API_KEY", ""))
    groq_base_url: str = field(default_factory=lambda: os.environ.get("GROQ_BASE_URL", "https://api.groq.com"))
    groq_model_text: str = field(default_factory=lambda: os.environ.get("GROQ_MODEL_TEXT", "llama-3.3-70b-versatile"))
    groq_model_vision: str = field(
        default_factory=lambda: os.environ.get("GROQ_MODEL_VISION", "qwen/qwen3.6-27b")
    )
    groq_model_asr: str = field(default_factory=lambda: os.environ.get("GROQ_MODEL_ASR", "whisper-large-v3-turbo"))
    log_level: str = field(default_factory=lambda: os.environ.get("LOG_LEVEL", "INFO"))

    # --- media layer ---------------------------------------------------- #
    media_cache_dir: Path = field(default_factory=lambda: _env_path("MEDIA_CACHE_DIR", ".cache/media"))
    media_rpm_vision: int = field(default_factory=lambda: int(os.environ.get("MEDIA_RPM_VISION", "10")))
    media_rpm_asr: int = field(default_factory=lambda: int(os.environ.get("MEDIA_RPM_ASR", "20")))
    media_retries: int = field(default_factory=lambda: int(os.environ.get("MEDIA_RETRIES", "2")))

    # --- data / output -------------------------------------------------- #
    dataset_dir: Path = field(default_factory=lambda: _env_path("DATASET_DIR", "dataset", anchor="repo"))
    output_path: Path = field(default_factory=lambda: _env_path("OUTPUT_PATH", "dataset/output.csv", anchor="repo"))

    # --- artifacts ------------------------------------------------------ #
    artifacts_dir: Path = field(default_factory=lambda: _env_path("ARTIFACTS_DIR", "artifacts"))
    preference_model_path: Path = field(
        default_factory=lambda: _env_path("ARTIFACTS_DIR", "artifacts") / "preference_model.joblib"
    )
    tfidf_vectorizer_path: Path = field(
        default_factory=lambda: _env_path("ARTIFACTS_DIR", "artifacts") / "tfidf_vectorizer.joblib"
    )
    llm_cache_dir: Path = field(
        default_factory=lambda: _env_path("ARTIFACTS_DIR", "artifacts") / "llm_cache"
    )

    # --- retrieval / llm ------------------------------------------------- #
    retrieval_top_k: int = field(default_factory=lambda: int(os.environ.get("RETRIEVAL_TOP_K", "3")))
    retrieval_min_similarity: float = field(
        default_factory=lambda: float(os.environ.get("RETRIEVAL_MIN_SIMILARITY", "0.1"))
    )
    retrieval_recency_decay_days: float = field(
        default_factory=lambda: float(os.environ.get("RETRIEVAL_RECENCY_DECAY_DAYS", "60"))
    )
    llm_temperature: float = 0.0
    llm_max_retries: int = 1
    preference_model_fallback_threshold: float = field(
        default_factory=lambda: float(os.environ.get("PREFERENCE_MODEL_FALLBACK_THRESHOLD", "0.5"))
    )


# Re-export helpers so callers (and tests) can resolve paths the same way.
__all__ = ["Settings", "load_env", "CODE_DIR", "REPO_DIR", "_repo_path", "_code_path"]