"""scoring: offline-trained preference model for notify-worthiness."""

from .interfaces import PreferenceModel, PreferenceScore
from .features import FeatureExtractor, FEATURE_NAMES, source_id_of, quiet_hours_overlap
from .preference_model import (
    HeuristicPreferenceModel,
    TrainedPreferenceModel,
    build_preference_model,
)

__all__ = [
    "PreferenceModel",
    "PreferenceScore",
    "FeatureExtractor",
    "FEATURE_NAMES",
    "source_id_of",
    "quiet_hours_overlap",
    "HeuristicPreferenceModel",
    "TrainedPreferenceModel",
    "build_preference_model",
]
