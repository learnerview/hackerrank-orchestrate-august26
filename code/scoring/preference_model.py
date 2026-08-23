"""Inference-time preference model (loaded from training artifacts).

Loads what scoring.train_preference_model produced:
- gradient_boosting -> TrainedPreferenceModel (sklearn classifier)
- weighted_engagement_rate -> HeuristicPreferenceModel (logistic-squashed
  weighted engagement rate — the sparsity fallback)

Both implement the PreferenceModel Protocol. The FeatureExtractor must be
provided by the caller so inference features match training features exactly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib

from scoring.features import FeatureExtractor
from scoring.interfaces import PreferenceModel, PreferenceScore

#: Priority below this maps to a preference-stage ``mute`` when a model does
#: not expose a learned band (heuristic model, or schema missing the key).
#: The value is documented, not magic: the logistic squash maps raw
#: weighted-engagement 0.0 (a neutral pair — no positive or negative signal)
#: to priority 0.5; raw -0.85 (weighted dismissals/reports outweighing opens)
#: lands at ~0.30. Anything below that is a *clearly negative* engagement
#: profile — the user demonstrably does not want this sender interrupting.
DEFAULT_MUTE_PRIORITY = 0.30


def map_priority_to_action(
    priority: float,
    mute_priority: float,
    notify_priority: float,
) -> str:
    """Map a preference priority to a three-way routing action.

    Bands, in priority order:

    - ``priority < mute_priority``      -> ``mute``  (low-value/repetitive
      for THIS user; never used for scam/spam — those need the safety rules
      or the LLM, whose message_type vocabulary includes them)
    - ``priority >= notify_priority``   -> ``notify``
    - otherwise                         -> ``digest``

    ``notify_priority`` is passed by the orchestrator (cascade constant
    ``PREF_NOTIFY_PRIORITY``) so this pure function carries no duplicated
    magic numbers; ``mute_priority`` is the model's own documented band.
    """
    if priority < mute_priority:
        return "mute"
    if priority >= notify_priority:
        return "notify"
    return "digest"


#: How much a promotional/greeting content signal suppresses notify-worthiness.
#: A fully promotional message (promo_signal = 1.0) loses at most 35% of its
#: priority; a mildly promotional one (signal 0.5) loses 17.5%. The number is
#: documented against the cascade contract, not magic: a priority of ~0.92
#: (typical high-open-rate sender) with signal 1.0 lands at ~0.60 — below the
#: notify threshold 0.65 but above the mute band, i.e. content OVERRIDES the
#: stale sender-history lift into a digest, while genuinely wanted content
#: (priority >= 0.95, e.g. engaged business offers) still notifies.
PROMO_SUPPRESSION = 0.35


#: Mirrors orchestration.router.PREF_NOTIFY_PRIORITY (asserted equal by a
#: unit test). The scoring layer needs the notify boundary for confidence
#: geometry but must NOT import the orchestration layer (layering); the
#: router's constant remains authoritative for the action mapping.
PREFERENCE_NOTIFY_PRIORITY = 0.65

#: One full "band gap": how far a maximally decisive priority sits from its
#: decision boundary (priority 1.0 vs the 0.65 notify line; priority 0.0 vs
#: the learned mute band ~0.37). Confidence normalizes by this gap so a
#: hairline decision scores ~0.5 and a maximally clear one scores 1.0.
DECISION_MARGIN_FULL = 0.55
DECISION_SHARPNESS = 1.5

#: History volume at which a (user, source) pair is treated as fully
#: observed for confidence purposes.
OBSERVATION_VOLUME = 10.0


def _decision_confidence(
    priority: float,
    mute_priority: float,
    notify_priority: float,
    volume: float,
) -> float:
    """Confidence = decision clarity x observation strength.

    Clarity is the distance from the priority to the boundary that ACTUALLY
    decided the action — the mute band for a mute, the notify line for a
    notify, and for a digest the NEAREST of the two (a digest at 0.5 is
    farther from both than a digest at 0.64, and its confidence must say
    so). The old formula measured distance from the classification midpoint
    (0.5), which is the wrong reference point: a hairline notify at 0.66
    got margin 0.32 and high confidence.

    Observation strength discounts sparse pairs: predicting from 1 message
    is inherently less reliable than from 24, so a cold pair can never pass
    the cascade's PREF_CONFIDENCE_MIN gate.

    The full-scale is set BEYOND the achievable margin range (a priority in
    [0,1] against a notify line at 0.65 can never produce a margin above
    0.35), so clarity never reaches 1.0: preference decisions are capped
    below rule-level certainty. Rules (deterministic, audited patterns) may
    legitimately claim 0.95; an engagement-history model must not.
    """
    if priority < mute_priority:
        margin = mute_priority - priority
    elif priority >= notify_priority:
        margin = priority - notify_priority
    else:
        margin = min(notify_priority - priority, priority - mute_priority)
    clarity = min(1.0, margin / DECISION_MARGIN_FULL) ** DECISION_SHARPNESS
    obs = min(1.0, volume / OBSERVATION_VOLUME)
    return (0.5 + 0.5 * clarity) * (0.5 + 0.5 * obs)


def _suppress_for_content(priority: float, features: dict[str, float]) -> float:
    """Content override: promo/greeting content pulls priority DOWN.

    This is the "content beats stale engagement history" policy: a sender
    with a high historical open-rate lifts EVERY message they send, even the
    promotional ones; the message-level promo_signal re-weights today's
    message by what it actually IS. Suppression is proportional to the
    signal and can only reduce notify-worthiness, never increase it.
    """
    promo = float(features.get("promo_signal", 0.0))
    if promo <= 0.0:
        return priority
    return priority * (1.0 - PROMO_SUPPRESSION * promo)


def _squash(raw: float) -> float:
    """Logistic squash of the raw weighted-engagement rate into (0,1)."""
    import numpy as np

    return float(1.0 / (1.0 + np.exp(-raw)))


class HeuristicPreferenceModel:
    """Weighted-engagement-rate fallback for sparse data.

    score = squash((open + 2*reply - dismiss - 3*report) / max(1, volume)).
    Confidence scales with log-volume: a pair with 0 history gets low
    confidence; >= 10 messages is treated as well-observed.

    mute_priority_threshold is a fixed documented constant (no training data):
    DEFAULT_MUTE_PRIORITY, i.e. clearly negative weighted engagement.
    """

    mute_priority_threshold: float = DEFAULT_MUTE_PRIORITY

    def __init__(self) -> None:
        pass

    @staticmethod
    def is_available() -> bool:
        return True

    def score(self, user_id: str, source_id: str, features: dict[str, float]) -> PreferenceScore:
        raw = float(features.get("weighted_engagement", 0.0))
        volume = float(features.get("volume", 0.0))
        priority = _suppress_for_content(_squash(raw), features)
        confidence = _decision_confidence(
            priority, self.mute_priority_threshold, PREFERENCE_NOTIFY_PRIORITY, volume
        )
        return PreferenceScore(priority=priority, confidence=confidence)


class TrainedPreferenceModel(PreferenceModel):
    """Gradient-boosting preference model loaded from artifacts/.

    priority = P(engaged) from the classifier; confidence is a calibrated
    blend of the model's margin and how much history the (user, source) pair
    has (a confident prediction on 1 message is still low-confidence).

    mute_priority_threshold is learned offline by train_preference_model.py
    (5th percentile of out-of-fold engaged priorities — at most ~5% of
    messages the user would engage with can fall in the mute band) and
    persisted in preference_schema.json. Models trained before the band
    existed fall back to the documented DEFAULT_MUTE_PRIORITY.
    """

    def __init__(self, model_path: str) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"preference model artifact not found: {path}")
        self._model = joblib.load(path)
        schema = path.with_name("preference_schema.json")
        if schema.exists():
            self._schema = json.loads(schema.read_text(encoding="utf-8"))
        else:
            self._schema = {}
        self._feature_names = list(self._schema.get("feature_names", []))
        self.mute_priority_threshold = float(
            self._schema.get("mute_priority_threshold", DEFAULT_MUTE_PRIORITY)
        )
        calibrator_path = path.with_name("preference_calibrator.joblib")
        self._calibrator = (
            joblib.load(calibrator_path) if calibrator_path.exists() else None
        )

    def is_available(self) -> bool:
        return self._model is not None

    def _to_vector(self, features: dict[str, float]) -> list[float]:
        if self._feature_names:
            return [float(features.get(name, 0.0)) for name in self._feature_names]
        return [float(features.get(name, 0.0)) for name in FeatureExtractor.feature_names()]

    def score(self, user_id: str, source_id: str, features: dict[str, float]) -> PreferenceScore:
        vector = [self._to_vector(features)]
        raw_priority = float(self._model.predict_proba(vector)[0, 1])
        # isotonic calibration (trained on out-of-fold predictions): GBM
        # probabilities are systematically overconfident; the calibrated
        # value is the one distances/confidence should be measured from
        if self._calibrator is not None:
            raw_priority = float(self._calibrator.transform([raw_priority])[0])
        # content override happens AFTER the model: promo/greeting content
        # can pull today's message down even when sender history says "loved"
        priority = _suppress_for_content(raw_priority, features)

        volume = float(features.get("volume", 0.0))
        confidence = _decision_confidence(
            priority, self.mute_priority_threshold, PREFERENCE_NOTIFY_PRIORITY, volume
        )
        return PreferenceScore(priority=priority, confidence=confidence)


def build_preference_model(
    artifacts_dir: str, features: dict[str, float] | None = None
) -> PreferenceModel:
    """Construct the model the training script chose (schema decides)."""
    artifacts = Path(artifacts_dir)
    schema_path = artifacts / "preference_schema.json"
    if not schema_path.exists():
        return HeuristicPreferenceModel()
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if schema.get("model") == "gradient_boosting":
        return TrainedPreferenceModel(str(artifacts / "preference_model.joblib"))
    return HeuristicPreferenceModel()


__all__ = [
    "PreferenceModel",
    "PreferenceScore",
    "DEFAULT_MUTE_PRIORITY",
    "map_priority_to_action",
    "PROMO_SUPPRESSION",
    "HeuristicPreferenceModel",
    "TrainedPreferenceModel",
    "build_preference_model",
]
