"""Preference-scoring contracts.

The PreferenceModel Protocol is the single scoring surface used by the
decision cascade. It predicts how likely the user is to ENGAGE with a message
from a given source, based on offline training over message_history.csv +
message_events.csv (no online learning — there is no live user in the loop
during evaluation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class PreferenceScore:
    """Output of one preference-scoring call.

    priority:   0-1 notify-worthiness (higher = more likely the user wants
                this message interrupted through).
    confidence: 0-1 — decision clarity times observation strength: how far
                the priority sits from the boundary that decided the action
                (mute band / notify line), discounted by how much history
                the (user, source) pair has. Sparse pairs and hairline
                decisions score low; clear decisions on well-observed pairs
                score high. This is what the cascade's PREF_CONFIDENCE_MIN
                gate consumes.
    """

    priority: float
    confidence: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.priority <= 1.0):
            raise ValueError(f"priority must be in [0, 1], got {self.priority}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


@runtime_checkable
class PreferenceModel(Protocol):
    """Scores how much the user wants messages from a source interrupted through.

    Implementations may be model-backed (trained classifier) or heuristic
    (smoothed engagement rates); callers only depend on this Protocol.
    """

    #: Priority BELOW this threshold maps to a preference-stage ``mute``:
    #: content is low-value/repetitive/uninteresting to this user — NOT
    #: unsafe. Scam/spam stays the exclusive domain of the safety rules
    #: and the LLM. Learned offline by scoring.train_preference_model
    #: (documented per model; see preference_model.DEFAULT_MUTE_PRIORITY).
    mute_priority_threshold: float

    def score(self, user_id: str, source_id: str, features: dict[str, float]) -> PreferenceScore:
        """Score one incoming message.

        Args:
            user_id: the receiving user (user_id in messages.csv).
            source_id: sender_user_id, group_id or business_id of the message.
            features: precomputed feature dict (see scoring.features).

        Returns:
            PreferenceScore with priority 0-1 and confidence 0-1.
        """
        ...

    def is_available(self) -> bool:
        """True if this model can produce scores right now."""
        ...
