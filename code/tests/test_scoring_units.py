"""Unit tests for the scoring layer: feature causality, heuristic fallback,
and the trained-model Protocol. No dataset files required.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from models import (  # noqa: E402
    GroupMemberContext,
    HistoricalMessage,
    IncomingMessage,
    MessageEvent,
    UserProfile,
)
from ingestion.repository import InMemoryDataRepository  # noqa: E402
from scoring.features import (  # noqa: E402
    FeatureExtractor,
    promo_signal,
    quiet_hours_overlap,
)
from scoring.interfaces import PreferenceScore  # noqa: E402
from scoring.preference_model import (  # noqa: E402
    DEFAULT_MUTE_PRIORITY,
    HeuristicPreferenceModel,
    map_priority_to_action,
)


def _hist(mid, sender, created, group=None, biz=None):
    return HistoricalMessage(
        message_id=mid,
        user_id="u1",
        conversation_type="group" if group else "personal",
        group_id=group,
        business_id=biz,
        sender_user_id=sender,
        created_at=created,
        message_text="hello",
        media_type="text",
        media_id=None,
        forwarded_count=0,
    )


def _event(mid, opened=True, replied=False, dismissed=False, reported=False):
    return MessageEvent(
        user_id="u1",
        message_id=mid,
        message_opened=opened,
        message_replied=replied,
        reaction_time_minutes=1,
        notification_dismissed=dismissed,
        muted_after_message=False,
        message_reported=reported,
    )


def _msg(text="hi", sender="s1", hour="2026-07-25 14:00:00", conv="personal", fwd=0):
    return IncomingMessage(
        message_id="m_in",
        user_id="u1",
        conversation_type=conv,
        group_id=None,
        business_id=None,
        sender_user_id=sender,
        created_at=hour,
        message_text=text,
        media_type="text",
        media_id=None,
        forwarded_count=fwd,
    )


def _repo(history, events, users=None, members=None):
    return InMemoryDataRepository.from_loaded(
        messages=[],
        history=history,
        samples=[],
        users=users or [],
        groups=[],
        group_members=members or [],
        businesses=[],
        user_business_history=[],
        events=events,
        images=[],
        voice_notes=[],
        daily_summaries=[],
    )


def test_quiet_hours_overlap_wrap_around():
    assert quiet_hours_overlap(23, "22:00-07:00") == 1.0
    assert quiet_hours_overlap(3, "22:00-07:00") == 1.0
    assert quiet_hours_overlap(12, "22:00-07:00") == 0.0
    assert quiet_hours_overlap(9, "08:00-18:00") == 1.0
    assert quiet_hours_overlap(20, None) == 0.0


def test_features_are_causal_exclude_current_message():
    # one opened + replied message BEFORE the query -> strong features
    history = [_hist("h1", "s1", "2026-07-01 10:00:00")]
    events = [_event("h1", opened=True, replied=True)]
    repo = _repo(history, events)
    fx = FeatureExtractor(repo)

    feats = fx.extract(_msg(sender="s1"))
    assert feats["volume"] == 1.0
    assert feats["open_rate"] == 1.0
    assert feats["reply_rate"] == 1.0
    assert feats["weighted_engagement"] == pytest.approx((1 + 2 - 0 - 0) / 1)

    # the message itself must never be its own evidence
    self_feats = fx.extract(history[0], as_of=None)
    assert self_feats["volume"] == 0.0


def test_features_include_message_level_signals():
    history = [_hist("h1", "s1", "2026-07-01 10:00:00")]
    repo = _repo(history, [_event("h1")])
    fx = FeatureExtractor(repo)
    feats = fx.extract(_msg(text="@u1 check this", hour="2026-07-25 23:00:00"))
    assert feats["at_mention"] == 1.0
    assert feats["is_night"] == 1.0
    assert feats["is_morning"] == 0.0
    assert feats["is_business"] == 0.0


def test_sender_role_and_group_muted_from_members():
    members = [
        GroupMemberContext(
            group_id="g1",
            user_id="s1",
            role="admin",
            joined_at="2026-01-01",
            messages_sent_30d=10,
            messages_read_30d=10,
            replies_sent_30d=2,
            notifications_dismissed_30d=1,
            group_muted_by_user=False,
        ),
        GroupMemberContext(
            group_id="g1",
            user_id="u1",
            role="member",
            joined_at="2026-01-01",
            messages_sent_30d=1,
            messages_read_30d=10,
            replies_sent_30d=1,
            notifications_dismissed_30d=0,
            group_muted_by_user=True,
        ),
    ]
    repo = _repo([], [], members=members)
    fx = FeatureExtractor(repo)
    msg = IncomingMessage(
        message_id="m_g",
        user_id="u1",
        conversation_type="group",
        group_id="g1",
        business_id=None,
        sender_user_id="s1",
        created_at="2026-07-25 10:00:00",
        message_text="meeting at 3",
        media_type="text",
        media_id=None,
        forwarded_count=0,
    )
    feats = fx.extract(msg)
    assert feats["sender_is_admin"] == 1.0
    assert feats["group_muted"] == 1.0


def test_heuristic_model_bounds_and_sparsity():
    model = HeuristicPreferenceModel()
    assert isinstance(model, HeuristicPreferenceModel)
    assert model.is_available()

    high = model.score("u1", "s1", {"weighted_engagement": 3.0, "volume": 10.0})
    low = model.score("u1", "s1", {"weighted_engagement": -3.0, "volume": 10.0})
    cold = model.score("u1", "s2", {"weighted_engagement": 0.0, "volume": 0.0})
    warm = model.score("u1", "s3", {"weighted_engagement": 0.0, "volume": 5.0})

    assert isinstance(high, PreferenceScore)
    assert high.priority > 0.7
    assert low.priority < 0.3
    assert high.confidence > cold.confidence  # well-observed pair -> higher confidence
    assert warm.confidence > cold.confidence
    assert cold.confidence < 0.6  # no history -> can never pass the cascade gate


def test_trained_model_loads_from_artifacts():
    artifacts = Path(__file__).resolve().parent.parent / "artifacts"
    if not (artifacts / "preference_schema.json").exists():
        pytest.skip("artifacts not trained yet")
    from scoring.preference_model import build_preference_model

    model = build_preference_model(str(artifacts))
    assert model.is_available()
    score = model.score(
        "u1", "s1", {"weighted_engagement": 2.0, "volume": 5.0}
    )
    assert 0.0 <= score.priority <= 1.0
    assert 0.0 <= score.confidence <= 1.0
    # the learned mute band must be exposed (schema contract) and well-formed
    assert 0.0 < model.mute_priority_threshold <= 0.5
    import json

    schema = json.loads((artifacts / "preference_schema.json").read_text(encoding="utf-8"))
    assert "mute_priority_threshold" in schema
    assert model.mute_priority_threshold == schema["mute_priority_threshold"]


def test_feature_names_stable():
    from scoring.features import FEATURE_NAMES

    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))
    assert "open_rate" in FEATURE_NAMES
    assert "quiet_hours_overlap" in FEATURE_NAMES


def test_map_priority_to_action_three_bands():
    # below the mute band -> mute; at/above notify -> notify; else digest
    assert map_priority_to_action(0.10, mute_priority=0.25, notify_priority=0.65) == "mute"
    assert map_priority_to_action(0.24, mute_priority=0.25, notify_priority=0.65) == "mute"
    assert map_priority_to_action(0.30, mute_priority=0.25, notify_priority=0.65) == "digest"
    assert map_priority_to_action(0.64, mute_priority=0.25, notify_priority=0.65) == "digest"
    assert map_priority_to_action(0.65, mute_priority=0.25, notify_priority=0.65) == "notify"
    assert map_priority_to_action(0.99, mute_priority=0.25, notify_priority=0.65) == "notify"


def test_map_priority_to_action_thresholds_are_explicit():
    # both thresholds must be passed: no hidden magic numbers in the mapping
    with pytest.raises(TypeError):
        map_priority_to_action(0.5)  # type: ignore[call-arg]


def test_heuristic_model_exposes_documented_mute_band():
    model = HeuristicPreferenceModel()
    assert model.mute_priority_threshold == DEFAULT_MUTE_PRIORITY
    assert model.mute_priority_threshold == pytest.approx(0.30)
    # raw -0.85 (weighted dismissals/reports dominate) sits just below the band
    low = model.score("u1", "s1", {"weighted_engagement": -0.85, "volume": 10.0})
    assert low.priority < model.mute_priority_threshold
    # a neutral pair (0.5) is digest territory, not mute
    neutral = model.score("u1", "s1", {"weighted_engagement": 0.0, "volume": 10.0})
    assert neutral.priority >= model.mute_priority_threshold


# --------------------------------------------------------------------------- #
# promo content signal + stale-engagement override
# --------------------------------------------------------------------------- #
def test_promo_signal_strong_content_saturates():
    # 2+ distinct pattern families -> 1.0 (unambiguous promotional content)
    assert promo_signal("Get 50% off now! Hurry, offer expires tonight.") == 1.0
    assert promo_signal(
        "Happy new year! Wishing you a wonderful festival. Stay positive, keep smiling."
    ) == 1.0
    assert promo_signal("Good morning. Stay positive, keep smiling.") == 1.0
    # a single family is ambiguous -> 0.5 (mild dampening only)
    assert promo_signal("I'm selling my old bike.") == 0.5
    assert promo_signal("Good morning everyone.") == 0.5
    # no signal -> 0.0 (personal messages untouched)
    assert promo_signal("Pick up some milk on the way home?") == 0.0
    assert promo_signal("") == 0.0


def test_promo_signal_media_intent_hint_is_strongest():
    assert promo_signal("", intent_hint="promotional_poster") == 1.0
    assert promo_signal("", intent_hint="greeting_card") == 1.0
    assert promo_signal("", intent_hint="payment_qr") == 0.0
    assert promo_signal("thanks for the lunch invite", intent_hint="other") == 0.0


def test_extract_uses_text_and_intent_hints_inference_only():
    repo = _repo([], [])
    fx = FeatureExtractor(repo)
    msg = _msg(text="see the poster file")
    assert fx.extract(msg)["promo_signal"] == 0.0
    # OCR text from the poster carries the keywords (inference-only hint)
    assert fx.extract(msg, text_hint="see the poster file\n50% off storewide, hurry")["promo_signal"] == 1.0
    # vision already classified the poster -> direct strong signal
    assert fx.extract(msg, intent_hint="promotional_poster")["promo_signal"] == 1.0
    assert "promo_signal" in fx.extract(msg)


def test_heuristic_suppresses_promotional_content():
    model = HeuristicPreferenceModel()
    high = model.score("u1", "s1", {"weighted_engagement": 3.0, "volume": 10.0})
    promo = model.score(
        "u1", "s1",
        {"weighted_engagement": 3.0, "volume": 10.0, "promo_signal": 1.0},
    )
    mild = model.score(
        "u1", "s1",
        {"weighted_engagement": 3.0, "volume": 10.0, "promo_signal": 0.5},
    )
    # content pulls priority down even though sender history is unchanged
    assert promo.priority == pytest.approx(high.priority * 0.65)
    assert mild.priority == pytest.approx(high.priority * 0.825)
    assert promo.priority < mild.priority < high.priority
    # but the dampened value is never below the mute band for a strong pair
    assert promo.priority > model.mute_priority_threshold


# --------------------------------------------------------------------------- #
# confidence calibration: distance from the DECISION boundary, not from 0.5
# --------------------------------------------------------------------------- #
def test_decision_confidence_measures_boundary_distance():
    from scoring.preference_model import _decision_confidence

    mute, notify, vol = 0.37, 0.65, 24.0
    # a hairline notify just above the line is LOW confidence (most error-prone)
    assert _decision_confidence(0.66, mute, notify, vol) < 0.6
    # a clear notify is high-confidence (capped below rule-level certainty:
    # the full-scale 0.55 exceeds the max achievable margin of 0.35)
    assert _decision_confidence(0.98, mute, notify, vol) > 0.7
    # a digest uses the NEAREST boundary: 0.5 is further from both than 0.64
    assert _decision_confidence(0.50, mute, notify, vol) > _decision_confidence(0.64, mute, notify, vol)
    # a deep mute is high confidence, a hairline mute is low
    assert _decision_confidence(0.05, mute, notify, vol) > 0.7
    assert _decision_confidence(0.36, mute, notify, vol) < 0.6
    # sparse pairs are discounted below the cascade gate
    assert _decision_confidence(0.98, mute, notify, 1.0) < 0.6
    assert _decision_confidence(0.98, mute, notify, 10.0) > 0.7


def test_decision_confidence_monotonic_in_clearness():
    from scoring.preference_model import _decision_confidence

    mute, notify, vol = 0.37, 0.65, 24.0
    # all three decision arms: confidence rises as the priority moves away
    # from the boundary that decides it
    arm_notify = [_decision_confidence(p, mute, notify, vol) for p in (0.66, 0.75, 0.90, 0.99)]
    assert arm_notify == sorted(arm_notify)
    arm_mute = [_decision_confidence(p, mute, notify, vol) for p in (0.35, 0.25, 0.10, 0.01)]
    assert arm_mute == sorted(arm_mute)  # deeper mute -> higher confidence
    # digest confidence is a V-shape in priority: highest mid-band (0.5),
    # lowest at BOTH edges (a digest hugging the mute band is also hairline)
    assert _decision_confidence(0.50, mute, notify, vol) > _decision_confidence(0.55, mute, notify, vol)
    assert _decision_confidence(0.50, mute, notify, vol) > _decision_confidence(0.42, mute, notify, vol)


def test_scoring_notify_constant_mirrors_cascade():
    from orchestration.router import PREF_NOTIFY_PRIORITY
    from scoring.preference_model import PREFERENCE_NOTIFY_PRIORITY

    assert PREFERENCE_NOTIFY_PRIORITY == PREF_NOTIFY_PRIORITY


def test_heuristic_confidence_moves_with_decision_clearness():
    model = HeuristicPreferenceModel()
    # same sender volume, three different weighted engagements -> confidence
    # must differ because the DECISION is clear-cut vs hairline
    clear_notify = model.score("u1", "s1", {"weighted_engagement": 3.0, "volume": 10.0})
    hairline_digest = model.score("u1", "s1", {"weighted_engagement": 0.5, "volume": 10.0})
    assert clear_notify.confidence > 0.7
    assert hairline_digest.confidence < clear_notify.confidence
    # and the hairline one still cannot pass the cascade gate
    from orchestration.router import PREF_CONFIDENCE_MIN

    assert hairline_digest.confidence < PREF_CONFIDENCE_MIN
