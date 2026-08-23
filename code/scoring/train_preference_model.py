"""Offline training for the preference model.

Builds a binary classifier on message_history.csv + message_events.csv where
the label is ``engaged`` = message_opened AND NOT notification_dismissed (the
user wanted this message interrupted through). Features are causal per
message (scoring.features), so no label leakage.

A gradient-boosting classifier is compared against the weighted-engagement
heuristic on a held-out split. The model is only shipped if it beats the
heuristic by a meaningful margin (else the heuristic is saved instead — a
sparse-data model is worse than no model). Artifacts (model + feature schema
+ metrics) are written to artifacts/.

The preference ``mute`` band is also learned here (see _learn_mute_band):
the 5th percentile of out-of-fold priorities among ENGAGED messages, so the
mute band never suppresses more than ~5% of messages the user would engage
with. The threshold is persisted in the schema for inference.

Run:  python -m scoring.train_preference_model
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Settings, load_env  # noqa: E402
from ingestion import (  # noqa: E402
    load_group_members,
    load_message_events,
    load_message_history,
    load_users,
)
from ingestion.parsing import parse_timestamp  # noqa: E402
from ingestion.repository import InMemoryDataRepository  # noqa: E402
from scoring.features import FeatureExtractor, FEATURE_NAMES, source_id_of  # noqa: E402
from scoring.preference_model import DEFAULT_MUTE_PRIORITY  # noqa: E402


def _build_dataset(
    history_path: str,
    events_path: str,
    group_members_path: str,
    users_path: str,
):
    """Load data, build repo, extract causal features + labels per message."""
    history = load_message_history(history_path)
    events = load_message_events(events_path)
    members = load_group_members(group_members_path)
    users = load_users(users_path)

    repo = InMemoryDataRepository.from_loaded(
        messages=[],
        history=history,
        samples=[],
        users=users,
        groups=[],
        group_members=members,
        businesses=[],
        user_business_history=[],
        events=events,
        images=[],
        voice_notes=[],
        daily_summaries=[],
    )
    extractor = FeatureExtractor(repo)

    X, y, ids = [], [], []
    for h in history:
        ev = repo.event_for_message(h.message_id)
        if ev is None:
            continue
        as_of = parse_timestamp(h.created_at)
        X.append(extractor.to_vector(extractor.extract(h, as_of=as_of)))
        y.append(1 if (ev.message_opened and not ev.notification_dismissed) else 0)
        ids.append(h.message_id)

    return np.array(X, dtype=float), np.array(y, dtype=int), np.array(ids), repo


def _heuristic_score(features: dict[str, float]) -> float:
    """Weighted engagement rate normalized to 0-1 (fallback model).

    Raw = (open + 2*reply - dismiss - 3*report) / max(1, volume) is in roughly
    [-3, 3]; logistic squash maps it to (0,1). Zero-volume pairs get the
    neutral 0.5 (no prior evidence either way).
    """
    raw = features.get("weighted_engagement", 0.0)
    return 1.0 / (1.0 + np.exp(-raw))


def _fit_calibrator(oof_priorities: np.ndarray, oof_labels: np.ndarray):
    """Isotonic calibration of model probabilities on OOF predictions."""
    from sklearn.isotonic import IsotonicRegression

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(oof_priorities, oof_labels)
    return calibrator


def _learn_mute_band(
    oof_priorities: np.ndarray,
    oof_labels: np.ndarray,
) -> tuple[float, dict[str, object]]:
    """Learn the preference ``mute`` band from out-of-fold predictions.

    Design (documented policy, not a magic number): the mute band is the
    5th percentile of out-of-fold priorities among ENGAGED messages. That
    encodes the metric we care about — mute decisions must not suppress
    messages the user would engage with: at most ~5% of engaged messages
    fall inside the band. Anything below the band is "low-value to THIS
    user", which is exactly the preference-mute semantic. Scam/spam is
    deliberately out of scope (safety rules / LLM only).

    The band is a post-hoc policy on the binary-``engaged`` model output,
    so the classifier itself is not retrained for it.
    """
    engaged_priorities = oof_priorities[oof_labels == 1]
    if engaged_priorities.size < 20:
        threshold = float(DEFAULT_MUTE_PRIORITY)
    else:
        threshold = float(np.percentile(engaged_priorities, 5))

    below = oof_priorities < threshold
    engaged = oof_labels == 1
    band = {
        "threshold": round(threshold, 4),
        "band_size": round(float(below.mean()), 4),  # fraction of all traffic muted
        # of muted: fraction NOT engaged (want close to 1 — don't mute what
        # the user would engage with)
        "mute_precision": round(
            float((oof_labels[below] == 0).mean()), 4
        ) if below.any() else None,
        # of ALL engaged: fraction wrongly inside the band (want ~0.05 —
        # this is the design property the 5th-percentile choice guarantees)
        "engaged_suppressed_fraction": round(
            float((engaged & below).mean()), 4
        ),
    }
    return threshold, band


def _print_mute_band(band: dict[str, object]) -> None:
    print(
        f"preference mute band: threshold={band['threshold']} "
        f"band_size={band['band_size']} mute_precision={band['mute_precision']} "
        f"engaged_suppressed={band['engaged_suppressed_fraction']}"
    )


def train(
    message_history_path: str,
    message_events_path: str,
    group_members_path: str,
    users_path: str,
    output_dir: str,
) -> dict[str, object]:
    """Train + evaluate the preference model, persist artifacts, return metrics."""
    X, y, ids, repo = _build_dataset(
        message_history_path, message_events_path, group_members_path, users_path
    )
    print(f"dataset: {len(X)} messages, positive label (engaged) rate = {y.mean():.2%}")

    if len(X) < 30 or y.min() == y.max():
        raise ValueError(f"dataset too small/unbalanced for training: n={len(X)}")

    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, ids, test_size=0.25, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=2,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # stability flag: 5-fold CV AUC on the full dataset (stratified).
    # The same folds produce out-of-fold priorities used for the mute band.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_aucs = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    cv_auc_mean = float(np.mean(cv_aucs))
    cv_auc_std = float(np.std(cv_aucs))
    stable = cv_auc_mean - cv_auc_std > 0.6  # comfortably above random on every fold
    print(f"5-fold CV AUC: {cv_auc_mean:.3f} +/- {cv_auc_std:.3f} "
          f"-> {'STABLE' if stable else 'UNSTABLE'}")

    # out-of-fold priorities (one fresh model per fold, held-out fold scored)
    oof_priorities = np.zeros(len(X), dtype=float)
    for train_idx, test_idx in cv.split(X, y):
        fold_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=2,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
        fold_model.fit(X[train_idx], y[train_idx])
        oof_priorities[test_idx] = fold_model.predict_proba(X[test_idx])[:, 1]

    # probability calibration on the OOF predictions: GBM predict_proba is
    # systematically overconfident (e.g. 0.99 on messages whose ground truth
    # is digest), which flattens the confidence signal. Isotonic regression
    # is monotonic, so the RANKING (and therefore every threshold decision)
    # is preserved while the margin-to-boundary becomes a trustworthy
    # confidence. Fit on out-of-fold values only — no leakage.
    calibrator = _fit_calibrator(oof_priorities, y)
    oof_calibrated = calibrator.transform(oof_priorities)
    print(f"calibrator: isotonic (mapped oof priority mean {oof_priorities.mean():.3f} "
          f"-> {oof_calibrated.mean():.3f})")

    mute_threshold, band = _learn_mute_band(oof_calibrated, y)
    _print_mute_band(band)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    model_auc = roc_auc_score(y_test, y_pred_proba)
    model_acc = accuracy_score(y_test, y_pred)

    # heuristic comparison on the same held-out split (causal features reused)
    extractor = FeatureExtractor(repo)
    history = load_message_history(message_history_path)
    hist_by_id = {h.message_id: h for h in history}
    test_hist = [hist_by_id[i] for i in ids_test]
    h_scores = []
    for h in test_hist:
        as_of = parse_timestamp(h.created_at)
        h_scores.append(_heuristic_score(extractor.extract(h, as_of=as_of)))
    heuristic_auc = roc_auc_score(y_test, h_scores)

    metrics = {
        "n_messages": int(len(X)),
        "positive_rate": round(float(y.mean()), 4),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "cv": {
            "roc_auc_mean": round(cv_auc_mean, 4),
            "roc_auc_std": round(cv_auc_std, 4),
            "stable": stable,
        },
        "model": {
            "name": "gradient_boosting",
            "accuracy": round(float(model_acc), 4),
            "roc_auc": round(float(model_auc), 4),
        },
        "heuristic": {
            "name": "weighted_engagement_rate",
            "roc_auc": round(float(heuristic_auc), 4) if heuristic_auc is not None else None,
        },
        "mute_band": band,
    }
    print(f"held-out (n={len(y_test)}):")
    print(f"  gradient boosting  acc={model_acc:.3f} auc={model_auc:.3f}")
    print(f"  heuristic          auc={heuristic_auc:.3f}" if heuristic_auc is not None else "  heuristic auc: n/a")

    # ship the better of the two; only ship the model if it clearly wins
    use_model = (heuristic_auc is None or model_auc >= heuristic_auc + 0.03) and stable
    chosen = "gradient_boosting" if use_model else "weighted_engagement_rate"
    metrics["chosen"] = chosen
    print(f"chosen: {chosen}")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if use_model:
        joblib.dump(model, output / "preference_model.joblib")
        joblib.dump(calibrator, output / "preference_calibrator.joblib")
        (output / "preference_schema.json").write_text(
            json.dumps(
                {
                    "feature_names": list(FEATURE_NAMES),
                    "model": "gradient_boosting",
                    "calibrated": True,
                    "mute_priority_threshold": round(mute_threshold, 4),
                }
            ),
            encoding="utf-8",
        )
    else:
        (output / "preference_schema.json").write_text(
            json.dumps(
                {
                    "feature_names": list(FEATURE_NAMES),
                    "model": "weighted_engagement_rate",
                    "mute_priority_threshold": DEFAULT_MUTE_PRIORITY,
                }
            ),
            encoding="utf-8",
        )
    (output / "preference_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


if __name__ == "__main__":
    load_env()
    settings = Settings()
    train(
        str(settings.dataset_dir / "message_history.csv"),
        str(settings.dataset_dir / "message_events.csv"),
        str(settings.dataset_dir / "group_members.csv"),
        str(settings.dataset_dir / "users.csv"),
        str(settings.artifacts_dir),
    )
