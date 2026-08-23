"""Offline evaluation of the router against dataset/sample_messages.csv.

The sample set carries ground-truth action/message_type/reason/confidence/
evidence_message_ids. Its message_ids are NOT part of messages.csv, so the
evaluator routes the sample rows through the SAME production pipeline that
writes output.csv (main.build_pipeline) and scores the resulting decisions
against the ground truth.

Metrics:
- action accuracy and message_type accuracy (overall + per message_type),
- evidence overlap: mean Jaccard of predicted vs expected evidence ids
  (computed only where the sample provides expected ids),
- confidence calibration: accuracy per confidence band plus the Pearson
  correlation between confidence and correctness,
- confusion matrices for action and message_type,
- stage-resolution breakdown per message_type (rules/preference/llm), so a
  weak type can be attributed to the responsible cascade stage.

Pure functions live here; the CLI wires the pipeline through the composition
root in main.py. No prediction logic is duplicated.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

from config import Settings, load_env
from ingestion import load_sample_messages
from models.decision import RoutingDecision
from models.message import SampleMessage

CALIBRATION_BANDS = [
    (0.50, 0.60),
    (0.60, 0.70),
    (0.70, 0.80),
    (0.80, 0.90),
    (0.90, 1.001),
]


# --------------------------------------------------------------------------- #
# pure metrics
# --------------------------------------------------------------------------- #
def pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation coefficient (pure python, no numpy)."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0.0 or vy == 0.0:
        return 0.0
    return cov / (vx * vy) ** 0.5


def _id_set(value: str) -> set[str]:
    return {i.strip() for i in value.split(";") if i.strip() and i.strip() != "none"}


def evidence_overlap(expected: str, predicted: str) -> Optional[float]:
    """Jaccard overlap of evidence id sets; None when no expected ids given."""
    exp, pred = _id_set(expected), _id_set(predicted)
    if not exp:
        return None
    if not pred:
        return 0.0
    return len(exp & pred) / len(exp | pred)


def confusion_matrix(actual: list[str], predicted: list[str]) -> tuple[list[str], dict[str, dict[str, int]]]:
    """Rows = ground truth, columns = predictions (sklearn convention)."""
    labels = sorted(set(actual) | set(predicted))
    matrix = {a: {p: 0 for p in labels} for a in labels}
    for a, p in zip(actual, predicted):
        matrix[a][p] += 1
    return labels, matrix


def _calibration_band(confidence: float) -> str:
    for lo, hi in CALIBRATION_BANDS:
        if lo <= confidence < hi:
            return f"{lo:.2f}-{hi if hi <= 1.0 else 1.0:.2f}"
    return "other"


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def score_samples(
    samples: list[SampleMessage],
    decisions: dict[str, RoutingDecision],
) -> dict[str, object]:
    """Score routed decisions against ground-truth samples. Requires a decision
    for every sample; raises otherwise (no silent drops)."""
    missing = [s.message_id for s in samples if s.message_id not in decisions]
    if missing:
        raise RuntimeError(f"No decision produced for samples: {missing}")

    correct_action = sum(1 for s in samples if decisions[s.message_id].action == s.action)
    correct_type = sum(1 for s in samples if decisions[s.message_id].message_type == s.message_type)

    overlaps = [
        ov
        for s in samples
        if (ov := evidence_overlap(s.evidence_message_ids, decisions[s.message_id].evidence_message_ids))
        is not None
    ]

    confidences = [decisions[s.message_id].confidence for s in samples]
    correctness = [1.0 if decisions[s.message_id].action == s.action else 0.0 for s in samples]

    bands: dict[str, dict[str, float]] = {}
    for band, correct, conf in zip(
        [_calibration_band(c) for c in confidences], correctness, confidences
    ):
        entry = bands.setdefault(band, {"n": 0.0, "correct": 0.0, "total_conf": 0.0})
        entry["n"] += 1
        entry["correct"] += correct
        entry["total_conf"] += conf

    action_labels, action_matrix = confusion_matrix(
        [s.action for s in samples], [decisions[s.message_id].action for s in samples]
    )
    type_labels, type_matrix = confusion_matrix(
        [s.message_type for s in samples], [decisions[s.message_id].message_type for s in samples]
    )

    # per-type: accuracy + stage attribution
    per_type: dict[str, dict[str, object]] = {}
    for s in samples:
        dec = decisions[s.message_id]
        entry = per_type.setdefault(
            s.message_type, {"n": 0, "correct": 0, "stages": Counter()}
        )
        entry["n"] += 1
        entry["correct"] += 1 if dec.action == s.action else 0
        entry["stages"][dec.resolved_by] += 1  # type: ignore[assignment]

    # per-stage accuracy
    per_stage: dict[str, dict[str, float]] = {}
    for s in samples:
        dec = decisions[s.message_id]
        entry = per_stage.setdefault(dec.resolved_by, {"n": 0.0, "correct": 0.0})
        entry["n"] += 1
        entry["correct"] += 1.0 if dec.action == s.action else 0.0

    return {
        "n": len(samples),
        "action_accuracy": correct_action / len(samples),
        "message_type_accuracy": correct_type / len(samples),
        "evidence_overlap_mean": sum(overlaps) / len(overlaps) if overlaps else None,
        "evidence_overlap_n": len(overlaps),
        "confidence_correctness_pearson": pearson(confidences, correctness),
        "calibration": {
            band: {"n": int(e["n"]), "accuracy": e["correct"] / e["n"] if e["n"] else 0.0}
            for band, e in sorted(bands.items())
        },
        "action_confusion_labels": action_labels,
        "action_confusion": action_matrix,
        "type_confusion_labels": type_labels,
        "type_confusion": type_matrix,
        "per_type": {
            t: {
                "n": e["n"],
                "action_accuracy": e["correct"] / e["n"],
                "stages": dict(e["stages"]),
            }
            for t, e in sorted(per_type.items())
        },
        "per_stage": {
            stage: {"n": int(e["n"]), "action_accuracy": e["correct"] / e["n"] if e["n"] else 0.0}
            for stage, e in sorted(per_stage.items())
        },
    }


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def render_report(report: dict[str, object]) -> str:
    n = report["n"]
    lines: list[str] = []
    lines.append(f"evaluated {n} samples")
    lines.append(
        f"action accuracy:      {report['action_accuracy']:.1%} "
        f"({int(report['action_accuracy'] * n)}/{n})"
    )
    lines.append(
        f"message_type accuracy: {report['message_type_accuracy']:.1%} "
        f"({int(report['message_type_accuracy'] * n)}/{n})"
    )
    ov = report["evidence_overlap_mean"]
    ov_n = report["evidence_overlap_n"]
    lines.append(f"evidence overlap (Jaccard): {ov:.2f} over {ov_n} samples with expected ids"
                 if ov is not None else "evidence overlap: no expected ids available")

    lines.append("")
    lines.append("confidence calibration (action correctness per band):")
    for band, entry in report["calibration"].items():  # type: ignore[index]
        lines.append(f"  conf {band:<9} acc={entry['accuracy']:.0%}  n={entry['n']}")
    lines.append(
        f"  pearson(confidence, correctness) = {report['confidence_correctness_pearson']:+.3f}"
    )

    lines.append("")
    lines.append("confusion matrix - action (rows=ground truth, cols=predicted):")
    labels = report["action_confusion_labels"]
    matrix = report["action_confusion"]
    lines.append("        " + "".join(f"{p:>9}" for p in labels))
    for a in labels:
        lines.append(f"{a:<8}" + "".join(f"{matrix[a][p]:>9}" for p in labels))  # type: ignore[index]

    lines.append("")
    lines.append("confusion matrix - message_type (rows=ground truth, cols=predicted):")
    tlabels = report["type_confusion_labels"]
    tmatrix = report["type_confusion"]
    lines.append("             " + "".join(f"{p:>12}" for p in tlabels))
    for a in tlabels:
        lines.append(f"{a:<12}" + "".join(f"{tmatrix[a][p]:>12}" for p in tlabels))  # type: ignore[index]

    lines.append("")
    lines.append("per message_type - action accuracy and producing stage:")
    lines.append(f"{'type':<15} {'n':>3} {'acc':>6}   rules  pref   llm")
    for t, entry in report["per_type"].items():  # type: ignore[index]
        stages = entry["stages"]
        lines.append(
            f"{t:<15} {entry['n']:>3} {entry['action_accuracy']:>5.0%}   "
            f"{stages.get('rules', 0):>4} {stages.get('preference_model', 0):>4} {stages.get('llm', 0):>4}"
        )

    lines.append("")
    lines.append("per stage - action accuracy:")
    for stage, entry in report["per_stage"].items():  # type: ignore[index]
        lines.append(f"  {stage:<16} acc={entry['action_accuracy']:.0%}  n={entry['n']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI: route samples through the production pipeline, then score
# --------------------------------------------------------------------------- #
def evaluate(predictions_path: str, sample_path: str) -> dict[str, object]:
    """Score generated output.csv against sample_messages.csv.

    Predictions are read from ``predictions_path`` when the sample ids appear
    there; otherwise the samples are routed through the production pipeline
    (main.build_pipeline) so every sample gets a real decision.
    """
    from main import build_pipeline

    load_env()
    settings = Settings()
    samples = load_sample_messages(sample_path)

    decisions: dict[str, RoutingDecision] = {}
    if Path(predictions_path).exists():
        with Path(predictions_path).open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if row["message_id"] in {s.message_id for s in samples}:
                    decisions[row["message_id"]] = RoutingDecision(
                        message_id=row["message_id"],
                        action=row["action"],
                        message_type=row["message_type"],
                        reason=row["reason"],
                        confidence=float(row["confidence"]),
                        evidence_message_ids=row["evidence_message_ids"],
                    )

    missing_ids = [s.message_id for s in samples if s.message_id not in decisions]
    if missing_ids:
        from orchestration.audit import DecisionAuditLog

        # in-memory audit: evaluation runs must not pollute the shared
        # artifacts/audit.jsonl (that is how sample ids leaked into the
        # degraded-ids dump in the past)
        repo, runner, orchestrator, audit = build_pipeline(settings, audit_log=DecisionAuditLog())
        samples_by_id = {s.message_id: s for s in samples}
        routed = runner.run([samples_by_id[i] for i in missing_ids])
        for decision in routed:
            decisions[decision.message_id] = decision

    return score_samples(samples, decisions)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    report = evaluate(
        str(repo_root / "dataset" / "output.csv"),
        str(repo_root / "dataset" / "sample_messages.csv"),
    )
    print(render_report(report))


if __name__ == "__main__":
    main()
