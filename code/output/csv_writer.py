"""Output writer: RoutingDecision -> output.csv.

Single responsibility: render a list of routing decisions to the exact
required schema. The writer knows nothing about rules, models or the LLM —
it only knows how to serialize a RoutingDecision deterministically.

Schema (exact columns, this order):
    message_id,action,message_type,reason,confidence,evidence_message_ids

Guarantees:
- one row per decision, in input order,
- ``evidence_message_ids`` is semicolon-separated or the literal string
  ``none`` when no evidence exists (empty values are normalized, never left
  blank),
- ``confidence`` is rendered without exponent notation,
- UTF-8 with ``\\n`` line endings and no BOM.
"""

from __future__ import annotations

import csv
from pathlib import Path

from models.decision import RoutingDecision

REQUIRED_COLUMNS = [
    "message_id",
    "action",
    "message_type",
    "reason",
    "confidence",
    "evidence_message_ids",
]


def _format_confidence(value: float) -> str:
    """Deterministic decimal rendering (no scientific notation, no trailing zeros)."""
    rendered = f"{value:.4f}".rstrip("0").rstrip(".")
    return rendered if rendered else "0"


def _format_evidence(evidence: str) -> str:
    """Normalize evidence ids: semicolon-separated or the literal 'none'."""
    if not evidence or not evidence.strip() or evidence.strip() == "none":
        return "none"
    ids = [i.strip() for i in evidence.split(";") if i.strip()]
    return ";".join(ids) if ids else "none"


def _rows(decisions: list[RoutingDecision]) -> list[dict[str, str]]:
    return [
        {
            "message_id": decision.message_id,
            "action": decision.action,
            "message_type": decision.message_type,
            "reason": decision.reason,
            "confidence": _format_confidence(decision.confidence),
            "evidence_message_ids": _format_evidence(decision.evidence_message_ids),
        }
        for decision in decisions
    ]


def write_output(decisions: list[RoutingDecision], output_path: str) -> None:
    """Write RoutingDecision list to output.csv with the exact required schema.

    Columns: message_id,action,message_type,reason,confidence,evidence_message_ids
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=REQUIRED_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(_rows(decisions))
