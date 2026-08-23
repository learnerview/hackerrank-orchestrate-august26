"""Unit tests for the output writer: exact schema, normalization, determinism."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.decision import RoutingDecision  # noqa: E402
from output.csv_writer import REQUIRED_COLUMNS, write_output  # noqa: E402


def _decision(message_id="m1", action="notify", evidence="none") -> RoutingDecision:
    return RoutingDecision(
        message_id=message_id,
        action=action,
        message_type="personal",
        reason="because",
        confidence=0.9,
        evidence_message_ids=evidence,
    )


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


class TestCsvWriter:
    def test_exact_schema_and_order(self, tmp_path: Path) -> None:
        out = tmp_path / "output.csv"
        write_output([_decision(), _decision("m2", "mute", "his_1;his_2")], str(out))
        with out.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            assert reader.fieldnames == REQUIRED_COLUMNS == [
                "message_id",
                "action",
                "message_type",
                "reason",
                "confidence",
                "evidence_message_ids",
            ]
            rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["message_id"] == "m1"
        assert rows[0]["action"] == "notify"
        assert rows[1]["evidence_message_ids"] == "his_1;his_2"

    def test_empty_evidence_normalized_to_none(self, tmp_path: Path) -> None:
        out = tmp_path / "output.csv"
        write_output([_decision(evidence=""), _decision(evidence="  "), _decision(evidence=";;")], str(out))
        rows = _read(out)
        assert all(r["evidence_message_ids"] == "none" for r in rows)

    def test_evidence_semicolons_trimmed(self, tmp_path: Path) -> None:
        out = tmp_path / "output.csv"
        write_output([_decision(evidence=" his_1 ; his_2 ;")], str(out))
        assert _read(out)[0]["evidence_message_ids"] == "his_1;his_2"

    def test_confidence_no_scientific_notation(self, tmp_path: Path) -> None:
        out = tmp_path / "output.csv"
        write_output([_decision()], str(out))
        rows = _read(out)
        assert rows[0]["confidence"] == "0.9"
        assert "e" not in rows[0]["confidence"].lower()

    def test_confidence_edge_values(self, tmp_path: Path) -> None:
        out = tmp_path / "output.csv"
        write_output(
            [
                RoutingDecision("m1", "digest", "event", "r", 1.0),
                RoutingDecision("m2", "digest", "event", "r", 0.33333),
            ],
            str(out),
        )
        rows = _read(out)
        assert rows[0]["confidence"] == "1"
        assert rows[1]["confidence"] == "0.3333"

    def test_matches_row_count_and_utf8(self, tmp_path: Path) -> None:
        out = tmp_path / "output.csv"
        write_output([_decision(f"msg_{i}") for i in range(50)], str(out))
        rows = _read(out)
        assert len(rows) == 50
        assert len({r["message_id"] for r in rows}) == 50
