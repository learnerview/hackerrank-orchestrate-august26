"""Targeted re-run of degraded messages (Groq outage fallbacks + media failures).

Reads artifacts/degraded_message_ids.txt, re-routes the subset that exists in
messages.csv through the SAME production pipeline (main.build_pipeline), and
patches dataset/output.csv IN PLACE for exactly those rows — preserving the
original row order and the exact required schema. A deliberate 2.5s sleep
precedes every routed message so Groq calls are paced even though the media
layer rate-limits (we burned quota once by underestimating call volume).

Ids that are NOT part of messages.csv (sample rows such as sample_msg_009)
cannot appear in output.csv; they are re-routed anyway so the LLM cache is
populated and their results are reported without touching the file.

Usage:
    python tools/rerun_degraded.py [--ids artifacts/degraded_message_ids.txt]
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Settings, load_env
from ingestion import load_sample_messages
from main import build_pipeline
from output.csv_writer import REQUIRED_COLUMNS, _format_confidence, _format_evidence

SLEEP_BETWEEN_MESSAGES = 2.5  # deliberate pacing for Groq calls


def _read_degraded_ids(path: Path) -> list[str]:
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.append(line)
    return sorted(set(ids))


def _row_from_decision(decision) -> dict[str, str]:
    return {
        "message_id": decision.message_id,
        "action": decision.action,
        "message_type": decision.message_type,
        "reason": decision.reason,
        "confidence": _format_confidence(decision.confidence),
        "evidence_message_ids": _format_evidence(decision.evidence_message_ids),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", type=str, default=None, help="degraded ids file (default: artifacts/degraded_message_ids.txt)")
    args = parser.parse_args()

    load_env()
    settings = Settings()
    ids_path = Path(args.ids) if args.ids else settings.artifacts_dir / "degraded_message_ids.txt"
    targets = _read_degraded_ids(ids_path)

    repo, runner, orchestrator, audit = build_pipeline(settings)
    all_messages = repo.all_messages()
    message_ids = {m.message_id for m in all_messages}

    in_messages = [t for t in targets if t in message_ids]
    samples_only = [t for t in targets if t not in message_ids]
    print(f"degraded targets: {len(targets)} unique ids "
          f"({len(in_messages)} in messages.csv, {len(samples_only)} sample-only)")

    # ------------------------------------------------------------------ #
    # before/after snapshot from the current output.csv
    # ------------------------------------------------------------------ #
    output_path = settings.output_path
    with output_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == REQUIRED_COLUMNS, f"schema mismatch: {reader.fieldnames}"
        rows = list(reader)
    before = {r["message_id"]: r for r in rows}
    assert len(rows) == len(before) == 110, f"output.csv rows: {len(rows)}"

    # ------------------------------------------------------------------ #
    # re-route the messages.csv subset with deliberate pacing
    # ------------------------------------------------------------------ #
    order = [m.message_id for m in all_messages]  # messages.csv order
    to_route = [m for m in all_messages if m.message_id in in_messages]
    by_id = {m.message_id: m for m in all_messages}

    new_rows = {}
    print(f"\nre-routing {len(to_route)} messages (sleep {SLEEP_BETWEEN_MESSAGES}s before each)...")
    for m in to_route:
        time.sleep(SLEEP_BETWEEN_MESSAGES)
        decision = runner.run([m])[0]
        new_rows[decision.message_id] = _row_from_decision(decision)

    # ------------------------------------------------------------------ #
    # sample-only ids: route for cache warm-up + reporting, never into output.csv
    # ------------------------------------------------------------------ #
    sample_results: dict[str, dict[str, str]] = {}
    if samples_only:
        samples = load_sample_messages(settings.dataset_dir / "sample_messages.csv")
        sample_by_id = {s.message_id: s for s in samples}
        print(f"\nre-routing {len(samples_only)} sample-only ids (cache warm-up)...")
        for sid in samples_only:
            sample = sample_by_id.get(sid)
            if sample is None:
                print(f"  !! {sid}: not found in sample_messages.csv — skipped")
                continue
            time.sleep(SLEEP_BETWEEN_MESSAGES)
            decision = runner.run([sample])[0]
            sample_results[decision.message_id] = _row_from_decision(decision)

    # ------------------------------------------------------------------ #
    # patch output.csv in place (preserve original row order), then validate
    # ------------------------------------------------------------------ #
    patched = [new_rows.get(r["message_id"], r) for r in rows]
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=REQUIRED_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(patched)

    with output_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == REQUIRED_COLUMNS, f"post-write schema mismatch: {reader.fieldnames}"
        final_rows = list(reader)
    final_ids = [r["message_id"] for r in final_rows]
    assert len(final_rows) == 110, f"post-write rows: {len(final_rows)}"
    assert len(set(final_ids)) == 110, "duplicate message_id after patch"
    assert set(final_ids) == message_ids, "id set drift after patch"
    assert final_ids == order, "row order changed after patch"
    print(f"\noutput.csv patched and validated: 110/110 rows, exact schema, "
          f"no dupes, no drops, order preserved.")

    # ------------------------------------------------------------------ #
    # before/after report
    # ------------------------------------------------------------------ #
    print(f"\n{'id':<14} {'action':<8} {'type':<16} {'conf':<6}  reason")
    for mid in order:
        if mid not in new_rows:
            continue
        b, a = before[mid], new_rows[mid]
        marker = "  <-- changed" if (b["action"], b["message_type"], b["confidence"]) != (a["action"], a["message_type"], a["confidence"]) else ""
        print(f"{mid:<14} {a['action']:<8} {a['message_type']:<16} {a['confidence']:<6}  {a['reason'][:64]}{marker}")
        if marker:
            print(f"{'':<14} before: {b['action']:<8} {b['message_type']:<16} {b['confidence']:<6}  {b['reason'][:64]}")
    for sid, row in sample_results.items():
        print(f"{sid:<14} {row['action']:<8} {row['message_type']:<16} {row['confidence']:<6}  {row['reason'][:64]} (sample-only, not in output.csv)")

    still_media_degraded = [m for m in runner.media_failures if m in set(targets)]
    if still_media_degraded:
        print(f"\nmedia still degraded: {still_media_degraded}")
    else:
        print("\nmedia: no remaining failures for the re-routed ids")

    print(f"\nnew decisions also appended to: {settings.artifacts_dir / 'audit.jsonl'}")


if __name__ == "__main__":
    main()
