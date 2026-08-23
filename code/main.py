"""CLI entrypoint for the Message Notification Router.

Composition root only: loads config, builds the repository, wires every layer
through its Protocol/interface (no business logic here), runs the full
pipeline over dataset/messages.csv, validates the output, and reports stage
usage. Run from the code directory:

    python main.py [--output PATH] [--limit N]
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path

from config import Settings, load_env
from ingestion import (
    load_business_accounts,
    load_group_members,
    load_groups,
    load_images,
    load_message_events,
    load_message_history,
    load_messages,
    load_user_business_history,
    load_users,
    load_voice_notes,
)
from ingestion.repository import InMemoryDataRepository
from llm.factory import build_routing_llm
from media.factory import build_audio_transcriber, build_image_analyzer
from orchestration.audit import DecisionAuditLog
from orchestration.pipeline import PipelineRunner
from orchestration.router import RoutingOrchestrator
from output.csv_writer import REQUIRED_COLUMNS, write_output
from personalization.overrides import PersonalizationOverrides
from retrieval.factory import build_retriever
from safety.rule_engine import SafetyRuleEngine
from scoring.features import FeatureExtractor
from scoring.preference_model import build_preference_model


def build_repository(dataset_dir: Path) -> InMemoryDataRepository:
    """Load every dataset CSV into the in-memory repository."""
    return InMemoryDataRepository.from_loaded(
        messages=load_messages(dataset_dir / "messages.csv"),
        history=load_message_history(dataset_dir / "message_history.csv"),
        samples=[],
        users=load_users(dataset_dir / "users.csv"),
        groups=load_groups(dataset_dir / "groups.csv"),
        group_members=load_group_members(dataset_dir / "group_members.csv"),
        businesses=load_business_accounts(dataset_dir / "business_accounts.csv"),
        user_business_history=load_user_business_history(dataset_dir / "user_business_history.csv"),
        events=load_message_events(dataset_dir / "message_events.csv"),
        images=load_images(dataset_dir / "images.csv"),
        voice_notes=load_voice_notes(dataset_dir / "voice_notes.csv"),
        daily_summaries=[],
    )


def build_pipeline(
    settings: Settings,
    audit_log: Optional[DecisionAuditLog] = None,
) -> tuple[InMemoryDataRepository, PipelineRunner, RoutingOrchestrator, DecisionAuditLog]:
    """Wire the full pipeline through Protocol/interface dependencies.

    ``audit_log`` is injectable so non-production callers (e.g. the offline
    evaluator) can keep their runs out of the shared artifacts/audit.jsonl.
    """
    repo = build_repository(settings.dataset_dir)

    rule_engine: SafetyRuleEngine = SafetyRuleEngine(repo)
    preference_model: PreferenceModel = build_preference_model(settings.artifacts_dir)
    retriever = build_retriever(settings, repo)
    llm: RoutingLLM = build_routing_llm(settings)
    personalization: PersonalizationOverrides = PersonalizationOverrides()
    if audit_log is None:
        audit_log = DecisionAuditLog(settings.artifacts_dir / "audit.jsonl")
    orchestrator = RoutingOrchestrator(
        rule_engine=rule_engine,
        preference_model=preference_model,
        llm=llm,
        personalization=personalization,
        audit_log=audit_log,
    )
    runner = PipelineRunner(
        settings=settings,
        repo=repo,
        image_analyzer=build_image_analyzer(settings),
        audio_transcriber=build_audio_transcriber(settings),
        retriever=retriever,
        features=FeatureExtractor(repo),
        orchestrator=orchestrator,
    )
    return repo, runner, orchestrator, audit_log


def validate_output(decisions: list, source_ids: list[str], output_path: Path) -> None:
    """Structural checks: one row per message, no dupes, no drops, exact schema."""
    written_ids = [d.message_id for d in decisions]
    if len(written_ids) != len(set(written_ids)):
        raise RuntimeError("Duplicate message_id in decisions — refusing to write.")
    if len(written_ids) != len(source_ids):
        raise RuntimeError(
            f"Row-count mismatch: {len(written_ids)} decisions vs {len(source_ids)} source rows."
        )
    if set(written_ids) != set(source_ids):
        missing = set(source_ids) - set(written_ids)
        extra = set(written_ids) - set(source_ids)
        raise RuntimeError(f"ID mismatch — missing: {sorted(missing)[:5]} extra: {sorted(extra)[:5]}")

    with output_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != REQUIRED_COLUMNS:
            raise RuntimeError(f"Output schema mismatch: {reader.fieldnames}")
        rows = list(reader)
    if len(rows) != len(source_ids):
        raise RuntimeError(f"Written file has {len(rows)} rows, expected {len(source_ids)}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Message Notification Router")
    parser.add_argument("--output", type=str, default=None, help="output.csv path (default: from config)")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N messages (dev)")
    args = parser.parse_args()

    load_env()
    settings = Settings()
    output_path = Path(args.output) if args.output else settings.output_path

    start = time.monotonic()
    repo, runner, orchestrator, audit_log = build_pipeline(settings)
    messages = repo.all_messages() if args.limit is None else repo.all_messages()[: args.limit]

    print(f"routing {len(messages)} messages from {settings.dataset_dir / 'messages.csv'} ...")
    decisions = runner.run(messages)
    elapsed = time.monotonic() - start

    write_output(decisions, str(output_path))
    validate_output(decisions, [m.message_id for m in messages], output_path)

    # ------------------------------------------------------------------ #
    # report
    # ------------------------------------------------------------------ #
    total = len(decisions)
    stages = Counter(e["stage"] for e in audit_log.records)
    personalized = sum(1 for e in audit_log.records if e["personalized"])
    actions = Counter(d.action for d in decisions)
    print()
    print(f"output written: {output_path} ({total} rows, validated)")
    print(f"total runtime:  {elapsed:.1f}s")
    for stage in ("rules", "preference_model", "llm"):
        n = stages.get(stage, 0)
        print(f"  {stage:<16} {n:>4}  {100.0 * n / total:>5.1f}%")
    print(f"  {'personalized':<16} {personalized:>4}  {100.0 * personalized / total:>5.1f}%")
    print(f"actions: {dict(actions)}")
    if runner.media_failures:
        print(f"media enrichment degraded (API/local unavailable) for: {', '.join(runner.media_failures)}")
    print(f"audit log: {settings.artifacts_dir / 'audit.jsonl'}")


if __name__ == "__main__":
    main()
