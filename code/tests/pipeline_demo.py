"""End-to-end pipeline demo: first 20 rows of dataset/messages.csv.

For each message: media merge (OCR/transcript) -> retrieval -> features ->
context -> cascade (rules -> preference -> LLM -> personalization) -> audit.
Prints the decision, producing stage, personalization flag and evidence.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Settings, load_env
from context.context_builder import MessageContext
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
from models.message import RoutedMessage
from orchestration.audit import DecisionAuditLog
from orchestration.router import RoutingOrchestrator
from personalization.overrides import PersonalizationOverrides
from retrieval.factory import build_retriever
from safety.rule_engine import SafetyRuleEngine
from scoring.features import FeatureExtractor
from scoring.preference_model import build_preference_model


def build_repo(dataset_dir: Path) -> InMemoryDataRepository:
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


def media_merge(repo, settings, message) -> RoutedMessage:
    effective = message.message_text or ""
    summary = None

    def resolve(p: str) -> str:
        path = Path(p)
        return str(path) if path.is_absolute() else str(settings.dataset_dir / path)

    if message.media_type == "image" and message.media_id:
        rec = repo.get_image(message.media_id)
        if rec:
            result = build_image_analyzer(settings).analyze(resolve(rec.file_path))
            if result.extracted_text:
                effective = f"{effective}\n{result.extracted_text}".strip()
            summary = result.visual_summary or f"image {message.media_id}"
    elif message.media_type == "voice" and message.media_id:
        rec = repo.get_voice_note(message.media_id)
        if rec:
            result = build_audio_transcriber(settings).transcribe(resolve(rec.file_path))
            effective = f"{effective}\n{result.transcript}".strip()
            summary = f"voice note transcript available ({len(result.transcript)} chars)"
    return RoutedMessage(message=message, effective_text=effective, media_summary=summary)


def main() -> None:
    load_env()
    settings = Settings()
    repo = build_repo(settings.dataset_dir)

    rules = SafetyRuleEngine(repo)
    model = build_preference_model(settings.artifacts_dir)
    fx = FeatureExtractor(repo)
    retriever = build_retriever(settings, repo)
    llm = build_routing_llm(settings)
    audit = DecisionAuditLog(settings.artifacts_dir / "audit.jsonl")
    orch = RoutingOrchestrator(rules, model, llm, PersonalizationOverrides(), audit_log=audit)

    history_ids = {h.message_id for h in repo.all_history()}
    by_history = {h.message_id: h for h in repo.all_history()}

    first_20 = repo.all_messages()[:20]
    print(f"{'message_id':<10} {'action':<8} {'type':<15} {'conf':<6} {'stage':<17} {'pers':<4} evidence")
    print("-" * 100)
    for m in first_20:
        routed = media_merge(repo, settings, m)
        evidence = retriever.retrieve(m, routed.effective_text)
        snippets = []
        for e in evidence:
            h = by_history.get(e.message_id)
            snippets.append(f'[{e.message_id}] "{(h.message_text if h else "?")[:100]}" (score {e.score:.2f})')
        ctx = MessageContext(
            message=m,
            routed=routed,
            user=repo.get_user(m.user_id),
            group=repo.get_group(m.group_id) if m.group_id else None,
            member=repo.get_group_member(m.user_id, m.group_id) if m.group_id else None,
            business=repo.get_business(m.business_id) if m.business_id else None,
            evidence=evidence,
            engagement_features=fx.extract(m),
            evidence_snippets=snippets,
            history_ids=history_ids,
        )
        decision = orch.route(m, ctx.to_dict())
        print(
            f"{m.message_id:<10} {decision.action:<8} {decision.message_type:<15} "
            f"{decision.confidence:<6.2f} {decision.resolved_by:<17} "
            f"{'yes' if decision.personalized else 'no':<4} {decision.evidence_message_ids}"
        )

    print("-" * 100)
    from collections import Counter

    stages = Counter(e["stage"] for e in audit.records)
    pers = sum(1 for e in audit.records if e["personalized"])
    print(f"stages: {dict(stages)} | personalized: {pers}/{len(audit.records)} | audit: {settings.artifacts_dir / 'audit.jsonl'}")


if __name__ == "__main__":
    main()
