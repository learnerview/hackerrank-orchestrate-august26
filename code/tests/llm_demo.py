"""Real-data demo of the LLM routing stage on ambiguous messages.

Selection criteria: no safety-rule hit AND preference model not confident
(mid priority 0.25-0.75 or low confidence < 0.6). Prints raw Groq responses
so the actual model output is visible. Runs the full pipeline for each:
media merge -> retrieval -> context summary -> LLM decide.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
from llm.groq_client import GroqRoutingLLM
from media.factory import build_audio_transcriber, build_image_analyzer
from retrieval.factory import build_retriever
from safety.rule_engine import SafetyRuleEngine
from scoring.features import FeatureExtractor, source_id_of
from scoring.preference_model import build_preference_model

SELECTED = [
    "msg_090",  # personal: unknown number asks for Arun
    "msg_104",  # personal: jacket pickup offer
    "msg_094",  # business: 40% off beauty welcome offer
    "msg_028",  # business: saved travel deal
    "msg_092",  # business: monthly card statement
    "msg_022",  # group: payment due today
    "msg_048",  # group: admin penalty list QR
    "msg_109",  # group: prompt-injection attempt
    "msg_084",  # business: voice note (no text)
]


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


def media_content(repo, settings, message) -> tuple[str, str | None]:
    """Merge OCR/transcript text and a media summary into effective text."""
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
    return effective, summary


def context_summary(repo, fx, message) -> str:
    lines = []
    u = repo.get_user(message.user_id)
    if u:
        lines.append(f"user: {u.user_id}, dnd_window={u.do_not_disturb_window!r}, "
                     f"opened_30d={u.messages_opened_30d}, reported_30d={u.messages_reported_30d}")
    gm = repo.get_group_member(message.user_id, message.sender_user_id) if message.sender_user_id else None
    g = repo.get_group(message.group_id) if message.group_id else None
    if g:
        lines.append(f"group: {g.group_name} ({g.group_type}, {g.member_count} members)")
    if gm:
        lines.append(f"sender role: {gm.role}, muted_by_user={gm.group_muted_by_user}, "
                     f"msgs_30d={gm.messages_sent_30d}")
    b = repo.get_business(message.business_id) if message.business_id else None
    if b:
        lines.append(f"business: {b.display_name} [{b.category}] verified={b.verified}, "
                     f"user_reports_30d={b.user_reports_30d}, msgs_30d={b.messages_sent_30d}")
    f = fx.extract(message)
    lines.append(
        f"engagement: open_rate={f['open_rate']:.2f} reply_rate={f['reply_rate']:.2f} "
        f"dismiss_rate={f['dismiss_rate']:.2f} volume={f['volume']:.0f} "
        f"noise_ratio={f['noise_ratio']:.2f} forwarded={f['forwarded_count']:.0f}"
    )
    return "\n".join(lines)


def main() -> None:
    load_env()
    settings = Settings()
    dataset = settings.dataset_dir
    repo = build_repo(dataset)
    rules = SafetyRuleEngine(repo)
    model = build_preference_model(settings.artifacts_dir)
    fx = FeatureExtractor(repo)
    retriever = build_retriever(settings, repo)
    llm = build_routing_llm(settings)
    orig_call = llm._call_groq

    def raw_call(prompt: str, attempt: int) -> str | None:
        raw = orig_call(prompt, attempt)
        print(f"    [RAW attempt {attempt + 1}] {raw}")
        return raw

    llm._call_groq = raw_call

    by_id = {m.message_id: m for m in repo.all_messages()}
    for mid in SELECTED:
        m = by_id[mid]
        hit = rules.evaluate(m)
        feats = fx.extract(m)
        src = source_id_of(m) or "unknown"
        pref = model.score(m.user_id, src, feats)
        print("=" * 100)
        print(f"{mid} [{m.conversation_type}] sender={m.sender_user_id} fwd={m.forwarded_count} media={m.media_type}")
        print(f"  text: {m.message_text!r}"[:160])
        print(f"  safety: {hit.action if hit else 'no hit'} | preference: priority={pref.priority:.3f} conf={pref.confidence:.2f}")
        effective, media_summary = media_content(repo, settings, m)
        if media_summary:
            print(f"  media summary: {media_summary}")
        evidence = retriever.retrieve(m, effective)
        snippets = []
        for e in evidence:
            h = repo.history_by_id(e.message_id)
            snippet = f"[{e.message_id}] \"{(h.message_text if h else '?')[:110]}\" (score {e.score:.2f})"
            snippets.append(snippet)
        print(f"  evidence: {len(evidence)} snippets")
        ctx = {
            "message_text": effective,
            "media_summary": media_summary,
            "context_summary": context_summary(repo, fx, m),
            "evidence_snippets": snippets,
        }
        judgment = llm.decide(m, ctx)
        if judgment is None:
            print("  -> LLM could not resolve (returned None)")
        else:
            print(f"  -> {judgment.action} | {judgment.message_type} | conf={judgment.confidence:.2f} "
                  f"| evidence={judgment.evidence_message_ids} | {judgment.reason}")


if __name__ == "__main__":
    main()
