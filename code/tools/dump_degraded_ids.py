"""Dump message_ids degraded by the Groq outage to a file, for targeted re-run.

Two sources:
- LLM-fallback rows: audit.jsonl entries whose reason is the designed
  'LLM could not resolve' fallback (their prompts were never answered by
  Groq, so the cache has nothing for them),
- media-degraded rows: messages whose image/voice enrichment failed
  (verified by re-probing only the media-bearing messages through the
  production runner - cached successes are served from disk, no API cost).

Writes artifacts/degraded_message_ids.txt (one id per line).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    from config import Settings, load_env
    from main import build_pipeline
    from orchestration.audit import DecisionAuditLog

    load_env()
    settings = Settings()

    # 1) LLM fallbacks from the persistent audit log
    llm_fallback: list[str] = []
    path = settings.artifacts_dir / "audit.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "LLM could not resolve" in line:
                entry = DecisionAuditLog()
                import json

                data = json.loads(line)
                if data.get("message_id"):
                    llm_fallback.append(data["message_id"])
    llm_fallback = sorted(set(llm_fallback))

    # 2) media-degraded rows: probe media messages through the pipeline
    repo, runner, orchestrator, audit = build_pipeline(settings)
    media_messages = [m for m in repo.all_messages() if m.media_type not in ("", "none")]
    for m in media_messages:
        runner.merge_media(m)  # cache hits are free; failures are recorded
    media_degraded = sorted(set(runner.media_failures))

    out = settings.artifacts_dir / "degraded_message_ids.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# degraded rows from the last full run (Groq unreachable)",
        "# LLM-fallback digests (re-resolve when Groq is back):",
        *llm_fallback,
        "# media enrichment failed (re-analyze when Groq is back):",
        *media_degraded,
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"llm_fallback({len(llm_fallback)}): {llm_fallback}")
    print(f"media_degraded({len(media_degraded)}): {media_degraded}")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
