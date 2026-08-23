"""Media smoke test: run the real analyzers against actual media files and
print the raw extracted output for manual quality inspection.

usage: python -m tests.media_demo  (from code/)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Settings, load_env  # noqa: E402
from ingestion import load_images, load_voice_notes  # noqa: E402
from media import (  # noqa: E402
    build_audio_transcriber,
    build_image_analyzer,
    resolve_media_path,
)

DATASET = Path(__file__).resolve().parent.parent.parent / "dataset"


def main() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_env(env_path)
    settings = Settings()
    image_analyzer = build_image_analyzer(settings)
    audio_transcriber = build_audio_transcriber(settings)

    images = load_images(DATASET / "images.csv")
    voice_notes = load_voice_notes(DATASET / "voice_notes.csv")

    print("=" * 90)
    print(f"IMAGES ({len(images)} total, analyzing first 5)")
    print("=" * 90)
    for record in images[:5]:
        path = resolve_media_path(DATASET, record.file_path)
        print(f"\n--- {record.image_id} @ {path} ---")
        try:
            result = image_analyzer.analyze(str(path))
            print(f"  detected_intent_hint : {result.detected_intent_hint}")
            print(f"  visual_summary       : {result.visual_summary}")
            print(f"  extracted_text       : {result.extracted_text!r}")
        except Exception as exc:  # noqa: BLE001 - demo must not die on one file
            print(f"  FAILED: {type(exc).__name__}: {exc}")

    print("\n" + "=" * 90)
    print(f"AUDIO: {len(voice_notes)} total, transcribing first 5")
    print("=" * 90)
    for record in voice_notes[:5]:
        path = resolve_media_path(DATASET, record.file_path)
        print(f"\n--- {record.voice_note_id} @ {path} ---")
        try:
            result = audio_transcriber.transcribe(str(path))
            print(f"  confidence  : {result.confidence}")
            print(f"  transcript  : {result.transcript!r}")
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}")

    print("\n--- analyzer counters ---")
    print(f"  image cache_hits={image_analyzer.cache_hits} "
          f"groq_calls={image_analyzer.groq_calls} fallback_calls={image_analyzer.fallback_calls}")
    print(f"  audio cache_hits={audio_transcriber.cache_hits} "
          f"groq_calls={audio_transcriber.groq_calls} fallback_calls={audio_transcriber.fallback_calls}")


if __name__ == "__main__":
    main()