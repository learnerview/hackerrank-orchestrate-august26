# Message Notification Router

An AI-powered WhatsApp notification router that decides, for every incoming
multimodal message, whether to `notify` (interrupt now), `digest` (show later), or
`mute` (low-value, repetitive, unwanted, suspicious, or unsafe).

> HackerRank Orchestrate — August 2026 hackathon submission. Runnable, tested, and
> documented, but **not production-hardened** — see
> [Limitations & Known Issues](#limitations--known-issues) and
> [`docs/PRODUCTION_READINESS.md`](docs/PRODUCTION_READINESS.md).

---

## Overview

WhatsApp delivers one undifferentiated stream — family chats, society notices, school
updates, co-worker pings, business promotions, image posters, voice notes, and scams
— so treating every message the same either hides important messages or taxes the
user with noise. This project builds a personalized router over the provided dataset.

For each row in `dataset/messages.csv` the system emits exactly one decision row in
`dataset/output.csv`:

```text
message_id,action,message_type,reason,confidence,evidence_message_ids
```

It reasons over text, image posters/screenshots (OCR + visual intent), and voice
notes (transcription), personalized by the receiving user's historical engagement
with each sender/group/business.

Full task specification: [`problem_statement.md`](problem_statement.md) and the
engineering spec [`SRS.md`](SRS.md) (with an implementation-status reconciliation in
§17).

### Repository layout

```text
.
├── README.md              # this file
├── SECURITY.md            # API-key handling
├── LICENSE                # MIT
├── problem_statement.md   # challenge statement (participant-facing)
├── SRS.md                 # engineering spec + implementation reconciliation
├── AGENTS.md / CLAUDE.md  # AI-coding-tool rules & transcript logging
├── docs/                  # architecture, configuration, deployment, readiness
├── dataset/               # input CSVs, media/, output.csv
└── code/                  # the implementation (Python)
```

---

## Features

- **Three-stage decision cascade** with a strict, safety-first order:
  1. deterministic safety rules (scam/spam — `code/safety/rule_engine.py`)
  2. an offline-trained preference model (learned mute band — `code/scoring/`)
  3. a Groq LLM fallback for genuinely ambiguous messages (`code/llm/`)
  then post-decision personalization overrides (`code/personalization/overrides.py`).
- **Multimodal ingestion** — image OCR + visual intent and voice transcription
  merged into one "effective text" before routing (`code/media/`, `code/orchestration/pipeline.py`).
- **Personalization** — quiet-hours downgrade and muted-group mention escalation.
- **Evidence retrieval** — scoped, recency-decayed TF-IDF that feeds
  `evidence_message_ids` (`code/retrieval/`).
- **Resilience** — content-hash caching, rate limiting, retry-with-backoff, and a
  local media fallback; LLM failure degrades to a fail-safe digest.
- **Auditability** — per-decision audit log with the producing stage
  (`code/artifacts/audit.jsonl`).
- **Determinism** — temperature 0, cached answers, seeded offline training.

---

## Architecture

A layered, dependency-inverted codebase: `orchestration/` depends on every other
layer through a Protocol, and nothing depends on `orchestration/`. Concrete
implementations are wired only in composition roots (`main.py`, `*/factory.py`).

```text
messages.csv → ingestion → media enrichment → retrieval → context
     → [safety rules] → [preference model] → [LLM] → personalization → output.csv
```

Full diagrams, the layer table, cascade thresholds, and the recent behavior changes
are in **[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)**.

---

## Tech Stack

- **Language:** Python (3.11+; developed/tested on CPython 3.14)
- **ML:** scikit-learn (`GradientBoostingClassifier`, isotonic calibration), numpy
- **LLM / vision / ASR:** Groq (`llama-3.3-70b-versatile`, `qwen/qwen3.6-27b`,
  `whisper-large-v3-turbo`)
- **Config/secrets:** `python-dotenv`
- **Media fallback:** `pytesseract` + `Pillow` (OCR), optionally `openai-whisper`
- **Testing:** pytest

---

## Prerequisites

- Python 3.11+ with `pip`.
- A Groq API key (`GROQ_API_KEY`) for live LLM/vision/ASR. Without it the pipeline
  still runs, degraded (local media fallback or raw text; low-confidence digest).
- Optional: a Tesseract OCR install on `PATH` and `pip install openai-whisper` for
  the offline media fallback.

---

## Quick Start

```powershell
# 1. create a virtual environment (from the repo root)
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows
# source .venv/bin/activate       # macOS / Linux

# 2. install dependencies
pip install -r code\requirements.txt

# 3. configure the API key (never commit this file)
Copy-Item code\.env.example code\.env
# edit code\.env and set GROQ_API_KEY=<your key>
# save as UTF-8 WITHOUT BOM (a BOM breaks python-dotenv here)

# 4. train the preference model (writes code/artifacts/)
python -m scoring.train_preference_model   # run from code/

# 5. route the dataset
python main.py                             # run from code/
```

`python main.py` writes `dataset/output.csv`, validates it (one row per source
message, exact schema, no drops), and prints per-stage resolution counts.

---

## Configuration

All configuration is read from environment variables / `code/.env` by
`code/config.py`. Relative paths are anchored to the repo root (for `dataset`/
`output`) or the code directory (for caches/artifacts), so runs are cwd-independent.
Full reference: **[`docs/CONFIGURATION.md`](docs/CONFIGURATION.md)**.

| Name | Purpose | Default | Required |
|---|---|---|---|
| `GROQ_API_KEY` | API key for text LLM, vision, ASR | *(empty)* | Yes (for live calls) |
| `GROQ_BASE_URL` | Groq API base URL | `https://api.groq.com` | No |
| `GROQ_MODEL_TEXT` | Text model (LLM stage) | `llama-3.3-70b-versatile` | No |
| `GROQ_MODEL_VISION` | Vision model (image OCR/intent) | `qwen/qwen3.6-27b` | No |
| `GROQ_MODEL_ASR` | ASR model (voice transcription) | `whisper-large-v3-turbo` | No |
| `LOG_LEVEL` | Log level (read but not yet wired to logging) | `INFO` | No |
| `DATASET_DIR` | Input CSV directory | `dataset` | No |
| `OUTPUT_PATH` | Output CSV path | `dataset/output.csv` | No |
| `ARTIFACTS_DIR` | Model/schema/metrics/cache/audit dir | `artifacts` | No |
| `MEDIA_CACHE_DIR` | Media analysis cache | `.cache/media` | No |
| `MEDIA_RPM_VISION` | Vision requests/minute | `10` | No |
| `MEDIA_RPM_ASR` | ASR requests/minute | `20` | No |
| `MEDIA_RETRIES` | Media retries before fallback | `2` | No |
| `RETRIEVAL_TOP_K` | Max evidence ids | `3` | No |
| `RETRIEVAL_MIN_SIMILARITY` | Min cosine similarity | `0.1` | No |
| `RETRIEVAL_RECENCY_DECAY_DAYS` | Evidence recency decay | `60` | No |
| `PREFERENCE_MODEL_FALLBACK_THRESHOLD` | Reserved — currently unused | `0.5` | No |

Cascade thresholds (fixed constants in `code/orchestration/router.py`):
`RULE_MIN_CONFIDENCE=0.70`, `PREF_CONFIDENCE_MIN=0.60`, `PREF_NOTIFY_PRIORITY=0.65`,
`LLM_FALLBACK_CONFIDENCE=0.50`.

---

## Usage

Run from the `code/` directory so the package modules import cleanly.

```powershell
# Train the preference model (offline; writes code/artifacts/)
python -m scoring.train_preference_model

# Route the full dataset -> dataset/output.csv
python main.py

# Options
python main.py --limit 10          # dev: first N messages only
python main.py --output out2.csv   # custom output path

# Evaluate against the labeled samples
python -m evaluation.evaluate
```

See [`code/README.md`](code/README.md) for the code-level guide and
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for the operational runbook.

### Submission (challenge-specific)

Per `AGENTS.md` §6 and `problem_statement.md`, the submission consists of the code
zip, the predictions CSV, and the chat transcript log (written outside the repo;
see [`AGENTS.md`](AGENTS.md) §2).

---

## Testing

```powershell
# from code/
python -m pytest
```

**135 tests pass** (measured `2026-10-16`). Coverage is by module:

| Test file | Tests |
|---|---|
| `tests/test_safety_units.py` | 27 |
| `tests/test_personalization_units.py` | 22 |
| `tests/test_llm_units.py` | 21 |
| `tests/test_scoring_units.py` | 21 |
| `tests/test_orchestration_units.py` | 18 |
| `tests/test_media_units.py` | 10 |
| `tests/test_retrieval_units.py` | 8 |
| `tests/test_output_units.py` | 6 |
| `tests/test_pipeline_units.py` | 1 |
| `tests/test_smoke.py` | 1 |
| **Total** | **135** |

The tests are unit-level with fakes (no live Groq/media calls). See
[Groundedness & Verification](#groundedness--verification) for what is *not* verified.

---

## Deployment

This is a **batch CLI**, not a service. Local deployment is:

```powershell
python -m scoring.train_preference_model   # once / when data changes
python main.py                             # produce dataset/output.csv
```

There is no container image, CI/CD, or hosted deployment in this repository.
Operational details, degraded/offline behavior, artifacts, and caches are in
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

---

## Limitations & Known Issues

This is a **hackathon submission**, not a production system. Honest and current:

- **It is a hackathon submission and not production-hardened.** No deploy/CI/
  service story, no health checks, no rollback. See the Deployment/Operability and
  Data-privacy dimensions in [`docs/PRODUCTION_READINESS.md`](docs/PRODUCTION_READINESS.md)
  (both 4/10).
- **Synthetic/mock data.** The dataset is the challenge-provided set; the code would
  run unchanged on real WhatsApp data, but nothing here has been validated on real
  traffic. Media files (`dataset/media/`, 20 images + 13 voice notes) live on disk
  but are git-ignored, so a fresh clone must obtain them separately.
- **External dependencies.** Routing depends on the Groq API (LLM/vision/ASR) and on
  network media access. Without a key or connectivity, the system degrades: local
  media fallback (if Tesseract/Whisper are installed) or raw text, and a
  low-confidence `digest` for messages the LLM would have handled.
- **Model choice differs from the original spec.** `SRS.md` described logistic
  regression (default) with gradient boosting as a fallback; the implementation
  ships a **gradient-boosting** classifier with a weighted-engagement-rate heuristic
  as the sparse-data fallback. The spec's vision-model examples also differ from the
  configured `qwen/qwen3.6-27b`. Reconciliations are listed in `SRS.md` §17.
- **Evaluation limits.** Metrics come from a **30-row** labeled sample set. Latest
  run: action accuracy **73.3%**, message_type accuracy **40.0%**, evidence Jaccard
  **0.12**, and weak confidence calibration (Pearson **+0.058**). Treat these as
  indicative, not conclusive.
- **Inert configuration.** `LOG_LEVEL` is read but not wired to logging;
  `PREFERENCE_MODEL_FALLBACK_THRESHOLD` is currently unused.
- **Robustness caveats.** Single LLM provider, no request timeouts, no circuit
  breaker, and no cost budget; media rate limiting does not cover the text LLM.

---

## Groundedness & Verification

### Verification status

Verified in this working tree (2026-10-16):

- **Tests:** `python -m pytest` in `code/` → **135 passed** (no failures, no skips).
- **Evaluation:** `python -m evaluation.evaluate` over 30 labeled samples → action
  accuracy 73.3% (22/30), message_type accuracy 40.0% (12/30), evidence Jaccard 0.12,
  Pearson(confidence, correctness) +0.058. Per-stage action accuracy: rules 100%
  (n=6), preference model 67% (n=15), LLM 67% (n=9).
- **Offline model:** `code/artifacts/preference_metrics.json` records 5-fold CV
  ROC-AUC 0.9318 ± 0.0201, held-out GBM ROC-AUC 0.9079, heuristic 0.8517, chosen
  model `gradient_boosting`, learned mute threshold 0.3769.

**Not verified** (explicitly): live media **downloads**, live Groq **LLM** responses
at scale, behaviour on any dataset other than the provided one, and long-running or
concurrent operation. The unit tests use fakes and the evaluation run is served from
on-disk caches on re-runs; no automated test asserts a live provider response.

### Data & evidence provenance

- **Inputs are the challenge-provided dataset** (`dataset/*.csv`): messages (110),
  history (412), events (412), users (54), groups (23), group members (401),
  business accounts (110), images (20), voice notes (13), daily summaries (756),
  user-business history (106), and 30 labeled samples. These are **synthetic/mock**
  challenge data, not real WhatsApp traffic.
- **Media** referenced by `images.csv`/`voice_notes.csv` exists under
  `dataset/media/` locally (33 files) but is **not git-tracked**.
- **External services** used: the Groq API for text/vision/ASR. Outputs from those
  services are cached under `code/artifacts/llm_cache/` and `code/.cache/media/`.
- **Local caches and artifacts are not committed**, so they are environment-specific.

### AI-output grounding

The system uses LLMs (text), a vision model (image OCR/intent), and an ASR model
(voice) — all of whose outputs are probabilistic and can be wrong.

- **Grounding mechanisms:** model output is constrained to a fixed JSON schema and
  validated against allow-listed actions/message types (`code/models/decision.py`);
  prompts include user/group/business context and retrieved historical evidence
  (`code/context/`, `code/llm/prompt_builder.py`); LLM-proposed evidence ids are
  validated against the known history set and hallucinated ids are dropped
  (`code/orchestration/router.py:67`); temperature is 0 for reproducibility.
- **Hallucination risk remains:** the reason text and confidence are model-generated,
  OCR/transcripts can be garbled, and the message_type accuracy of 40% shows the
  semantic labels are frequently approximate.
- **Decisions are advisory and approximate.** They are **not** a substitute for
  human judgment and **must not** be relied upon for real safety-critical moderation.
  The safety rules are intentionally conservative (precision over recall) and will
  both miss novel scams and occasionally over-mute legitimate messages.

---

## Roadmap

- Improve `message_type` classification and confidence calibration.
- Add request timeouts, a circuit breaker, and LLM rate limiting/cost accounting.
- PII redaction and a retention policy for prompts/transcripts/audit logs.
- CI (lint, tests, coverage), packaging, and a container image.
- Persist/recover caches so a cold clone is reproducible without live calls.
- Optional embedding-based retrieval (see `SRS.md` §15).

---

## Contributing

- Keep the dependency direction intact: orchestration depends on Protocols, never
  the reverse; wire concrete classes only in `main.py` / `factory.py`.
- Run `python -m pytest` from `code/` before proposing changes; add unit tests for
  new rules, features, or overrides.
- Never commit secrets, the transcript log, caches, or media files.
- This project follows the tooling rules in [`AGENTS.md`](AGENTS.md); see
  [`CLAUDE.md`](CLAUDE.md) for the Claude Code entry point.

---

## Security

See [`SECURITY.md`](SECURITY.md). In short: secrets come from environment variables
only, `code/.env` is git-ignored and must be saved as UTF-8 **without BOM**, and no
key material may appear in the chat transcript log.

---

## License

Released under the [MIT License](LICENSE). Copyright (c) 2026 Sumit Shukla.
