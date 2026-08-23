# Message Notification Router — Code

AI-powered WhatsApp notification router for the HackerRank Orchestrate
challenge. For every incoming multimodal message (`dataset/messages.csv`) the
system decides `notify` (interrupt now), `digest` (batch later), or `mute`
(low-value, repetitive, unwanted, suspicious, or unsafe).

## Setup

```powershell
# 1. create and activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows
# source .venv/bin/activate       # macOS / Linux

# 2. install dependencies
pip install -r requirements.txt

# 3. configure secrets (never commit these)
#    copy code/.env.example to code/.env and fill in GROQ_API_KEY
```

All settings come from environment variables / `code/.env` (see
`code/.env.example`). `code/.env` is gitignored. `config.Settings` resolves
paths relative to the code directory, so scripts work from any cwd.

Optional local media fallbacks (used when the Groq media models are
unavailable): install tesseract (OCR) and `pip install openai-whisper` for
voice transcription. The media layer degrades gracefully without them.

## Training the preference model

```powershell
python scoring/train_preference_model.py
```

- Trains a gradient-boosting model on `P(engaged)` from
  `message_history.csv` + `message_events.csv` (+ `group_members.csv`,
  `users.csv` for features), with 5-fold CV and an out-of-fold isotonic
  probability calibration.
- Learns the **mute band**: 5th percentile of out-of-fold engaged priorities.
- Ships the trained model only if it beats the weighted-engagement-rate
  heuristic by >= 0.03 AUC and the CV is stable; otherwise the heuristic is
  used.
- Writes to `artifacts/`: `preference_model.joblib`,
  `preference_calibrator.joblib`, `preference_schema.json`
  (`mute_priority_threshold`, `feature_names`, `calibrated`),
  `preference_metrics.json`.

## Running the full pipeline

```powershell
python main.py                 # routes dataset/messages.csv -> dataset/output.csv
python main.py --output out2.csv   # custom output path
python main.py --limit 10          # dev: first N messages only
```

- Writes exactly one row per message in `dataset/messages.csv` to
  `dataset/output.csv` with the exact required schema
  `message_id,action,message_type,reason,confidence,evidence_message_ids`.
- Validates before finishing: 110/110 rows, exact schema, no duplicates, no
  drops; refuses to write otherwise.
- Appends every decision to `artifacts/audit.jsonl`.

## Running evaluation

```powershell
python -m evaluation.evaluate
```

- Scores the router against the 30 labeled rows in
  `dataset/sample_messages.csv` (action accuracy, message_type accuracy,
  evidence Jaccard, confidence-calibration bands + Pearson, confusion
  matrices, per-stage attribution).
- Predictions are read from `dataset/output.csv` when the sample ids appear
  there; ids that are absent are routed live through the production pipeline
  (with an in-memory audit log, so evaluation never pollutes
  `artifacts/audit.jsonl`).
- Live routing is served from the on-disk caches (`artifacts/llm_cache`,
  `.cache/media`) on re-runs, so re-evaluating is cheap.

## Architecture (matches spec section 1)

The system is a decision **cascade** of three stages, each with a single
responsibility and a Protocol interface, wired by the composition root in
`main.py`:

```
messages.csv
   │
   ▼
[1] SAFETY RULES (deterministic) ── mute: scam / spam (explainable, high precision)
   │  no hit
   ▼
[2] PREFERENCE MODEL (personalized) ── notify / digest / mute (learned mute band)
   │  low confidence
   ▼
[3] LLM JUDGMENT (Groq, cached) ──── notify / digest / mute for ambiguous cases
   │
   ▼
personalization overrides (quiet-hours downgrade, muted-group mention escalation)
   │
   ▼
output.csv
```

Supporting layers:

- **ingestion/** — CSV loaders + an in-memory indexed repository (all dataset
  context: users, groups, businesses, history, events, media).
- **media/** — image OCR + voice transcription (Groq vision/ASR with local
  fallback), content-addressed disk cache, rate limiter, retries.
- **retrieval/** — TF-IDF evidence retrieval (sender → group → business scope)
  with recency decay, feeding `evidence_message_ids`.
- **scoring/** — engagement feature engineering, the trained preference model
  (with learned mute band), and the training script.
- **safety/** — the deterministic rule engine (scam/spam patterns, business
  impersonation, prompt injection, broadcast spam).
- **context/** — assembles the per-message context dict consumed by the
  orchestrator and the LLM prompt.
- **orchestration/** — the router (cascade) and the batch pipeline runner.
- **personalization/** — post-decision overrides (quiet hours, muted groups).
- **output/** — exact-schema CSV writer.
- **evaluation/** — offline scoring against the labeled samples.
- **tools/** — maintenance scripts (degraded-id re-run, Groq probe, etc.).

The multimodal input path: image posters/screenshots and voice notes are
enriched in the media layer (OCR text / transcript + intent hint) before the
cascade runs, so the same three stages see the effective text of every
message type.
