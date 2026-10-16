# Deployment & Operations

This is a **hackathon-grade, batch** application: it runs once from the terminal
over a static dataset and writes `output.csv`. It is not a long-running service and
is **not production-hardened** (see [`PRODUCTION_READINESS.md`](./PRODUCTION_READINESS.md)).

---

## 1. Environment

- Python 3.11+ (developed and tested on CPython 3.14).
- Dependencies: `code/requirements.txt` (`groq`, `python-dotenv`, `scikit-learn`,
  `pytesseract`, `Pillow`; `pytest` for tests).
- Optional local fallbacks (used only when Groq media models are unavailable):
  a Tesseract OCR install on `PATH` and `pip install openai-whisper`. The media
  layer degrades gracefully without them.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows; or: source .venv/bin/activate
pip install -r code\requirements.txt
```

---

## 2. Run the pipeline

```powershell
# from code/ so the package modules import cleanly
python main.py                       # routes dataset/messages.csv -> dataset/output.csv
python main.py --output out2.csv     # custom output path
python main.py --limit 10            # dev: first N messages only
```

`main.py` is the composition root: it loads config, builds the repository, wires
every layer through its Protocol, runs the pipeline, writes the CSV, then
**validates** before finishing (one row per source message, no duplicates, no drops,
exact schema) and refuses to write otherwise (`main.py:102-122`). It reports total
runtime, per-stage resolution counts, action counts, and any media-degraded ids
(`main.py:146-163`).

---

## 3. Train the preference model

```powershell
# from code/
python -m scoring.train_preference_model
```

Trains a `GradientBoostingClassifier` on `P(engaged)` from
`message_history.csv` + `message_events.csv`, with stratified 5-fold CV, out-of-fold
isotonic calibration, and a learned mute band (5th percentile of out-of-fold engaged
priorities). Ships the model only if it beats the weighted-engagement heuristic by
≥ 0.03 AUC and CV is stable; otherwise persists the heuristic
(`scoring/train_preference_model.py:268`). Writes artifacts described in
[`CONFIGURATION.md`](./CONFIGURATION.md#5-training-artifacts-and-their-schema).

Committed training metrics (`code/artifacts/preference_metrics.json`):

| Metric | Value |
|---|---|
| Messages | 412 |
| Positive (engaged) rate | 0.6748 |
| 5-fold CV ROC-AUC | 0.9318 ± 0.0201 (stable) |
| Gradient boosting held-out ROC-AUC / accuracy | 0.9079 / 0.8932 |
| Heuristic held-out ROC-AUC | 0.8517 |
| Chosen model | `gradient_boosting` |
| Learned mute threshold | 0.3769 (mute precision 0.8739; engaged-suppressed fraction 0.034) |

---

## 4. Evaluate

```powershell
# from code/
python -m evaluation.evaluate
```

Scores the router against the 30 labeled rows in `dataset/sample_messages.csv`
(action accuracy, message-type accuracy, evidence Jaccard, calibration bands +
Pearson, confusion matrices, per-stage attribution). Predictions are read from
`dataset/output.csv` when the sample ids appear there; otherwise samples are routed
through the **same production pipeline** with an in-memory audit log so evaluation
never pollutes `artifacts/audit.jsonl` (`evaluation/evaluate.py:247-287`).

> The sample ids in `dataset/messages.csv` and `dataset/sample_messages.csv` are
> disjoint, so `dataset/output.csv` never contains sample ids and the evaluator
> always routes the samples live (served from the on-disk caches on re-runs).

Results from a local run in this environment (2026-10-16): action accuracy **73.3%
(22/30)**, message_type accuracy **40.0% (12/30)**, evidence Jaccard **0.12**
(28 samples with expected ids), Pearson(confidence, correctness) **+0.058**. Per-stage
action accuracy: rules 100% (n=6), preference model 67% (n=15), LLM 67% (n=9). See
`SRS.md` §12 for the full table and `docs/PRODUCTION_READINESS.md` for interpretation
(30 samples is a small evaluation set).

---

## 5. Artifacts and caches on disk

| Path | Contents | Git-tracked? |
|---|---|---|
| `code/artifacts/preference_model.joblib` | trained classifier | No (`.gitignore`) |
| `code/artifacts/preference_calibrator.joblib` | isotonic calibrator | No |
| `code/artifacts/preference_schema.json` | feature schema + mute band | No |
| `code/artifacts/preference_metrics.json` | training metrics | No |
| `code/artifacts/llm_cache/` | LLM judgments by content hash | No |
| `code/artifacts/audit.jsonl` | one JSON record per decision (stage, action, confidence, …) | No |
| `code/artifacts/degraded_message_ids.txt` | notes on a past outage (see below) | No |
| `code/.cache/media/` | content-addressed OCR/transcript cache | No |
| `dataset/media/` | media files referenced by `images.csv` / `voice_notes.csv` | No |
| `dataset/output.csv` | the submission predictions | Yes |

Because caches and media are git-ignored, a clean clone **must** obtain
`dataset/media/` separately and will re-derive caches (incurring live Groq calls
unless the caller restores the caches).

---

## 6. Degraded / offline operation

- **No `GROQ_API_KEY`:** media falls back to local OCR/Whisper (if installed);
  if that is unavailable the media analysis raises `MediaAnalysisError`, the
  pipeline records the id in `runner.media_failures` and continues with raw text
  (`orchestration/pipeline.py:90`). The LLM stage returns `None` and the router
  emits a low-confidence `digest` fallback (`orchestration/router.py:219`).
- **Provider outage (rate limit / auth / connection / 5xx):** media calls retry
  with exponential backoff, then fall back to local (`media/resilient.py`).
- `artifacts/degraded_message_ids.txt` documents a resolved 2026-08-02 outage whose
  root cause was a UTF-8 BOM on the first line of `code/.env` (python-dotenv parsed
  the key as `\ufeffGROQ_API_KEY`). The file records that all degraded rows were
  re-routed and `output.csv` is fully resolved. **When editing `.env`, save it as
  UTF-8 without BOM.**

---

## 7. Operational notes

- Keep `.env` out of version control; inject `GROQ_API_KEY` through the environment
  in any shared/CI context.
- The run is deterministic given an unchanged `dataset/` and warm caches; cold runs
  depend on live Groq availability.
- There is no server, scheduler, container image, or health endpoint — see the
  Deployment/Operability dimension in [`PRODUCTION_READINESS.md`](./PRODUCTION_READINESS.md).
