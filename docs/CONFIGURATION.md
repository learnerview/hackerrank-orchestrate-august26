# Configuration

All runtime configuration is read from environment variables by
`code/config.py` (`Settings`, a frozen dataclass). Secrets are read from the
environment only. `code/.env` is loaded once at process start by `load_env()`
(`config.py:21`) and **only fills variables that are not already set**; it is
git-ignored (`.gitignore`). `code/.env.example` is the documented template.

---

## 1. Path anchoring

Relative paths resolve against fixed anchors, never the process working directory,
so the pipeline behaves identically from any cwd (`config.py:28-50`):

- `DATASET_DIR`, `OUTPUT_PATH` are anchored to the **repository root** (`code/..`).
- `MEDIA_CACHE_DIR`, `ARTIFACTS_DIR` (and everything under it) are anchored to the
  **code directory** (`code/`).

Absolute values pass through unchanged.

---

## 2. Environment variables

Defaults match `code/config.py` and `code/.env.example`.

| Name | Purpose | Default | Required |
|---|---|---|---|
| `GROQ_API_KEY` | Groq API key used for text LLM, vision, and ASR calls | *(empty)* | **Yes** for live LLM/vision/ASR. Absent → no live Groq calls; media degrades to local fallback; LLM returns `None` and the router falls back to a low-confidence `digest` |
| `GROQ_BASE_URL` | Groq API base URL | `https://api.groq.com` | No |
| `GROQ_MODEL_TEXT` | Text model for the LLM cascade stage | `llama-3.3-70b-versatile` | No |
| `GROQ_MODEL_VISION` | Vision model for image OCR + intent | `qwen/qwen3.6-27b` | No |
| `GROQ_MODEL_ASR` | ASR model for voice transcription | `whisper-large-v3-turbo` | No |
| `LOG_LEVEL` | Logging level string (read into `settings.log_level`; **not yet wired to `logging.basicConfig`**, so it currently has no runtime effect) | `INFO` | No |
| `DATASET_DIR` | Directory holding the input CSVs | `dataset` | No |
| `OUTPUT_PATH` | Where `output.csv` is written (CLI `--output` overrides) | `dataset/output.csv` | No |
| `ARTIFACTS_DIR` | Trained model, schema, metrics, LLM cache, audit log | `artifacts` | No |
| `MEDIA_CACHE_DIR` | Content-addressed media analysis cache | `.cache/media` | No |
| `MEDIA_RPM_VISION` | Vision requests/minute rate limit (token bucket) | `10` | No |
| `MEDIA_RPM_ASR` | ASR requests/minute rate limit | `20` | No |
| `MEDIA_RETRIES` | Retries per media call before local fallback | `2` | No |
| `RETRIEVAL_TOP_K` | Max evidence ids returned per message | `3` | No |
| `RETRIEVAL_MIN_SIMILARITY` | Minimum cosine similarity for evidence | `0.1` | No |
| `RETRIEVAL_RECENCY_DECAY_DAYS` | Recency-decay scale for evidence ranking | `60` | No |
| `PREFERENCE_MODEL_FALLBACK_THRESHOLD` | **Reserved — currently unused.** Defined in `config.py:96` but read by no code path as of this writing | `0.5` | No |

> The effective vision model is whatever `GROQ_MODEL_VISION` resolves to (wired in
> `media/factory.py:32`). The class-level default inside `GroqVisionAnalyzer`
> (`llama-4-scout-17b-16e-instruct`, `media/groq_vision_analyzer.py:39`) is only used
> if a caller constructs the analyzer without passing a model; the production
> composition root always passes `settings.groq_model_vision`.

---

## 3. Cascade thresholds (compile-time, not env-configurable)

These are documented constants in `code/orchestration/router.py:54-57` and
`code/scoring/preference_model.py`:

| Constant | Value | Meaning |
|---|---|---|
| `RULE_MIN_CONFIDENCE` | `0.70` | A safety-rule hit at/above this is decisive |
| `PREF_CONFIDENCE_MIN` | `0.60` | Preference model must be this confident to skip the LLM |
| `PREF_NOTIFY_PRIORITY` | `0.65` | Preference priority ≥ this maps to `notify` |
| `LLM_FALLBACK_CONFIDENCE` | `0.50` | Confidence of the digest fallback when the LLM fails |
| `DEFAULT_MUTE_PRIORITY` | `0.30` | Fallback mute band when no learned band exists |
| `OBSERVATION_VOLUME` | `10.0` | History volume treated as "fully observed" for confidence |

---

## 4. Setting it up

```powershell
# from the repository root
Copy-Item code\.env.example code\.env
# then edit code\.env and set GROQ_API_KEY=<your key>
```

Never commit `code/.env`. See [`../SECURITY.md`](../SECURITY.md).

---

## 5. Training artifacts and their schema

`python -m scoring.train_preference_model` writes to `ARTIFACTS_DIR`:

- `preference_model.joblib` — the fitted `GradientBoostingClassifier` (only written
  when the model beats the heuristic by ≥ 0.03 AUC and CV is stable).
- `preference_calibrator.joblib` — isotonic probability calibrator.
- `preference_schema.json` — `{feature_names, model, calibrated,
  mute_priority_threshold}`. `model` selects the runtime implementation in
  `build_preference_model` (`scoring/preference_model.py:237`): `gradient_boosting`
  → `TrainedPreferenceModel`; anything else → `HeuristicPreferenceModel`.
- `preference_metrics.json` — training metrics (see `docs/DEPLOYMENT.md`).

Committed artifact values: `model = "gradient_boosting"`, `mute_priority_threshold =
0.3769`.
