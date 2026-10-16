# Production Readiness Assessment

Honest, evidence-based self-assessment of the **current** implementation. Scores are
out of 10 and reflect fitness for a real deployment, **not** fitness for the
hackathon (where the same artifact would score much higher). Every score cites
evidence and the gap that keeps it below 10.

Evaluated against the working tree on **2026-10-16**. `python -m pytest` in `code/`:
**135 passed**. `python -m evaluation.evaluate`: action accuracy **73.3% (22/30)**,
message_type accuracy **40.0% (12/30)**, evidence Jaccard **0.12**, Pearson(conf,
correctness) **+0.058**.

| # | Dimension | Score | Evidence | Gap |
|---|---|---|---|---|
| 1 | Requirement clarity & completeness | 8/10 | Full spec in `SRS.md` + `problem_statement.md`; implemented reality reconciled in `SRS.md` §17; architecture documented in `docs/ARCHITECTURE.md` | The top-level spec drifted from the code (classifier, model names, §12) and required a reconciliation pass — the drift itself is the gap |
| 2 | Correctness / model quality | 6/10 | Offline preference model ROC-AUC 0.91 (committed `artifacts/preference_metrics.json`); deterministic rules 100% action accuracy on 6/6 sample rule cases | End-to-end action accuracy 73.3%, **message_type accuracy only 40.0%**, evidence Jaccard 0.12, confidence is poorly calibrated (Pearson +0.058); evaluation set is only 30 samples |
| 3 | Testing & verification | 7/10 | 135 unit tests across 10 files; strict loaders, rule engine, scoring, personalization, retrieval, output, orchestration all unit-tested; smoke test validates all FK joins | Exactly **1** end-to-end pipeline test (`tests/test_pipeline_units.py`); no test exercises a live Groq/media call; no CI workflow; no coverage threshold |
| 4 | Documentation | 8/10 | This `docs/` set, root `README`, `code/README`, `SECURITY.md`, `LICENSE`; every claim cited to source | Docs are hand-written and can drift; no doc build/lint in CI |
| 5 | Configuration & secrets management | 7/10 | Secrets from env only (`config.py:55`); `.env` git-ignored; `.env.example` documents keys; path anchoring is cwd-independent | A past outage was caused by a **UTF-8 BOM in `.env`** (`artifacts/degraded_message_ids.txt`); two config keys (`LOG_LEVEL`, `PREFERENCE_MODEL_FALLBACK_THRESHOLD`) are currently inert; no secret scanning / rotation |
| 6 | Reproducibility & determinism | 8/10 | Groq calls at temperature 0; content-hash caches for media and LLM; seeded offline training (fixed `random_state=42`); rule engine pure | Caches and `dataset/media/` are git-ignored, so a cold clone re-derives caches and depends on live provider availability/version; no pinned model snapshot hashes |
| 7 | Reliability & resilience | 7/10 | Retry-with-backoff then local fallback (`media/resilient.py`); LLM failure degrades to a fail-safe `digest` (`orchestration/router.py:219`); media failure degrades to raw text | Single provider (Groq); no circuit breaker; no configurable request timeouts; media fallback requires locally-installed Tesseract/Whisper that are not bundled |
| 8 | Observability & auditability | 6/10 | Per-decision audit log `artifacts/audit.jsonl` with stage attribution (`orchestration/router.py:242`); per-stage and per-media-failure console reporting | `LOG_LEVEL` is **not wired** to logging; no metrics, tracing, or alerting; audit log is per-run and unrotated |
| 9 | Security | 5/10 | Secrets via env only; prompt-injection rule + hardened system prompt (`safety/rule_engine.py:278`, `llm/groq_client.py:188`); no secrets in the repo | No PII redaction in reasons/audit; no input-size limits; no dependency/secret scanning; a single shared API key with no rotation; see `SECURITY.md` for the full posture |
| 10 | Performance & cost efficiency | 6/10 | Media results and LLM judgments cached by content hash; LLM used only for ambiguous cases (9/30 samples ≈ 30% of the evaluation run) | No request timeouts; rate limiting covers media only, not the text LLM; no cost budget or per-run cost accounting |
| 11 | Data privacy & governance | 4/10 | Reason strings are human-readable; no live user PII beyond the provided dataset | No data-retention policy; message text, OCR text and transcripts are written into the audit log and prompts; no consent/anonymization story; the hackathon dataset is synthetic/mock but the code would run unchanged on real data |
| 12 | Deployment & operability | 4/10 | Runs from the terminal with one command (`python main.py`); single-command training and evaluation | No container image, no packaging, no CI/CD, no service/health endpoint, no rollback or versioned releases; it is a batch script, not an operable service |

**Average: 6.3 / 10.**

Lowest-scoring dimensions: **Data privacy & governance (4/10)** and **Deployment &
operability (4/10)**, followed by **Security (5/10)**.

## Summary

Strong as a hackathon submission: modular, tested, deterministic, auditable, and
documented. Not production-ready: the weak areas are operationalization (no
deploy/CI/service story), data governance (no retention/PII handling), security
hardening (scanning, redaction, key rotation), and end-to-end accuracy — especially
`message_type` classification and confidence calibration on a small evaluation set.
