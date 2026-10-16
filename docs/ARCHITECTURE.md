# Architecture

Message Notification Router — HackerRank Orchestrate, August 2026.

This document describes the **implemented** system as it exists in `code/`. Every
claim is tied to a source file so it can be verified. If this document and the code
ever disagree, the code is authoritative.

---

## 1. What the system does

For every row in `dataset/messages.csv` the system emits exactly one routing
decision (`notify` / `digest` / `mute`) with a message type, a short reason, a
confidence, and evidence ids, written to a CSV with the exact schema
`message_id,action,message_type,reason,confidence,evidence_message_ids`
(`code/output/csv_writer.py`, validated in `code/main.py:102`).

---

## 2. High-level shape

```text
                    dataset/*.csv
                         │
                 ingestion (loaders + repository)
                         │
   image/voice ── media/ ─┤  (OCR / transcript merged into effective text)
                         │
              retrieval/ │  (TF-IDF evidence, scoped + recency-decayed)
                         │
                  context/  (per-message context dict)
                         │
              orchestration/router  ── decision cascade ──┐
                         │                                │
                         │   [1] safety rules (deterministic)
                         │   [2] preference model (learned)
                         │   [3] LLM judgment (Groq, cached)
                         │        ↓
                         │   personalization overrides
                         │        ↓
                    output/csv_writer ──► output.csv
                         │
                  orchestration/audit ──► artifacts/audit.jsonl
```

---

## 3. Layers and dependency direction

The orchestration layer depends on every other layer **through its Protocol /
interface**; nothing depends on orchestration. Concrete implementations are wired
only in composition roots (`code/main.py`, `code/*/factory.py`).

| Layer | Responsibility | Key files |
|---|---|---|
| `models/` | Pure data (frozen dataclasses), no I/O, minimal validation | `code/models/decision.py` (allowed actions / message types), `message.py`, `user.py`, `group.py`, `business.py`, `event.py`, `media.py` |
| `ingestion/` | CSV loaders, strict field parsing, in-memory indexed repository (Protocol) | `loaders.py`, `parsing.py`, `repository.py` |
| `media/` | OCR + voice transcription (Groq primary, local fallback), cache, rate limiter, retries | `groq_vision_analyzer.py`, `groq_whisper_transcriber.py`, `resilient.py`, `factory.py`, `local_fallback.py` |
| `retrieval/` | Historical evidence ranking | `tfidf_retriever.py`, `factory.py` |
| `safety/` | Deterministic safety rule engine (first cascade stage) | `rule_engine.py` |
| `scoring/` | Engagement feature engineering + trained preference model + training script | `features.py`, `preference_model.py`, `train_preference_model.py` |
| `llm/` | Groq-backed structured reasoning (third cascade stage) | `groq_client.py`, `prompt_builder.py`, `cache.py`, `factory.py` |
| `context/` | Assembles the per-message context dict consumed by the router and LLM prompt | `context_builder.py` |
| `orchestration/` | The cascade + batch runner + audit log | `router.py`, `pipeline.py`, `audit.py` |
| `personalization/` | Post-decision overrides (quiet hours, muted-group mention escalation) | `overrides.py` |
| `output/` | Exact-schema CSV writer | `csv_writer.py` |
| `evaluation/` | Offline scoring against labeled samples | `evaluate.py` |
| `tools/` | Maintenance scripts (Groq probe, degraded-id re-run/dump) | `tools/` |

The spec's layer list (`SRS.md` §3) predates the addition of the `context/` layer;
the implemented codebase includes it (`code/context/context_builder.py`).

---

## 4. Decision cascade

Order is strict; each stage runs only if the previous one is not decisive.
Thresholds are module constants in `code/orchestration/router.py:54-57`.

| Stage | Trigger / boundary | Produces | Source |
|---|---|---|---|
| 1. Safety rules | deterministic; a hit with confidence ≥ `RULE_MIN_CONFIDENCE` (0.70) short-circuits | `mute` (`scam`/`spam`) | `orchestration/router.py:161`, `safety/rule_engine.py` |
| 2. Preference model | model confidence ≥ `PREF_CONFIDENCE_MIN` (0.60) | `notify` / `digest` / `mute` | `orchestration/router.py:175`, `scoring/preference_model.py` |
| 3. LLM fallback | everything else; a `None` return degrades to a low-confidence `digest` (`LLM_FALLBACK_CONFIDENCE` 0.50) | `notify`/`digest`/`mute` | `orchestration/router.py:210`, `llm/groq_client.py` |
| 4. Personalization | always applied last | may raise/quiet-hour-downgrade | `personalization/overrides.py` |

Preference-model bands (`scoring/preference_model.py:33`, `code/orchestration/router.py:56`):

- `priority < mute_priority_threshold` → `mute` (low-value to **this** user; never `scam`/`spam`)
- `priority >= PREF_NOTIFY_PRIORITY` (0.65) → `notify`
- otherwise → `digest`

The mute band is learned offline (5th percentile of out-of-fold engaged priorities,
persisted as `mute_priority_threshold` in `artifacts/preference_schema.json`; the
committed artifact value is `0.3769`) and falls back to the documented
`DEFAULT_MUTE_PRIORITY = 0.30` when absent (`scoring/preference_model.py:30,202`).

**Invariant:** a rule-engine `scam`/`spam` mute is never modified by
personalization (`personalization/overrides.py:43`).

---

## 5. Media enrichment

`orchestration/pipeline.py:64` merges image OCR text / voice transcript into the
message's "effective text" before the cascade runs, so all three stages reason over
one text field regardless of modality. Enrichment is best-effort: a failure degrades
to the raw text and the message id is recorded in `runner.media_failures`
(`orchestration/pipeline.py:90-92`).

The media layer is wrapped as: disk cache → rate limiter → Groq primary
(retry with backoff) → local fallback (`media/resilient.py`). Provider-availability
errors (`RateLimitError`, `AuthenticationError`, `APIConnectionError`,
`InternalServerError`) trigger retry then fallback; other errors propagate so real
bugs are not masked (`media/resilient.py:34-39,61-74`).

---

## 6. Evidence retrieval

`retrieval/tfidf_retriever.py` scopes candidates to the same sender → group →
business, ranks by TF-IDF cosine similarity combined with recency decay, and returns
the top-k (`RETRIEVAL_TOP_K`, default 3) above `RETRIEVAL_MIN_SIMILARITY` (default
0.1). No match renders as `none`. On the LLM path, retriever ids are authoritative
and LLM-proposed ids are appended only if they exist in the known history set
(hallucinations dropped) — `orchestration/router.py:67`.

---

## 7. Recent behavior changes (documented)

These are current working-tree behaviors and are covered by tests:

1. **Case-insensitive domain normalization.** `ingestion/parsing.py:66`
   (`normalize_domain`) lower-cases, trims whitespace and a trailing root dot, and
   maps empty → `None`. It is used in loaders and in the safety engine
   (`safety/rule_engine.py:29,131,134,348`), so domain comparisons are case-insensitive.
2. **Token-boundary mention parsing.** Mentions are parsed by a shared
   `mentioned_user_ids` / `mentions_user` (`ingestion/parsing.py:23,37`) using a
   delimiter-anchored regex, so `@sam123` does **not** match user `sam`. Used by
   personalization (`personalization/overrides.py:52`) and by scoring features
   (`scoring/features.py:253`).
3. **LLM 5xx passthrough preserves fallback.** `groq.InternalServerError` is
   re-raised unchanged by both media analyzers
   (`media/groq_vision_analyzer.py:74`, `media/groq_whisper_transcriber.py:56`) and is
   part of `_AVAILABILITY_ERRORS` in the resilient wrapper
   (`media/resilient.py:34-39`), so a transient 5xx retries then falls back locally.
4. **Guarded group-member lookup.** The pipeline only looks up a group member when
   `group_id` is present:
   `get_group_member(message.user_id, message.group_id) if message.group_id else None`
   (`orchestration/pipeline.py:112-116`).

---

## 8. Determinism

- All Groq calls use temperature 0 (`config.py:94`, callers).
- Media analyses and LLM judgments are cached on disk by content hash
  (`media/cache.py`, `llm/cache.py`), so re-runs do not re-call the API.
- The preference model is trained offline with a fixed seed and persisted; it is
  never retrained implicitly at inference (`scoring/train_preference_model.py:186`).
- The rule engine and retrieval are pure functions of the input.

See `docs/PRODUCTION_READINESS.md` for known deviations and `SRS.md` §13 for the
original determinism contract.

---

## 9. Extension points

Every swappable component sits behind a Protocol/interface: `ImageAnalyzer`,
`AudioTranscriber`, `EvidenceRetriever`, `RoutingLLM`, `PreferenceModel`,
`DataRepository`. Substituting an implementation requires no change to
`orchestration/router.py`.
