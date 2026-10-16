# Software Requirements Specification
## WhatsApp Message Notification Router
### HackerRank Orchestrate — August 2026

**Author:** Sumit Shukla
**Status:** Originally a pre-implementation planning document. Sections 1–11 and 13–16 are
preserved as the **original specification**. The system has since been implemented in
`code/`; several design decisions changed during the build. **§17 (Implementation Status &
Reconciliation) is authoritative where it differs from the original text**, and stale passages
below carry inline "Implemented:" notes. §12 now carries the real post-implementation numbers.

---

## Table of Contents

1. Introduction & Problem Statement
2. Goals & Non-Goals
3. System Architecture
4. Data Model & Schema
5. Decision Cascade — Design & Rationale
6. Media Pipeline
7. Retrieval & Evidence Selection
8. Preference Model
9. LLM Integration
10. Personalization & Override Logic
11. Safety & Scam Handling
12. Evaluation Methodology & Results
13. Determinism & Reproducibility
14. Limitations
15. Future Work
16. Appendices
17. Implementation Status & Reconciliation (authoritative where it differs)

---

## 1. Introduction & Problem Statement

WhatsApp delivers a single, undifferentiated stream of messages to every user: family chats,
society/class group notices, co-worker pings, business promotions, image posters, voice notes,
and scams all arrive with equal visual and notification weight. This creates two failure modes
simultaneously:

- **Signal loss** — an important message (an exam reminder buried in a normally-noisy class
  group, a payment confirmation, a direct mention) is missed because the surrounding channel is
  usually low-value and the user has learned to ignore it.
- **Attention tax** — low-value or unsafe messages (repetitive promotions, scam links, irrelevant
  group chatter) interrupt the user regardless of whether they'd actually want to see them.

The system specified here is a **Message Notification Router**: for every incoming WhatsApp
message, it decides one of three actions — `notify` (interrupt now), `digest` (show later),
or `mute` (suppress) — personalized to the receiving user, using message content (text, OCR'd
images, transcribed voice notes), sender/group/business metadata, and the user's historical
engagement behavior.

This document is the engineering plan for that system, written before implementation, so that
every downstream build decision (by the developer or by an AI coding agent) traces back to an
explicit, reviewable design choice rather than being improvised mid-build.

---

## 2. Goals & Non-Goals

### Goals

- Produce a personalized `notify` / `digest` / `mute` decision for every row in
  `dataset/messages.csv`, with a `message_type` classification, a short human-readable `reason`,
  a calibrated `confidence`, and relevant `evidence_message_ids`.
- Reason over three content modalities: plain text, image posters/screenshots (via OCR + visual
  understanding), and voice notes (via transcription).
- Personalize using per-user historical engagement (opens, replies, dismissals, reports) rather
  than treating all users identically.
- Guarantee that clear scam/safety-risk messages are muted regardless of a user's usual
  engagement pattern with that sender — safety overrides personalization, never the reverse.
- Preserve high-value signal inside otherwise-noisy channels — e.g., a normally-mutable class
  group should still surface a genuinely urgent message (exam reminder, direct mention) rather
  than being uniformly suppressed. This is the specific behavior motivated by real observed
  usage: class groups generate high message volume that trains users to tune out, but the
  important announcements (schedule changes, exam reminders) also flow through that same
  channel, so blanket muting loses exactly the messages that matter most.
- Be modular (SOLID principles — each module has one responsibility, dependencies point through
  interfaces/Protocols, not concrete implementations) so components (LLM provider, retrieval
  strategy, media backend) can be swapped without touching decision logic.
- Be deterministic and reproducible on repeated runs against the same input.
- Keep LLM usage minimal and justified — call an LLM only when deterministic rules and a
  historically-trained preference model are both insufficiently confident, for cost, latency,
  and auditability reasons.

### Non-Goals (explicitly out of scope, and why)

- **True online reinforcement learning.** RL requires an agent acting repeatedly in an
  environment and receiving reward signal from those actions to update policy. The evaluation
  harness for this project generates `output.csv` once against a static dataset and compares it
  to hidden ground-truth labels — there is no live user in the loop during evaluation, and
  therefore no environment to actually run RL against. What this system builds instead is an
  **offline-trained preference model** using historical engagement data (`message_events.csv`)
  as a reward proxy — this is honest, tractable in the available time, and is the right
  foundation to later extend into true online learning (see §15). Claiming RL without this
  distinction would not survive scrutiny in the AI Judge interview.
- **Live/continuous re-training during the hackathon submission.** The preference model is
  trained offline once against provided historical data; a documented retraining path is
  specified (§8.4) but not executed as part of this submission.
- **Any provider other than Groq for LLM/vision/ASR calls.** Local fallbacks exist only for
  resilience against rate limits, not as a primary path.
- **A production notification-delivery system.** This system produces routing *decisions*; it
  does not implement push delivery, digest scheduling UI, or a live WhatsApp integration.

---

## 3. System Architecture

The codebase is organized into layers with a single dependency direction: orchestration depends
on interfaces defined by every other layer; nothing depends on orchestration. This is what lets
any individual component (LLM provider, retrieval algorithm, media backend, preference model
type) be replaced without touching the decision logic itself — a direct application of the
Dependency Inversion and Single Responsibility principles.

```
models/          — pure data (dataclasses), no behavior
ingestion/        — CSV loading + in-memory indexed repository (Protocol-backed)
media/            — image/audio understanding (Groq primary, local fallback)
retrieval/        — historical evidence ranking
safety/           — deterministic rule engine (highest priority, runs first)
scoring/          — offline-trained preference model
llm/              — Groq-backed structured reasoning fallback
orchestration/    — the decision cascade that ties the above together
personalization/  — post-decision overrides (quiet hours, mention escalation)
output/           — CSV writer matching the required schema exactly
evaluation/       — scoring against sample_messages.csv
```

Each layer other than `models`, `orchestration`, and `output` exposes its capability behind a
`Protocol` (structural interface) defined in that layer's own `interfaces.py`, so the
orchestrator depends on an abstraction ("something that can retrieve evidence") rather than a
concrete class ("the TF-IDF retriever"). No file is permitted to mix I/O, decision logic, and
formatting concerns — this is the direct answer to "not clutter a single file with a huge
codebase."

**Rationale for this shape specifically:** the task has three genuinely separable concerns —
*understanding* the message (media/retrieval/context), *deciding* what to do with it
(safety/scoring/llm/orchestration), and *personalizing* the decision to the specific user
(personalization). Collapsing these into one pipeline function would make it impossible to unit
test the safety rules independently of the LLM, or to swap the preference model without touching
media handling — both of which this project needs to do iteratively.

---

## 4. Data Model & Schema

The authoritative input schema, per the problem statement, for `dataset/messages.csv`:

| Field | Meaning |
|---|---|
| `message_id` | unique incoming message ID |
| `user_id` | receiving user |
| `conversation_type` | `personal`, `group`, or `business` |
| `group_id` | present if group message |
| `business_id` | present if business message |
| `sender_user_id` | present if from a user |
| `created_at` | timestamp |
| `message_text` | text content; empty for voice notes |
| `media_type` | empty, `image`, or `voice` |
| `media_id` | links to `images.csv` / `voice_notes.csv` |
| `forwarded_count` | forwarding signal |

Supporting context files (`users.csv`, `groups.csv`, `group_members.csv`,
`business_accounts.csv`, `user_business_history.csv`, `message_history.csv`,
`message_events.csv`, `images.csv`, `voice_notes.csv`, `daily_notification_summary.csv`) are
joined in at inference time via the `DataRepository` (§3) — indexed by `user_id`, `group_id`,
`business_id`, and `sender_user_id` for O(1)-class lookups.

Required output schema for `output.csv` (exact column order, one row per input row):

```
message_id,action,message_type,reason,confidence,evidence_message_ids
```

- `action` ∈ {`notify`, `digest`, `mute`}
- `message_type` ∈ {`personal`, `urgent`, `event`, `payment`, `business_update`, `promotion`,
  `greeting`, `forward`, `spam`, `scam`, `unknown`}
- `confidence` ∈ [0, 1]
- `evidence_message_ids` — semicolon-separated historical message IDs, or the literal string
  `none`

**Note:** exact column headers and types for the supporting CSVs are to be confirmed against the
actual files at build time (opencode Prompt 2) rather than assumed here — if any discrepancy from
this document is found, it should be logged and this section updated, not silently worked around.

> **Implemented:** headers were confirmed at build time; the loaders validate every header and
> parse strictly (`code/ingestion/loaders.py`, `code/ingestion/parsing.py`, smoke-tested in
> `code/tests/test_smoke.py`). The input schema in this section matches the dataset. Domains are
> now normalized case-insensitively (`normalize_domain`, `code/ingestion/parsing.py:66`).

---

## 5. Decision Cascade — Design & Rationale

Every message passes through up to four stages, in strict order:

1. **Safety rule engine** (deterministic, §11) — can force `mute` immediately with high
   confidence. Short-circuits everything downstream. Never overridden by personalization.
2. **Preference model** (offline-trained, §8) — for anything the rule engine didn't catch,
   compute a priority score from the user's historical engagement with this specific
   sender/group/business. If confident, this alone produces the action/message_type/reason
   without an LLM call.
3. **LLM fallback** (Groq, §9) — invoked only when stages 1–2 are low-confidence or the content
   genuinely requires semantic reasoning (novel sender, ambiguous phrasing, interpreting OCR'd
   poster text or a voice-note transcript).
4. **Personalization overrides** (§10) — applied last: quiet hours, muted-group-but-directly-
   mentioned escalation, per-user sensitivity.

**Why this order, specifically:**

- Safety-first is non-negotiable: a scam should never be notified because a preference model
  or LLM was uncertain. Determinism and explainability matter most here, so rules run first and
  are allowed to veto everything else.
- Preference-model-before-LLM keeps the system fast, cheap, and mostly deterministic — the
  majority of messages (repeat senders, established groups, familiar businesses) should be
  resolvable from historical engagement alone without paying LLM latency/cost or introducing
  LLM non-determinism.
- LLM-last means Groq is reserved for the cases that actually need reasoning — new senders,
  ambiguous or multi-modal content, edge cases the historical data can't disambiguate. This is
  both a cost-control decision and an interpretability one: the system should be able to report,
  per run, what fraction of decisions needed the LLM at all (§12).
- Personalization-last is deliberate: it's easier to reason about a small set of override rules
  applied to an already-reasonable base decision than to bake every personalization edge case
  into the scoring/LLM stage itself.

**Motivating case study (class-group scenario):** A university class WhatsApp group generates
high message volume — much of it low-value chatter — so a user's historical engagement with that
group trains the preference model toward `digest`/`mute` as the default action. But a message
containing exam/deadline language, or one where the sender holds an admin/teacher-like role in
`group_members.csv`, should not be silently muted just because the group's baseline is noisy.
This is handled by two mechanisms working together: the preference model scores at the
message level (not just the group level) using features like urgency keywords and sender role,
and the personalization layer explicitly escalates a muted-group message that directly mentions
the user. Neither the group's mute state nor its noise history is allowed to be the sole
determinant of an individual message's routing.

---

## 6. Media Pipeline

Two content types beyond plain text must be understood before routing:

- **Images** (posters, screenshots, promotional graphics) — handled by
  `media/groq_vision_analyzer.py`, using a Groq vision-capable model (Llama 4 Scout/Maverick or
  Qwen vision) to extract both the literal text (OCR) and a visual-intent summary (e.g. "sale
  poster," "payment QR code," "official notice screenshot") in a single call, which is cheaper
  and more useful than a separate OCR step followed by a separate reasoning step.

  > **Implemented:** the wired vision model is **`qwen/qwen3.6-27b`**, taken from
  > `GROQ_MODEL_VISION` (`code/config.py:59`, `code/.env.example`) and passed by
  > `code/media/factory.py:32`. The class-level default
  > (`llama-4-scout-17b-16e-instruct`, `code/media/groq_vision_analyzer.py:39`) is only a
  > fallback for direct construction and is not used by the production composition root.
- **Voice notes** — handled by `media/groq_whisper_transcriber.py`, using Groq's Whisper Large v3
  Turbo endpoint (OpenAI-compatible, ~228× real-time, generous free-tier daily allowance).

Both are behind Protocol interfaces (`ImageAnalyzer`, `AudioTranscriber`) with a local fallback
(`media/local_fallback.py`: pytesseract + local Whisper) that activates automatically on a
rate-limit or auth error from Groq, so the pipeline degrades gracefully rather than halting on
a full dataset run.

**Caching:** every media analysis result is cached on disk keyed by file content hash, so
repeated pipeline runs (during development, or re-running for the SRS evaluation numbers) never
re-incur Groq API calls for the same file. A simple rate limiter (token-bucket or paced sleep)
protects against exceeding Groq's free-tier per-minute limits on a full dataset run.

Extracted text/summary from media is merged into the message's effective text before it reaches
retrieval, safety rules, the preference model's text-derived features, and the LLM prompt — so
downstream stages don't need to know whether content originated as typed text, an image, or a
voice note.

---

## 7. Retrieval & Evidence Selection

`retrieval/tfidf_retriever.py` selects historical evidence for `evidence_message_ids`:

1. Scope candidates to `message_history.csv` entries from the same sender, group, or business as
   the current message — cross-source evidence is rarely meaningful for this task, so scoping
   happens before ranking, not after.
2. Rank scoped candidates by TF-IDF cosine similarity between the current message's (possibly
   media-derived) text and each candidate's text, combined with a recency decay factor so a
   similar-but-stale message doesn't outrank a similar-and-recent one.
3. Return the top-k (default 3) IDs with scores; return an empty list — rendered as `none` in
   the output — when nothing in scope clears a minimum similarity threshold, rather than forcing
   a weak match.

Embeddings/vector search were deliberately not chosen for v1: at this dataset scale, TF-IDF plus
scoping plus recency gives good enough ranking without adding an embedding-model dependency and
its own rate-limit/cost surface. This is documented as a candidate improvement in §15.

---

## 8. Preference Model

### 8.1 Purpose

Represents "how has this specific user actually engaged with this specific sender/group/business
in the past" as a score the orchestrator can act on without an LLM call for the common case.

### 8.2 Features

Derived per (user, source) pair from `message_history.csv` + `message_events.csv` +
`group_members.csv` + `users.csv`:

- Historical open rate, reply rate, dismiss rate, report count for this source.
- Message volume / noise ratio for the source over time (distinguishes a chatty-but-valuable
  group from a chatty-and-ignorable one).
- Time-of-day pattern relative to the user's typical engagement window and quiet hours.
- Whether the current message directly `@mentions` the user.
- Sender role signal where derivable (admin/teacher-like role in `group_members.csv`).

### 8.3 Model & Training

`scoring/train_preference_model.py` trains a scikit-learn classifier (logistic regression as the
default, gradient boosting as a fallback if the label distribution supports it) using engagement
outcomes from `message_events.csv` as the reward proxy — an "opened/replied" outcome is a
positive label for `notify`-worthiness, "dismissed/reported" is a negative one. The fitted model
and its feature schema are persisted to an `artifacts/` directory. If historical data proves too
sparse for a stable fit (a real risk given hackathon-scale data), the design explicitly permits
falling back to a simpler weighted-heuristic engagement score rather than shipping an unstable
model — this is a documented decision point, not a silent failure mode.

> **Implemented (differs from spec):** the shipped classifier is a
> **`GradientBoostingClassifier`** (`code/scoring/train_preference_model.py:180`), not logistic
> regression. `build_preference_model` selects the runtime implementation from the schema's
> `model` field (`code/scoring/preference_model.py:237`); the committed artifact is
> `"model": "gradient_boosting"` (`code/artifacts/preference_schema.json`). The sparse-data
> fallback is the **weighted-engagement-rate heuristic**, `HeuristicPreferenceModel`
> (`code/scoring/preference_model.py:147`) — not a "simpler logistic-regression model". Training
> also persists an isotonic calibrator and a learned mute band. Full details in §17.

### 8.4 "Learns and responds as it grows" — the honest version

The system is designed with a clear retraining path: as more `message_events.csv`-equivalent
data accumulates in a real deployment, `train_preference_model.py` can be re-run periodically
(or triggered on a data-volume threshold) to update the artifact the live system loads. This is
the mechanism that actually satisfies "continues to learn and respond as it grows with the
environment" for this project — see §15 for how this would extend into true online learning in
a production setting.

---

## 9. LLM Integration

Used only as the third stage of the cascade (§5), via Groq (recommended default: Llama 3.3 70B
for general reasoning quality/speed balance on Groq's LPU hardware).

- **Prompting:** `llm/prompt_builder.py` assembles the message (with media-derived text merged
  in), a condensed context summary (user/group/business relevant fields), retrieved evidence
  snippets, and 4–6 diverse few-shot examples selected from `sample_messages.csv` — chosen to
  cover a spread of actions and message_types, not just the first N rows in the file.
- **Structured output:** the model is instructed to return strict JSON matching
  `{action, message_type, reason, confidence, evidence_message_ids}`; responses are validated
  against the `RoutingDecision` model, with a single corrective retry on malformed JSON.
- **Determinism:** temperature 0 for every call.
- **Cost/latency discipline:** responses are cached by a hash of (message content + context
  summary), and the system logs what fraction of total messages actually reached this stage —
  a low fraction is evidence the rules/preference-model stages are doing their job, and is a
  reportable metric (§12).

---

## 10. Personalization & Override Logic

Applied after the base decision (rule engine, preference model, or LLM), never before:

- **Quiet hours** (`users.csv`): downgrade `notify` → `digest`, *unless* the message was flagged
  urgent/unsafe by the rule engine, in which case quiet hours never suppress a genuine safety or
  urgency signal.
- **Muted-group-but-mentioned escalation**: a message from a group the user has muted
  (`group_members.csv`) that directly `@mentions` the user is never silently muted — it's
  escalated to at least `digest`, or `notify` if urgency signals (deadline/exam-type language,
  admin/teacher sender role) are present. This directly implements the class-group case study
  from §5.
- **Non-negotiable boundary:** personalization can raise an action's priority but can never
  downgrade or override a safety-rule `mute` on a scam/spam decision. Convenience never trumps
  safety in this design.

---

## 11. Safety & Scam Handling

`safety/rule_engine.py` runs first and can force `mute` with `message_type` `scam` or `spam`
before any probabilistic scoring occurs. Rules include, at minimum:

- Unverified business sender + payment/urgent-action language + a sender domain that doesn't
  match any known/verified pattern in `business_accounts.csv`.
- Sender or business with a meaningful prior report count (`message_events.csv` /
  `business_accounts.csv`).
- Known scam-pattern language: prize/lottery framing, urgent account-suspension threats,
  OTP-sharing requests, suspicious shortened links, identity-mismatch signals.
- Extreme `forwarded_count` combined with generic/broadcast-style phrasing.

Each rule returns a structured hit (action, message_type, reason, confidence) or `None`; if
multiple rules fire, the highest-confidence hit wins. Rules are intentionally conservative —
false positives here wrongly suppress legitimate messages, which is exactly the failure mode
this whole system exists to avoid, so precision is prioritized over recall in this layer
specifically (recall on genuinely ambiguous risk is a job better suited to the LLM stage, which
can express lower confidence rather than a hard mute).

---

## 12. Evaluation Methodology & Results

### Methodology

`evaluation/evaluate.py` compares generated `output.csv` rows against the labeled rows in
`dataset/sample_messages.csv` and reports:

- Action accuracy (confusion matrix over `notify`/`digest`/`mute`).
- `message_type` accuracy (confusion matrix over all 11 categories).
- Evidence overlap — whether generated `evidence_message_ids` intersect with the expected ones
  where the sample provides them.
- Confidence calibration — whether higher-confidence predictions correlate with correctness.
- Stage-resolution breakdown — what fraction of decisions were resolved by rules vs. the
  preference model vs. the LLM fallback, which doubles as a cost/latency health metric.

### Results — measured post-implementation

Real output of `python -m evaluation.evaluate` (from `code/`, 2026-10-16) over the 30
labeled rows in `dataset/sample_messages.csv`. The samples are disjoint from
`messages.csv`, so the evaluator routes them through the production pipeline; re-runs are
served from the on-disk caches.

| Metric | Value |
|---|---|
| Action accuracy | 0.733 (22/30) |
| Message-type accuracy | 0.400 (12/30) |
| Evidence overlap rate (mean Jaccard; 28 samples with expected ids) | 0.12 |
| Pearson(confidence, correctness) | +0.058 |
| Calibration — conf 0.50–0.60 | acc 75%, n=4 |
| Calibration — conf 0.60–0.70 | acc 100%, n=3 |
| Calibration — conf 0.70–0.80 | acc 60%, n=15 |
| Calibration — conf 0.80–0.90 | acc 100%, n=2 |
| Calibration — conf 0.90–1.00 | acc 83%, n=6 |
| % resolved by rule engine | 20% (6/30), action accuracy 100% |
| % resolved by preference model | 50% (15/30), action accuracy 67% |
| % resolved by LLM fallback | 30% (9/30), action accuracy 67% |
| Full-dataset runtime | reported by `main.py` per run (not captured here) |

**Interpretation:** action routing is reasonable (73%); **message-type classification is
weak (40%)** and confidence is poorly calibrated (Pearson +0.058). Rule-engine decisions
are perfectly precise on the small sample; the preference-model and LLM stages each carry
the errors. The sample is only 30 rows, so treat these numbers as indicative.

*(The original planning note said to paste Prompt 10's output here; this table instead
records a measured run of the implemented code. `docs/PRODUCTION_READINESS.md` scores the
implications honestly.)*

---

## 13. Determinism & Reproducibility

- All Groq calls (text, vision, ASR) use temperature 0.
- Every media analysis and LLM call is cached on disk by content hash, so repeated runs against
  unchanged input produce identical output without re-incurring API calls.
- The preference model is trained offline with a fixed random seed and persisted as an artifact,
  not retrained implicitly at inference time.
- The rule engine and retrieval scoring are purely deterministic functions of the input data.

Given these, two runs of `main.py` against an unchanged `dataset/` should produce byte-identical
`output.csv` files, modulo timestamps in logs.

---

## 14. Limitations

- **Data sparsity risk in the preference model.** Hackathon-scale historical data may not
  provide enough (user, source) engagement history to train a stable classifier; the design
  explicitly permits falling back to a heuristic engagement score rather than shipping an
  overfit model (§8.3).
- **Media quality dependency.** Routing quality for image/voice messages is bounded by OCR and
  transcription accuracy; a garbled poster or noisy voice note can degrade the effective text the
  rest of the pipeline reasons over.
- **Rule engine precision/recall tradeoff.** Deliberately conservative rules (§11) reduce false
  positives on legitimate messages but may under-catch novel scam patterns not covered by the
  rule set — that residual risk is intentionally deferred to the LLM stage, which can flag lower
  confidence rather than silently pass content through.
- **No live personalization signal during evaluation.** The system reasons from historical
  engagement only; it cannot ask the user a clarifying question or observe a live reaction during
  the graded run (see §15 for how this would change in production).

---

## 15. Future Work

- **True online learning.** In a production deployment with a live user, this design extends
  naturally into a contextual bandit (e.g. Thompson sampling) over notify/digest/mute per
  sender-cluster, updated after each real user action — closing the loop that a static offline
  evaluation cannot.
- **Active preference elicitation.** After a new type of sender/group/business is encountered,
  the system could occasionally surface a lightweight in-app prompt ("Want to be notified for
  messages like this?") and feed the response back into the preference model as a direct label —
  a more targeted signal than passively inferred engagement.
- **Richer retrieval.** Replace or augment TF-IDF with embedding-based semantic retrieval once
  volume justifies the added infrastructure and cost.
- **Multi-modal fusion.** Combine image and text signals more tightly for messages that mix a
  poster with a caption, rather than treating OCR output as plain merged text.
- **Per-user sensitivity tuning UI.** Expose the personalization thresholds (§10) as user-facing
  settings rather than fixed defaults.

---

## 16. Appendices

### A. Full input schema reference
See §4 — cross-checked against actual CSV headers at build time. The loaders validate headers
and parse strictly (`code/ingestion/loaders.py`, `code/ingestion/parsing.py`); `tests/test_smoke.py`
verifies row counts and joins. No discrepancy was found.

### B. Prompt templates
LLM few-shot prompt structure and rule-engine rule definitions are maintained as versioned code
in `llm/prompt_builder.py` and `safety/rule_engine.py` respectively — referenced here rather than
duplicated, so this document never drifts out of sync with the actual implementation.

### C. Sample input/output pairs
`dataset/sample_messages.csv` (30 labeled rows) and `dataset/output.csv` (the generated
predictions) are the concrete input/output pair. `evaluation/evaluate.py` compares them
(§12). The generated rows can be inspected directly with `tests/peek_output.py`.

---

## 17. Implementation Status & Reconciliation

This section reconciles the original pre-implementation specification with the code that
now exists in `code/`. **Where this section and an earlier section disagree, this section
and the code are authoritative.** The original problem statement is preserved (see
[`problem_statement.md`](problem_statement.md)) and nothing above was deleted.

### 17.1 Stale facts corrected

| # | Original spec said | Implemented reality | Evidence |
|---|---|---|---|
| 1 | §8.3: default classifier is **logistic regression**, gradient boosting as a fallback | Ships a **`GradientBoostingClassifier`**; the fallback is the **weighted-engagement-rate heuristic** (`HeuristicPreferenceModel`), not logistic regression | `code/scoring/train_preference_model.py:180`, `code/scoring/preference_model.py:147,246`, `code/artifacts/preference_schema.json` (`"model":"gradient_boosting"`) |
| 2 | §6: vision model examples "Llama 4 Scout/Maverick or Qwen vision" | Wired vision model is **`qwen/qwen3.6-27b`** (`GROQ_MODEL_VISION`) | `code/config.py:59`, `code/.env.example`, `code/media/factory.py:32` |
| 3 | §12: Results "**pending**" / a template | Filled with measured numbers (action 73.3%, type 40.0%, evidence Jaccard 0.12, Pearson +0.058) | `python -m evaluation.evaluate`, see §12 |
| 4 | §4/§16.A: CSV headers "**to be confirmed at build time**" | Confirmed; strict header validation in the loaders, smoke-tested | `code/ingestion/loaders.py`, `code/ingestion/parsing.py`, `code/tests/test_smoke.py` |
| 5 | §3 layer list: `models, ingestion, media, retrieval, safety, scoring, llm, orchestration, personalization, output, evaluation` | Adds a **`context/`** layer (per-message context dict) plus `tools/` | `code/context/context_builder.py`, `code/tools/` |
| 6 | §8.3: persists "fitted model and its feature schema" | Also persists an **isotonic calibrator** and a **learned mute band**; plus `preference_metrics.json` | `code/scoring/train_preference_model.py:218,276-302` |
| 7 | Naming `scoring/train_preference_model.py` (file path) | Invoked as a module: `python -m scoring.train_preference_model` | `code/scoring/train_preference_model.py:306` |
| 8 | §9: text model "recommended default: Llama 3.3 70B" | Exact string is **`llama-3.3-70b-versatile`** | `code/config.py:57`, `code/.env.example` |
| 9 | §10: mentions "**per-user sensitivity**" as an implemented override | Implemented overrides are **quiet-hours downgrade** and **muted-group mention escalation** only; no separate per-user sensitivity knob | `code/personalization/overrides.py:77-142` |
| 10 | §8.2/§10: "directly `@mentions` the user" (ambiguous) | Mentions are parsed at **token boundaries** via shared helpers, so `@sam123` does not match `sam` | `code/ingestion/parsing.py:23,37`; `code/personalization/overrides.py:52`; `code/scoring/features.py:253` |
| 11 | §11: domain/identity comparisons (implied literal) | Domains are compared **case-insensitively** after `normalize_domain` (trim + lowercase + strip root dot) in both loaders and the rule engine | `code/ingestion/parsing.py:66`; `code/safety/rule_engine.py:29,131,134,348` |
| 12 | §6: media "local fallback activates on rate-limit or auth error" | Broadened to any **provider-availability** error, including transient **5xx `InternalServerError`**; retries with backoff first | `code/media/resilient.py:34-39,61-74`; `code/media/groq_vision_analyzer.py:74`; `code/media/groq_whisper_transcriber.py:56` |
| 13 | §13: "byte-identical `output.csv`" claim | Holds given unchanged input **and warm caches**; a cold clone depends on live provider availability/version because caches are git-ignored | `code/media/cache.py`, `code/llm/cache.py`, `.gitignore` |

### 17.2 Config keys as implemented

`code/config.py` defines settings whose keys/defaults are listed in
[`docs/CONFIGURATION.md`](docs/CONFIGURATION.md). Two keys are currently **inert**:
`LOG_LEVEL` is read into `Settings` but never wired to `logging.basicConfig`, and
`PREFERENCE_MODEL_FALLBACK_THRESHOLD` is read by no code path.

### 17.3 Requirement status (§ by §)

| SRS area | Status |
|---|---|
| §5 decision cascade (4 stages) | Implemented as specified; stage order and safety-first invariant hold (`code/orchestration/router.py`) |
| §6 media pipeline | Implemented (Groq vision/ASR + local fallback, disk cache, rate limiter); see corrected model names |
| §7 retrieval | Implemented (TF-IDF, sender→group→business scoping, recency decay, top-k) (`code/retrieval/tfidf_retriever.py`) |
| §8 preference model | Implemented with the corrected classifier/fallback; trained artifact committed under `code/artifacts/` |
| §9 LLM integration | Implemented (Groq, JSON response mode, temperature 0, single corrective retry, content-hash cache) (`code/llm/groq_client.py`) |
| §10 personalization | Implemented as quiet-hours + muted-group mention escalation (per-user sensitivity not implemented) |
| §11 safety rules | Implemented as 10 deterministic rules with fixed confidences, highest-confidence hit wins (`code/safety/rule_engine.py`) |
| §12 evaluation | Implemented and populated with measured numbers |
| §13 determinism | Implemented (temperature 0, caches, seeded offline training) |
| §14 limitations, §15 future work | Still accurate; extended by `README.md` and `docs/PRODUCTION_READINESS.md` |
