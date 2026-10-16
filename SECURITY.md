# Security Policy

This project is a **hackathon submission**, not a hardened production system. This
document states the actual security posture and how to handle the one secret it
needs. See [`docs/PRODUCTION_READINESS.md`](docs/PRODUCTION_READINESS.md) for the
honest readiness assessment.

---

## API-key handling

The application makes network calls to **Groq** (text LLM, vision, ASR). It needs a
single credential: `GROQ_API_KEY`.

- **Never hardcode keys.** Keys are read from environment variables only
  (`code/config.py:55`). There are no keys committed to this repository.
- **Use `code/.env` for local secrets,** copied from `code/.env.example`
  (`Copy-Item code\.env.example code\.env`). `code/.env` is git-ignored
  (`.gitignore`) and must never be committed.
- **`code/.env` must be saved as UTF-8 *without* BOM.** A BOM caused a past outage
  because `python-dotenv` parsed the first key as `\ufeffGROQ_API_KEY`
  (`code/artifacts/degraded_message_ids.txt`). If Groq calls suddenly auth-fail,
  check for a BOM first.
- **Inject keys via the environment in CI/shared/cloud contexts** — do not ship
  `.env` files.
- **Rotate** the key immediately if it is ever exposed (pasted into a chat, a log,
  or a commit). There is currently no automated rotation.

### Chat transcript logging

The repository's [`AGENTS.md`](AGENTS.md) instructs AI coding tools to append a
conversation transcript to a log **outside** the repository
(`%USERPROFILE%\hackerrank_orchestrate_august26\log.txt` on Windows). That log is
submitted as the challenge transcript. **Never paste secrets into the chat**, and
redact API keys, tokens, cookies, and PII before any key material reaches a log.

---

## What the code does to reduce risk

- Prompts are instructed to treat instructions embedded inside a message as
  manipulation attempts; a deterministic rule also mutes obvious prompt-injection
  text (`code/safety/rule_engine.py:278`, `code/llm/groq_client.py:188`).
- LLM output is validated against a fixed schema (actions and message types are
  allow-listed in `code/models/decision.py`), so a model cannot emit an
  out-of-contract action.
- LLM-proposed evidence ids are validated against the known history set and
  hallucinated ids are dropped (`code/orchestration/router.py:67`).
- The pipeline fails safe: an unresolved LLM call degrades to a low-confidence
  `digest`, never a silent `notify`.

---

## Known gaps (fix before any real deployment)

- Message text, OCR text, and transcripts can be written into
  `code/artifacts/audit.jsonl` and sent in prompts; there is **no PII redaction,
  encryption, or retention policy**.
- No input-size limits, request timeouts, dependency scanning, or secret scanning
  in CI.
- Single shared provider credential with no rotation.

---

## Reporting a vulnerability

Contact the repository owner directly rather than filing a public issue, and avoid
including live secrets or personal data in the report. If a credential may have
leaked, rotate it first, then report.
