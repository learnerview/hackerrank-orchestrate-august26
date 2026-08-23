"""Unit tests for the LLM stage: prompt construction, judgment validation,
cache behavior and the Groq client's parse/validate/retry/cache flow, using
fakes so no network access is required.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from config import Settings  # noqa: E402
from llm.cache import LLMJsonCache  # noqa: E402
from llm.groq_client import GroqRoutingLLM, _coerce_judgment, extract_json_object  # noqa: E402
from llm.interfaces import LLMJudgment  # noqa: E402
from llm.prompt_builder import (  # noqa: E402
    OUTPUT_CONTRACT,
    FewShotExample,
    build_prompt,
    select_few_shot_examples,
)
from models.message import IncomingMessage, SampleMessage  # noqa: E402


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        groq_api_key="test-key",
        llm_cache_dir=tmp_path / "llm_cache",
        dataset_dir=tmp_path,
    )


def _message(msg_id: str = "msg_1", text: str = "Hello there") -> IncomingMessage:
    return IncomingMessage(
        message_id=msg_id,
        user_id="u_001",
        conversation_type="personal",
        group_id=None,
        business_id=None,
        sender_user_id="u_002",
        created_at="2026-07-20T10:00:00Z",
        message_text=text,
        media_type="none",
        media_id=None,
        forwarded_count=0,
    )


class FakeCompletions:
    """Fake Groq chat completions returning scripted contents."""

    def __init__(self, contents: list[str]) -> None:
        self.contents = list(contents)
        self.calls: list[str] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(str(kwargs.get("messages")))
        class _Choice:
            def __init__(self, content: str) -> None:
                self.message = type("M", (), {"content": content})()
        class _Resp:
            def __init__(self, content: str) -> None:
                self.choices = [_Choice(content)]
        content = self.contents.pop(0) if self.contents else '{"action":"digest"}'
        return _Resp(content)


class FakeClient:
    def __init__(self, contents: list[str]) -> None:
        self.chat = type("C", (), {"completions": FakeCompletions(contents)})()


# ---------------------------------------------------------------------- #
# prompt builder
# ---------------------------------------------------------------------- #
class TestPromptBuilder:
    def test_output_contract_enforced(self) -> None:
        prompt = build_prompt("hi", None, "", [], [])
        assert '"action"' in prompt
        assert '"message_type"' in prompt
        assert '"reason"' in prompt
        assert '"confidence"' in prompt
        assert '"evidence_message_ids"' in prompt
        assert "only" in prompt.lower() or "ONLY" in prompt

    def test_media_summary_merged(self) -> None:
        prompt = build_prompt("Buy 1 get 1", "poster with 50% off text", "ctx", [], [])
        assert "Buy 1 get 1" in prompt
        assert "poster with 50% off text" in prompt

    def test_evidence_and_examples_rendered(self) -> None:
        ex = FewShotExample("Good morning", "digest", "greeting", "low value", "0.8")
        prompt = build_prompt(
            "m", None, "ctx", ["<his_1> hi from raj", "<his_2> meeting moved"],
            [ex],
        )
        assert "his_1" in prompt and "his_2" in prompt
        assert '"action": "digest"' in prompt
        assert '"message_type": "greeting"' in prompt

    def test_selection_covers_all_actions_and_types(self) -> None:
        rows = [
            SampleMessage(
                message_id=f"s{i}",
                user_id="u_0",
                conversation_type="personal",
                group_id=None,
                business_id=None,
                sender_user_id="u_1",
                created_at="2026-07-01T00:00:00Z",
                message_text=f"Text number {i} with some more words to make it richer",
                media_type="none",
                media_id=None,
                forwarded_count=0,
                action=a,
                message_type=t,
                reason="r",
                confidence=0.9,
                evidence_message_ids="none",
            )
            for i, (a, t) in enumerate(
                [
                    ("notify", "urgent"),
                    ("notify", "event"),
                    ("digest", "promotion"),
                    ("digest", "greeting"),
                    ("mute", "scam"),
                    ("mute", "spam"),
                    ("mute", "forward"),
                    ("digest", "business_update"),
                ]
            )
        ]
        picked = select_few_shot_examples(rows, n=5)
        actions = {e.action for e in picked}
        types = {e.message_type for e in picked}
        assert actions == {"notify", "digest", "mute"}
        assert len(types) >= 4

    def test_selection_never_first_n_rows(self) -> None:
        combos = [
            ("mute", "spam"),
            ("mute", "spam"),
            ("mute", "spam"),
            ("mute", "spam"),
            ("digest", "promotion"),
            ("notify", "urgent"),
            ("digest", "greeting"),
            ("mute", "forward"),
        ]
        rows = [
            SampleMessage(
                message_id=f"s{i}",
                user_id="u_0",
                conversation_type="personal",
                group_id=None,
                business_id=None,
                sender_user_id="u_1",
                created_at="2026-07-01T00:00:00Z",
                message_text=f"row {i}",
                media_type="none",
                media_id=None,
                forwarded_count=0,
                action=combos[i][0],
                message_type=combos[i][1],
                reason="r",
                confidence=0.9,
                evidence_message_ids="none",
            )
            for i in range(8)
        ]
        picked = select_few_shot_examples(rows, n=4)
        ids = {e.message_text for e in picked}
        first_n = {f"row {i}" for i in range(4)}
        assert ids != first_n  # duplicate first rows must be skipped for diversity
        assert {e.action for e in picked} == {"mute", "digest", "notify"}


# ---------------------------------------------------------------------- #
# judgment parsing/validation
# ---------------------------------------------------------------------- #
class TestJudgmentParsing:
    def test_coerce_valid(self) -> None:
        j = _coerce_judgment(
            {
                "action": "notify",
                "message_type": "urgent",
                "reason": "real time",
                "confidence": 0.92,
                "evidence_message_ids": "his_5;his_9",
            }
        )
        assert j.action == "notify"
        assert j.confidence == 0.92

    def test_coerce_evidence_list_to_semicolons(self) -> None:
        j = _coerce_judgment(
            {
                "action": "digest",
                "message_type": "event",
                "reason": "r",
                "confidence": 0.5,
                "evidence_message_ids": ["his_1", "his_2"],
            }
        )
        assert j.evidence_message_ids == "his_1;his_2"

    def test_coerce_rejects_bad_action(self) -> None:
        with pytest.raises(ValueError):
            _coerce_judgment(
                {"action": "explode", "message_type": "urgent", "reason": "r", "confidence": 0.5}
            )

    def test_extract_fenced_and_plain_json(self) -> None:
        assert extract_json_object('```json\n{"action":"mute"}\n```') == {"action": "mute"}
        assert extract_json_object('Sure! Here: {"action":"digest","message_type":"event"} ok') == {
            "action": "digest",
            "message_type": "event",
        }

    def test_extract_missing_returns_none(self) -> None:
        assert extract_json_object("sorry I cannot answer") is None
        assert extract_json_object("") is None

    def test_judgment_as_routing_decision(self) -> None:
        j = LLMJudgment("mute", "spam", "junk", 0.8, "his_1")
        d = j.as_routing_decision("msg_7")
        assert d.message_id == "msg_7"
        assert d.resolved_by == "llm"
        assert d.action == "mute"


# ---------------------------------------------------------------------- #
# cache
# ---------------------------------------------------------------------- #
class TestLLMCache:
    def test_roundtrip(self, tmp_path: Path) -> None:
        c = LLMJsonCache(tmp_path / "c")
        content = "msg:hi|ctx:verified business"
        assert c.get(content) is None
        c.put(content, {"action": "notify"})
        assert c.get(content) == {"action": "notify"}

    def test_key_stability_and_sensitivity(self, tmp_path: Path) -> None:
        c = LLMJsonCache(tmp_path / "c")
        a = c.key_for("same content")
        b = c.key_for("same content")
        d = c.key_for("same content!")
        assert a == b
        assert a != d

    def test_corrupt_entry_returns_none(self, tmp_path: Path) -> None:
        c = LLMJsonCache(tmp_path / "c")
        content = "x"
        c.put(content, {"action": "mute"})
        entry = tmp_path / "c" / f"{c.key_for(content)}.json"
        entry.write_text("{not json", encoding="utf-8")
        assert c.get(content) is None


# ---------------------------------------------------------------------- #
# GroqRoutingLLM flow
# ---------------------------------------------------------------------- #
class TestGroqRoutingLLM:
    def test_happy_path_parses_and_caches(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        fake = FakeClient(
            [
                '{"action":"notify","message_type":"urgent","reason":"direct ask","confidence":0.91,"evidence_message_ids":"his_3"}'
            ]
        )
        llm = GroqRoutingLLM(settings, few_shot_examples=[], client=fake, cache=LLMJsonCache(settings.llm_cache_dir))
        j = llm.decide(_message(), {})
        assert j is not None
        assert j.action == "notify"
        assert j.evidence_message_ids == "his_3"
        assert len(fake.chat.completions.calls) == 1

    def test_cache_hit_skips_api(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        fake = FakeClient(
            [
                '{"action":"digest","message_type":"promotion","reason":"low","confidence":0.6,"evidence_message_ids":"none"}'
            ]
        )
        llm = GroqRoutingLLM(settings, few_shot_examples=[], client=fake, cache=LLMJsonCache(settings.llm_cache_dir))
        ctx = {"context_summary": "verified business, low volume"}
        assert llm.decide(_message(), ctx) is not None
        assert llm.decide(_message(), ctx) is not None
        assert len(fake.chat.completions.calls) == 1

    def test_retry_once_on_malformed_json(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        fake = FakeClient(
            [
                "Sorry, here is my analysis in prose: the message is promotional...",
                '{"action":"mute","message_type":"spam","reason":"unwanted promo","confidence":0.87,"evidence_message_ids":"none"}',
            ]
        )
        llm = GroqRoutingLLM(settings, few_shot_examples=[], client=fake, cache=LLMJsonCache(settings.llm_cache_dir))
        j = llm.decide(_message(), {})
        assert j is not None
        assert j.action == "mute"
        assert len(fake.chat.completions.calls) == 2
        assert "invalid JSON" in fake.chat.completions.calls[1]

    def test_two_failures_returns_none(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        fake = FakeClient(["nope", "still nope"])
        llm = GroqRoutingLLM(settings, few_shot_examples=[], client=fake, cache=LLMJsonCache(settings.llm_cache_dir))
        assert llm.decide(_message(), {}) is None
        assert len(fake.chat.completions.calls) == 2

    def test_invalid_action_triggers_retry(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        fake = FakeClient(
            [
                '{"action":"explode","message_type":"urgent","reason":"r","confidence":0.5}',
                '{"action":"digest","message_type":"event","reason":"fine","confidence":0.5,"evidence_message_ids":"none"}',
            ]
        )
        llm = GroqRoutingLLM(settings, few_shot_examples=[], client=fake, cache=LLMJsonCache(settings.llm_cache_dir))
        j = llm.decide(_message(), {})
        assert j is not None and j.action == "digest"
        assert len(fake.chat.completions.calls) == 2

    def test_api_error_returns_none(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)

        class Boom:
            def create(self, **kwargs: object) -> object:
                raise RuntimeError("network down")

        class BoomClient:
            chat = type("C", (), {"completions": Boom()})()

        llm = GroqRoutingLLM(settings, few_shot_examples=[], client=BoomClient(), cache=LLMJsonCache(settings.llm_cache_dir))
        assert llm.decide(_message(), {}) is None

    def test_content_from_context_wins(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        fake = FakeClient(
            [
                '{"action":"notify","message_type":"personal","reason":"media poster","confidence":0.7,"evidence_message_ids":"none"}'
            ]
        )
        llm = GroqRoutingLLM(settings, few_shot_examples=[], client=fake, cache=LLMJsonCache(settings.llm_cache_dir))
        j = llm.decide(_message(), {"message_text": "50% OFF TODAY", "media_summary": "sale poster"})
        assert j is not None
        assert "50% OFF TODAY" in fake.chat.completions.calls[0]
