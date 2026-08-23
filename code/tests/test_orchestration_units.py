"""Unit tests for the decision cascade orchestrator: stage ordering, the
documented thresholds, evidence reconciliation, and audit logging — all with
fakes, no network or real models required.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from llm.interfaces import LLMJudgment  # noqa: E402
from models.decision import RuleHit, RoutingDecision  # noqa: E402
from models.message import EvidenceRef, IncomingMessage  # noqa: E402
from orchestration.audit import DecisionAuditLog  # noqa: E402
from orchestration.router import RoutingOrchestrator  # noqa: E402
from personalization.overrides import PersonalizationOverrides  # noqa: E402
from scoring.interfaces import PreferenceScore  # noqa: E402


def _msg(text="hi", conv="personal", hour="2026-07-20T10:00:00Z") -> IncomingMessage:
    return IncomingMessage(
        message_id="msg_1",
        user_id="u_001",
        conversation_type=conv,
        group_id=None,
        business_id=None,
        sender_user_id="u_002",
        created_at=hour,
        message_text=text,
        media_type="none",
        media_id=None,
        forwarded_count=0,
    )


def _context(**overrides: object) -> dict[str, object]:
    ctx: dict[str, object] = {
        "features": {"volume": 5.0, "open_rate": 0.5},
        "source_id": "u_002",
        "evidence_refs": [EvidenceRef("his_1", 0.9), EvidenceRef("his_2", 0.7)],
        "history_ids": frozenset({"his_1", "his_2", "his_9"}),
        "message_text": "hi",
        "context_summary": "ctx",
        "evidence_snippets": ["[his_1] old msg"],
    }
    ctx.update(overrides)
    return ctx


class FakeRules:
    def __init__(self, hit: RuleHit | None = None) -> None:
        self._hit = hit
        self.calls = 0

    def evaluate(self, message: IncomingMessage) -> RuleHit | None:
        self.calls += 1
        return self._hit


class FakePreference:
    mute_priority_threshold: float = 0.25

    def __init__(self, score: PreferenceScore) -> None:
        self._score = score
        self.calls = 0

    def score(self, user_id: str, source_id: str, features: dict[str, float]) -> PreferenceScore:
        self.calls += 1
        return self._score


class FakeLLM:
    def __init__(self, judgment: LLMJudgment | None) -> None:
        self._judgment = judgment
        self.calls = 0

    def decide(self, message: IncomingMessage, context: dict[str, object]) -> LLMJudgment | None:
        self.calls += 1
        return self._judgment


def _score(priority: float, confidence: float) -> PreferenceScore:
    return PreferenceScore(priority=priority, confidence=confidence)


def _judgment(action="digest", message_type="promotion", evidence="none") -> LLMJudgment:
    return LLMJudgment(
        action=action, message_type=message_type, reason="llm says so", confidence=0.8,
        evidence_message_ids=evidence,
    )


def _orchestrator(rules, pref, llm, audit: DecisionAuditLog | None = None) -> RoutingOrchestrator:
    return RoutingOrchestrator(
        rule_engine=rules,
        preference_model=pref,
        llm=llm,
        personalization=PersonalizationOverrides(),
        audit_log=audit,
    )


class TestStageOrdering:
    def test_rule_high_confidence_returns_immediately(self) -> None:
        pref = FakePreference(_score(0.99, 0.99))
        llm = FakeLLM(None)
        hit = RuleHit("mute", "scam", "otp leak", 0.95)
        orch = _orchestrator(FakeRules(hit), pref, llm)
        decision = orch.route(_msg(), _context())
        assert decision.action == "mute"
        assert decision.resolved_by == "rules"
        assert pref.calls == 0 and llm.calls == 0

    def test_rule_low_confidence_falls_through(self) -> None:
        pref = FakePreference(_score(0.8, 0.9))
        llm = FakeLLM(None)
        hit = RuleHit("mute", "scam", "weak rule", 0.5)
        orch = _orchestrator(FakeRules(hit), pref, llm)
        decision = orch.route(_msg(), _context())
        assert decision.resolved_by == "preference_model"

    def test_confident_preference_skips_llm(self) -> None:
        llm = FakeLLM(_judgment())
        orch = _orchestrator(FakeRules(None), FakePreference(_score(0.8, 0.9)), llm)
        decision = orch.route(_msg(), _context())
        assert decision.action == "notify"
        assert decision.resolved_by == "preference_model"
        assert llm.calls == 0

    def test_preference_low_priority_maps_to_digest(self) -> None:
        orch = _orchestrator(FakeRules(None), FakePreference(_score(0.40, 0.9)), FakeLLM(None))
        decision = orch.route(_msg(), _context())
        assert decision.action == "digest"
        assert decision.resolved_by == "preference_model"

    def test_preference_below_mute_band_maps_to_mute(self) -> None:
        llm = FakeLLM(None)
        orch = _orchestrator(FakeRules(None), FakePreference(_score(0.1, 0.9)), llm)
        decision = orch.route(_msg(), _context())
        assert decision.action == "mute"
        assert decision.resolved_by == "preference_model"
        assert llm.calls == 0

    def test_preference_mute_band_uses_model_threshold(self) -> None:
        pref = FakePreference(_score(0.24, 0.9))
        pref.mute_priority_threshold = 0.30
        orch = _orchestrator(FakeRules(None), pref, FakeLLM(None))
        decision = orch.route(_msg(), _context())
        assert decision.action == "mute"

    def test_preference_mute_keeps_non_safety_message_type(self) -> None:
        orch = _orchestrator(FakeRules(None), FakePreference(_score(0.05, 0.9)), FakeLLM(None))
        decision = orch.route(_msg(conv="business"), _context())
        assert decision.action == "mute"
        assert decision.message_type == "business_update"  # low-value, not scam/spam

    def test_preference_low_confidence_falls_to_llm(self) -> None:
        llm = FakeLLM(_judgment("mute", "spam"))
        orch = _orchestrator(FakeRules(None), FakePreference(_score(0.9, 0.3)), llm)
        decision = orch.route(_msg(), _context())
        assert decision.resolved_by == "llm"
        assert decision.action == "mute"
        assert decision.message_type == "spam"

    def test_preference_message_type_mapping(self) -> None:
        orch = _orchestrator(
            FakeRules(None), FakePreference(_score(0.9, 0.9)), FakeLLM(None)
        )
        business = orch.route(_msg(conv="business"), _context())
        group = orch.route(_msg(conv="group"), _context())
        assert business.message_type == "business_update"
        assert group.message_type == "event"

    def test_llm_none_returns_digest_fallback(self) -> None:
        orch = _orchestrator(FakeRules(None), FakePreference(_score(0.5, 0.3)), FakeLLM(None))
        decision = orch.route(_msg(), _context())
        assert decision.action == "digest"
        assert decision.message_type == "unknown"
        assert decision.resolved_by == "llm"
        assert decision.confidence == 0.5


class TestEvidenceReconciliation:
    def test_retriever_evidence_attached_on_preference_path(self) -> None:
        orch = _orchestrator(
            FakeRules(None), FakePreference(_score(0.8, 0.9)), FakeLLM(None)
        )
        decision = orch.route(_msg(), _context())
        assert decision.evidence_message_ids == "his_1;his_2"

    def test_llm_ids_within_retriever_use_retriever(self) -> None:
        llm = FakeLLM(_judgment(evidence="his_2"))
        orch = _orchestrator(FakeRules(None), FakePreference(_score(0.5, 0.3)), llm)
        decision = orch.route(_msg(), _context())
        assert decision.evidence_message_ids == "his_1;his_2"

    def test_llm_specific_ids_appended_after_retriever(self) -> None:
        llm = FakeLLM(_judgment(evidence="his_9"))
        orch = _orchestrator(FakeRules(None), FakePreference(_score(0.5, 0.3)), llm)
        decision = orch.route(_msg(), _context())
        assert decision.evidence_message_ids == "his_1;his_2;his_9"

    def test_hallucinated_llm_ids_dropped(self) -> None:
        llm = FakeLLM(_judgment(evidence="his_999;his_1"))
        orch = _orchestrator(FakeRules(None), FakePreference(_score(0.5, 0.3)), llm)
        decision = orch.route(_msg(), _context())
        assert decision.evidence_message_ids == "his_1;his_2"

    def test_no_evidence_renders_none(self) -> None:
        llm = FakeLLM(_judgment(evidence="none"))
        orch = _orchestrator(FakeRules(None), FakePreference(_score(0.5, 0.3)), llm)
        decision = orch.route(_msg(), _context(evidence_refs=[]))
        assert decision.evidence_message_ids == "none"


class TestAudit:
    def test_audit_records_stage_and_personalization(self) -> None:
        audit = DecisionAuditLog()
        orch = _orchestrator(
            FakeRules(None), FakePreference(_score(0.8, 0.9)), FakeLLM(None), audit=audit
        )
        orch.route(_msg(), _context())
        assert len(audit) == 1
        entry = audit.records[0]
        assert entry["message_id"] == "msg_1"
        assert entry["stage"] == "preference_model"
        assert entry["action"] == "notify"
        assert "personalized" in entry

    def test_audit_required_keys_enforced(self) -> None:
        audit = DecisionAuditLog()
        with pytest.raises(ValueError):
            audit.record({"message_id": "m"})

    def test_audit_writes_jsonl(self, tmp_path: Path) -> None:
        path = tmp_path / "audit.jsonl"
        audit = DecisionAuditLog(path)
        audit.record({"message_id": "m1", "stage": "rules", "action": "mute", "message_type": "scam", "confidence": 0.95})
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        assert '"stage": "rules"' in lines[0]
