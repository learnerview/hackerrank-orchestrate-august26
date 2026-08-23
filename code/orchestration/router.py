"""Decision cascade orchestrator.

Routes one message through, in order:

1. ``rules``        — deterministic safety engine. A hit with confidence >=
                      RULE_MIN_CONFIDENCE (0.70) is decisive and returns
                      immediately. Rules are deterministic, so any hit that
                      fires this high is trusted without further reasoning.
 2. ``preference_model`` — statistical engagement model. If the model's own
                       confidence >= PREF_CONFIDENCE_MIN (0.60) it decides
                       directly, three bands:
                         - priority < model.mute_priority_threshold -> mute
                           (content is low-value/repetitive for THIS user,
                           NOT unsafe — message_type never scam/spam here),
                         - priority >= PREF_NOTIFY_PRIORITY (0.65) -> notify,
                         - otherwise -> digest.
                       The mute band is a learned, documented threshold
                       (scoring.preference_model.DEFAULT_MUTE_PRIORITY when
                       no learned band exists). Scam/spam is exclusively the
                       domain of the rules and the LLM — their message_type
                       vocabulary includes those; the preference path's does
                       not.
 3. ``llm``           — semantic judgment for everything else. A None return
                       (API failure / unresolvable output) falls back to a
                       low-confidence digest.
4. ``personalization`` — post-decision overrides (quiet hours, muted-group
                      mentions). Never touches rule-engine scam/spam mutes.

Evidence: retrieval output (context["evidence_refs"]) is attached to every
decision. On the LLM path the model's own evidence ids are reconciled:
retriever ids are authoritative and appear first; LLM ids that exist in the
known history set and are not already present are appended (the LLM found
something more specific). Hallucinated ids are dropped.

Every decision is recorded in the audit log with its producing stage and
personalization flag — required for SRS metrics.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

from llm.interfaces import RoutingLLM
from models.decision import RoutingDecision
from models.message import IncomingMessage
from orchestration.audit import DecisionAuditLog
from personalization.overrides import PersonalizationOverrides
from safety.rule_engine import SafetyRuleEngine
from scoring.interfaces import PreferenceModel
from scoring.preference_model import DEFAULT_MUTE_PRIORITY, map_priority_to_action

# --- cascade thresholds (documented contract) --------------------------- #
RULE_MIN_CONFIDENCE = 0.70  # deterministic rule hits at/above this are decisive
PREF_CONFIDENCE_MIN = 0.60  # preference model must be this confident to skip the LLM
PREF_NOTIFY_PRIORITY = 0.65  # priority >= this maps to notify (above the mute band)
LLM_FALLBACK_CONFIDENCE = 0.50  # confidence of the digest fallback when LLM fails


def _render_evidence(evidence_refs: list[object]) -> str:
    """Render retriever evidence refs as the semicolon-separated id string."""
    if not evidence_refs:
        return "none"
    return ";".join(str(ref.message_id) for ref in evidence_refs)


def _reconcile_evidence(
    evidence_refs: list[object],
    llm_evidence: str,
    history_ids: set[str] | frozenset[str] | None,
) -> str:
    """Merge LLM evidence ids into the retriever's, retriever first.

    - ids already covered by the retriever are left as-is (retriever is
      authoritative for what it found),
    - LLM ids that exist in ``history_ids`` and are not yet present are
      appended (the LLM found something clearly more specific),
    - ids that cannot be validated against history (hallucinations) are
      dropped; without a history set the LLM ids are trusted as-is.
    """
    base = _render_evidence(evidence_refs)
    if not llm_evidence or llm_evidence == "none":
        return base
    llm_ids = [i.strip() for i in llm_evidence.split(";") if i.strip()]
    if not llm_ids:
        return base
    base_ids = [i.strip() for i in base.split(";") if i.strip()] if base != "none" else []
    if history_ids is not None:
        llm_ids = [i for i in llm_ids if i in set(history_ids)]
        if not llm_ids:
            return base
    merged = list(base_ids)
    for i in llm_ids:
        if i not in merged:
            merged.append(i)
    return ";".join(merged)


def _preference_message_type(message: IncomingMessage) -> str:
    """Deterministic message_type label for the preference-model path."""
    if message.conversation_type == "business":
        return "business_update"
    if message.conversation_type == "group":
        return "event"
    return "personal"


def _topic_hint(message: IncomingMessage, features: dict[str, float]) -> str:
    if features.get("promo_signal", 0.0) > 0.5:
        return "promotional content"
    return {"business_update": "business update", "event": "group event", "personal": "personal update"}.get(
        _preference_message_type(message), _preference_message_type(message)
    )


class RoutingOrchestrator:
    """Ties the decision cascade together: rules -> preference model -> LLM -> personalization.

    Each stage is invoked only when the previous stage is insufficient.
    The orchestrator depends on all layers through their Protocol/interface
    definitions, not concrete implementations, so any component can be swapped
    without touching the decision logic.
    """

    def __init__(
        self,
        rule_engine: SafetyRuleEngine,
        preference_model: PreferenceModel,
        llm: RoutingLLM,
        personalization: PersonalizationOverrides,
        audit_log: Optional[DecisionAuditLog] = None,
    ) -> None:
        self._rule_engine = rule_engine
        self._preference_model = preference_model
        self._llm = llm
        self._personalization = personalization
        self._audit_log = audit_log

    # ------------------------------------------------------------------ #
    # cascade
    # ------------------------------------------------------------------ #
    def route(self, message: IncomingMessage, context: dict[str, object]) -> RoutingDecision:
        evidence_refs = list(context.get("evidence_refs") or [])
        history_ids = context.get("history_ids")
        features = dict(context.get("features") or {})
        source_id = context.get("source_id")
        user = context.get("user")
        member = context.get("member")
        base_evidence = _render_evidence(evidence_refs)

        decision = self._rules_stage(message, base_evidence)
        if decision is None:
            decision = self._preference_stage(message, features, source_id, base_evidence)
        if decision is None:
            decision = self._llm_stage(message, context, evidence_refs, history_ids, base_evidence)

        final = self._personalization.apply(decision, user, member, message)
        self._record(message, decision, final)
        return final

    def _rules_stage(self, message: IncomingMessage, base_evidence: str) -> Optional[RoutingDecision]:
        hit = self._rule_engine.evaluate(message)
        if hit is None or hit.confidence < RULE_MIN_CONFIDENCE:
            return None
        return RoutingDecision(
            message_id=message.message_id,
            action="mute",
            message_type=hit.message_type,
            reason=hit.reason,
            confidence=hit.confidence,
            evidence_message_ids=base_evidence,
            resolved_by="rules",
        )

    def _preference_stage(
        self,
        message: IncomingMessage,
        features: dict[str, float],
        source_id: object,
        base_evidence: str,
    ) -> Optional[RoutingDecision]:
        source = str(source_id) if source_id else "unknown"
        score = self._preference_model.score(message.user_id, source, features)
        if score.confidence < PREF_CONFIDENCE_MIN:
            return None
        mute_priority = float(
            getattr(self._preference_model, "mute_priority_threshold", DEFAULT_MUTE_PRIORITY)
        )
        action = map_priority_to_action(score.priority, mute_priority, PREF_NOTIFY_PRIORITY)
        if action == "mute":
            engagement = "very low (low-value/repetitive for this user; not unsafe)"
        elif action == "notify":
            engagement = "high"
        else:
            engagement = "moderate (above the mute band)"
        return RoutingDecision(
            message_id=message.message_id,
            action=action,
            message_type=_preference_message_type(message),
            reason=(
                f"user's engagement with this sender is {engagement}: priority "
                f"{score.priority:.2f} at model confidence {score.confidence:.2f}"
                f" — {_topic_hint(message, features)}"
            ),
            confidence=score.confidence,
            evidence_message_ids=base_evidence,
            resolved_by="preference_model",
        )

    def _llm_stage(
        self,
        message: IncomingMessage,
        context: dict[str, object],
        evidence_refs: list[object],
        history_ids: object,
        base_evidence: str,
    ) -> RoutingDecision:
        judgment = self._llm.decide(message, context)
        if judgment is None:
            return RoutingDecision(
                message_id=message.message_id,
                action="digest",
                message_type="unknown",
                reason="LLM could not resolve this message; defaulted to digest",
                confidence=LLM_FALLBACK_CONFIDENCE,
                evidence_message_ids=base_evidence,
                resolved_by="llm",
            )
        decision = judgment.as_routing_decision(message.message_id)
        reconciled = _reconcile_evidence(
            evidence_refs,
            decision.evidence_message_ids,
            history_ids if history_ids is not None else None,
        )
        if reconciled != decision.evidence_message_ids:
            decision = replace(decision, evidence_message_ids=reconciled)
        return decision

    # ------------------------------------------------------------------ #
    # audit
    # ------------------------------------------------------------------ #
    def _record(self, message: IncomingMessage, base: RoutingDecision, final: RoutingDecision) -> None:
        if self._audit_log is None:
            return
        self._audit_log.record(
            {
                "message_id": message.message_id,
                "stage": base.resolved_by,
                "personalized": final.personalized,
                "action": final.action,
                "message_type": final.message_type,
                "confidence": round(final.confidence, 3),
                "evidence_message_ids": final.evidence_message_ids,
                "reason": final.reason,
            }
        )
