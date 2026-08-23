"""safety: deterministic pre-model rules that can force mute (scam/spam)."""

from .rule_engine import SafetyRuleEngine, RuleContext, RULES

__all__ = ["SafetyRuleEngine", "RuleContext", "RULES"]
