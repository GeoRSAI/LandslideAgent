"""Stable public boundary for the LangGraph-based free Agent mode."""

from src.orchestration.agent import RuleGuidedAgent

# Agent-specific loop control lives in RuleGuidedAgent; the alias makes the
# orchestration role explicit at service composition sites.
AgentRunner = RuleGuidedAgent

__all__ = ["AgentRunner", "RuleGuidedAgent"]
