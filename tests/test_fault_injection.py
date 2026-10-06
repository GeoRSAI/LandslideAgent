import pytest

from src.agent.protocol import ToolRegistry, ToolSpec


def _registry():
    r = ToolRegistry()
    r.register(ToolSpec(name="geo.nearby", description="", input_schema={}), lambda a: {"ok": True})
    return r


def test_no_injection_by_default(monkeypatch):
    monkeypatch.delenv("AGENT_FAULT_INJECT", raising=False)
    assert _registry().call_tool("geo.nearby", {}) == {"ok": True}


def test_injected_tool_fails(monkeypatch):
    monkeypatch.setenv("AGENT_FAULT_INJECT", "geo.nearby, seg.llm_review")
    with pytest.raises(RuntimeError, match="unavailable"):
        _registry().call_tool("geo.nearby", {})
