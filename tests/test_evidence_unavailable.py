"""Rule-mode degradation: missing coordinates or a persistently failing
auxiliary tool is recorded as unavailable evidence instead of deadlocking the
contract, and the report must say so. Free mode is untouched."""
from src.agent.controller import LandslidePolicy, unavailable_placeholder
from src.pipelines.stage5_fusion import _state_unavailable_evidence
from tests.test_agent_loop_guards import _FakeRegistry, _competent_model, _run
from tests.test_tool_precondition_guard import _ctx, _full_outputs
from src.agent.rule_guided_agent import RuleGuidedAgent


def test_fusion_accepts_unavailable_geo_and_lists_it():
    p = LandslidePolicy()
    outputs = _full_outputs()
    outputs["geo.nearby"] = unavailable_placeholder("geo.nearby", "failed 2 times (timeout)")
    outputs["geo.background"] = unavailable_placeholder("geo.background", "no coordinates were supplied with the request")
    args = p.prepare_tool_call("fuse.decision", {}, outputs, _ctx())
    assert {g["tool"] for g in args["unavailable_evidence"]} == {"geo.nearby", "geo.background"}
    assert p.missing_fusion_requirements(args) == []


def test_unavailable_geo_is_not_flagged_for_retry():
    p = LandslidePolicy()
    outputs = _full_outputs()
    outputs["geo.nearby"] = unavailable_placeholder("geo.nearby", "down")
    assert not any("retry geo.nearby" in v for v in p.consistency_violations(outputs, geo_expected=True))


def test_report_states_unavailable_evidence():
    text = "### Conclusion\nx\n### Uncertainty Analysis\nbody\n### Causal Inference\ny"
    out = _state_unavailable_evidence(text, [{"tool": "geo.nearby", "reason": "failed 2 times"}])
    section = out.split("### Uncertainty Analysis\n", 1)[1]
    assert section.startswith("Observed: The following evidence was not available")
    assert "geo.nearby" in section and "body" in section


def _agent(tmp_path, monkeypatch, *, lat, lon, enforce_contract=True, registry=None):
    img = tmp_path / "img.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    monkeypatch.setenv("AGENT_MAX_TURNS", "60")
    agent = RuleGuidedAgent(model_fn=_competent_model, report_write_required=True, image_path=str(img),
                            latitude=lat, longitude=lon, nearby_radius=300, enforce_contract=enforce_contract)
    agent.registry = registry or _FakeRegistry()
    agent.tools = []
    return agent


def _states(events, tool):
    return [e["data"]["execution_state"] for e in events
            if e.get("type") == "tool_result" and e["data"].get("tool") == tool]


def test_rule_without_coordinates_completes(tmp_path, monkeypatch):
    agent = _agent(tmp_path, monkeypatch, lat=None, lon=None)
    events, last = _run(agent)
    assert "declared_unavailable" in _states(events, "geo.nearby")
    assert "declared_unavailable" in _states(events, "geo.background")
    assert last.get("type") != "fallback", last
    assert "geo.nearby" not in agent.registry.calls


class _FailingNearby(_FakeRegistry):
    def call_tool(self, name, args):
        if name == "geo.nearby":
            self.calls.append(name)
            raise RuntimeError("geo.nearby is unavailable: backend service did not respond")
        return super().call_tool(name, args)


def _retrying_model(messages, tools):
    """Retries geo.nearby after a failure, as a real model tends to."""
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    if tool_msgs and tool_msgs[-1].get("name") == "geo.nearby" and "\"error\"" in str(tool_msgs[-1].get("content")) \
            and sum(1 for m in tool_msgs if m.get("name") == "geo.nearby") < 2:
        return {"message": {"role": "assistant", "content": "", "tool_calls": [
            {"id": "retry", "type": "function", "function": {"name": "geo.nearby", "arguments": "{}"}}]}, "raw": ""}
    return _competent_model(messages, tools)


def test_rule_degrades_after_repeated_tool_failure(tmp_path, monkeypatch):
    agent = _agent(tmp_path, monkeypatch, lat=29.6, lon=103.0, registry=_FailingNearby())
    agent.model_fn = _retrying_model
    events, last = _run(agent)
    assert agent.registry.calls.count("geo.nearby") == 2
    assert last.get("type") != "fallback", last


def test_free_mode_is_unchanged_without_coordinates(tmp_path, monkeypatch):
    agent = _agent(tmp_path, monkeypatch, lat=None, lon=None, enforce_contract=False)
    events, _ = _run(agent)
    assert "declared_unavailable" not in _states(events, "geo.nearby")
    assert "geo.nearby" in agent.registry.calls


class _DegradedBackground(_FakeRegistry):
    def call_tool(self, name, args):
        if name == "geo.background":
            self.calls.append(name)
            return {
                "source_status": "degraded",
                "terrain": {"slope_deg": None, "aspect_deg": None},
                "geology": {},
                "warnings": ["DEM service timed out"],
            }
        return super().call_tool(name, args)


def test_degraded_geo_is_retried_then_recorded_unavailable(tmp_path, monkeypatch):
    registry = _DegradedBackground()
    agent = _agent(tmp_path, monkeypatch, lat=29.6, lon=103.0, registry=registry)
    call = {
        "role": "assistant", "content": "",
        "tool_calls": [{"id": "geo", "type": "function", "function": {
            "name": "geo.background", "arguments": "{}",
        }}],
    }
    first = agent._tools_node({"messages": [call], "outputs": {}})
    assert registry.calls.count("geo.background") == 1
    assert _states(first["events"], "geo.background") == ["degraded"]

    second = agent._tools_node({"messages": [call], "outputs": first["outputs"]})
    assert registry.calls.count("geo.background") == 2
    assert _states(second["events"], "geo.background") == ["declared_unavailable"]
    assert second["outputs"]["geo.background"]["evidence_unavailable"] is True
    assert second["events"][-1]["data"]["output"]["evidence_unavailable"] is True

    agent._tools_node({"messages": [call], "outputs": second["outputs"]})
    assert registry.calls.count("geo.background") == 2


def test_degraded_geo_cannot_enter_fusion():
    import pytest
    from src.agent.controller import ToolPreconditionError

    policy = LandslidePolicy()
    outputs = _full_outputs()
    outputs["geo.background"] = {
        "source_status": "degraded",
        "terrain": {"slope_deg": None, "aspect_deg": None},
        "geology": {"source": "unavailable"},
    }
    with pytest.raises(ToolPreconditionError, match="degraded"):
        policy.prepare_tool_call("fuse.decision", {}, outputs, _ctx())
    outputs["geo.background"] = unavailable_placeholder("geo.background", "DEM failed twice")
    policy.prepare_tool_call("fuse.decision", {}, outputs, _ctx())


def test_report_write_warning_waits_for_valid_fusion():
    policy = LandslidePolicy(require_report_write=True)
    outputs = _full_outputs()
    assert not any("report.write" in v for v in policy.verify_analysis(outputs))
    outputs["fuse.decision"] = {"has_landslide": True, "final_description": "confirmed"}
    assert any("report.write" in v for v in policy.verify_analysis(outputs))
