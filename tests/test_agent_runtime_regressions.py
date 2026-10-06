import json

import pytest

from scripts.llm_service import ChatMessage, ChatRequest, _build_agent_runner, _inject_forced_system_messages
from tests.test_agent_loop_guards import _FakeRegistry, _make_agent


def _call(name, args="{}", call_id="c1"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": args}}


def _step(agent, calls, outputs=None):
    return agent._tools_node({"messages": [{"role": "assistant", "content": "", "tool_calls": calls}],
                              "outputs": outputs or {}})


def _traces(result):
    return [e["data"] for e in result["events"] if e["type"] == "tool_result"]


def test_cached_result_cannot_bypass_image_precondition(monkeypatch, tmp_path):
    agent = _make_agent(lambda *_: {}, monkeypatch, tmp_path)
    first = _step(agent, [_call("seg.run")])
    second = _step(agent, [_call("seg.run", json.dumps({"image_path": "/other.png"}))], first["outputs"])
    assert _traces(second)[0]["execution_state"] == "refused"
    assert agent.registry.calls == ["seg.run"]
    assert second["outputs"]["seg.run"] == first["outputs"]["seg.run"]


def test_cache_respects_changed_model_arguments_in_free_mode(monkeypatch, tmp_path):
    agent = _make_agent(lambda *_: {}, monkeypatch, tmp_path, flags={"enforce_contract": False})
    first = _step(agent, [_call("seg.run", '{"image_path":"/a.png"}')])
    second = _step(agent, [_call("seg.run", '{"image_path":"/b.png"}')], first["outputs"])
    assert agent.registry.calls == ["seg.run", "seg.run"]
    assert _traces(second)[0]["cached"] is False


@pytest.mark.parametrize("args", ['{"image_path":', '[]', 'null', '3', '"x"'])
def test_invalid_arguments_are_observations_and_never_execute(monkeypatch, tmp_path, args):
    agent = _make_agent(lambda *_: {}, monkeypatch, tmp_path)
    result = _step(agent, [_call("seg.run", args)])
    assert not agent.registry.calls
    assert _traces(result)[0]["status"] == "error"
    assert "arguments" in _traces(result)[0]["output"]["error"]
    assert result["messages"][0]["tool_call_id"] == "c1"


def test_duplicate_failure_keeps_failure_status(monkeypatch, tmp_path):
    agent = _make_agent(lambda *_: {}, monkeypatch, tmp_path)
    result = _step(agent, [_call("region.locate", call_id="a"), _call("region.locate", call_id="b")])
    assert [item["status"] for item in _traces(result)] == ["error", "error"]
    assert [item["tool_call_id"] for item in result["messages"]] == ["a", "b"]
    assert not agent.registry.calls


def test_pause_resolves_every_tool_call_without_running_remaining_plan(monkeypatch, tmp_path):
    agent = _make_agent(lambda *_: {}, monkeypatch, tmp_path)
    agent.nearby_radius = None
    result = _step(agent, [_call("geo.nearby", call_id="a"), _call("seg.run", call_id="b")])
    assert result["paused_reason"] == "nearby_radius"
    assert [item["tool_call_id"] for item in result["messages"]] == ["a", "b"]
    assert not agent.registry.calls
    assert all(item["status"] != "ok" for item in _traces(result))


def test_backend_declaration_is_not_a_cache_hit(monkeypatch, tmp_path):
    class BrokenRegistry(_FakeRegistry):
        def call_tool(self, name, args):
            self.calls.append(name)
            raise RuntimeError("backend offline")

    agent = _make_agent(lambda *_: {}, monkeypatch, tmp_path)
    agent.registry = BrokenRegistry()
    first = _step(agent, [_call("geo.nearby")])
    second = _step(agent, [_call("geo.nearby")], first["outputs"])
    trace = _traces(second)[0]
    assert trace["execution_state"] == "declared_unavailable"
    assert trace["cached"] is False


def test_model_failure_returns_fallback_with_evidence(monkeypatch, tmp_path):
    def broken_model(*_):
        raise RuntimeError("model service offline")

    agent = _make_agent(broken_model, monkeypatch, tmp_path)
    evidence = {"image_path": agent.image_path, "width": 512, "height": 512}
    events = list(agent.stream([{"role": "tool", "name": "tiff.info", "content": json.dumps(evidence)}]))
    assert events[-1]["type"] == "fallback"
    assert "model service offline" in events[-1]["reason"]
    assert events[-1]["outputs"]["tiff.info"] == evidence


def test_resume_uses_separate_trace_when_model_message_is_truncated(monkeypatch, tmp_path):
    img = tmp_path / "img.png"
    img.write_bytes(b"image")
    evidence = {"image_path": str(img), "width": 512, "height": 512, "description": "x" * 7000}
    req = ChatRequest(messages=[ChatMessage(role="user", content=[{"type": "image", "image_path": str(img)}])],
                      agent_mode="agent", agent_trace=[{"tool": "tiff.info", "status": "ok",
                      "execution_state": "completed", "input": {"image_path": str(img)}, "output": evidence}])
    agent = _build_agent_runner(req=req, thresholds_path="configs/thresholds.json", image_path=str(img),
                                report_write_required=False)
    agent.model_fn = lambda *_: {"message": {"role": "assistant", "content": "done"}}
    agent.max_identical_critic_checks = 1
    events = list(agent.stream([{"role": "tool", "name": "tiff.info", "content": '{"image_path": "broken'}]))
    assert events[-1]["outputs"]["tiff.info"] == evidence


def test_unknown_radius_is_not_fabricated_as_immutable_input():
    req = ChatRequest(messages=[ChatMessage(role="user", content=[{"type": "image", "image_path": "/img.png"}])],
                      agent_mode="agent")
    messages = _inject_forced_system_messages([{"role": "user", "content": "analyze"}], req)
    assert "nearby_radius_m=" not in messages[0]["content"]


def test_changed_evidence_invalidates_cached_refinement(monkeypatch, tmp_path):
    agent = _make_agent(lambda *_: {}, monkeypatch, tmp_path)
    outputs = {"tiff.info": {"image_path": agent.image_path, "width": 512, "height": 512},
               "seg.run": {"area_ratio": 0.4, "mask_path": "first.png"}}
    first = _step(agent, [_call("seg.refine")], outputs)
    first["outputs"]["seg.run"] = {"area_ratio": 0.1, "mask_path": "second.png"}
    second = _step(agent, [_call("seg.refine")], first["outputs"])
    assert agent.registry.calls == ["seg.refine", "seg.refine"]
    assert _traces(second)[0]["input"]["segmentation"]["mask_path"] == "second.png"


def test_cached_coordinate_conflict_is_refused(monkeypatch, tmp_path):
    agent = _make_agent(lambda *_: {}, monkeypatch, tmp_path)
    first = _step(agent, [_call("geo.background")])
    second = _step(agent, [_call("geo.background", '{"lat":35,"lon":110}')], first["outputs"])
    assert _traces(second)[0]["execution_state"] == "refused"
    assert agent.registry.calls == ["geo.background"]


def test_review_pause_resolves_unexecuted_calls(monkeypatch, tmp_path):
    agent = _make_agent(lambda *_: {}, monkeypatch, tmp_path)
    agent.lazy_review_threshold = True
    agent.review_threshold = None
    outputs = {"tiff.info": {"image_path": agent.image_path, "width": 512, "height": 512},
               "seg.run": {"area_ratio": 0.12}}
    result = _step(agent, [_call("seg.refine", call_id="a"), _call("cls.run", call_id="b")], outputs)
    assert result["paused_reason"] == "review_threshold"
    assert [m["tool_call_id"] for m in result["messages"]] == ["a", "b"]
    assert agent.registry.calls == ["seg.refine"]
    assert "cls.run" not in result["outputs"]


def test_fusion_cutoff_resolves_remaining_calls(monkeypatch, tmp_path):
    from tests.test_tool_precondition_guard import _full_outputs

    agent = _make_agent(lambda *_: {}, monkeypatch, tmp_path)
    outputs = _full_outputs()
    outputs["tiff.info"]["image_path"] = agent.image_path
    result = _step(agent, [_call("fuse.decision", call_id="a"), _call("image.tile", call_id="b")], outputs)
    assert agent.registry.calls == ["fuse.decision"]
    assert [m["tool_call_id"] for m in result["messages"]] == ["a", "b"]
    assert _traces(result)[-1]["execution_state"] == "deferred"


def test_critic_progress_is_measured_by_evidence_not_only_obligation_text(monkeypatch, tmp_path):
    agent = _make_agent(lambda *_: {}, monkeypatch, tmp_path)
    first = agent._critic_node({"outputs": {}, "turns": 1})
    outputs = {"seg.refine": {"regions": [], "area_ratio": 0.4}}
    second = agent._critic_node({**first, "outputs": outputs, "turns": 2})
    assert second["stalled_critic_checks"] == 1
    assert second["critic_signature"] != first["critic_signature"]


def test_resume_carries_backend_failure_budget(monkeypatch, tmp_path):
    img = tmp_path / "img.png"
    img.write_bytes(b"image")
    req = ChatRequest(messages=[], agent_mode="agent", latitude=29.6, longitude=103,
                      nearby_radius=300, agent_turns_used=5,
                      agent_trace=[{"tool": "geo.nearby", "status": "error", "execution_state": "failed",
                                    "input": {"lat": 29.6, "lon": 103, "radius": 300}, "output": {"error": "offline"}}])
    agent = _build_agent_runner(req=req, thresholds_path="configs/thresholds.json", image_path=str(img),
                                report_write_required=False)
    class BrokenRegistry(_FakeRegistry):
        def call_tool(self, name, args):
            self.calls.append(name)
            raise RuntimeError("offline")
    agent.registry = BrokenRegistry()
    agent.model_fn = lambda *_: {"message": {"role": "assistant", "tool_calls": [_call("geo.nearby")]}}
    agent.max_turns = 6
    events = list(agent.stream([{"role": "user", "content": "resume"}]))
    assert any(e["data"]["execution_state"] == "declared_unavailable"
               for e in events if e["type"] == "tool_result")
    assert events[-1]["turns_used"] == 6
    assert agent.registry.calls == ["geo.nearby"]


def test_model_visible_verification_is_not_hydrated_as_evidence(monkeypatch, tmp_path):
    from src.orchestration.agent import _hydrate_outputs_from_messages

    payload = {"verification": {"status": "admitted", "checked": ["image_path"]},
               "area_ratio": 0.4, "mask_path": "mask.png"}
    outputs = _hydrate_outputs_from_messages([{"role": "tool", "name": "seg.run", "content": json.dumps(payload)}])
    assert outputs["seg.run"] == {"area_ratio": 0.4, "mask_path": "mask.png"}


@pytest.mark.parametrize("budget", [0, -1])
def test_nonpositive_turn_budget_is_rejected(budget):
    with pytest.raises(ValueError):
        ChatRequest(messages=[], max_turns=budget)


def test_reexecuted_segmentation_invalidates_downstream_evidence(monkeypatch, tmp_path):
    class ChangingRegistry(_FakeRegistry):
        def call_tool(self, name, args):
            self.calls.append(name)
            return {"area_ratio": 0.05, "mask_path": "new-mask.png"}

    agent = _make_agent(lambda *_: {}, monkeypatch, tmp_path, flags={"loop_hygiene": False})
    agent.registry = ChangingRegistry()
    outputs = {"seg.run": {"area_ratio": 0.4, "mask_path": "old-mask.png"},
               "seg.refine": {"regions": [{"bbox": [0, 0, 10, 10]}]},
               "region.locate": {"available": True}, "seg.llm_review": {"llm_second_pass": {"decision": "positive"}},
               "fuse.decision": {"has_landslide": True}, "report.write": {"report_path": "old.json"},
               "cls.run": {"class_name": "earth_flow"}}
    result = _step(agent, [_call("seg.run")], outputs)
    assert set(result["outputs"]) == {"seg.run", "cls.run"}
    invalidated = _traces(result)[0]["verification"]["invalidated_evidence"]
    assert set(invalidated) == {"seg.refine", "region.locate", "seg.llm_review", "fuse.decision", "report.write"}
    assert agent.registry.calls == ["seg.run"]


def test_failed_backend_observation_does_not_resurrect_old_success():
    from src.orchestration.agent import _hydrate_outputs_from_messages

    outputs = _hydrate_outputs_from_messages([
        {"role": "tool", "name": "seg.run", "content": json.dumps({"area_ratio": 0.4})},
        {"role": "tool", "name": "seg.refine", "content": json.dumps({"regions": []})},
        {"role": "tool", "name": "seg.run", "content": json.dumps({"verification": {"status": "admitted"}, "error": "offline"})},
    ])
    assert outputs == {}


def test_refused_call_does_not_erase_prior_success():
    from src.orchestration.agent import _hydrate_outputs_from_messages

    outputs = _hydrate_outputs_from_messages([
        {"role": "tool", "name": "seg.run", "content": json.dumps({"area_ratio": 0.4})},
        {"role": "tool", "name": "seg.run", "content": json.dumps({"verification": {"status": "refused"}, "error": "wrong image"})},
    ])
    assert outputs == {"seg.run": {"area_ratio": 0.4}}


def test_same_batch_retry_is_admitted_after_prerequisite_arrives(monkeypatch, tmp_path):
    agent = _make_agent(lambda *_: {}, monkeypatch, tmp_path)
    outputs = {"tiff.info": {"image_path": agent.image_path, "width": 512, "height": 512}}
    result = _step(agent, [_call("region.locate", call_id="a"), _call("seg.run", call_id="b"),
                           _call("seg.refine", call_id="c"), _call("region.locate", call_id="d")], outputs)
    assert [t["status"] for t in _traces(result)] == ["error", "ok", "ok", "ok"]
    assert agent.registry.calls == ["seg.run", "seg.refine", "region.locate"]


def test_same_batch_repeat_observes_changed_upstream_evidence(monkeypatch, tmp_path):
    agent = _make_agent(lambda *_: {}, monkeypatch, tmp_path)
    outputs = {"tiff.info": {"image_path": agent.image_path, "width": 512, "height": 512},
               "seg.run": {"area_ratio": 0.1}}
    result = _step(agent, [_call("seg.refine", call_id="a"),
                           _call("seg.run", json.dumps({"image_path": agent.image_path}), call_id="b"),
                           _call("seg.refine", call_id="c")], outputs)
    assert agent.registry.calls == ["seg.refine", "seg.run", "seg.refine"]
    assert all(t["status"] == "ok" for t in _traces(result))


def test_resume_does_not_count_reused_failure_as_another_backend_execution(tmp_path):
    img = tmp_path / "img.png"
    img.write_bytes(b"image")
    failed = {"tool": "geo.nearby", "status": "error", "execution_state": "failed",
              "input": {"lat": 29.6, "lon": 103, "radius": 300}, "output": {"error": "offline"}}
    req = ChatRequest(messages=[], agent_mode="agent", latitude=29.6, longitude=103, nearby_radius=300,
                      max_turns=1, agent_trace=[{**failed, "cached": False}, {**failed, "cached": True}])
    agent = _build_agent_runner(req=req, thresholds_path="configs/thresholds.json", image_path=str(img),
                                report_write_required=False)
    class BrokenRegistry(_FakeRegistry):
        def call_tool(self, name, args):
            self.calls.append(name)
            raise RuntimeError("offline")
    agent.registry = BrokenRegistry()
    agent.model_fn = lambda *_: {"message": {"role": "assistant", "tool_calls": [_call("geo.nearby")]}}
    events = list(agent.stream([{"role": "user", "content": "resume"}]))
    evidence = events[-1]["outputs"]["geo.nearby"]
    assert "failed or degraded 2 times" in evidence["reason"]
    assert agent.registry.calls == ["geo.nearby"]
