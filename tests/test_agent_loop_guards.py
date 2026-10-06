"""Loop-guard behaviour of RuleGuidedAgent under the precondition contract.

The contract layer may refuse a requested call and may report unmet deliverable
obligations, but it never selects, reorders or executes a tool on the model's
behalf. These tests pin that boundary.
"""
import json

from scripts.llm_service import _compress_followup_report_messages
from src.agent.rule_guided_agent import RuleGuidedAgent, _truncate_for_model


def test_truncate_for_model_caps_length_and_marks_it():
    assert _truncate_for_model("abc", 100) == "abc"
    out = _truncate_for_model("x" * 5000, 500)
    assert out.startswith("x" * 500)
    assert "truncated 4500 chars" in out
    assert _truncate_for_model("x" * 5000, 0) == "x" * 5000  # 0 disables the cap


class _FakeRegistry:
    """Stand-in for the JSON-RPC tool registry with canned, valid tool outputs."""

    def __init__(self):
        self.calls: list[str] = []

    def list_tools(self):
        return []

    def call_tool(self, name, args):
        self.calls.append(name)
        canned = {
            "tiff.info": {"image_path": "/img.png", "width": 512, "height": 512},
            "llm.first_pass": {"has_landslide": True, "score": 0.8, "assessment_label": "likely"},
            "seg.run": {"area_ratio": 0.4, "landslide_pixels": 100, "polygon_count": 1},
            "seg.refine": {"regions": [{"bbox": [0, 0, 1, 1], "score": 0.7}], "area_ratio": 0.4},
            "vlm.describe": {"fields": {"presence": "Yes", "type": "Earthflow", "morphology": "lobate deposit"}, "raw_text": "Landslide presence: Yes"},
            "cls.run": {"class_name": "earth_flow", "confidence": 0.7, "topk": []},
            "geo.background": {
                "terrain": {"slope_deg": 20.0, "aspect_deg": 180.0},
                "geology": {"lithology": "sandstone"},
            },
            "geo.nearby": {"count": 1, "features": [{"id": "way/1", "type": "road"}]},
            "fuse.decision": {"has_landslide": True, "final_description": "Earth flow confirmed."},
            "report.write": {"report_path": "outputs/reports/x.json"},
        }
        return dict(canned.get(name, {"ok": True}))


def _make_agent(model_fn, monkeypatch, tmp_path, *, flags=None, **env):
    img = tmp_path / "img.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    for k, v in {"AGENT_MAX_TURNS": "60", **env}.items():
        monkeypatch.setenv(k, v)
    agent = RuleGuidedAgent(
        model_fn=model_fn,
        report_write_required=True,
        image_path=str(img),
        latitude=29.6,
        longitude=103.0,
        nearby_radius=300,
        **(flags or {}),
    )
    agent.registry = _FakeRegistry()
    agent.tools = []
    return agent


# The agent no longer tells the model what to call next, so a fake model that
# is meant to finish the analysis has to carry its own plan.
_WORKFLOW = ["tiff.info", "llm.first_pass", "seg.run", "seg.refine", "region.locate", "vlm.describe",
             "seg.llm_review", "cls.run", "geo.background", "geo.nearby",
             "fuse.decision", "report.write"]


def _competent_model(_messages, _tools):
    """A cooperative model that drives the analysis from its own domain plan."""
    done = {m.get("name") for m in _messages if m.get("role") == "tool"}
    for name in _WORKFLOW:
        if name not in done:
            return {"message": {"role": "assistant", "content": "",
                                "tool_calls": [{"id": name, "type": "function",
                                                "function": {"name": name, "arguments": "{}"}}]},
                    "raw": ""}
    return {"message": {"role": "assistant", "content": "done"}, "raw": ""}


def _run(agent):
    events = list(agent.stream([{"role": "user", "content": "analyze"}]))
    return events, events[-1]


def test_repeated_closing_line_is_not_re_rendered_as_a_chat_bubble(monkeypatch, tmp_path):
    """A stalled model re-emits its closing line every critic round-trip.

    The frontend renders one bubble per ``assistant`` event, so the same
    "analysis completed / report saved" sentence used to pile up in the chat.
    Show it once; route the verbatim repeats to the token stream.
    """
    line = "Landslide analysis completed. Report saved to outputs/reports/x.json."

    def repeating_model(_messages, _tools):
        return {"message": {"role": "assistant", "content": line}, "raw": ""}

    agent = _make_agent(repeating_model, monkeypatch, tmp_path)
    events, _last = _run(agent)

    bubbles = [
        e for e in events
        if e.get("type") == "assistant"
        and " ".join(str(e.get("content") or "").split()) == line
        and not e.get("tool_calls")
    ]
    assert len(bubbles) == 1, f"closing line rendered {len(bubbles)} times"

    repeats = [e for e in events if e.get("source") == "agent-repeat"]
    assert repeats, "repeats should still be visible in the token stream"
    assert all(e.get("type") == "model_raw" for e in repeats)


def test_distinct_narrations_are_all_shown(monkeypatch, tmp_path):
    """Only verbatim repeats are suppressed -- genuine narration still renders."""
    state = {"i": 0}

    def varying_model(_messages, _tools):
        state["i"] += 1
        return {"message": {"role": "assistant", "content": f"step {state['i']}"}, "raw": ""}

    agent = _make_agent(varying_model, monkeypatch, tmp_path)
    events, _last = _run(agent)

    texts = [
        " ".join(str(e.get("content") or "").split())
        for e in events
        if e.get("type") == "assistant" and not e.get("tool_calls") and str(e.get("content") or "").strip()
    ]
    assert len(texts) == len(set(texts))
    assert len(texts) > 1
    assert not [e for e in events if e.get("source") == "agent-repeat"]


def test_stalled_model_is_never_rescued_by_deterministic_execution(monkeypatch, tmp_path):
    """A stalled model is reported on, not substituted for.

    The critic states the unmet deliverable obligations and stops there. Nothing
    is executed on the model's behalf, so a model that never emits a usable tool
    call spends its turn budget and hands off.
    """

    def stalling_model(_messages, _tools):
        return {"message": {"role": "assistant", "content": ""}, "raw": "<tool_call>\n</tool_call>"}

    agent = _make_agent(stalling_model, monkeypatch, tmp_path, AGENT_MAX_TURNS="6")
    events, last = _run(agent)

    assert last["type"] == "fallback", last
    assert "no progress after 4 identical contract checks" in last["reason"]
    assert agent.registry.calls == [], agent.registry.calls
    assert not [e for e in events if "auto-advance" in str(e.get("content", ""))]
    assert any("contract check" in str(e.get("content", "")) for e in events)


def _scripted_model(script):
    """A model that emits the given tool calls in order, then answers."""
    state = {"i": 0}

    def model(_messages, _tools):
        i = state["i"]
        state["i"] += 1
        if i >= len(script):
            return {"message": {"role": "assistant", "content": "done"}, "raw": ""}
        call = script[i]
        return {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": f"c{i}", "type": "function",
                    "function": {"name": call["name"],
                                 "arguments": json.dumps(call.get("arguments", {}))},
                }],
            },
            "raw": "",
        }

    return model


def test_premature_fuse_decision_is_refused_by_its_precondition(monkeypatch, tmp_path):
    """fuse.decision called before its evidence exists is refused, not repaired.

    The refusal reaches the model as an ordinary tool observation. The contract
    layer does not reorder the plan and does not run the missing tools itself,
    so the underlying tool is never reached.
    """
    agent = _make_agent(_scripted_model([{"name": "fuse.decision"}]),
                        monkeypatch, tmp_path, AGENT_MAX_TURNS="4")
    events, _last = _run(agent)

    results = [e["data"] for e in events if e.get("type") == "tool_result"]
    assert results, "fuse.decision produced no observation"
    assert results[0]["tool"] == "fuse.decision"
    assert results[0]["status"] == "error"
    assert "precondition" in json.dumps(results[0]["output"]).lower() or \
           "prerequisite" in json.dumps(results[0]["output"]).lower()
    assert agent.registry.calls == [], agent.registry.calls


def test_region_locate_without_refinement_is_refused_not_backfilled(monkeypatch, tmp_path):
    """region.locate no longer triggers a controller-side seg.refine run."""
    agent = _make_agent(_scripted_model([{"name": "region.locate"}]),
                        monkeypatch, tmp_path, AGENT_MAX_TURNS="4")
    events, _last = _run(agent)

    results = [e["data"] for e in events if e.get("type") == "tool_result"]
    assert results and results[0]["tool"] == "region.locate"
    assert results[0]["status"] == "error"
    assert "seg.refine" not in agent.registry.calls, agent.registry.calls


def test_model_chosen_order_is_admitted_when_preconditions_hold(monkeypatch, tmp_path):
    """No ordering is imposed beyond the declared evidence dependencies.

    geo.background before llm.first_pass is not a prescribed order, but both
    preconditions hold, so both calls execute as the model requested them.
    """
    agent = _make_agent(
        _scripted_model([
            {"name": "tiff.info"},
            {"name": "geo.background"},
            {"name": "llm.first_pass"},
        ]),
        monkeypatch, tmp_path,
    )
    _events, _last = _run(agent)

    assert agent.registry.calls[:3] == ["tiff.info", "geo.background", "llm.first_pass"], \
        agent.registry.calls


def test_repeated_idempotent_call_served_from_cache(monkeypatch, tmp_path):
    """Model loops on seg.run after it already succeeded -> second call is cached."""
    script = [
        {"name": "tiff.info", "arguments": {}},
        {"name": "seg.run", "arguments": {}},
        {"name": "seg.run", "arguments": {}},   # repeat -> should hit cache
    ]
    state = {"i": 0}

    def scripted_model(_messages, _tools):
        i = state["i"]
        state["i"] += 1
        if i >= len(script):
            return {"message": {"role": "assistant", "content": "done"}, "raw": ""}
        call = script[i]
        return {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": f"c{i}", "type": "function",
                    "function": {"name": call["name"],
                                 "arguments": json.dumps(call["arguments"])},
                }],
            },
            "raw": "",
        }

    agent = _make_agent(scripted_model, monkeypatch, tmp_path)
    list(agent.stream([{"role": "user", "content": "analyze"}]))

    assert agent.registry.calls.count("seg.run") == 1, agent.registry.calls


def test_contract_off_lets_the_model_terminate_on_its_own(monkeypatch, tmp_path):
    """Without the deliverable contract a narration-only turn ends the run."""

    def one_shot_model(_messages, _tools):
        return {"message": {"role": "assistant", "content": "No landslide."}, "raw": ""}

    agent = _make_agent(one_shot_model, monkeypatch, tmp_path,
                        flags={"enforce_contract": False})
    _events, last = _run(agent)

    assert last["type"] == "final_core", last
    assert last["data"]["message"]["content"] == "No landslide."
    assert agent.registry.calls == [], agent.registry.calls


def test_contract_off_passes_model_arguments_through_unchanged(monkeypatch, tmp_path):
    """No precondition and no binding: the tool sees exactly what the model sent."""
    seen = {}

    class RecordingRegistry(_FakeRegistry):
        def call_tool(self, name, args):
            seen[name] = dict(args)
            return super().call_tool(name, args)

    agent = _make_agent(
        _scripted_model([{"name": "seg.run", "arguments": {"image_path": "/x.png"}}]),
        monkeypatch, tmp_path, flags={"enforce_contract": False},
    )
    agent.registry = RecordingRegistry()
    _run(agent)

    assert seen.get("seg.run") == {"image_path": "/x.png"}, seen


def test_loop_hygiene_is_independent_of_the_contract(monkeypatch, tmp_path):
    """Idempotent reuse is a resource control that can be switched off on its own.

    Both arms must run with the same setting, otherwise a difference in tool
    executions would reflect caching rather than rule enforcement.
    """
    script = [{"name": "tiff.info"}, {"name": "seg.run"}, {"name": "seg.run"}]

    hygienic = _make_agent(_scripted_model(script), monkeypatch, tmp_path,
                           AGENT_MAX_TURNS="6")
    _run(hygienic)
    assert hygienic.registry.calls.count("seg.run") == 1, hygienic.registry.calls

    raw = _make_agent(_scripted_model(script), monkeypatch, tmp_path,
                      flags={"loop_hygiene": False}, AGENT_MAX_TURNS="6")
    _run(raw)
    assert raw.registry.calls.count("seg.run") == 2, raw.registry.calls


def test_turn_budget_bounds_the_contract_free_arm_too(monkeypatch, tmp_path):
    """'Failure' means the same thing in both arms: the same turn budget."""

    def looping_model(_messages, _tools):
        return {"message": {"role": "assistant", "content": "",
                            "tool_calls": [{"id": "t", "type": "function",
                                            "function": {"name": "tiff.info",
                                                         "arguments": "{}"}}]},
                "raw": ""}

    agent = _make_agent(looping_model, monkeypatch, tmp_path,
                        flags={"enforce_contract": False}, AGENT_MAX_TURNS="5")
    events, last = _run(agent)

    # stopped on the turn budget, not on the graph's recursion guard
    assert last["type"] == "final_core", last
    assert any("turn budget (5) reached" in str(e.get("content", "")) for e in events), events
    assert not [e for e in events if e.get("type") == "fallback"]


def test_contradicting_argument_reaches_the_model_and_the_tool_does_not_run(
    monkeypatch, tmp_path
):
    """A supplied coordinate that contradicts the request is refused, not replaced."""
    agent = _make_agent(
        _scripted_model([{"name": "geo.background", "arguments": {"lat": 2.96, "lon": 103.0}}]),
        monkeypatch, tmp_path, AGENT_MAX_TURNS="4",
    )
    events, _last = _run(agent)

    results = [e["data"] for e in events if e.get("type") == "tool_result"]
    assert results and results[0]["tool"] == "geo.background"
    assert results[0]["status"] == "error"
    assert results[0]["execution_state"] == "refused", results[0]
    assert results[0]["verification"]["status"] == "refused", results[0]
    assert "lat" in json.dumps(results[0]["output"]), results[0]
    assert "geo.background" not in agent.registry.calls, agent.registry.calls


def test_admitted_call_reports_its_verification_to_the_model(monkeypatch, tmp_path):
    """The observation says what was checked and what was resolved, and from where."""
    seen_by_model = []
    scripted = _scripted_model([{"name": "tiff.info"},
                                {"name": "geo.background", "arguments": {"lat": 29.6}}])

    def recording_model(messages, tools):
        seen_by_model[:] = list(messages)
        return scripted(messages, tools)

    agent = _make_agent(recording_model, monkeypatch, tmp_path, AGENT_MAX_TURNS="4")
    events, last = _run(agent)

    geo = [e["data"] for e in events
           if e.get("type") == "tool_result" and e["data"]["tool"] == "geo.background"][0]
    assert geo["execution_state"] == "completed", geo
    assert geo["verification"] == {
        "status": "admitted",
        "checked": ["lat"],
        "resolved": {"lon": "request: scene longitude"},
    }, geo

    # The model receives the same record at the head of the observation ...
    observation = [m for m in seen_by_model
                   if m.get("role") == "tool" and m.get("name") == "geo.background"]
    assert observation, seen_by_model
    assert json.loads(observation[0]["content"])["verification"]["status"] == "admitted"

    # ... while the evidence ledger keeps the bare tool result.
    ledger = last.get("outputs") or (last.get("data") or {}).get("outputs") or {}
    assert ledger.get("geo.background"), ledger
    assert "verification" not in ledger["geo.background"], ledger


def test_contract_free_arm_observations_carry_no_verification(monkeypatch, tmp_path):
    agent = _make_agent(
        _scripted_model([{"name": "geo.background", "arguments": {"lat": 1.0, "lon": 2.0}}]),
        monkeypatch, tmp_path, flags={"enforce_contract": False},
    )
    events, _last = _run(agent)
    geo = [e["data"] for e in events if e.get("type") == "tool_result"][0]
    assert "verification" not in geo, geo
    assert agent.registry.calls == ["geo.background"], agent.registry.calls


def test_big_tool_result_is_truncated_in_the_history(monkeypatch, tmp_path):
    class BigRegistry(_FakeRegistry):
        def call_tool(self, name, args):
            result = super().call_tool(name, args)
            if name == "tiff.info":
                result = {**result, "blob": "x" * 40000}
            return result

    agent = _make_agent(_competent_model, monkeypatch, tmp_path, AGENT_TOOL_RESULT_CHARS="500")
    agent.registry = BigRegistry()
    _, last = _run(agent)
    tool_msgs = [m for m in last["data"]["history"] if m.get("role") == "tool"]
    assert tool_msgs, "no tool message in history"
    assert all(len(m["content"]) < 2000 for m in tool_msgs)
    assert any("truncated" in m["content"] for m in tool_msgs)


def test_followup_legacy_report_is_omitted_from_model_context():
    report = "### Final Decision Report\n" + ("evidence " * 400) + "\n### Conclusion\nLikely landslide.\n### Final Determination\nPositive."
    messages = [
        {"role": "user", "content": "analyze image"},
        {"role": "assistant", "content": report},
        {"role": "user", "content": "why?"},
    ]
    compressed = _compress_followup_report_messages(messages)
    assert compressed[1]["content"] == "[Previous report omitted from model context]"


class _FailedReportRegistry(_FakeRegistry):
    def call_tool(self, name, args):
        if name == "report.write":
            self.calls.append(name)
            return {"error": "disk full"}
        return super().call_tool(name, args)


def test_completed_fusion_checks_report_persistence(monkeypatch, tmp_path):
    agent = _make_agent(_competent_model, monkeypatch, tmp_path)
    agent.registry = _FailedReportRegistry()
    _events, last = _run(agent)
    assert last["type"] == "fallback", last
    assert "report.write failed: disk full" in last["reason"]
    assert agent.registry.calls.count("report.write") == 1
