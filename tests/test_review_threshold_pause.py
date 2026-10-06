import json

from src.orchestration.agent import RuleGuidedAgent, _hydrate_outputs_from_messages


def _run_refinement(threshold):
    agent = RuleGuidedAgent.__new__(RuleGuidedAgent)
    agent.lazy_review_threshold = True
    agent.review_threshold = threshold
    agent.loop_hygiene = False
    agent.rule_mode = True
    agent.tool_result_char_limit = 10000

    def execute(name, args, outputs):
        result = {"area_ratio": 0.12, "overlay_path": "refined.png"}
        outputs[name] = result
        return result, args, None

    agent._guarded_execute = execute
    state = {"messages": [{"tool_calls": [{"id": "review-1", "function": {
        "name": "seg.refine", "arguments": "{}",
    }}]}], "outputs": {"seg.run": {"area_ratio": 0.12}}}
    return agent._tools_node(state)


def test_agent_asks_after_refinement_and_preserves_evidence():
    result = _run_refinement(None)
    assert result["paused_reason"] == "review_threshold"
    assert [event["data"]["tool"] for event in result["events"] if event["type"] == "tool_result"] == ["seg.refine"]
    resumed = _hydrate_outputs_from_messages([{
        "role": "tool", "name": "seg.refine", "content": json.dumps(result["outputs"]["seg.refine"]),
    }])
    assert resumed["seg.refine"]["area_ratio"] == 0.12


def test_agent_continues_when_threshold_is_supplied():
    result = _run_refinement(0.20)
    assert "paused_reason" not in result
    assert result["outputs"]["seg.refine"]["area_ratio"] == 0.12
