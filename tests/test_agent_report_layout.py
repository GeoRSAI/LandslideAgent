from scripts.llm_service import ChatRequest, _attach_structured_report, _graph_report_trace, _prefer_structured_final_message
from src.pipelines.report_writer import compose_report
from src.pipelines.structured_report import FIELD_TITLES


def _ten_fields():
    return "\n".join(f"**{title}:** value {index}" for index, title in enumerate(FIELD_TITLES.values(), 1))


def test_composed_report_separates_the_batch_fields():
    result = compose_report({"fields": {}}, lambda *_args, **_kwargs: {"content": _ten_fields()})
    assert result.startswith("### Landslide Assessment Report\n\n")
    assert result.count("\n\n**") == 10
    for title in FIELD_TITLES.values():
        assert result.count(f"**{title}:**") == 1


def test_agent_final_uses_same_ten_fields_as_batch_report(monkeypatch):
    monkeypatch.setenv("FUSE_LENIENT", "0")
    monkeypatch.setenv("REPORT_COMPOSER", "llm")
    expected = "### Landslide Assessment Report\n\n" + _ten_fields().replace("\n", "\n\n")
    monkeypatch.setattr("src.pipelines.report_writer.compose_report", lambda *_args, **_kwargs: expected)
    sections = "\n\n".join(f"### Section {index}\nField {index}: value" for index in range(1, 7))
    fusion = {"final_description": sections, "has_landslide": True}
    message = _prefer_structured_final_message(
        {"role": "assistant", "content": "Report saved."}, {"fuse.decision": fusion}
    )
    response = {"choices": [{"index": 0, "message": message, "finish_reason": "stop"}], "mode": "agent"}
    trace = [{"tool": "fuse.decision", "execution_state": "completed", "input": {}, "output": fusion}]
    result = _attach_structured_report(response, trace, ChatRequest(messages=[]), enabled=True)
    assert result["choices"][0]["message"]["content"] == expected
    assert result["report_composer"] == "llm"
    assert "fields" in result["structured_report"]


def test_workflow_final_uses_same_ten_fields(monkeypatch):
    monkeypatch.setenv("FUSE_LENIENT", "0")
    monkeypatch.setenv("REPORT_COMPOSER", "llm")
    expected = "### Landslide Assessment Report\n\n" + _ten_fields().replace("\n", "\n\n")
    monkeypatch.setattr("src.pipelines.report_writer.compose_report", lambda *_args, **_kwargs: expected)
    state = {
        "stage1": {"has_landslide": True},
        "segmentation": {"area_ratio": 0.2},
        "geo_context": {"background": {"terrain": {}}, "nearby": {"count": 0}},
        "final_report": {"final_description": "### Final Decision Report\nLegacy workflow report"},
    }
    trace = _graph_report_trace(state)
    assert [item["tool"] for item in trace][-3:] == ["geo.background", "geo.nearby", "fuse.decision"]
    response = {"choices": [{"message": {"role": "assistant", "content": state["final_report"]["final_description"]}}]}
    result = _attach_structured_report(response, trace, ChatRequest(messages=[]), enabled=True)
    assert result["choices"][0]["message"]["content"] == expected
    assert result["report_composer"] == "llm"
