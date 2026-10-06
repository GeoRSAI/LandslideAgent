import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from scripts import llm_service as service
from tests.test_agent_loop_guards import _FakeRegistry, _WORKFLOW


def _install(monkeypatch, tmp_path):
    from PIL import Image

    image = tmp_path / "scene.png"
    Image.new("RGB", (16, 16)).save(image)
    class Registry(_FakeRegistry):
        def call_tool(self, name, args):
            if name == "tiff.info":
                self.calls.append(name)
                return {"image_path": str(image), "width": 16, "height": 16, "description": "x" * 7000}
            return super().call_tool(name, args)
    registry = Registry()
    monkeypatch.setattr("src.orchestration.agent.create_default_server", lambda *_args, **_kw: SimpleNamespace(registry=registry))
    monkeypatch.setattr(service, "_attach_structured_report", lambda response, *_args, **_kw: response)
    monkeypatch.setenv("AGENT_FALLBACK_TO_GRAPH", "0")

    def planning(req):
        done = set()
        for message in req.messages:
            if message.role != "tool":
                continue
            text = str(message.content)
            try:
                output = json.JSONDecoder().raw_decode(text[text.index("{"):])[0]
            except (ValueError, json.JSONDecodeError):
                if "...[truncated" in text and '"error":' not in text:
                    done.add(message.name)
                continue
            if not output.get("error"):
                done.add(message.name)
        name = next((name for name in _WORKFLOW if name not in done), None)
        message = {"role": "assistant", "content": "done"}
        if name:
            message["tool_calls"] = [{"id": name, "type": "function", "function": {"name": name, "arguments": "{}"}}]
        return {"choices": [{"message": message}]}
    monkeypatch.setattr(service, "chat_completions", planning)
    payload = {"messages": [{"role": "user", "content": [{"type": "image", "image_path": str(image)}]}],
               "latitude": 29.6, "longitude": 103, "agent_mode": "agent", "max_turns": 25}
    return TestClient(service.app), registry, payload


def test_sync_agent_exposes_spent_turns_and_preserves_autonomous_plan(monkeypatch, tmp_path):
    client, registry, payload = _install(monkeypatch, tmp_path)
    payload.update(nearby_radius=250, review_threshold=0.2)
    response = client.post("/v1/agent/analyze", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert result["critic"]["degraded"] is False
    assert result["agent_turns_used"] == 11
    assert registry.calls == _WORKFLOW
    assert result["mode"] == "agent"


def test_stream_resumes_both_pauses_from_trace_with_truncated_history(monkeypatch, tmp_path):
    client, registry, payload = _install(monkeypatch, tmp_path)
    trace = []
    for expected_pause, selected_value in [("need_review_threshold", {"review_threshold": 0.2}),
                                           ("need_nearby_radius", {"nearby_radius": 250})]:
        response = client.post("/v1/agent/analyze_stream", json=payload)
        assert response.status_code == 200
        events = [json.loads(line) for line in response.text.splitlines()]
        assert events[-1]["type"] == expected_pause, events[-1]
        payload["agent_turns_used"] = events[-1]["agent_turns_used"]
        for event in events:
            if event["type"] != "tool_result":
                continue
            item = event["data"]
            trace.append(item)
            payload["messages"].append({"role": "tool", "name": item["tool"],
                                        "content": ('{"truncated":' if item["tool"] == "tiff.info"
                                                    else json.dumps(item["output"]))})
        payload["agent_trace"] = trace
        payload.update(selected_value)

    response = client.post("/v1/agent/analyze_stream", json=payload)
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[-1]["type"] == "final"
    result = events[-1]["data"]
    assert result["critic"]["degraded"] is False
    assert result["agent_turns_used"] == 14  # includes model re-issues after UI pauses
    assert registry.calls == _WORKFLOW
    assert result["mode"] == "agent"


def test_stream_model_failure_retains_gathered_artifacts(monkeypatch, tmp_path):
    client, registry, payload = _install(monkeypatch, tmp_path)
    original = service.chat_completions
    count = 0
    def fail_second(req):
        nonlocal count
        count += 1
        if count > 1:
            raise RuntimeError("model offline")
        return original(req)
    monkeypatch.setattr(service, "chat_completions", fail_second)
    response = client.post("/v1/agent/analyze_stream", json=payload)
    events = [json.loads(line) for line in response.text.splitlines()]
    result = events[-1]["data"]
    assert result["critic"]["degraded"] is True
    assert result["agent_turns_used"] == 2
    assert "model offline" in result["choices"][0]["message"]["content"]
    assert result["artifacts"]["original"]
    assert registry.calls == ["tiff.info"]
