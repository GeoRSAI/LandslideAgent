import json
import pytest
from src.models import llm_client

_completion = llm_client._openai_chat_completion


@pytest.mark.parametrize("trace,expected", [
    ("tool.llm.first_pass", "dualhead"),
    ("tool.seg.llm_review", "base"),
    ("tool.vlm.describe", "dualhead"),
    ("report", None),
])
def test_default_adapter_routing(monkeypatch, trace, expected):
    for key in ("LLM_FIRST_PASS_ADAPTER", "LLM_VISUAL_EVIDENCE_ADAPTER", "LLM_DESCRIBE_ADAPTER"):
        monkeypatch.delenv(key, raising=False)
    captured = {}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self):
            return json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()

    def urlopen(req, **kwargs):
        captured.update(json.loads(req.data))
        return Response()

    monkeypatch.setattr(llm_client.request, "urlopen", urlopen)
    _completion([{"role": "user", "content": "test"}], trace_label=trace)
    assert captured.get("adapter_mode") == expected
