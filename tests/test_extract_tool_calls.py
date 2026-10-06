import json

from scripts.llm_service import _extract_tool_calls, _filter_tool_calls_to_available


def _names(calls):
    return [c["function"]["name"] for c in calls]


def _args(call):
    return json.loads(call["function"]["arguments"])


def test_json_style_tool_call_block():
    text = '<tool_call>\n{"name": "tiff.info", "arguments": {"image_path": "/a/b.png"}}\n</tool_call>'
    content, calls = _extract_tool_calls(text)
    assert _names(calls) == ["tiff.info"]
    assert _args(calls[0]) == {"image_path": "/a/b.png"}
    # no scaffolding left behind (this was the empty-<tool_call> feedback bug)
    assert "<tool_call>" not in content and "</tool_call>" not in content


def test_xml_function_style():
    text = (
        "<tool_call><function=geo.nearby>"
        "<parameter=lat>29.6</parameter><parameter=lon>103.0</parameter>"
        "</function></tool_call>"
    )
    content, calls = _extract_tool_calls(text)
    assert _names(calls) == ["geo.nearby"]
    assert _args(calls[0]) == {"lat": 29.6, "lon": 103.0}


def test_multiple_tool_calls_in_one_completion():
    text = (
        '<tool_call>{"name": "llm.first_pass", "arguments": {}}</tool_call>\n'
        '<tool_call>{"name": "seg.run", "arguments": {}}</tool_call>'
    )
    _, calls = _extract_tool_calls(text)
    assert _names(calls) == ["llm.first_pass", "seg.run"]


def test_duplicate_calls_are_deduped():
    text = (
        '<tool_call>{"name": "seg.run", "arguments": {}}</tool_call>\n'
        '<tool_call>{"name": "seg.run", "arguments": {}}</tool_call>'
    )
    _, calls = _extract_tool_calls(text)
    assert _names(calls) == ["seg.run"]


def test_empty_tool_call_block_is_ignored():
    content, calls = _extract_tool_calls("<tool_call>\n</tool_call>")
    assert calls == []


def test_arguments_as_json_string():
    text = '<tool_call>{"name": "cls.run", "arguments": "{\\"image_info\\": {\\"w\\": 1}}"}</tool_call>'
    _, calls = _extract_tool_calls(text)
    assert _names(calls) == ["cls.run"]
    assert _args(calls[0]) == {"image_info": {"w": 1}}


def test_fenced_json_call():
    text = 'here you go\n```json\n{"name": "tiff.info", "arguments": {"image_path": "/x.tif"}}\n```'
    _, calls = _extract_tool_calls(text)
    assert _names(calls) == ["tiff.info"]


def test_bare_json_call_without_tags():
    text = 'I will call {"name": "seg.run", "arguments": {"image_info": {}}} now.'
    _, calls = _extract_tool_calls(text)
    assert _names(calls) == ["seg.run"]


def test_truncated_tool_call_recovered():
    text = '<tool_call>\n{"name": "tiff.info", "arguments": {"image_path": "/root/data/'
    _, calls = _extract_tool_calls(text)
    assert _names(calls) == ["tiff.info"]
    assert _args(calls[0]).get("image_path", "").startswith("/root/data/")


def test_plain_text_answer_has_no_tool_calls():
    content, calls = _extract_tool_calls("The image shows an earthflow on a forested slope.")
    assert calls == []
    assert "earthflow" in content


def test_hidden_or_unexposed_tool_calls_are_filtered():
    _, calls = _extract_tool_calls('<tool_call>{"name": "fuse.decision", "arguments": {}}</tool_call>')
    assert _filter_tool_calls_to_available(calls, []) == []
    assert _filter_tool_calls_to_available(calls, [{"type": "function", "function": {"name": "tiff.info"}}]) == []
    kept = _filter_tool_calls_to_available(calls, [{"type": "function", "function": {"name": "fuse.decision"}}])
    assert _names(kept) == ["fuse.decision"]
