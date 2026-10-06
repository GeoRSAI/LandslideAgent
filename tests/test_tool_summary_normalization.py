from scripts.llm_service import (
    _select_tool_summary_text,
)


def test_tool_summary_prefers_raw_response_when_available():
    raw = "A short complete summary from the model."
    content = "Different parsed content."
    assert _select_tool_summary_text(raw, content) == "A short complete summary from the model."


def test_tool_summary_falls_back_to_message_content_when_raw_response_is_empty():
    raw = ""
    content = "Parsed message content from the model."
    assert _select_tool_summary_text(raw, content) == "Parsed message content from the model."


def test_tool_summary_preserves_model_output_without_sentence_rewriting():
    raw = "A concise summary without terminal punctuation"
    assert _select_tool_summary_text(raw, "") == "A concise summary without terminal punctuation"
