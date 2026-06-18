import json

from scripts import llm_service


def _collect_stream_events(response) -> list[dict]:
    return [
        json.loads(chunk)
        for chunk in response.body_iterator
    ]


def test_should_require_report_write_only_for_image_turns(monkeypatch):
    monkeypatch.setenv("AGENT_ENABLE_REPORT_WRITE", "1")

    with_image = [
        llm_service.ChatMessage(
            role="user",
            content=[
                {"type": "image", "image_path": "/tmp/scene.png"},
                {"type": "text", "text": "Analyze this image."},
            ],
        )
    ]
    followup_only = [
        llm_service.ChatMessage(
            role="user",
            content=[{"type": "text", "text": "Why did you make that conclusion?"}],
        )
    ]

    assert llm_service._should_require_report_write(with_image) is True
    assert llm_service._should_require_report_write(followup_only) is False


def test_followup_stream_does_not_fail_mandatory_report_write(monkeypatch):
    monkeypatch.setenv("AGENT_ENABLE_REPORT_WRITE", "1")

    def fake_chat_completions(req):
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "The conclusion came from the prior screening and segmentation evidence.",
                    }
                }
            ],
            "raw_response": "The conclusion came from the prior screening and segmentation evidence.",
        }

    monkeypatch.setattr(llm_service, "chat_completions", fake_chat_completions)

    req = llm_service.ChatRequest(
        messages=[
            llm_service.ChatMessage(
                role="user",
                content=[{"type": "text", "text": "Why did you make that conclusion?"}],
            )
        ]
    )

    response = llm_service.agent_analyze_stream(req)
    events = _collect_stream_events(response)

    assert all(event.get("type") != "error" for event in events)
    final_events = [event for event in events if event.get("type") == "final"]
    assert len(final_events) == 1
    final_message = final_events[0]["data"]["choices"][0]["message"]
    assert "prior screening and segmentation evidence" in final_message["content"]
