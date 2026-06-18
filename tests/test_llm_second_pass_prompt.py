from src.models import llm_client


def test_second_pass_prompt_uses_boxed_whole_image_guidance(monkeypatch):
    captured: dict = {}

    def fake_chat_completion(messages, **kwargs):
        captured["messages"] = messages
        return {
            "content": "Support | boxed regions align with the exposed slope scar in the middle-right hillside.",
        }

    monkeypatch.setattr(llm_client, "_openai_chat_completion", fake_chat_completion)

    result = llm_client.llm_second_pass_on_boxed_image(
        {
            "image_path": "/tmp/original.png",
            "review_image_path": "/tmp/overlay.png",
            "review_mode": "seg_boundary_whole_image",
            "overlay_source": "segmentation_mask_boundary",
            "regions": [{"bbox": [12, 18, 80, 70], "score": 0.91, "class_id": 0}],
        }
    )

    messages = captured["messages"]
    user_text = messages[1]["content"][1]["text"]
    system_text = messages[0]["content"]

    assert "segmentation-mask boundary" in user_text
    assert "spatial guidance, not as cropped patches" in user_text
    assert "where the suspected landslide sits in the whole scene" in user_text
    assert "segmentation-mask boundary" in system_text
    assert result["decision"] == "positive"
    assert result["review_mode"] == "seg_boundary_whole_image"
    assert "score" not in result
