from scripts.llm_service import _deterministic_tool_summary


def test_tool_summary_uses_deterministic_complete_sentence():
    summary = _deterministic_tool_summary(
        "seg.run",
        "ok",
        {"area_ratio": 0.0163, "landslide_pixels": 4281},
    )

    assert summary == "Segmentation finished: landslide area ratio 1.63%, pixels=4281."


def test_tool_summary_keeps_classifier_confidence_only_for_classification():
    summary = _deterministic_tool_summary(
        "cls.run",
        "ok",
        {"class_name": "Earthflow", "confidence": 0.889},
    )

    assert summary == "Classification finished: Earthflow (0.89)."


def test_tool_summary_reports_failure_as_complete_sentence():
    summary = _deterministic_tool_summary(
        "seg.run",
        "error",
        {"error": "model unavailable"},
    )

    assert summary == "seg.run failed: model unavailable."
