"""A failed agent run states its failure without reviving the retired report format."""
from scripts.llm_service import _failed_run_message

REPORT = "### Final Decision Report\n" + "".join(
    f"### Section {i}\nbody {i}\n" for i in range(8)
) + "### Final Determination\nPositive."


def test_failure_does_not_attach_retired_report_format():
    outputs = {"fuse.decision": {"has_landslide": True, "final_description": REPORT}}
    msg = _failed_run_message("final report not written", outputs, include_report=True)
    content = msg["content"]
    assert content.startswith("**Analysis incomplete.**"), content
    assert "final report not written" in content
    assert REPORT not in content


def test_no_assessment_means_only_the_failure_is_reported():
    msg = _failed_run_message("initial cross-check incomplete", {}, include_report=True)
    assert msg["content"] == (
        "**Analysis incomplete.** The agent did not satisfy the task rules: "
        "initial cross-check incomplete."
    )


def test_follow_up_turns_do_not_re_dump_the_report():
    outputs = {"fuse.decision": {"has_landslide": True, "final_description": REPORT}}
    msg = _failed_run_message("x", outputs, include_report=False)
    assert REPORT not in msg["content"]
