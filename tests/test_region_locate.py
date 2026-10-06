"""The region.locate tool: frame-relative position of the primary candidate.

Descriptive only - it must never influence the landslide decision.
"""
import pytest

from src.utils.geometry import locate_primary_candidate
from src.agent.controller import LandslidePolicy, ToolCallContext, ToolPreconditionError


def test_locate_uses_image_info_frame_and_thirds_grid():
    refinement = {"regions": [{"bbox": [10, 10, 30, 30]}]}   # centre (20,20)
    out = locate_primary_candidate(refinement, {"width": 100, "height": 100})
    assert out["available"] is True
    assert out["position"] == "upper-left"          # 0.2, 0.2 -> both < 1/3
    assert out["rel_center"] == [0.2, 0.2]
    assert out["frame_size"] == [100.0, 100.0]


def test_locate_centre_and_lower_right():
    out = locate_primary_candidate(
        {"regions": [{"bbox": [40, 80, 60, 100]}]}, {"width": 100, "height": 100}
    )
    assert out["position"] == "lower-center"


def test_locate_degrades_when_no_region_or_no_frame():
    assert locate_primary_candidate({"regions": []}, {"width": 100})["available"] is False
    # region present but frame size unknown (no image_info, no candidate_tiles)
    out = locate_primary_candidate({"regions": [{"bbox": [1, 2, 3, 4]}]}, None)
    assert out["available"] is False
    assert "bbox" in out


def _evidence_outputs() -> dict:
    return {
        "tiff.info": {"image_path": "/i.png", "width": 10, "height": 10},
        "llm.first_pass": {"assessment_label": "likely", "has_landslide": True},
        "seg.run": {"area_ratio": 0.4},
        "seg.refine": {"regions": [{"bbox": [0, 0, 1, 1], "score": 0.7}], "area_ratio": 0.4},
        "vlm.describe": {"fields": {"presence": "Yes", "type": "Earthflow"}},
        "cls.run": {"class_name": "earth_flow", "confidence": 0.7},
        "geo.background": {"terrain": {"slope_deg": 1, "aspect_deg": 1}, "geology": {"source": "t"}},
        "geo.nearby": {"count": 0, "features": []},
        "fuse.decision": {"has_landslide": True, "final_description": "x"},
    }


def test_registered_as_a_tool():
    server = __import__(
        "src.agent.default_server", fromlist=["create_default_server"]
    ).create_default_server("configs/thresholds.json")
    names = {t["name"] for t in server.registry.list_tools()}
    assert "region.locate" in names


def test_region_locate_is_a_required_workflow_step():
    """It used to be registered but unreachable: nothing in the required order,
    the per-turn hint or the critic ever named it, so no model called it."""
    p = LandslidePolicy()
    assert "region.locate" in p.evidence_order()
    assert "region.locate" in p.workflow_instruction()

    without = _evidence_outputs()
    assert p.next_required_tool(without) == "region.locate"
    assert "region.locate" in p.next_step_hint(without)
    assert any("region.locate" in v for v in p.verify_analysis(without))

    complete = dict(without, **{"region.locate": {"available": True, "position": "middle-center"}})
    assert p.next_required_tool(complete) is None
    assert p.verify_analysis(complete) == []


def test_region_locate_requirement_can_be_switched_off():
    """The flag is the only thing gating it -- off restores the old contract."""
    p = LandslidePolicy(require_region_locate=False)
    outputs = _evidence_outputs()
    assert "region.locate" not in p.evidence_order()
    assert p.next_required_tool(outputs) is None
    assert p.verify_analysis(outputs) == []

    # per-call override wins over the policy field, in both directions
    on = LandslidePolicy()
    assert on.next_required_tool(outputs, require_region_locate=False) is None
    assert p.next_required_tool(outputs, require_region_locate=True) == "region.locate"


def test_frame_position_section_uses_computed_value_not_llm_prose():
    """The narrative model contradicts the computed position (it has written
    "lower right quadrant" for a middle-center candidate); the mask wins."""
    from src.pipelines.stage5_fusion import REQUIRED_SECTIONS, _normalize_llm_sectioned_report

    llm_text = "\n\n".join(
        f"### {s}\n" + ("Lower right quadrant, toward the bottom-right corner."
                        if s == "Relative Position Within Image Frame" else "Body text.")
        for s in REQUIRED_SECTIONS
    )
    out = _normalize_llm_sectioned_report(
        llm_text,
        overrides={"Relative Position Within Image Frame": "Primary candidate region lies in the middle-center part of the frame."},
    )
    assert "middle-center" in out
    assert "Lower right quadrant" not in out
    # every other section keeps the model's prose
    assert out.count("Body text.") == len(REQUIRED_SECTIONS) - 1

    # no override -> unchanged behaviour
    plain = _normalize_llm_sectioned_report(llm_text)
    assert "Lower right quadrant" in plain


def test_prepare_tool_call_requires_segmentation_derived_regions():
    p = LandslidePolicy()
    ctx = ToolCallContext()
    # nothing segmented yet -> must fail, pointing at seg.run/seg.refine
    with pytest.raises(ToolPreconditionError, match="seg.refine"):
        p.prepare_tool_call("region.locate", {}, {}, ctx)
    # seg.refine available -> dependencies wired in from prior outputs
    outs = {"seg.refine": {"regions": [{"bbox": [1, 1, 2, 2]}]}, "tiff.info": {"width": 4, "height": 4}}
    args = p.prepare_tool_call("region.locate", {}, outs, ctx)
    assert args["refinement"] == outs["seg.refine"]
    assert args["image_info"] == outs["tiff.info"]
