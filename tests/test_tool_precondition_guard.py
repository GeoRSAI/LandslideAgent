"""The single hard-precondition layer: LandslidePolicy.prepare_tool_call.

These tests pin the behaviour that the free-agent runtime
(`RuleGuidedAgent._guarded_execute`) and the FastAPI service
(`create_default_server`'s guarded executor) both now delegate to, so the two
entry points cannot drift apart again.
"""
import pytest

from src.agent.controller import (
    LandslidePolicy,
    ToolCallContext,
    ToolPreconditionError,
)

CANNED = {
    "tiff.info": {"image_path": "/img.png", "width": 512, "height": 512},
    "llm.first_pass": {"has_landslide": True, "score": 0.8, "assessment_label": "likely"},
    "seg.run": {"area_ratio": 0.4, "landslide_pixels": 100, "polygon_count": 1},
    "seg.refine": {"regions": [{"bbox": [0, 0, 1, 1], "score": 0.7}], "area_ratio": 0.4},
    "vlm.describe": {"fields": {"presence": "Yes", "type": "Earthflow", "morphology": "lobate deposit"}, "raw_text": "Landslide presence: Yes"},
    "cls.run": {"class_name": "earth_flow", "confidence": 0.7, "topk": []},
    "geo.background": {
        "terrain": {"slope_deg": 20.0, "aspect_deg": 180.0},
        "geology": {"lithology": "sandstone"},
    },
    "geo.nearby": {"count": 1, "features": [{"id": "way/1", "type": "road"}]},
    "region.locate": {
        "primary_region": {"bbox": [0, 0, 1, 1], "score": 0.7},
        "frame_position": "center",
        "grid_position": {"row": 1, "col": 1, "label": "center"},
    },
    "seg.llm_review": {"llm_second_pass": {"decision": "confirmed", "support": 0.9}},
}


def _ctx(**kw):
    def run_tool(name, args):
        return dict(CANNED.get(name, {"ok": True}))

    kw.setdefault("run_tool", run_tool)
    return ToolCallContext(**kw)


def _full_outputs(**overrides):
    out = {k: dict(v) for k, v in CANNED.items() if k != "seg.llm_review"}
    out.update(overrides)
    return out


def test_initial_cross_check_is_required_but_order_is_flexible():
    p = LandslidePolicy()
    # Whole-scene tools can run before metadata; the final contract still
    # requires tiff.info as part of the initial cross-check.
    seg_args = p.prepare_tool_call("seg.run", {}, {}, _ctx())
    first_args = p.prepare_tool_call("llm.first_pass", {}, {}, _ctx())
    assert seg_args["image_info"] == {}
    assert first_args["image_info"] == {}
    assert any("tiff.info" in item for item in p.verify_analysis({
        "seg.run": CANNED["seg.run"], "llm.first_pass": CANNED["llm.first_pass"],
    }))
    args = p.prepare_tool_call("seg.run", {}, {"tiff.info": CANNED["tiff.info"]}, _ctx())
    assert args["image_info"] == CANNED["tiff.info"]


def test_geo_tools_need_coordinates_and_take_them_from_context():
    p = LandslidePolicy()
    with pytest.raises(ToolPreconditionError):
        p.prepare_tool_call("geo.nearby", {}, {}, _ctx())
    args = p.prepare_tool_call("geo.nearby", {}, {}, _ctx(latitude=29.6, longitude=103.0))
    assert args["lat"] == 29.6 and args["lon"] == 103.0 and args["radius"] == 300


def test_fuse_decision_blocks_until_all_evidence_present():
    p = LandslidePolicy()
    partial = {"llm.first_pass": CANNED["llm.first_pass"], "seg.run": CANNED["seg.run"]}
    with pytest.raises(ToolPreconditionError):
        p.prepare_tool_call("fuse.decision", {}, partial, _ctx())

    outputs = _full_outputs()
    args = p.prepare_tool_call("fuse.decision", {}, outputs, _ctx())
    assert args["classification"]["class_name"] == "earth_flow"
    assert args["geo_context"]["background"] == CANNED["geo.background"]
    assert p.missing_fusion_requirements(args) == []


def test_tiny_area_forces_seg_llm_review_before_fusion():
    p = LandslidePolicy()
    outputs = _full_outputs(
        seg={"area_ratio": 0.02},
        **{"seg.run": {"area_ratio": 0.02}, "seg.refine": {"regions": [], "area_ratio": 0.02}},
    )
    with pytest.raises(ToolPreconditionError, match="seg.llm_review"):
        p.prepare_tool_call("fuse.decision", {}, outputs, _ctx())

    outputs["seg.llm_review"] = CANNED["seg.llm_review"]
    args = p.prepare_tool_call("fuse.decision", {}, outputs, _ctx())
    assert args["llm_second_pass"] == CANNED["seg.llm_review"]["llm_second_pass"]


def test_missing_seg_refine_is_refused_not_silently_computed():
    """The contract layer never executes a tool the agent did not request.

    A missing dependency is reported as an unmet precondition; it is not
    manufactured behind the agent's back, so the evidence ledger stays an
    accurate record of what the agent actually chose to do.
    """
    p = LandslidePolicy()
    calls = []

    def run_tool(name, args):
        calls.append(name)
        return dict(CANNED.get(name, {"ok": True}))

    outputs = {
        "tiff.info": CANNED["tiff.info"],
        "llm.first_pass": CANNED["llm.first_pass"],
        "seg.run": CANNED["seg.run"],
    }
    ctx = ToolCallContext(run_tool=run_tool)
    with pytest.raises(ToolPreconditionError):
        p.prepare_tool_call("seg.llm_review", {}, outputs, ctx)
    assert "seg.refine" not in outputs
    assert calls == []


def test_request_parameters_are_checked_not_overwritten():
    """A supplied coordinate or radius must match the request; it is never replaced."""
    p = LandslidePolicy()
    ctx = _ctx(latitude=29.6, longitude=103.0, nearby_radius=300)

    with pytest.raises(ToolPreconditionError) as err:
        p.prepare_tool_call("geo.background", {"lat": 92.6, "lon": 1030.0}, {}, ctx)
    assert "lat" in str(err.value) and "29.6" in str(err.value), err.value

    with pytest.raises(ToolPreconditionError):   # swapped lat/lon
        p.prepare_tool_call("geo.background", {"lat": 103.0, "lon": 29.6}, {}, ctx)

    with pytest.raises(ToolPreconditionError) as err:
        p.prepare_tool_call("geo.nearby", {"lat": 29.6, "lon": 103.0, "radius": 1000}, {}, ctx)
    assert "radius" in str(err.value), err.value

    # A faithful, rounded reference is admitted and bound to the exact request value.
    args = p.prepare_tool_call(
        "geo.background", {"lat": 29.64, "lon": 103.0}, {},
        _ctx(latitude=29.6412, longitude=103.0),
    )
    assert args["lat"] == 29.6412 and args["lon"] == 103.0, args


def test_contradicting_evidence_argument_is_refused_not_replaced():
    p = LandslidePolicy()
    with pytest.raises(ToolPreconditionError) as err:
        p.prepare_tool_call(
            "fuse.decision",
            {"classification": {"class_name": "rock_fall"}},
            _full_outputs(),
            _ctx(),
        )
    text = str(err.value)
    assert "classification.class_name" in text, text
    assert "rock_fall" in text and "earth_flow" in text, text


def test_consistent_abbreviated_evidence_is_admitted_and_bound_to_the_record():
    p = LandslidePolicy()
    args = p.prepare_tool_call(
        "fuse.decision",
        {
            "classification": {"class_name": "earth_flow"},
            "geo_context": {"background": {"terrain": {"slope_deg": 20}}},
        },
        _full_outputs(),
        _ctx(),
    )
    assert args["classification"] == CANNED["cls.run"]
    assert args["geo_context"]["background"] == CANNED["geo.background"]
    assert args["geo_context"]["nearby"] == CANNED["geo.nearby"]


def test_invented_field_counts_as_a_conflict():
    p = LandslidePolicy()
    with pytest.raises(ToolPreconditionError) as err:
        p.prepare_tool_call(
            "fuse.decision",
            {"classification": {"class_name": "earth_flow", "risk": "high"}},
            _full_outputs(),
            _ctx(),
        )
    assert "classification.risk" in str(err.value), err.value


def test_evidence_argument_without_a_recorded_source_is_refused():
    p = LandslidePolicy()
    outputs = {k: CANNED[k] for k in ("tiff.info", "seg.run", "seg.refine")}
    with pytest.raises(ToolPreconditionError) as err:
        p.prepare_tool_call(
            "seg.llm_review",
            {"stage1": {"assessment_label": "likely"}},   # llm.first_pass never ran
            outputs,
            _ctx(),
        )
    assert "llm.first_pass" in str(err.value), err.value


def test_image_reference_must_name_the_image_under_analysis(tmp_path):
    image = tmp_path / "scene.png"
    image.write_bytes(b"x")
    other = tmp_path / "other.png"
    other.write_bytes(b"x")
    p = LandslidePolicy()

    with pytest.raises(ToolPreconditionError):
        p.prepare_tool_call("tiff.info", {"image_path": str(other)}, {}, _ctx(image_path=str(image)))

    args = p.prepare_tool_call("tiff.info", {}, {}, _ctx(image_path=str(image)))
    assert args["image_path"] == str(image)


def test_verification_record_lists_checked_and_resolved_arguments():
    p = LandslidePolicy()
    record = {}
    p.prepare_tool_call(
        "geo.nearby", {"lat": 29.6}, {},
        _ctx(latitude=29.6, longitude=103.0, nearby_radius=300),
        record=record,
    )
    assert record["checked"] == ["lat"], record
    assert record["resolved"]["lon"].startswith("request"), record
    assert record["resolved"]["radius"].startswith("request"), record

    record = {}
    p.prepare_tool_call(
        "fuse.decision", {"classification": {"class_name": "earth_flow"}},
        _full_outputs(), _ctx(), record=record,
    )
    assert "classification" in record["checked"], record
    assert record["resolved"]["refinement"] == "evidence ledger: seg.refine", record
    assert "classification" not in record["resolved"], record


def test_evidence_from_the_wrong_tool_is_diagnosed_as_such():
    """seg.run output passed as `stage1` is named as a source mix-up."""
    p = LandslidePolicy()
    with pytest.raises(ToolPreconditionError) as err:
        p.prepare_tool_call(
            "seg.llm_review", {"stage1": dict(CANNED["seg.run"])}, _full_outputs(), _ctx()
        )
    text = str(err.value)
    assert "matches the recorded seg.run result" in text, text
    assert "refers to the llm.first_pass result" in text, text


def test_unknown_fields_are_named_with_the_reference_fields():
    p = LandslidePolicy()
    with pytest.raises(ToolPreconditionError) as err:
        p.prepare_tool_call(
            "seg.llm_review", {"stage1": {"bogus": 1}}, _full_outputs(), _ctx()
        )
    text = str(err.value)
    assert "`stage1.bogus` is not a field of the recorded llm.first_pass result" in text, text
    assert "`assessment_label`" in text, text


def test_long_values_keep_their_distinguishing_tail():
    from src.agent.controller import _short

    shown = _short("outputs/masks/debris flow102656_Level_16_1789548087164497340_mask.png")
    assert shown.endswith('_mask.png"'), shown
    assert len(shown) <= 60, shown


def test_composed_report_is_refused_with_an_explicit_reason():
    p = LandslidePolicy()
    outputs = {"fuse.decision": {"has_landslide": True, "final_description": "x"}}
    with pytest.raises(ToolPreconditionError) as err:
        p.prepare_tool_call(
            "report.write", {"report": {"summary": "my own words"}}, outputs,
            _ctx(image_path="/img.png"),
        )
    assert "cannot be written" in str(err.value), err.value
    assert "omitted `report`" in str(err.value), err.value


def test_rounding_rule_for_numeric_evidence():
    from src.agent.controller import _evidence_conflicts

    assert _evidence_conflicts(42.7, 42.71, "x") == []          # valid rounding
    assert _evidence_conflicts(43, 42.71, "x") == []            # valid rounding
    assert _evidence_conflicts(42.8, 42.71, "x") != []          # not a rounding
    assert _evidence_conflicts(300, 300, "r") == []
    assert _evidence_conflicts(301, 300, "r") != []             # integers are exact
    assert _evidence_conflicts("earthflow", "earth_flow", "c") != []


def test_model_supplied_coordinates_are_range_checked_without_a_request_georeference():
    """With no georeference on record the model may supply one, but not a bogus one."""
    p = LandslidePolicy()
    ok = p.prepare_tool_call("geo.background", {"lat": 29.6, "lon": 103.0}, {}, _ctx())
    assert ok["lat"] == 29.6 and ok["lon"] == 103.0

    with pytest.raises(ToolPreconditionError):
        p.prepare_tool_call("geo.background", {"lat": 999.0, "lon": 103.0}, {}, _ctx())

    with pytest.raises(ToolPreconditionError):
        p.prepare_tool_call("geo.background", {}, {}, _ctx())


def test_tool_layer_reports_absent_inputs_in_actionable_terms():
    """Both arms get a legible error, not a bare KeyError.

    The unconstrained arm has only error feedback to recover from, so an
    uninformative message would penalise it for reasons unrelated to rule
    enforcement.
    """
    from src.agent.default_server import create_default_server

    registry = create_default_server("configs/thresholds.json").registry

    with pytest.raises(Exception) as fuse_error:
        registry.call_tool("fuse.decision", {})
    text = str(fuse_error.value)
    assert "stage1" in text and "refinement" in text, text
    assert text != "'stage1'", text

    with pytest.raises(Exception) as report_error:
        registry.call_tool("report.write", {"report": {"a": 1}})
    assert "out_path" in str(report_error.value), report_error.value


def test_report_write_is_single_shot():
    p = LandslidePolicy()
    outputs = {"fuse.decision": {"has_landslide": True, "final_description": "x"}}
    args = p.prepare_tool_call("report.write", {}, outputs, _ctx(image_path="/img.png"))
    assert args["report"] == outputs["fuse.decision"]
    assert args["out_path"].endswith(".json")
    with pytest.raises(ToolPreconditionError):
        p.prepare_tool_call("report.write", {}, outputs, _ctx(report_written=True))
