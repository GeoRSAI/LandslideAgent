from src.pipelines.structured_report import (
    PARTIAL, UNAVAILABLE, UNVERIFIED, VERIFIED, build_structured_report, render_structured_report,
)

FP = {"has_landslide": True, "score": 0.85, "assessment_label": "likely", "scene_description": "A scarp above a fan."}
SEG = {"area_ratio": 0.12, "landslide_pixels": 3000, "polygon_count": 1, "mask_path": "m.png"}
REFINE = {"regions": [{"bbox": [0, 0, 100, 100]}], "area_ratio": 0.12}
LOC = {"available": True, "position": "lower-center", "rel_center": [0.5, 0.8], "bbox": [150, 350, 350, 512],
       "position_source": "mask_centroid"}
DESC = {"fields": {
    "presence": "Yes", "type": "Debris flow", "position": "The affected area is located in the bottom center.",
    "morphology": "elongated, tongue-shaped deposit with lateral levees",
    "material": "coarse sediment and boulders", "movement": "rapid fluid-like transport",
    "environment": "densely forested steep terrain", "impact": "No impact on human facilities was observed.",
    "reason": "channelized flow deposit", "causation": "likely intense rainfall",
}}
CLS = {"class_name": "Debris flow", "confidence": 0.6, "resolution": "subclass_agreement",
       "sources": {"vlm": {"class_name": "Debris flow"}, "image_classifier": {"class_name": "Debris flow"}}}
BG = {"terrain": {"elevation_m": 900, "slope_deg": 30.0, "aspect_deg": 90.0}, "geology": {"lithology": "shale"}}
NB = {"radius_m": 300, "count": 0, "features": [], "source_status": "ok"}
REVIEW = {"llm_second_pass": {"decision": "positive", "evidence": "boundary fits scarp"}}


def call(tool, output, state="completed", inp=None):
    return {"tool": tool, "execution_state": state, "input": inp or {}, "output": output}


def fuse(**inp):
    return call("fuse.decision", {"has_landslide": True, "landslide_type": "Debris flow",
                                  "classification_reconciliation": {"status": "agreement"}}, inp=inp)


def full_trace(desc=DESC, nearby=NB, **fuse_overrides):
    inp = dict(stage1=FP, segmentation=SEG, refinement=REFINE, classification=CLS,
               geo_context={"background": BG, "nearby": nearby}, llm_second_pass=REVIEW["llm_second_pass"])
    inp.update(fuse_overrides)
    return [call("tiff.info", {"width": 512, "height": 512}), call("llm.first_pass", FP), call("seg.run", SEG),
            call("seg.refine", REFINE), call("region.locate", LOC), call("vlm.describe", desc),
            call("seg.llm_review", REVIEW), call("cls.run", CLS), call("geo.background", BG),
            call("geo.nearby", nearby), fuse(**inp)]


def status(r, key):
    return r["fields"][key]["status"]


def test_complete_run_merges_image_and_tool_evidence():
    r = build_structured_report(full_trace(), latitude=30.0, longitude=100.0)
    assert all(f["status"] == VERIFIED for f in r["fields"].values())
    env = r["fields"]["environment"]["parts"]
    assert env[0]["kind"] == "image" and env[0]["text"] == DESC["fields"]["environment"]  # verbatim
    assert env[1]["kind"] == "tool" and "slope 30.0" in env[1]["text"] and env[1]["source"] == "geo.background#9"
    assert r["flags"] == []
    text = render_structured_report(r)
    assert "**Morphological characteristics:** elongated" in text and "vlm.describe#6" in text
    assert "The affected area is located in the bottom center." in text


def test_misplaced_argument_is_flagged():
    r = build_structured_report(full_trace(stage1=SEG))
    part = r["fields"]["presence"]["parts"][0]
    assert status(r, "presence") == UNVERIFIED and "seg.run#3" in part["note"] and "放错" in part["note"]


def test_fabricated_classification_is_flagged():
    r = build_structured_report(full_trace(classification={**CLS, "class_name": "Rock slide", "confidence": 0.99}))
    assert status(r, "type") == UNVERIFIED


def test_missing_description_leaves_image_fields_unavailable():
    trace = [c for c in full_trace() if c["tool"] != "vlm.describe"]
    r = build_structured_report(trace)
    assert status(r, "morphology") == UNAVAILABLE
    assert "未调用 vlm.describe" in r["fields"]["morphology"]["parts"][0]["note"]
    assert status(r, "environment") == PARTIAL  # terrain still measured


def test_missing_coordinates_and_failed_tool_keep_image_part():
    placeholder = {"evidence_unavailable": True, "reason": "no coordinates were supplied with the request"}
    trace = full_trace()[:8] + [
        call("geo.background", placeholder, state="declared_unavailable"),
        call("geo.nearby", {"error": "backend did not respond"}, state="failed"),
        call("geo.nearby", {"error": "backend did not respond"}, state="failed"),
        fuse(stage1=FP, segmentation=SEG, classification=CLS),
    ]
    r = build_structured_report(trace)
    assert status(r, "environment") == PARTIAL and "未提供经纬度" in r["fields"]["environment"]["parts"][1]["note"]
    assert status(r, "impact") == PARTIAL and "失败 2 次" in r["fields"]["impact"]["parts"][1]["note"]


def test_disagreements_between_image_and_tools_are_flagged():
    desc = {"fields": {**DESC["fields"], "type": "Rock fall", "position": "top left"}}
    nearby = {"radius_m": 300, "count": 3, "features": [{"type": "road"}] * 3, "source_status": "ok"}
    r = build_structured_report(full_trace(desc=desc, nearby=nearby))
    flags = " ".join(r["flags"])
    assert "Rock fall" in flags and "运动方式大类也不同" in flags
    assert "位置" in flags
    assert "3 处已登记设施" in flags


def test_negative_decision_marks_type_and_position_not_applicable():
    trace = full_trace()
    trace[-1]["output"] = {"has_landslide": False, "landslide_type": "unknown"}
    r = build_structured_report(trace)
    assert r["fields"]["type"]["parts"][0]["text"].startswith("Not applicable")
    assert r["fields"]["position"]["parts"][0]["text"].startswith("Not applicable")


def test_no_fusion_means_no_conclusion():
    r = build_structured_report([call("tiff.info", {"error": "file not found"}, state="failed")])
    assert status(r, "presence") == UNAVAILABLE and r["counts"][VERIFIED] == 0


def test_boundary_review_not_required_for_large_confident_target():
    big = {**SEG, "area_ratio": 0.4}
    r = build_structured_report([call("llm.first_pass", FP), call("seg.run", big), fuse(stage1=FP, segmentation=big)])
    assert r["boundary_review"]["text"].startswith("not required")


def test_parse_fine_tuned_description():
    from src.models.llm_client import parse_scene_description
    text = ("Landslide presence: Yes\n\nLandslide type:\nDebris flow\n\nImage relative position within the image frame:\n"
            "The affected area is located in the bottom center.\n\nMorphological characteristics:\nelongated deposit\n"
            "with lateral levees\n\nLandslide causation inference:\nlikely rainfall")
    f = parse_scene_description(text)
    assert f["presence"] == "Yes" and f["type"] == "Debris flow"
    assert f["morphology"] == "elongated deposit with lateral levees"
    assert f["causation"] == "likely rainfall"
