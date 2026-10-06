"""Cross-module consistency gate in LandslidePolicy.verify_analysis.

This is contribution 3 ("confidence-triggered conditional re-perception")
generalised past the tiny-area rule: when first-pass screening, segmentation,
and fusion disagree, a VLM boundary re-check (seg.llm_review) becomes a hard
requirement before the analysis may finalise. Toggle via AGENT_CONSISTENCY_GATE.
"""
from src.agent.controller import LandslidePolicy

BASE = {
    "tiff.info": {"image_path": "/img.png", "width": 512, "height": 512},
    "llm.first_pass": {"has_landslide": False, "assessment_label": "unlikely", "score": 0.2},
    "seg.run": {"area_ratio": 0.30},
    "seg.refine": {"regions": [{"bbox": [0, 0, 1, 1], "score": 0.7}], "area_ratio": 0.30},
    "cls.run": {"class_name": "earth_flow", "confidence": 0.7, "topk": []},
    "geo.background": {"terrain": {"slope_deg": 20.0, "aspect_deg": 180.0}, "geology": {"lithology": "x"}},
    "geo.nearby": {"count": 0, "features": []},
    "fuse.decision": {"has_landslide": True, "final_description": "confirmed"},
}


def _gate_on():
    return LandslidePolicy(enforce_consistency_gate=True)


def test_screening_vs_segmentation_disagreement_requires_review():
    p = _gate_on()
    out = {k: dict(v) for k, v in BASE.items() if k != "fuse.decision"}
    v = p.verify_analysis(out)
    assert any("disagreement" in x for x in v), v
    assert p.consistency_needs_second_pass(out) is True

    out["seg.llm_review"] = {"llm_second_pass": {"decision": "confirmed"}}
    assert not any("disagreement" in x for x in p.verify_analysis(out))


def test_fusion_overturning_screening_is_flagged_as_disagreement():
    p = _gate_on()
    out = {k: dict(v) for k, v in BASE.items()}   # non-tiny seg area (0.30), stage1 unlikely
    out["fuse.decision"] = {"has_landslide": True}
    assert p.consistency_needs_second_pass(out) is True
    v = p.verify_analysis(out)
    assert any("disagreement" in x for x in v), v


def test_gate_can_be_disabled_for_ablation():
    p = LandslidePolicy(enforce_consistency_gate=False)
    out = {k: dict(v) for k, v in BASE.items() if k != "fuse.decision"}
    assert not any("disagreement" in x for x in p.verify_analysis(out))


def test_degraded_geo_flagged_only_when_coordinates_expected():
    p = _gate_on()
    out = {k: dict(v) for k, v in BASE.items()}
    out["llm.first_pass"] = {"has_landslide": True, "assessment_label": "likely"}
    out["geo.background"] = {"terrain": {"slope_deg": None, "aspect_deg": None}, "source_status": "error"}
    assert not any("degraded" in x for x in p.verify_analysis(out, geo_expected=False))
    assert any("degraded" in x for x in p.verify_analysis(out, geo_expected=True))


def test_env_toggle(monkeypatch):
    monkeypatch.setenv("AGENT_CONSISTENCY_GATE", "0")
    assert LandslidePolicy.from_environment("configs/thresholds.json").enforce_consistency_gate is False
    monkeypatch.setenv("AGENT_CONSISTENCY_GATE", "1")
    assert LandslidePolicy.from_environment("configs/thresholds.json").enforce_consistency_gate is True
