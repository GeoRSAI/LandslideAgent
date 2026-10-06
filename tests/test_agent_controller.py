from src.agent.controller import LandslidePolicy


def test_policy_preserves_initial_cross_check_and_tiny_area_rule():
    policy = LandslidePolicy()
    assert policy.requires_initial_cross_check(set()) == ["tiff.info", "llm.first_pass", "seg.run"]
    assert policy.requires_initial_cross_check({"tiff.info", "seg.run"}) == ["llm.first_pass"]
    assert policy.review_required({"area_ratio": 0.03}, {"area_ratio": 0.03}) is True
    assert policy.review_required({"area_ratio": 0.20}, None) is False


def test_policy_validates_fusion_requirements():
    policy = LandslidePolicy()
    args = {
        "classification": {"class_name": "debris flow"},
        "geo_context": {
            "background": {"terrain": {"slope_deg": 12, "aspect_deg": 90}, "geology": {"source": "test"}},
            "nearby": {"count": 0, "features": []},
        },
    }
    assert policy.missing_fusion_requirements(args) == []
    assert "classification.class_name" in policy.missing_fusion_requirements({"classification": {}})
