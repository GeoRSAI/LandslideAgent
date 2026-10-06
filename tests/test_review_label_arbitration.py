"""Boundary re-check verdict semantics: seg.llm_review arbitrates through its
three-valued verdict label alone (Support/NotSupport, i.e. decision
``positive``/``negative``); there is no numeric pass threshold. The rule arm
completes without asking the caller anything about the review, and the review
score (a hardcoded label placeholder in the tool output) never enters the
fused confidence.
"""
from tests.test_agent_loop_guards import (
    _FakeRegistry,
    _competent_model,
    _make_agent,
    _run,
)


# stage-1 negative + small measured area + one region -> the review runs as a
# verification arbiter purely on its label; no threshold is consulted.
_NEG_SMALL = {
    "tiff.info": {"image_path": "/img.png", "width": 512, "height": 512},
    "llm.first_pass": {"has_landslide": False, "score": 0.2, "assessment_label": "unlikely"},
    "seg.run": {"area_ratio": 0.03, "landslide_pixels": 8000, "polygon_count": 1},
    "seg.refine": {"regions": [{"bbox": [0, 0, 10, 10], "score": 0.6}], "area_ratio": 0.03},
    "cls.run": {"class_name": "rock_fall", "confidence": 0.4, "topk": []},
    "geo.background": {
        "terrain": {"slope_deg": 35.0, "aspect_deg": 90.0},
        "geology": {"lithology": "granite"},
    },
    "geo.nearby": {"count": 0, "features": []},
    "seg.llm_review": {"decision": "negative", "score": 0.28, "purpose": "verification"},
    "fuse.decision": {"has_landslide": True, "final_description": "ok"},
    "report.write": {"report_path": "outputs/reports/x.json"},
}


class _CannedRegistry(_FakeRegistry):
    def __init__(self, canned):
        super().__init__()
        self._canned = canned

    def call_tool(self, name, args):
        self.calls.append(name)
        return dict(self._canned.get(name, {"ok": True}))


def test_review_runs_without_asking_for_a_threshold(monkeypatch, tmp_path):
    agent = _make_agent(_competent_model, monkeypatch, tmp_path)
    agent.registry = _CannedRegistry(_NEG_SMALL)
    agent.tools = []

    events, last = _run(agent)
    assert last["type"] == "final_core", last
    assert not [e for e in events if e.get("type") == "need_second_pass_threshold"]
    assert "seg.llm_review" in agent.registry.calls


def test_fused_confidence_never_uses_the_review_placeholder_score(monkeypatch, tmp_path):
    """The review's hardcoded score (0.78/0.28/0.5) must not become confidence:
    the review verdict is a label, not a calibrated model score. With stage-1
    scoring 0.2 and the review supporting the landslide, confidence must come
    from the first pass."""
    from src.agent.controller import get_policy

    policy = get_policy()
    decision = policy.fuse_decision(
        stage1={"has_landslide": False, "score": 0.2, "assessment_label": "unlikely"},
        segmentation={"area_ratio": 0.03, "landslide_pixels": 8000},
        refinement={"regions": [{"bbox": [0, 0, 10, 10]}], "area_ratio": 0.03},
        llm_second_pass={"decision": "positive", "score": 0.78, "purpose": "verification"},
    )
    assert decision["has_landslide"] is True
    assert decision["confidence_source"] == "vlm_first_pass"
    assert decision["confidence"] == 0.2
    assert decision["modalities"]["second_pass_positive"] is True


def test_review_positive_label_alone_resolves_disagreement(monkeypatch, tmp_path):
    """In agreement mode, a Support label (whatever its placeholder score)
    resolves a modality disagreement to positive."""
    from src.agent.controller import LandslidePolicy

    policy = LandslidePolicy(require_modality_agreement=True)
    decision = policy.fuse_decision(
        stage1={"has_landslide": False, "score": 0.2, "assessment_label": "unlikely"},
        segmentation={"area_ratio": 0.03, "landslide_pixels": 8000},
        llm_second_pass={"decision": "positive", "score": 0.78, "purpose": "verification"},
    )
    assert decision["has_landslide"] is True
    assert "re-check" in decision["decision_basis"]
