"""Stage-1 orchestration hardening:

* ``LandslidePolicy.fuse_decision_is_stale`` + wiring: a fusion computed before
  ``seg.llm_review`` must be recomputed, not finalised.
* ``RuleGuidedAgent`` critic reports unmet deliverable obligations and never
  executes a tool on the model's behalf.
"""
import json

from src.agent.controller import LandslidePolicy
from tests.test_agent_loop_guards import _make_agent, _run


# --------------------------------------------------------------------------- #
# stale fusion
# --------------------------------------------------------------------------- #
_OUT = {
    "tiff.info": {"image_path": "/img.png", "width": 512, "height": 512},
    "llm.first_pass": {"has_landslide": False, "assessment_label": "unlikely", "score": 0.2},
    "seg.run": {"area_ratio": 0.30},
    "seg.refine": {"regions": [{"bbox": [0, 0, 1, 1], "score": 0.7}], "area_ratio": 0.30},
    "vlm.describe": {"fields": {"presence": "Yes", "type": "Earthflow"}},
    "region.locate": {"available": True, "position": "middle-center"},
    "cls.run": {"class_name": "earth_flow", "confidence": 0.7, "topk": []},
    "geo.background": {"terrain": {"slope_deg": 20.0, "aspect_deg": 180.0}, "geology": {"lithology": "x"}},
    "geo.nearby": {"count": 0, "features": []},
}


def _fuse(role):
    return {"has_landslide": True, "final_description": "### a\n### b\n", "decision_support": {"second_pass_role": role}}


def _review(decision):
    return {"llm_second_pass": {"decision": decision, "evidence": "boundary looks like a scarp"}}


def test_fusion_without_review_is_stale_when_review_now_exists():
    p = LandslidePolicy()
    out = dict(_OUT, **{"fuse.decision": _fuse("none"), "seg.llm_review": _review("negative")})
    assert p.fuse_decision_is_stale(out) is True
    assert p.next_required_tool(out) == "fuse.decision"
    assert any("stale fusion" in v for v in p.verify_analysis(out))


def test_fusion_that_used_the_review_is_not_stale():
    p = LandslidePolicy()
    out = dict(_OUT, **{"fuse.decision": _fuse("arbiter"), "seg.llm_review": _review("negative")})
    assert p.fuse_decision_is_stale(out) is False
    assert p.next_required_tool(out) is None


def test_descriptive_only_review_does_not_mark_fusion_stale():
    p = LandslidePolicy()
    out = dict(_OUT, **{"fuse.decision": _fuse("none"), "seg.llm_review": _review("descriptive")})
    assert p.fuse_decision_is_stale(out) is False


# --------------------------------------------------------------------------- #
# graduated critic nudge
# --------------------------------------------------------------------------- #
def test_critic_reports_obligations_without_executing_anything(monkeypatch, tmp_path):
    """The critic states what the contract still requires, and stops there."""

    def stalling_model(_messages, _tools):
        return {"message": {"role": "assistant", "content": ""}, "raw": "<tool_call>\n</tool_call>"}

    agent = _make_agent(stalling_model, monkeypatch, tmp_path, AGENT_MAX_TURNS="6")
    events, last = _run(agent)

    assert last["type"] == "fallback", last
    critic = [str(e.get("content", "")) for e in events if e.get("source") == "critic"]
    assert critic, events
    assert all("auto-advance" not in c and "targeted nudge" not in c for c in critic)
    assert agent.registry.calls == [], agent.registry.calls
