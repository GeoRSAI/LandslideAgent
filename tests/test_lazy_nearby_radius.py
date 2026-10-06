"""Lazy OSM nearby-probe radius: the agent pauses right before geo.* and
emits ``need_nearby_radius`` when the radius was not supplied; once it is
supplied (resume, via the hydrated tool ledger) the workflow completes.
"""
from tests.test_agent_loop_guards import (
    _FakeRegistry,
    _competent_model,
    _make_agent,
    _run,
)
from src.agent.rule_guided_agent import RuleGuidedAgent


# Nothing in the conversation names the next tool any more, so the cooperative
# model supplies its own plan.
_completing_model = _competent_model


def test_agent_pauses_for_nearby_radius_when_not_supplied(monkeypatch, tmp_path):
    img = tmp_path / "img.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    monkeypatch.setenv("AGENT_MAX_TURNS", "60")
    agent = RuleGuidedAgent(
        model_fn=_completing_model,
        report_write_required=True,
        image_path=str(img),
        latitude=29.6,
        longitude=103.0,
        # nearby_radius omitted -> None -> lazy
    )
    agent.registry = _FakeRegistry()
    agent.tools = []

    events = list(agent.stream([{"role": "user", "content": "analyze"}]))
    assert events[-1]["type"] == "need_nearby_radius"
    assert events[-1]["turns_used"] == 10
    called = agent.registry.calls
    # everything up to geo.background ran; geo.nearby did not.
    assert "cls.run" in called
    assert "geo.nearby" not in called
    assert "fuse.decision" not in called


def test_agent_completes_when_radius_supplied(monkeypatch, tmp_path):
    _, last = _run(_make_agent(_completing_model, monkeypatch, tmp_path))
    assert last["type"] == "final_core"



def test_resume_leaves_geo_nearby_to_the_model(monkeypatch, tmp_path):
    """Resuming with a radius does not execute geo.nearby on the model's behalf.

    The confirmed radius is an argument the contract layer supplies, not a
    queued action: the model re-issues the call itself and the contract admits
    it with the user-confirmed value.
    """
    import json as _json

    img = tmp_path / "img.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    monkeypatch.setenv("AGENT_MAX_TURNS", "60")

    agent = RuleGuidedAgent(
        model_fn=_completing_model,
        report_write_required=True,
        image_path=str(img),
        latitude=29.6,
        longitude=103.0,
        nearby_radius=250,  # supplied -> this is a resume
    )
    agent.registry = _FakeRegistry()
    agent.tools = []

    done = {
        "tiff.info": {"image_path": "/img.png", "width": 512, "height": 512},
        "llm.first_pass": {"has_landslide": True, "score": 0.8, "assessment_label": "likely"},
        "seg.run": {"area_ratio": 0.4, "landslide_pixels": 100, "polygon_count": 1},
        "seg.refine": {"regions": [{"bbox": [0, 0, 1, 1], "score": 0.7}], "area_ratio": 0.4},
        "region.locate": {"available": True, "position": "middle-center"},
        "cls.run": {"class_name": "earth_flow", "confidence": 0.7, "topk": []},
        "geo.background": {"terrain": {"slope_deg": 20.0, "aspect_deg": 180.0}, "geology": {"lithology": "x"}},
    }
    msgs = [{"role": "user", "content": "analyze"}]
    for name, out in done.items():
        msgs.append({"role": "tool", "name": name, "content": _json.dumps(out)})

    events = list(agent.stream(msgs))
    tool_calls = [e["name"] for e in events if e.get("type") == "tool_call"]

    nearby = [e for e in events
              if e.get("type") == "tool_call" and e.get("name") == "geo.nearby"]
    assert nearby, tool_calls
    # the model issued the call; the contract layer supplied the confirmed radius
    assert nearby[0]["arguments"].get("radius") == 250, nearby[0]
    assert events[-1]["type"] == "final_core"


def test_model_calling_geo_nearby_triggers_the_radius_pause(monkeypatch, tmp_path):
    """The pause is tied to the model actually calling geo.nearby, not to order."""
    img = tmp_path / "img.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    monkeypatch.setenv("AGENT_MAX_TURNS", "60")

    def geo_first_model(_messages, _tools):
        # deliberately reach for geo.nearby early, before cls.run
        done = {m.get("name") for m in _messages if m.get("role") == "tool"}
        if "geo.nearby" not in done:
            return {"message": {"role": "assistant", "content": "",
                                "tool_calls": [{"id": "g", "type": "function",
                                                "function": {"name": "geo.nearby", "arguments": "{}"}}]},
                    "raw": ""}
        return {"message": {"role": "assistant", "content": "done"}, "raw": ""}

    agent = RuleGuidedAgent(
        model_fn=geo_first_model,
        report_write_required=True,
        image_path=str(img),
        latitude=29.6,
        longitude=103.0,
        # nearby_radius omitted
    )
    agent.registry = _FakeRegistry()
    agent.tools = []

    events = list(agent.stream([{"role": "user", "content": "analyze"}]))
    assert events[-1]["type"] == "need_nearby_radius"
    assert events[-1]["turns_used"] == 1
    assert "geo.nearby" not in agent.registry.calls
