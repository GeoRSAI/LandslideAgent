from pathlib import Path

from PIL import Image

from src.graph.landslide_graph import GraphDependencies, build_landslide_graph, invoke_landslide_graph


def _fake_deps(calls):
    def image_info(path):
        calls.append("input")
        return {"image_path": path, "width": 100, "height": 100}

    def first_pass(info):
        calls.append("first_pass")
        return {"has_landslide": True, "score": 0.8, "assessment_label": "likely", "evidence": "Likely"}

    def segmentation(info):
        calls.append("segmentation")
        return {"area_ratio": 0.01, "landslide_pixels": 100, "polygon_count": 1, "mask_path": "", "overlay_path": ""}

    def refinement(*args, **kwargs):
        calls.append("refinement")
        return {"regions": [{"bbox": [1, 2, 20, 30], "score": 0.8, "class_id": 0}], "area_ratio": 0.01}

    def second_pass(*args, **kwargs):
        calls.append("second_pass")
        return {"llm_second_pass": {"decision": "positive", "score": 0.8, "reviewed_regions": 1}}

    def classification(info):
        calls.append("classification")
        return {"class_name": "debris flow", "confidence": 0.7, "topk": []}

    def geo_background(lat, lon):
        calls.append("geo_background")
        return {"terrain": {"slope_deg": 20, "aspect_deg": 120}, "geology": {}}

    def geo_nearby(lat, lon, radius):
        calls.append("geo_nearby")
        return {"count": 0, "features": []}

    def fusion(**kwargs):
        calls.append("fusion")
        return {"has_landslide": True, "final_description": "ok"}

    def report(report, path):
        calls.append("report")
        return path

    return GraphDependencies(
        read_image_info=image_info,
        first_pass=first_pass,
        segmentation=segmentation,
        refinement=refinement,
        second_pass=second_pass,
        classification=classification,
        geo_background=geo_background,
        geo_nearby=geo_nearby,
        fusion=fusion,
        report=report,
    )


def test_graph_runs_explicit_workflow_and_accumulates_trace(tmp_path: Path):
    calls = []
    deps = _fake_deps(calls)
    graph = build_landslide_graph(deps=deps)
    image = tmp_path / "scene.png"
    Image.new("RGB", (100, 100), color=(100, 100, 100)).save(image)

    result = invoke_landslide_graph(
        image_path=str(image),
        latitude=30.0,
        longitude=100.0,
        report_out_path=str(tmp_path / "report.json"),
        enable_second_pass=True,
        second_pass_area_ratio=0.20,
        graph=graph,
    )

    assert result["final_report"]["has_landslide"] is True
    assert result["report_path"].endswith("report.json")
    assert "second_pass" in calls
    assert calls.index("refinement") < calls.index("second_pass") < calls.index("classification")
    assert [item["node"] for item in result["trace"]] == [
        "input", "first_pass", "segmentation", "refinement", "region_locate", "second_pass_review",
        "classification", "geo_context", "fusion", "report"
    ]


def test_graph_runs_mandatory_tiny_area_review(tmp_path: Path):
    calls = []
    deps = _fake_deps(calls)
    graph = build_landslide_graph(deps=deps)
    image = tmp_path / "scene.png"
    Image.new("RGB", (100, 100), color=(100, 100, 100)).save(image)

    result = invoke_landslide_graph(image_path=str(image), graph=graph)

    assert result["final_report"]["has_landslide"] is True
    assert "second_pass" in calls
    assert "geo_context" in [item["node"] for item in result["trace"]]


def test_graph_routes_material_screening_disagreement_to_review():
    from src.graph.landslide_graph import route_second_pass

    state = {
        "stage1": {"has_landslide": False, "assessment_label": "unlikely"},
        "segmentation": {"area_ratio": 0.10, "landslide_pixels": 10000},
        "refinement": {"area_ratio": 0.30, "regions": [{"bbox": [0, 0, 10, 10]}]},
        "enable_second_pass": False,
        "second_pass_area_ratio": 0.20,
    }
    assert route_second_pass(state) == "review"


def test_graph_route_is_registered_without_replacing_legacy_route():
    import scripts.llm_service as service

    paths = {getattr(route, "path", "") for route in service.app.routes}
    assert "/v1/graph/analyze" in paths
    assert "/v1/agent/analyze" in paths
    assert "/v1/agent/analyze_stream" in paths
