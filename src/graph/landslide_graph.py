from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Callable, TypedDict
import operator

from langgraph.graph import END, START, StateGraph

from src.agent.controller import get_policy
from src.domain.schemas import (
    ClassificationResult,
    FirstPassResult,
    GeoContext,
    ImageInfo,
    RefinementResult,
    SegmentationResult,
)
from src.pipelines.stage1_llm_judge import run_stage1
from src.pipelines.stage2_segmentation import run_stage2
from src.pipelines.stage3_classification import run_stage3
from src.pipelines.stage4_segmentation_refine import run_stage4, run_stage4_llm_review
from src.pipelines.stage5_fusion import run_stage5
from src.pipelines.stage6_report import run_stage6
from src.tools.geo_background_tool import query_geo_background_safe
from src.tools.osm_tool import query_osm_nearby_safe
from src.tools.tiff_info_tool import read_tiff_info
from src.utils.geometry import locate_primary_candidate


class GraphState(TypedDict, total=False):
    image_info: dict[str, Any]
    stage1: dict[str, Any]
    segmentation: dict[str, Any]
    refinement: dict[str, Any]
    region_location: dict[str, Any]
    classification: dict[str, Any]
    geo_context: dict[str, Any]
    llm_second_pass: dict[str, Any]
    final_report: dict[str, Any]
    report_path: str
    report_out_path: str
    latitude: float | None
    longitude: float | None
    nearby_radius: int
    enable_second_pass: bool
    second_pass_area_ratio: float
    errors: Annotated[list[str], operator.add]
    trace: Annotated[list[dict[str, Any]], operator.add]


@dataclass(frozen=True)
class GraphDependencies:
    read_image_info: Callable[[str], dict[str, Any]] = read_tiff_info
    first_pass: Callable[[dict[str, Any]], dict[str, Any]] = run_stage1
    segmentation: Callable[[dict[str, Any]], dict[str, Any]] = run_stage2
    classification: Callable[[dict[str, Any]], dict[str, Any]] = run_stage3
    refinement: Callable[..., dict[str, Any]] = run_stage4
    region_locate: Callable[..., dict[str, Any]] = locate_primary_candidate
    second_pass: Callable[..., dict[str, Any]] = run_stage4_llm_review
    geo_background: Callable[[float, float], dict[str, Any]] = query_geo_background_safe
    geo_nearby: Callable[[float, float, int], dict[str, Any]] = query_osm_nearby_safe
    fusion: Callable[..., dict[str, Any]] = run_stage5
    report: Callable[[dict[str, Any], str], str] = run_stage6


def _trace(state: GraphState, node: str, started: float, status: str = "ok", error: str = "") -> dict[str, Any]:
    item: dict[str, Any] = {
        "node": node,
        "status": status,
        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
    }
    if error:
        item["error"] = error
    return {"trace": [item]}


def _append_error(state: GraphState, error: Exception) -> list[str]:
    # errors is an additive LangGraph channel; return only the new item.
    return [f"{type(error).__name__}: {error}"]


def _input_node(state: GraphState, deps: GraphDependencies) -> GraphState:
    started = time.perf_counter()
    try:
        info = dict(state.get("image_info") or {})
        image_path = str(info.get("image_path", "") or "").strip()
        if not image_path:
            raise ValueError("image_info.image_path is required")
        if "width" not in info or "height" not in info:
            info = {**deps.read_image_info(image_path), **info}
        validated = ImageInfo.model_validate(info).model_dump()
        return {"image_info": validated, **_trace(state, "input", started)}
    except Exception as exc:
        return {"errors": _append_error(state, exc), **_trace(state, "input", started, "error", str(exc))}


def _first_pass_node(state: GraphState, deps: GraphDependencies) -> GraphState:
    started = time.perf_counter()
    try:
        result = FirstPassResult.model_validate(deps.first_pass(state["image_info"])).model_dump()
        return {"stage1": result, **_trace(state, "first_pass", started)}
    except Exception as exc:
        return {"stage1": {"assessment_label": "error", "evidence": str(exc)}, "errors": _append_error(state, exc), **_trace(state, "first_pass", started, "error", str(exc))}


def _segmentation_node(state: GraphState, deps: GraphDependencies) -> GraphState:
    started = time.perf_counter()
    try:
        result = SegmentationResult.model_validate(deps.segmentation(state["image_info"])).model_dump()
        return {"segmentation": result, **_trace(state, "segmentation", started)}
    except Exception as exc:
        return {"segmentation": {"area_ratio": 0.0, "landslide_pixels": 0, "polygon_count": 0}, "errors": _append_error(state, exc), **_trace(state, "segmentation", started, "error", str(exc))}


def _refinement_node(state: GraphState, deps: GraphDependencies) -> GraphState:
    started = time.perf_counter()
    try:
        result = deps.refinement(
            [],
            image_info=state["image_info"],
            stage1=state.get("stage1"),
            segmentation=state.get("segmentation"),
            run_llm_second_pass=False,
            llm_second_pass_max_area_ratio=state.get("second_pass_area_ratio", 0.20),
        )
        result = RefinementResult.model_validate(result).model_dump()
        return {"refinement": result, **_trace(state, "refinement", started)}
    except Exception as exc:
        return {"refinement": {"regions": [], "area_ratio": 0.0}, "errors": _append_error(state, exc), **_trace(state, "refinement", started, "error", str(exc))}



def _region_locate_node(state: GraphState, deps: GraphDependencies) -> GraphState:
    started = time.perf_counter()
    try:
        result = deps.region_locate(
            state.get("refinement") or {},
            state.get("image_info") or {},
        )
        if not isinstance(result, dict):
            result = {"value": result}
        return {"region_location": result, **_trace(state, "region_locate", started)}
    except Exception as exc:
        return {
            "region_location": {"available": False, "position": "unknown"},
            "errors": _append_error(state, exc),
            **_trace(state, "region_locate", started, "error", str(exc)),
        }


def route_second_pass(state: GraphState) -> str:
    import dataclasses

    policy = get_policy()
    threshold = float(state.get("second_pass_area_ratio", policy.tiny_area_review_threshold) or policy.tiny_area_review_threshold)
    scoped_policy = dataclasses.replace(policy, tiny_area_review_threshold=threshold)
    outputs = {
        "llm.first_pass": state.get("stage1"),
        "seg.run": state.get("segmentation"),
        "seg.refine": state.get("refinement"),
    }
    # Keep the fixed workflow on the same hard-review triggers as the Agent.
    mandatory_reason = scoped_policy.mandatory_second_pass_reason(outputs)
    if mandatory_reason or bool(state.get("enable_second_pass", False)):
        return "review"
    return "skip"


def _second_pass_node(state: GraphState, deps: GraphDependencies) -> GraphState:
    started = time.perf_counter()
    try:
        result = deps.second_pass(
            state.get("refinement"),
            image_info=state.get("image_info"),
            stage1=state.get("stage1"),
            llm_second_pass_max_area_ratio=state.get("second_pass_area_ratio", 0.20),
        )
        review = result.get("llm_second_pass") if isinstance(result, dict) else None
        return {"llm_second_pass": review or {}, **_trace(state, "second_pass_review", started)}
    except Exception as exc:
        return {"llm_second_pass": {"decision": "error", "evidence": str(exc)}, "errors": _append_error(state, exc), **_trace(state, "second_pass_review", started, "error", str(exc))}


def _classification_node(state: GraphState, deps: GraphDependencies) -> GraphState:
    started = time.perf_counter()
    try:
        result = ClassificationResult.model_validate(deps.classification(state["image_info"])).model_dump()
        return {"classification": result, **_trace(state, "classification", started)}
    except Exception as exc:
        return {"classification": {"class_name": "unknown", "confidence": 0.0, "topk": []}, "errors": _append_error(state, exc), **_trace(state, "classification", started, "error", str(exc))}


def _geo_node(state: GraphState, deps: GraphDependencies) -> GraphState:
    started = time.perf_counter()
    lat, lon = state.get("latitude"), state.get("longitude")
    if lat is None or lon is None:
        context = {
            "background": {"terrain": {"slope_deg": None, "aspect_deg": None}, "geology": {}, "source_status": "unavailable"},
            "nearby": {"count": 0, "features": [], "source_status": "unavailable"},
            "available": False,
            "warnings": ["latitude/longitude not provided; geographic enrichment skipped"],
        }
        return {"geo_context": context, **_trace(state, "geo_context", started)}
    try:
        background = deps.geo_background(float(lat), float(lon))
        nearby = deps.geo_nearby(float(lat), float(lon), int(state.get("nearby_radius", 300) or 300))
        context = GeoContext(background=background, nearby=nearby, available=True, warnings=[]).model_dump()
        return {"geo_context": context, **_trace(state, "geo_context", started)}
    except Exception as exc:
        context = {
            "background": {"terrain": {"slope_deg": None, "aspect_deg": None}, "geology": {}, "source_status": "error"},
            "nearby": {"count": 0, "features": [], "source_status": "error"},
            "available": False,
            "warnings": [str(exc)],
        }
        context = GeoContext.model_validate(context).model_dump()
        return {"geo_context": context, "errors": _append_error(state, exc), **_trace(state, "geo_context", started, "degraded", str(exc))}


def _fusion_node(state: GraphState, deps: GraphDependencies) -> GraphState:
    started = time.perf_counter()
    try:
        refinement = state.get("refinement") or {}
        result = deps.fusion(
            stage1=state.get("stage1") or {},
            refinement=refinement,
            classification=state.get("classification") or {"class_name": "unknown", "confidence": 0.0, "topk": []},
            geo_context=state.get("geo_context") or {},
            segmentation=state.get("segmentation"),
            llm_second_pass=state.get("llm_second_pass"),
            gate={"area_ratio": float(refinement.get("area_ratio", 0.0) or 0.0)},
        )
        return {"final_report": result, **_trace(state, "fusion", started)}
    except Exception as exc:
        return {"final_report": {"has_landslide": None, "error": str(exc)}, "errors": _append_error(state, exc), **_trace(state, "fusion", started, "error", str(exc))}


def _report_node(state: GraphState, deps: GraphDependencies) -> GraphState:
    started = time.perf_counter()
    out_path = str(state.get("report_out_path", "") or "").strip()
    if not out_path:
        return {**_trace(state, "report", started)}
    try:
        report_path = deps.report(state.get("final_report") or {}, out_path)
        return {"report_path": report_path, **_trace(state, "report", started)}
    except Exception as exc:
        return {"errors": _append_error(state, exc), **_trace(state, "report", started, "error", str(exc))}


def build_landslide_graph(*, deps: GraphDependencies | None = None, checkpointer: Any = None):
    dependencies = deps or GraphDependencies()
    builder = StateGraph(GraphState)
    builder.add_node("input", lambda state: _input_node(state, dependencies))
    builder.add_node("first_pass", lambda state: _first_pass_node(state, dependencies))
    builder.add_node("segmentation", lambda state: _segmentation_node(state, dependencies))
    builder.add_node("refinement", lambda state: _refinement_node(state, dependencies))
    builder.add_node("region_locate", lambda state: _region_locate_node(state, dependencies))
    builder.add_node("second_pass_review", lambda state: _second_pass_node(state, dependencies))
    builder.add_node("classification", lambda state: _classification_node(state, dependencies))
    builder.add_node("geo_context", lambda state: _geo_node(state, dependencies))
    builder.add_node("fusion", lambda state: _fusion_node(state, dependencies))
    builder.add_node("report", lambda state: _report_node(state, dependencies))

    builder.add_edge(START, "input")
    builder.add_edge("input", "first_pass")
    builder.add_edge("first_pass", "segmentation")
    builder.add_edge("segmentation", "refinement")
    builder.add_edge("refinement", "region_locate")
    builder.add_conditional_edges("region_locate", route_second_pass, {"review": "second_pass_review", "skip": "classification"})
    builder.add_edge("second_pass_review", "classification")
    builder.add_edge("classification", "geo_context")
    builder.add_edge("geo_context", "fusion")
    builder.add_edge("fusion", "report")
    builder.add_edge("report", END)
    return builder.compile(checkpointer=checkpointer)


def _build_graph_input_state(
    *,
    image_path: str,
    latitude: float | None,
    longitude: float | None,
    nearby_radius: int,
    report_out_path: str,
    enable_second_pass: bool,
    second_pass_area_ratio: float | None,
) -> GraphState:
    policy = get_policy()
    return {
        "image_info": {"image_path": str(Path(image_path))},
        "latitude": latitude,
        "longitude": longitude,
        "nearby_radius": nearby_radius,
        "report_out_path": report_out_path,
        "enable_second_pass": enable_second_pass,
        "second_pass_area_ratio": (
            policy.tiny_area_review_threshold
            if second_pass_area_ratio is None
            else float(second_pass_area_ratio)
        ),
        "errors": [],
        "trace": [],
    }


def _graph_config(thread_id: str | None) -> dict[str, Any]:
    return {
        "configurable": {
            "thread_id": thread_id or f"landslide-{int(time.time() * 1000)}"
        }
    }


def stream_landslide_graph(
    *,
    image_path: str,
    latitude: float | None = None,
    longitude: float | None = None,
    nearby_radius: int = 300,
    report_out_path: str = "",
    enable_second_pass: bool = False,
    second_pass_area_ratio: float | None = None,
    thread_id: str | None = None,
    graph: Any = None,
):
    """Yield one ``{node, update}`` item whenever a graph node completes."""
    compiled = graph or build_landslide_graph()
    input_state = _build_graph_input_state(
        image_path=image_path,
        latitude=latitude,
        longitude=longitude,
        nearby_radius=nearby_radius,
        report_out_path=report_out_path,
        enable_second_pass=enable_second_pass,
        second_pass_area_ratio=second_pass_area_ratio,
    )
    for updates in compiled.stream(
        input_state,
        config=_graph_config(thread_id),
        stream_mode="updates",
    ):
        for node, update in (updates or {}).items():
            yield {"node": str(node), "update": dict(update or {})}


def invoke_landslide_graph(
    *,
    image_path: str,
    latitude: float | None = None,
    longitude: float | None = None,
    nearby_radius: int = 300,
    report_out_path: str = "",
    enable_second_pass: bool = False,
    second_pass_area_ratio: float | None = None,
    thread_id: str | None = None,
    graph: Any = None,
) -> GraphState:
    compiled = graph or build_landslide_graph()
    input_state = _build_graph_input_state(
        image_path=image_path,
        latitude=latitude,
        longitude=longitude,
        nearby_radius=nearby_radius,
        report_out_path=report_out_path,
        enable_second_pass=enable_second_pass,
        second_pass_area_ratio=second_pass_area_ratio,
    )
    config = _graph_config(thread_id)
    return compiled.invoke(input_state, config=config)
