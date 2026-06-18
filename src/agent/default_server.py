from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.agent.protocol import JsonRpcAgentServer, ToolRegistry, ToolSpec
from src.models.llm_client import chat_with_tools
from src.pipelines.stage1_llm_judge import run_stage1
from src.pipelines.stage2_segmentation import run_stage2
from src.pipelines.stage3_classification import run_stage3
from src.pipelines.stage4_segmentation_refine import run_stage4, run_stage4_llm_review
from src.pipelines.stage5_fusion import run_stage5
from src.pipelines.stage6_report import run_stage6
from src.tools.crop_tool import crop_or_tile
from src.tools.osm_tool import query_osm_nearby_safe
from src.tools.geo_background_tool import query_geo_background_safe
from src.tools.tiff_info_tool import read_tiff_info


def load_thresholds(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _as_openai_tools(registry: ToolRegistry) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": spec["name"],
                "description": spec["description"],
                "parameters": spec["input_schema"],
            },
        }
        for spec in registry.list_tools()
    ]


def _looks_like_region_item(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    bbox = item.get("bbox")
    return isinstance(bbox, list) and len(bbox) == 4


def _looks_like_refinement_result(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    regions = value.get("regions")
    if not isinstance(regions, list):
        return False
    return all(_looks_like_region_item(d) for d in regions)


def _region_count(value: Any) -> int:
    if not isinstance(value, dict):
        return 0
    regions = value.get("regions")
    if not isinstance(regions, list):
        return 0
    return sum(1 for item in regions if _looks_like_region_item(item))


def _resolve_image_info(
    args: dict[str, Any],
    outputs: dict[str, Any],
) -> dict[str, Any]:
    image_info = args.get("image_info")
    if isinstance(image_info, dict):
        image_path = str(image_info.get("image_path", "") or "")
        has_size = ("width" in image_info) and ("height" in image_info)
        if image_path and not has_size:
            try:
                enriched = read_tiff_info(image_path)
                merged = dict(enriched)
                merged.update(image_info)
                return merged
            except Exception:
                return image_info
        return image_info

    if "tiff.info" in outputs and isinstance(outputs["tiff.info"], dict):
        return outputs["tiff.info"]

    image_path = str(args.get("image_path", "") or "")
    if image_path:
        return read_tiff_info(image_path)

    raise ValueError("missing image_info/image_path. Provide image_path or call tiff.info.")


def _parse_second_pass_area_ratio_limit(raw: str) -> float | None:
    value = str(raw or "").strip()
    if not value:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0.0:
        return None
    return parsed


def _mandatory_seg_llm_review_area_ratio(limit: float | None) -> float:
    if limit is None:
        return 0.20
    try:
        value = float(limit)
    except (TypeError, ValueError):
        return 0.20
    if value <= 0.0:
        return 0.20
    return value


def _resolve_refinement_area_ratio(refinement: Any, segmentation: Any) -> float | None:
    for candidate in (refinement, segmentation):
        if not isinstance(candidate, dict):
            continue
        raw = candidate.get("area_ratio")
        try:
            ratio = float(raw)
        except (TypeError, ValueError):
            continue
        if ratio < 0.0:
            continue
        return ratio
    return None


def _extract_latest_user_image_path(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image":
                    image_path = part.get("image_path") or part.get("image")
                    if image_path:
                        return str(image_path)
        return ""
    return ""


def _latest_user_has_image(messages: list[dict[str, Any]]) -> bool:
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image":
                    return True
        return False
    return False


def _is_existing_file(path: str) -> bool:
    try:
        return bool(path) and Path(path).exists() and Path(path).is_file()
    except Exception:
        return False


def _default_report_out_path(image_path: str | None = None) -> str:
    image_stem = Path(str(image_path or "")).stem.strip() if image_path else ""
    base_name = image_stem or "landslide_report"
    return str(Path("outputs") / "reports" / f"{base_name}_{uuid4().hex[:8]}.json")


def _fuse_decision_argument_hint(missing: list[str]) -> str:
    missing_text = ", ".join(missing) if missing else "unknown"
    return (
        "fuse.decision input is incomplete. "
        f"Missing: {missing_text}. "
        "Required top-level args: stage1, refinement, classification, geo_context. "
        "Required fields: classification.class_name; "
        "geo_context.background.terrain.slope_deg/aspect_deg; "
        "geo_context.background.geology; "
        "geo_context.nearby.count/features. "
        "Build arguments by reusing tool outputs: "
        "classification <- cls.run, "
        "geo_context.background <- geo.background, "
        "geo_context.nearby <- geo.nearby. "
        "Then call fuse.decision again with explicit classification + geo_context. "
        "Minimal argument template: "
        "{\"classification\":{\"class_name\":\"<from cls.run>\",\"confidence\":0.0,\"topk\":[]},"
        "\"geo_context\":{\"background\":{\"terrain\":{\"slope_deg\":0.0,\"aspect_deg\":0.0},\"geology\":{}},"
        "\"nearby\":{\"count\":0,\"features\":[]}}}."
    )


def _fuse_decision_required_call_instruction() -> str:
    return (
        "Hard requirement for fuse.decision arguments: never call with empty arguments {}. "
        "You must explicitly provide classification and geo_context. "
        "Build arguments by reusing tool outputs exactly as: classification <- cls.run; "
        "geo_context.background <- geo.background; geo_context.nearby <- geo.nearby. "
        "Minimal argument template: "
        "{\"classification\":{\"class_name\":\"<from cls.run>\",\"confidence\":0.0,\"topk\":[]},"
        "\"geo_context\":{\"background\":{\"terrain\":{\"slope_deg\":0.0,\"aspect_deg\":0.0},\"geology\":{}},"
        "\"nearby\":{\"count\":0,\"features\":[]}}}. "
        "If fuse.decision returns input incomplete, immediately call fuse.decision again with corrected explicit arguments."
    )


def _seg_llm_review_required_call_instruction() -> str:
    return (
        "Hard requirement before fuse.decision for tiny-area cases: "
        "if area_ratio is below 0.20, call seg.llm_review first, then call fuse.decision again. "
        "Use arguments exactly as: refinement <- seg.refine; stage1 <- llm.first_pass; image_info <- tiff.info. "
        "Then set llm_second_pass <- seg.llm_review.llm_second_pass when calling fuse.decision."
    )


def _fuse_retry_system_instruction_from_error(error_text: str) -> str | None:
    text = str(error_text or "").strip()
    lowered = text.lower()
    if "fuse.decision" not in lowered:
        return None
    if (
        "input is incomplete" in lowered
        or "missing:" in lowered
        or "requires" in lowered
        or "classification" in lowered
        or "geo_context" in lowered
    ):
        return _fuse_decision_required_call_instruction()
    if "seg.llm_review" in lowered and ("mandatory" in lowered or "call seg.llm_review first" in lowered):
        return _seg_llm_review_required_call_instruction()
    return None


def _validate_fuse_required_arguments(args: dict[str, Any]) -> None:
    missing: list[str] = []
    classification = args.get("classification")
    if not isinstance(classification, dict):
        missing.append("classification")
        class_name = ""
    else:
        class_name = str(classification.get("class_name", "") or "").strip()
        if not class_name:
            missing.append("classification.class_name")

    geo_context = args.get("geo_context")
    if not isinstance(geo_context, dict):
        missing.append("geo_context")
        raise ValueError(_fuse_decision_argument_hint(missing))

    background = geo_context.get("background")
    nearby = geo_context.get("nearby")
    if not isinstance(background, dict):
        missing.append("geo_context.background")
        terrain: Any = None
        geology: Any = None
    else:
        terrain = background.get("terrain")
        geology = background.get("geology")
    if not isinstance(nearby, dict):
        missing.append("geo_context.nearby")

    if not isinstance(terrain, dict):
        missing.append("geo_context.background.terrain")
    if isinstance(terrain, dict) and "slope_deg" not in terrain:
        missing.append("geo_context.background.terrain.slope_deg")
    if isinstance(terrain, dict) and "aspect_deg" not in terrain:
        missing.append("geo_context.background.terrain.aspect_deg")
    if not isinstance(geology, dict):
        missing.append("geo_context.background.geology")
    elif not any(key in geology for key in ("lithology", "unit_name", "description", "age", "source")):
        missing.append("geo_context.background.geology.(lithology|unit_name|description|age|source)")

    if isinstance(nearby, dict) and "count" not in nearby:
        missing.append("geo_context.nearby.count")
    if isinstance(nearby, dict) and "features" not in nearby:
        missing.append("geo_context.nearby.features")
    if isinstance(nearby, dict) and "features" in nearby and not isinstance(nearby.get("features"), list):
        missing.append("geo_context.nearby.features(list)")

    if missing:
        raise ValueError(_fuse_decision_argument_hint(missing))


def _extract_last_tool_error(history: list[dict[str, Any]], tool_name: str) -> str:
    for msg in reversed(history):
        if msg.get("role") != "tool":
            continue
        if str(msg.get("name", "")) != tool_name:
            continue
        raw_content = msg.get("content", "")
        if isinstance(raw_content, str):
            try:
                parsed = json.loads(raw_content)
            except json.JSONDecodeError:
                continue
        elif isinstance(raw_content, dict):
            parsed = raw_content
        else:
            continue
        if isinstance(parsed, dict):
            err = str(parsed.get("error", "") or "").strip()
            if err:
                return err
    return ""


def create_default_server(
    thresholds_path: str = "configs/thresholds.json",
    *,
    enable_seg_llm_second_pass: bool | None = None,
) -> JsonRpcAgentServer:
    thresholds = load_thresholds(thresholds_path)
    enable_report_write = os.getenv("AGENT_ENABLE_REPORT_WRITE", "0") in ("1", "true", "True")
    if enable_seg_llm_second_pass is None:
        enable_seg_llm_second_pass = os.getenv("SEG_ENABLE_LLM_SECOND_PASS", "0") in ("1", "true", "True")
    seg_llm_second_pass_max_area_ratio = _parse_second_pass_area_ratio_limit(
        os.getenv("SEG_LLM_SECOND_PASS_MAX_AREA_RATIO", "0.20")
    )
    registry = ToolRegistry()

    registry.register(
        ToolSpec(
            name="geo.nearby",
            description=(
                "Query nearby human facilities from OpenStreetMap around (lat, lon). "
                "Use this to assess exposure context near the landslide site. "
                "Returns observation_point, radius_m, count, features[], warnings, source_status."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "lat": {"type": "number"},
                    "lon": {"type": "number"},
                    "radius": {"type": "integer", "default": 300},
                },
                "required": ["lat", "lon"],
            },
        ),
        lambda args: query_osm_nearby_safe(
            float(args["lat"]),
            float(args["lon"]),
            int(args.get("radius", 300)),
        ),
    )

    registry.register(
        ToolSpec(
            name="geo.background",
            description=(
                "Query geographic background for (lat, lon). "
                "Returns address, terrain (elevation_m, slope_deg, aspect_deg, dem_source), "
                "geology (lithology/unit_name/description/age/source), and warnings."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "lat": {"type": "number"},
                    "lon": {"type": "number"},
                },
                "required": ["lat", "lon"],
            },
        ),
        lambda args: query_geo_background_safe(
            float(args["lat"]),
            float(args["lon"]),
        ),
    )

    registry.register(
        ToolSpec(
            name="tiff.info",
            description=(
                "Read raster image metadata from image_path. "
                "Supports GeoTIFF and common raster formats (PNG/JPG/JPEG/TIFF). "
                "Returns width, height, bands, dtype, image_path, and geospatial fields when available (e.g., crs/resolution/bounds)."
            ),
            input_schema={
                "type": "object",
                "properties": {"image_path": {"type": "string"}},
                "required": ["image_path"],
            },
        ),
        lambda args: read_tiff_info(args["image_path"]),
    )

    registry.register(
        ToolSpec(
            name="image.tile",
            description=(
                "Split a large image into tiles for downstream localized analysis. "
                "Prefer skipping this when width <= 1024 and height <= 1024. "
                "Returns tiles[] with tile coordinates and paths."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "image_info": {"type": "object"},
                    "image_path": {"type": "string"},
                    "tile_size": {"type": "integer", "default": 512},
                },
                "required": [],
            },
        ),
        lambda args: {"tiles": crop_or_tile(args["image_info"], int(args.get("tile_size", 512)))},
    )

    registry.register(
        ToolSpec(
            name="llm.first_pass",
            description=(
                "Whole-image first-pass landslide screening by VLM. "
                "Returns has_landslide, assessment_label, scene_description, evidence. "
                "Does not return numeric risk scores or confidence values."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "image_info": {"type": "object"},
                    "image_path": {"type": "string"},
                },
                "required": [],
            },
        ),
        lambda args: run_stage1(args["image_info"]),
    )

    registry.register(
        ToolSpec(
            name="seg.run",
            description=(
                "Semantic segmentation on the full image for landslide area extraction. "
                "Returns area_ratio, landslide_pixels, mask_path, overlay_path, polygon_count (if available)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "image_info": {"type": "object"},
                    "image_path": {"type": "string"},
                },
                "required": [],
            },
        ),
        lambda args: run_stage2(args["image_info"]),
    )

    registry.register(
        ToolSpec(
            name="cls.run",
            description=(
                "Landslide subtype classification (reference evidence, not final yes/no by itself). "
                "Returns class_name, confidence, class_id, topk."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "image_info": {"type": "object"},
                    "image_path": {"type": "string"},
                },
                "required": [],
            },
        ),
        lambda args: run_stage3(args["image_info"]),
    )

    registry.register(
        ToolSpec(
            name="seg.refine",
            description=(
                "Segmentation-guided refinement to derive candidate landslide regions from segmentation output and image context. "
                "Returns regions[] (bbox/class_id/source/area_ratio without confidence scoring), area_ratio, optional overlay_path/mask_path."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "tiles": {"type": "array"},
                    "image_info": {"type": "object"},
                    "segmentation": {"type": "object"},
                    "regions": {"type": "array"},
                },
                "required": [],
            },
        ),
        lambda args: run_stage4(
            args.get("tiles", []),
            image_info=args.get("image_info"),
            segmentation=args.get("segmentation"),
            regions=args.get("regions"),
            run_llm_second_pass=False,
            llm_second_pass_max_area_ratio=seg_llm_second_pass_max_area_ratio,
        ),
    )

    registry.register(
        ToolSpec(
            name="seg.llm_review",
            description=(
                "Second-pass VLM review on the full image with segmentation-boundary overlay. "
                "Used for verification or description enrichment after refinement. "
                "Returns llm_second_pass (decision/support label/evidence) and review metadata. "
                "Does not return numeric risk scores or confidence values."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "refinement": {"type": "object"},
                    "stage1": {"type": "object"},
                    "image_info": {"type": "object"},
                    "image_path": {"type": "string"}
                },
                "required": [],
            },
        ),
        lambda args: run_stage4_llm_review(
            args.get("refinement"),
            image_info=args.get("image_info"),
            stage1=args.get("stage1"),
            llm_second_pass_max_area_ratio=seg_llm_second_pass_max_area_ratio,
        ),
    )

    registry.register(
        ToolSpec(
            name="fuse.decision",
            description=(
                "Fuse multi-stage evidence into final decision/report fields. "
                "Requires stage1 + refinement + explicit classification + explicit geo_context "
                "(background terrain/geology and nearby facilities). "
                "Do not call with empty arguments; always pass explicit classification and geo_context. "
                "Minimal argument template: "
                "{\"classification\":{\"class_name\":\"<from cls.run>\",\"confidence\":0.0,\"topk\":[]},"
                "\"geo_context\":{\"background\":{\"terrain\":{\"slope_deg\":0.0,\"aspect_deg\":0.0},\"geology\":{}},"
                "\"nearby\":{\"count\":0,\"features\":[]}}}. "
                "Returns has_landslide, severity, summary, recommendations, final_description. "
                "No calibrated overall confidence is returned; cls.run confidence remains reference-only."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "stage1": {"type": "object"},
                    "refinement": {"type": "object"},
                    "classification": {
                        "type": "object",
                        "properties": {
                            "class_name": {"type": "string"},
                            "confidence": {"type": "number"},
                            "topk": {"type": "array", "items": {"type": "object"}},
                        },
                        "required": ["class_name"],
                    },
                    "segmentation": {"type": "object"},
                    "geo_context": {
                        "type": "object",
                        "properties": {
                            "background": {
                                "type": "object",
                                "properties": {
                                    "terrain": {
                                        "type": "object",
                                        "properties": {
                                            "slope_deg": {"type": ["number", "null"]},
                                            "aspect_deg": {"type": ["number", "null"]},
                                        },
                                        "required": ["slope_deg", "aspect_deg"],
                                    },
                                    "geology": {"type": "object"},
                                },
                                "required": ["terrain", "geology"],
                            },
                            "nearby": {
                                "type": "object",
                                "properties": {
                                    "count": {"type": "integer"},
                                    "features": {"type": "array", "items": {"type": "object"}},
                                },
                                "required": ["count", "features"],
                            },
                        },
                        "required": ["background", "nearby"],
                    },
                    "llm_second_pass": {"type": "object"},
                },
                "required": ["stage1", "refinement", "classification", "geo_context"],
            },
        ),
        lambda args: run_stage5(
            stage1=args["stage1"],
            refinement=args["refinement"],
            classification=args.get("classification"),
            geo_context=args.get("geo_context"),
            gate={"area_ratio": float(args["refinement"].get("area_ratio", 0.0) or 0.0)},
            segmentation=args.get("segmentation"),
            llm_second_pass=args.get("llm_second_pass"),
            llm_second_pass_threshold=float(thresholds["llm_second_pass_threshold"]),
            min_region_score=float(thresholds.get("min_region_score", 0.45)),
        ),
    )

    if enable_report_write:
        registry.register(
            ToolSpec(
                name="report.write",
                description=(
                    "Write the final report object to local disk as JSON. "
                    "If report is omitted, reuse the latest fuse.decision output. "
                    "If out_path is omitted, auto-write under outputs/reports/. "
                    "Returns report_path."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "report": {"type": "object"},
                        "out_path": {"type": "string"},
                    },
                    "required": [],
                },
            ),
            lambda args: {"report_path": run_stage6(args["report"], args["out_path"])},
        )

    def chat_handler(params: dict[str, Any]) -> dict[str, Any]:
        messages = params.get("messages", []) or []
        latitude = params.get("latitude")
        longitude = params.get("longitude")
        has_geo_inputs = latitude is not None and longitude is not None
        latest_user_image_path = _extract_latest_user_image_path(messages)
        current_turn_has_image = _latest_user_has_image(messages)
        report_write_required = enable_report_write and current_turn_has_image
        if not messages or messages[0].get("role") != "system":
            if current_turn_has_image:
                system_content = "".join(
                    [
                        "You are a landslide analysis agent. ",
                        "For image analysis, always complete this initial cross-check before final decision/report: ",
                        "tiff.info, llm.first_pass, seg.run. ",
                        "Intermediate tool usage is flexible and not fixed by a required sequence. ",
                        "When landslide area ratio is very small (< 0.20), you must invoke seg.llm_review using the segmentation-boundary highlighted overlay image ",
                        "to perform a second-pass verification and enrich the final narrative description. ",
                        (
                            "Before finishing, call fuse.decision and then report.write to write the final report JSON to local disk. "
                            "After report.write succeeds, end the current assistant turn immediately without generating another assistant message in the same turn. "
                            if report_write_required
                            else "Before finishing, call fuse.decision to produce the final decision/report output. "
                        ),
                        "Final conclusions and reports must include landslide subtype reference, ",
                        "terrain slope/aspect and geological background evidence, and nearby human-facility context. ",
                        _fuse_decision_required_call_instruction(),
                    ]
                )
            else:
                system_content = (
                    "This is a follow-up turn without a new image upload. "
                    "Focus on answering questions about previous conclusions and evidence. "
                    "Do not force a new analysis workflow unless the user explicitly asks to rerun tools."
                )
            messages = [
                {
                    "role": "system",
                    "content": system_content,
                }
            ] + messages

        if has_geo_inputs:
            coord_instruction = (
                "Coordinates provided for this request: "
                f"lat={float(latitude):.8f}, lon={float(longitude):.8f}. "
                "Use these exact values whenever collecting geographic evidence."
            )
            if messages and messages[0].get("role") == "system":
                messages[0]["content"] = f"{messages[0].get('content', '')}\n\n{coord_instruction}"
            else:
                messages = [{"role": "system", "content": coord_instruction}] + messages

        raw_max_turns = params.get("max_turns")
        max_turns: int | None
        if raw_max_turns is None:
            max_turns = None
        else:
            parsed_max_turns = int(raw_max_turns)
            max_turns = parsed_max_turns if parsed_max_turns > 0 else None

        tool_state: dict[str, Any] = {
            "outputs": {},
            "fuse_called": 0,
            "report_written": 0,
            "call_counts": {},
        }

        def guarded_tool_executor(name: str, raw_args: dict[str, Any]) -> dict[str, Any]:
            args = dict(raw_args or {})
            outputs = tool_state["outputs"]
            call_counts = tool_state["call_counts"]
            current_calls = int(call_counts.get(name, 0))

            if name == "tiff.info":
                image_path = str(args.get("image_path", "") or "").strip()
                if not _is_existing_file(image_path):
                    if _is_existing_file(latest_user_image_path):
                        args["image_path"] = latest_user_image_path
                    elif image_path and Path(image_path).exists() and Path(image_path).is_dir():
                        raise ValueError(f"tiff.info requires an image file path, got directory: {image_path}")
                    else:
                        raise ValueError(
                            "tiff.info requires a valid image_path file; no usable image path was found in tool arguments or latest user image."
                        )
            elif name in ("geo.nearby", "geo.background"):
                if ("lat" not in args or "lon" not in args) and has_geo_inputs:
                    args["lat"] = float(latitude)
                    args["lon"] = float(longitude)
                    if name == "geo.nearby" and "radius" not in args:
                        args["radius"] = 300
                if "lat" not in args or "lon" not in args:
                    raise ValueError("geo tools require lat and lon.")
            elif name == "image.tile":
                args["image_info"] = _resolve_image_info(args, outputs)
            elif name in ("llm.first_pass", "seg.run"):
                if "tiff.info" not in outputs:
                    raise ValueError(f"{name} requires tiff.info first for initial cross-check.")
                args["image_info"] = outputs["tiff.info"]
            elif name == "cls.run":
                args["image_info"] = _resolve_image_info(args, outputs)
            elif name == "seg.refine":
                if "image_info" not in args:
                    args["image_info"] = _resolve_image_info(args, outputs)
                if "segmentation" not in args and "seg.run" in outputs:
                    args["segmentation"] = outputs["seg.run"]
                if not isinstance(args.get("segmentation"), dict):
                    raise ValueError("seg.refine requires segmentation output (call seg.run first).")
                if "regions" in args and not all(_looks_like_region_item(d) for d in (args.get("regions") or [])):
                    args.pop("regions", None)
            elif name == "seg.llm_review":
                if "refinement" not in args and "seg.refine" in outputs:
                    args["refinement"] = outputs["seg.refine"]
                if "refinement" not in args and "tiff.info" in outputs and "seg.run" in outputs:
                    args["refinement"] = run_stage4(
                        [],
                        image_info=outputs["tiff.info"],
                        stage1=outputs.get("llm.first_pass"),
                        segmentation=outputs["seg.run"],
                        run_llm_second_pass=False,
                        llm_second_pass_max_area_ratio=seg_llm_second_pass_max_area_ratio,
                    )
                if "refinement" not in args:
                    raise ValueError("seg.llm_review requires segmentation-guided refinement context.")
                if "stage1" not in args and "llm.first_pass" in outputs:
                    args["stage1"] = outputs["llm.first_pass"]
                args["image_info"] = _resolve_image_info(args, outputs)
            elif name == "fuse.decision":
                if "llm.first_pass" not in outputs:
                    raise ValueError(
                        "fuse.decision prerequisite missing: llm.first_pass. "
                        + _fuse_decision_argument_hint(["stage1", "classification", "geo_context"])
                    )
                if "seg.run" not in outputs:
                    raise ValueError(
                        "fuse.decision prerequisite missing: seg.run. "
                        + _fuse_decision_argument_hint(["refinement/segmentation", "classification", "geo_context"])
                    )
                if "cls.run" not in outputs:
                    raise ValueError(
                        "fuse.decision prerequisite missing: cls.run. "
                        + _fuse_decision_argument_hint(["classification", "classification.class_name"])
                    )
                if "geo.nearby" not in outputs:
                    raise ValueError(
                        "fuse.decision prerequisite missing: geo.nearby. "
                        + _fuse_decision_argument_hint(["geo_context.nearby.count", "geo_context.nearby.features"])
                    )
                if "geo.background" not in outputs:
                    raise ValueError(
                        "fuse.decision prerequisite missing: geo.background. "
                        + _fuse_decision_argument_hint(
                            [
                                "geo_context.background.terrain.slope_deg",
                                "geo_context.background.terrain.aspect_deg",
                                "geo_context.background.geology",
                            ]
                        )
                    )
                args["stage1"] = outputs["llm.first_pass"]
                if "segmentation" not in args:
                    args["segmentation"] = outputs["seg.run"]
                if "refinement" not in args and "seg.refine" in outputs:
                    args["refinement"] = outputs["seg.refine"]
                if "refinement" not in args:
                    args["refinement"] = run_stage4(
                        [],
                        image_info=outputs["tiff.info"] if isinstance(outputs.get("tiff.info"), dict) else None,
                        stage1=outputs.get("llm.first_pass"),
                        segmentation=args.get("segmentation"),
                        run_llm_second_pass=False,
                        llm_second_pass_max_area_ratio=seg_llm_second_pass_max_area_ratio,
                    )
                if "segmentation" not in args and isinstance(args.get("refinement"), dict):
                    derived_seg = args["refinement"].get("segmentation")
                    if isinstance(derived_seg, dict):
                        args["segmentation"] = derived_seg
                mandatory_review_threshold = _mandatory_seg_llm_review_area_ratio(seg_llm_second_pass_max_area_ratio)
                refinement_area_ratio = _resolve_refinement_area_ratio(args.get("refinement"), args.get("segmentation"))
                review_is_mandatory = (
                    refinement_area_ratio is not None and refinement_area_ratio < mandatory_review_threshold
                )
                if review_is_mandatory:
                    review_output = outputs.get("seg.llm_review")
                    if not isinstance(review_output, dict):
                        ratio_text = f"{refinement_area_ratio:.6f}" if refinement_area_ratio is not None else "unknown"
                        raise ValueError(
                            "fuse.decision prerequisite missing: seg.llm_review. "
                            f"seg.llm_review is mandatory when area_ratio ({ratio_text}) is below {mandatory_review_threshold:.2f}. "
                            "Call seg.llm_review first, then call fuse.decision again."
                        )
                    args["llm_second_pass"] = review_output.get("llm_second_pass")
                elif "llm_second_pass" not in args and "seg.llm_review" in outputs:
                    args["llm_second_pass"] = outputs["seg.llm_review"].get("llm_second_pass")
                if "classification" not in args:
                    args["classification"] = outputs["cls.run"]
                if "geo_context" not in args:
                    args["geo_context"] = {
                        "background": outputs["geo.background"],
                        "nearby": outputs["geo.nearby"],
                    }
                _validate_fuse_required_arguments(args)
                if "refinement" in args and not _looks_like_refinement_result(args["refinement"]):
                    if "seg.refine" in outputs and _looks_like_refinement_result(outputs["seg.refine"]):
                        args["refinement"] = outputs["seg.refine"]
                if "refinement" in args and not _looks_like_refinement_result(args["refinement"]):
                    raise ValueError("fuse.decision requires segmentation-guided refinement output.")
                if "stage1" not in args or "refinement" not in args:
                    raise ValueError("fuse.decision requires stage1 and refinement outputs.")
            elif name == "report.write":
                if "report" not in args and "fuse.decision" not in outputs:
                    raise ValueError("report.write requires fuse.decision first.")
                if "report" not in args and isinstance(outputs.get("fuse.decision"), dict):
                    args["report"] = outputs["fuse.decision"]
                if "out_path" not in args or not str(args.get("out_path", "") or "").strip():
                    fused_image_path = ""
                    if isinstance(outputs.get("tiff.info"), dict):
                        fused_image_path = str(outputs["tiff.info"].get("image_path", "") or "").strip()
                    if not fused_image_path:
                        fused_image_path = str(latest_user_image_path or "").strip()
                    args["out_path"] = _default_report_out_path(fused_image_path)
                if not isinstance(args.get("report"), dict):
                    raise ValueError("report.write requires a report object; call fuse.decision first.")
                if tool_state["report_written"] >= 1:
                    raise ValueError("report.write should only be called once.")

            result = registry.call_tool(name, args)
            outputs[name] = result
            call_counts[name] = current_calls + 1
            if name == "fuse.decision":
                tool_state["fuse_called"] += 1
            if name == "report.write":
                tool_state["report_written"] += 1
            return result

        tools = _as_openai_tools(registry)
        result = chat_with_tools(
            messages,
            tools,
            guarded_tool_executor,
            max_turns=max_turns,
            stop_after_tools={"report.write"} if report_write_required else None,
        )
        fuse_retry_retries = 0
        while (
            tool_state["fuse_called"] >= 1
            and "fuse.decision" not in tool_state["outputs"]
            and fuse_retry_retries < 2
        ):
            fuse_retry_retries += 1
            fuse_error_text = _extract_last_tool_error(list(result["history"]), "fuse.decision")
            fuse_retry_instruction = _fuse_retry_system_instruction_from_error(fuse_error_text)
            retry_message = (
                fuse_retry_instruction
                if fuse_retry_instruction
                else _fuse_decision_required_call_instruction()
            )
            result = chat_with_tools(
                list(result["history"]) + [{"role": "system", "content": retry_message}],
                tools,
                guarded_tool_executor,
                max_turns=max_turns,
                stop_after_tools={"report.write"} if report_write_required else None,
            )
        if report_write_required and tool_state["report_written"] < 1:
            report_guard_retries = 0
            while report_guard_retries < 2 and tool_state["report_written"] < 1:
                report_guard_retries += 1
                fuse_error_text = _extract_last_tool_error(list(result["history"]), "fuse.decision")
                fuse_retry_instruction = _fuse_retry_system_instruction_from_error(fuse_error_text)
                reminder_message = {
                    "role": "system",
                    "content": (
                        "Do not finish yet. Mandatory finalization is incomplete: "
                        "call fuse.decision if needed, then call report.write with a valid local out_path."
                        + ((" " + fuse_retry_instruction) if fuse_retry_instruction else "")
                    ),
                }
                result = chat_with_tools(
                    list(result["history"]) + [reminder_message],
                    tools,
                    guarded_tool_executor,
                    max_turns=max_turns,
                    stop_after_tools={"report.write"},
                )
            if tool_state["report_written"] < 1:
                return {
                    "message": {
                        "role": "assistant",
                        "content": "Analysis stopped because mandatory report.write was not completed.",
                    },
                    "history": list(result["history"]),
                }
        fuse_output = tool_state["outputs"].get("fuse.decision")
        structured_final = ""
        if isinstance(fuse_output, dict):
            structured_final = str(fuse_output.get("final_description", "") or "").strip()
        if structured_final:
            final_message = {"role": "assistant", "content": structured_final}
            final_history = list(result["history"])
            if final_history and final_history[-1].get("role") == "assistant" and not (final_history[-1].get("tool_calls") or []):
                final_history[-1] = final_message
            else:
                final_history.append(final_message)
            return {
                "message": final_message,
                "history": final_history,
            }
        return {
            "message": result["message"],
            "history": result["history"],
        }

    return JsonRpcAgentServer(registry, chat_handler=chat_handler)
