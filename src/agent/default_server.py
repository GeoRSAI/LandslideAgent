from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.agent.controller import ToolCallContext, get_policy, format_rule_violations
from src.agent.protocol import JsonRpcAgentServer, ToolRegistry, ToolSpec
from src.models.llm_client import chat_with_tools, llm_describe_scene
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
from src.utils.geometry import locate_primary_candidate


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


def _image_info_for_tool(args: dict[str, Any]) -> dict[str, Any]:
    """Accept either image_info.image_path or the public top-level image_path.

    The contract layer supplies a full ``image_info`` block, but an agent running
    without it can only reasonably pass a path. Both arms must reach the same
    tool implementation, so the tool boundary accepts either form.
    """
    raw_info = args.get("image_info")
    image_info = dict(raw_info) if isinstance(raw_info, dict) else {}
    top_level_path = str(args.get("image_path", "") or "").strip()
    nested_path = str(image_info.get("image_path", "") or "").strip()
    if top_level_path and not nested_path:
        image_info["image_path"] = top_level_path
    if not str(image_info.get("image_path", "") or "").strip():
        raise ValueError("missing image_path for image analysis tool")
    return image_info


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
    return get_policy().tiny_area_review_threshold

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
    return get_policy().fusion_argument_hint(missing)

def _fuse_decision_required_call_instruction() -> str:
    return get_policy().fusion_required_call_instruction()


def _run_seg_llm_review(args: dict[str, Any], max_area_ratio: float) -> dict[str, Any]:
    """Rebuild the boundary input when the model calls review too early."""
    refinement = args.get("refinement") if isinstance(args.get("refinement"), dict) else {}
    segmentation = args.get("segmentation") if isinstance(args.get("segmentation"), dict) else None
    if not refinement.get("regions") and segmentation:
        refinement = run_stage4(
            [],
            image_info=args.get("image_info"),
            stage1=args.get("stage1"),
            segmentation=segmentation,
            run_llm_second_pass=False,
            llm_second_pass_max_area_ratio=max_area_ratio,
        )
    result = run_stage4_llm_review(
        refinement,
        image_info=args.get("image_info"),
        stage1=args.get("stage1"),
        llm_second_pass_max_area_ratio=max_area_ratio,
    )
    if not result.get("llm_second_pass") and not result.get("llm_second_pass_skipped_for_large_area"):
        result["error"] = "seg.llm_review produced no second-pass result"
    return result

def _seg_llm_review_required_call_instruction() -> str:
    return get_policy().second_pass_required_instruction()

def _fuse_retry_system_instruction_from_error(error_text: str) -> str | None:
    return get_policy().retry_instruction_from_error(error_text)

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


def _fuse_lenient() -> bool:
    """True on the free-arm container: fuse.decision has no required inputs."""
    return os.getenv("FUSE_LENIENT", "0") in ("1", "true", "True")


_LENIENT_FUSE_DESCRIPTION = (
    "Signal that you have finished gathering evidence. Call it when you are ready for the "
    "final report; no arguments are needed. It makes no decision itself: the analysis ends "
    "and the final report is written from the tool results you have collected."
)


def _run_fuse_decision(args: dict[str, Any], policy: Any) -> dict[str, Any]:
    """Execute fusion, reporting absent required inputs in actionable terms.

    Presence check only: it names which declared input is missing or malformed
    so the caller receives a usable observation instead of a bare KeyError. It
    makes no judgement about whether the evidence is sufficient - that is a
    domain rule and lives in the contract layer, not here.
    """
    if _fuse_lenient():
        # Free arm (FUSE_LENIENT=1): fuse.decision is only the agent's signal that
        # evidence gathering is finished. It makes no decision; the final report
        # (and its verdict) is written from the tool results the agent collected.
        return {
            "report_requested": True,
            "note": "Evidence gathering closed; the final report is written from the tool results collected so far.",
        }
    missing = [key for key in ("stage1", "refinement") if not isinstance(args.get(key), dict)]
    if missing:
        raise ValueError(
            "fuse.decision needs `stage1` and `refinement` as objects; "
            + ", ".join("`%s`" % key for key in missing)
            + (" is" if len(missing) == 1 else " are")
            + " absent or not an object."
        )
    refinement = args["refinement"]
    return run_stage5(
        stage1=args["stage1"],
        refinement=refinement,
        classification=args.get("classification"),
        geo_context=args.get("geo_context"),
        gate={"area_ratio": float(refinement.get("area_ratio", 0.0) or 0.0)},
        segmentation=args.get("segmentation"),
        llm_second_pass=args.get("llm_second_pass"),
        unavailable_evidence=args.get("unavailable_evidence"),
    )


def _run_report_write(args: dict[str, Any]) -> dict[str, Any]:
    """Persist a report, reporting absent required inputs in actionable terms."""
    problems = []
    if not isinstance(args.get("report"), dict):
        problems.append("`report` is absent or not an object")
    if not str(args.get("out_path", "") or "").strip():
        problems.append("`out_path` is absent or empty")
    if problems:
        raise ValueError("report.write needs both inputs: " + "; ".join(problems) + ".")
    return {"report_path": run_stage6(args["report"], str(args["out_path"]).strip())}


def create_default_server(
    thresholds_path: str = "configs/thresholds.json",
    *,
    enable_seg_llm_second_pass: bool | None = None,
    review_threshold: float | None = None,
) -> JsonRpcAgentServer:
    thresholds = load_thresholds(thresholds_path)
    policy = get_policy(thresholds_path)
    enable_report_write = policy.require_report_write
    if enable_seg_llm_second_pass is None:
        enable_seg_llm_second_pass = os.getenv("SEG_ENABLE_LLM_SECOND_PASS", "0") in ("1", "true", "True")
    seg_llm_second_pass_max_area_ratio = review_threshold if review_threshold is not None else policy.tiny_area_review_threshold
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
        lambda args: {"tiles": crop_or_tile(_image_info_for_tool(args), int(args.get("tile_size", 512)))},
    )

    registry.register(
        ToolSpec(
            name="llm.first_pass",
            description=(
                "Whole-image first-pass landslide screening by VLM. "
                "Returns has_landslide, score, assessment_label, scene_description, evidence."
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
        lambda args: run_stage1(_image_info_for_tool(args)),
    )

    registry.register(
        ToolSpec(
            name="vlm.describe",
            description=(
                "Whole-image landslide description by the fine-tuned vision-language model. Returns "
                "fields: presence, type, position, morphology, material, movement, environment, "
                "impact, reason, causation."
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
        lambda args: llm_describe_scene(str(_image_info_for_tool(args).get("image_path", "") or "")),
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
        lambda args: run_stage2(_image_info_for_tool(args)),
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
        lambda args: run_stage3(_image_info_for_tool(args)),
    )

    registry.register(
        ToolSpec(
            name="seg.refine",
            description=(
                "Segmentation-guided refinement to derive candidate landslide regions from segmentation output and image context. "
                "Returns regions[] (bbox/score/class_id), area_ratio, optional overlay_path/mask_path."
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
            name="region.locate",
            description=(
                "Run AFTER seg.refine. Describe where the primary landslide candidate sits within "
                "the image frame (3x3 grid: upper/middle/lower by left/center/right), with its "
                "normalized centre and bbox. Descriptive only - it does not affect the landslide "
                "decision. Takes the seg.refine output and optionally image_info for the frame size."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "refinement": {"type": "object"},
                    "image_info": {"type": "object"},
                    "image_path": {"type": "string"},
                },
                "required": [],
            },
        ),
        lambda args: locate_primary_candidate(args.get("refinement"), args.get("image_info")),
    )

    registry.register(
        ToolSpec(
            name="seg.llm_review",
            description=(
                "Second-pass VLM review on the full image with segmentation-boundary overlay. "
                "Used for verification or description enrichment after refinement. "
                "Returns llm_second_pass (decision/support score/evidence) and review metadata."
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
        lambda args: _run_seg_llm_review(
            args,
            seg_llm_second_pass_max_area_ratio,
        ),
    )

    registry.register(
        ToolSpec(
            name="fuse.decision",
            description=_LENIENT_FUSE_DESCRIPTION if _fuse_lenient() else (
                "Fuse multi-stage evidence into the final decision/report fields. "
                "Call this with NO arguments once tiff.info, llm.first_pass, seg.run, "
                "cls.run, geo.background and geo.nearby have run (plus seg.llm_review "
                "when it was required): stage1, refinement, segmentation, classification "
                "and geo_context are assembled automatically from those tool outputs. "
                "Pass an argument only to override an auto-filled value - do not "
                "hand-construct classification or geo_context. "
                "Returns has_landslide, classification_confidence, severity, summary, recommendations, final_description."
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
                "required": [],
            },
        ),
        lambda args: _run_fuse_decision(args, policy),
    )

    if enable_report_write:
        registry.register(
            ToolSpec(
                name="report.write",
                description=(
                    "Write the final report object to local disk as JSON. "
                    "Both report and out_path must be supplied to this tool; when the "
                    "domain contract is active it resolves them from the evidence "
                    "ledger before the call. Returns report_path."
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
            lambda args: _run_report_write(args),
        )

    def chat_handler(params: dict[str, Any]) -> dict[str, Any]:
        messages = params.get("messages", []) or []
        latitude = params.get("latitude")
        longitude = params.get("longitude")
        nearby_radius = int(params.get("nearby_radius", 300) or 300)
        has_geo_inputs = latitude is not None and longitude is not None
        latest_user_image_path = _extract_latest_user_image_path(messages)
        report_write_required_for_request = enable_report_write and bool(
            latest_user_image_path
        )
        finalization_instruction = (
            "Before finishing, call fuse.decision and then report.write to write the final report JSON to local disk. "
            if report_write_required_for_request
            else "Before finishing, call fuse.decision to produce the final decision/report output. "
        )
        if not messages or messages[0].get("role") != "system":
            messages = [
                {
                    "role": "system",
                    "content": "".join(
                        [
                            "You are a landslide analysis agent. ",
                            "For image analysis, always complete this initial cross-check before final decision/report: ",
                            "tiff.info, llm.first_pass, seg.run. ",
                            "Intermediate tool usage is flexible and not fixed by a required sequence. ",
                            "When landslide area ratio is very small (< 0.20), you must invoke seg.llm_review using the segmentation-boundary highlighted overlay image ",
                            "to perform a second-pass verification and enrich the final narrative description. ",
                            finalization_instruction,
                            "Final conclusions and reports must include landslide subtype reference, ",
                            "terrain slope/aspect and geological background evidence, and nearby human-facility context. ",
                            _fuse_decision_required_call_instruction(),
                        ]
                    ),
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
            outputs = tool_state["outputs"]
            call_counts = tool_state["call_counts"]
            current_calls = int(call_counts.get(name, 0))

            # Hard-precondition + dependency-assembly layer: single implementation
            # in LandslidePolicy, shared verbatim with the free-agent runtime.
            ctx = ToolCallContext(
                image_path=str(latest_user_image_path or ""),
                latitude=float(latitude) if has_geo_inputs else None,
                longitude=float(longitude) if has_geo_inputs else None,
                nearby_radius=nearby_radius,
                report_written=tool_state["report_written"] >= 1,
                run_tool=registry.call_tool,
                read_image_info=read_tiff_info,
            )
            args = policy.prepare_tool_call(name, raw_args, outputs, ctx)

            result = registry.call_tool(name, args)
            if not isinstance(result, dict):
                result = {"value": result}
            # Keep external geo-service failures inside the tool protocol so
            # the controller can retry or report degraded evidence instead of
            # terminating the whole agent stream.
            if name == "geo.background" and result.get("error"):
                result.setdefault("warnings", []).append(str(result["error"]))
                result["source_status"] = "degraded"
            outputs[name] = result
            call_counts[name] = current_calls + 1
            if name == "fuse.decision":
                tool_state["fuse_called"] += 1
            if name == "report.write":
                tool_state["report_written"] += 1
            return result

        tools = _as_openai_tools(registry)
        if not report_write_required_for_request:
            tools = [
                tool
                for tool in tools
                if (tool.get("function") or {}).get("name") != "report.write"
            ]
        result = chat_with_tools(
            messages,
            tools,
            guarded_tool_executor,
            max_turns=max_turns,
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
            )
        if (
            report_write_required_for_request
            and tool_state["report_written"] < 1
            and isinstance(tool_state["outputs"].get("fuse.decision"), dict)
            and not tool_state["outputs"].get("fuse.decision", {}).get("error")
        ):
            # fuse.decision is terminal; persist report directly.
            try:
                guarded_tool_executor("report.write", {})
            except Exception:
                logging.exception("deterministic report.write after fuse.decision failed")
            if tool_state["report_written"] < 1:
                return {
                    "message": {
                        "role": "assistant",
                        "content": "Analysis stopped because mandatory report.write was not completed.",
                    },
                    "history": list(result["history"]),
                }
        # Critic is skipped after successful fusion.
        critic_max_retries = 0 if (isinstance(tool_state["outputs"].get("fuse.decision"), dict) and not tool_state["outputs"].get("fuse.decision", {}).get("error")) else int(os.getenv("AGENT_CRITIC_MAX_RETRIES", "3") or "3")
        critic_retries = 0
        while critic_retries < critic_max_retries:
            critic_violations = policy.verify_analysis(
                tool_state["outputs"],
                require_report_write=report_write_required_for_request,
                geo_expected=has_geo_inputs,
            )
            if not critic_violations:
                break
            critic_retries += 1
            result = chat_with_tools(
                list(result["history"])
                + [{"role": "system", "content": format_rule_violations(critic_violations)}],
                tools,
                guarded_tool_executor,
                max_turns=max_turns,
            )

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
