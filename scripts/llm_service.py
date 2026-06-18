import importlib.util

from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import os
import time
import subprocess
import threading
from PIL import Image
import json
import re
import ssl
from pathlib import Path
from urllib import request
from urllib import error as urlerror
from urllib.parse import quote
from uuid import uuid4

from src.tools.osm_tool import query_osm_nearby_safe
from src.models.llm_client import capture_model_raw_events

DEFAULT_LLM_API_MODEL_NAME = "qwen3-vl-8b-instruct"
DEFAULT_MODEL_PATH = ""
LLM_API_MODEL_NAME = os.getenv("LLM_API_MODEL_NAME", DEFAULT_LLM_API_MODEL_NAME)

app = FastAPI(title="Qwen Multimodal Service")
logging.basicConfig(level=logging.INFO)

# Add CORS support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
tokenizer = None
processor = None
mock_mode = False
lora_loaded = False
model_status = "not_started"
model_error = ""
model_loader_thread: threading.Thread | None = None
model_state_lock = threading.Lock()
multipart_available = importlib.util.find_spec("multipart") is not None

MODEL_PATH = os.getenv("LLM_MODEL_PATH", DEFAULT_MODEL_PATH)
LORA_PATH = os.getenv("LLM_LORA_PATH", "")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

def _resolve_seg_llm_second_pass_max_area_ratio() -> float | None:
    raw = str(os.getenv("SEG_LLM_SECOND_PASS_MAX_AREA_RATIO", "0.20") or "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logging.warning("Invalid SEG_LLM_SECOND_PASS_MAX_AREA_RATIO=%r; fallback to 0.20", raw)
        return 0.20
    if value <= 0.0:
        return None
    return value


def _segmentation_positive(segmentation: dict[str, Any] | None) -> bool:
    segmentation = segmentation or {}
    seg_ratio = float(segmentation.get("area_ratio", 0.0) or 0.0)
    seg_pixels = int(segmentation.get("landslide_pixels", 0) or 0)
    return bool(seg_ratio >= 0.01 or (seg_ratio >= 0.005 and seg_pixels >= 512))


def _cross_check_positive(stage1: dict[str, Any] | None, segmentation: dict[str, Any] | None) -> bool | None:
    if not isinstance(stage1, dict) or not isinstance(segmentation, dict):
        return None
    return bool(stage1.get("has_landslide", False)) or _segmentation_positive(segmentation)


def _is_existing_file(path: str) -> bool:
    try:
        return bool(path) and Path(path).exists() and Path(path).is_file()
    except Exception:
        return False


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


def _report_write_enabled() -> bool:
    return os.getenv("AGENT_ENABLE_REPORT_WRITE", "0") in ("1", "true", "True")


def _analysis_workflow_instruction() -> str:
    finalization_instruction = (
        "Before finishing, call fuse.decision and then report.write to write the final report JSON to local disk. "
        "After report.write succeeds, end the current assistant turn immediately without generating another assistant message in the same turn."
        if _report_write_enabled()
        else "Before finishing, call fuse.decision to produce the final decision/report output."
    )
    return (
        "For image analysis, always complete this initial cross-check before final decision/report: "
        "tiff.info, llm.first_pass, seg.run. "
        "Intermediate tool usage is flexible and not fixed by a required sequence. "
        "When landslide area ratio is very small (< 0.20), you must invoke seg.llm_review using the segmentation-boundary highlighted overlay image "
        "to perform a second-pass verification and enrich the final narrative description. "
        "Before writing the final report, prefer to gather supporting evidence when available: "
        "landslide subtype classification via cls.run; "
        "terrain slope/aspect and geological background via geo.background; "
        "nearby human-facility context via geo.nearby. "
        "Use your judgment to decide whether these tools are needed for the current case, "
        "but aim to produce the most complete evidence-backed report practical for the case. "
        + finalization_instruction
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


def _latest_fuse_tool_error_text(tool_trace: list[dict[str, Any]]) -> str:
    for item in reversed(tool_trace):
        if item.get("tool") != "fuse.decision" or item.get("status") != "error":
            continue
        output = item.get("output")
        if isinstance(output, dict):
            return str(output.get("error", "") or "")
    return ""


def _default_report_out_path(image_path: str | None = None) -> str:
    image_stem = Path(str(image_path or "")).stem.strip() if image_path else ""
    base_name = image_stem or "landslide_report"
    return str(Path("outputs") / "reports" / f"{base_name}_{uuid4().hex[:8]}.json")


def _prune_payload_for_summary(value: Any, *, depth: int = 0) -> Any:
    if depth >= 3:
        if isinstance(value, dict):
            return "{...}"
        if isinstance(value, list):
            return ["..."]
        return value
    if isinstance(value, dict):
        items = list(value.items())
        pruned: dict[str, Any] = {}
        for index, (key, item) in enumerate(items):
            if index >= 12:
                pruned["__truncated_keys__"] = len(items) - 12
                break
            pruned[str(key)] = _prune_payload_for_summary(item, depth=depth + 1)
        return pruned
    if isinstance(value, list):
        pruned_list = [_prune_payload_for_summary(item, depth=depth + 1) for item in value[:8]]
        if len(value) > 8:
            pruned_list.append(f"... ({len(value) - 8} more items)")
        return pruned_list
    return value


def _serialize_summary_payload(value: Any, *, max_chars: int = 1800) -> str:
    pruned = _prune_payload_for_summary(value)
    try:
        text = json.dumps(pruned, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        text = str(pruned)
    compact = " ".join(str(text or "").split()).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 16].rstrip() + "...(truncated)"


def _deterministic_tool_summary(tool_name: str, status: str, output: Any) -> str:
    data = output if isinstance(output, dict) else {}
    if status != "ok" or "error" in data:
        error_text = " ".join(str(data.get("error", "") or "").split()).strip()
        return f"{tool_name} failed." if not error_text else f"{tool_name} failed: {error_text}."

    if tool_name == "tiff.info":
        width = data.get("width", "?")
        height = data.get("height", "?")
        bands = data.get("bands", "?")
        return f"Image metadata loaded: {width} x {height}, bands={bands}."

    if tool_name == "llm.first_pass":
        if data.get("has_landslide") is True:
            decision = "likely landslide"
        elif data.get("has_landslide") is False:
            decision = "likely non-landslide"
        else:
            decision = "inconclusive"
        return f"First-pass screening finished: {decision}."

    if tool_name == "seg.run":
        try:
            ratio_text = f"{float(data.get('area_ratio', 0.0) or 0.0) * 100:.2f}%"
        except Exception:
            ratio_text = "unknown"
        pixels = int(data.get("landslide_pixels", 0) or 0)
        return f"Segmentation finished: landslide area ratio {ratio_text}, pixels={pixels}."

    if tool_name == "seg.refine":
        regions = data.get("regions")
        count = len(regions) if isinstance(regions, list) else 0
        return f"Segmentation-guided refinement finished: {count} candidate region(s)."

    if tool_name == "seg.llm_review":
        if data.get("llm_second_pass_skipped_for_large_area"):
            return "Second-pass LLM review was skipped because the candidate area exceeded the review threshold."
        reviewed = int(data.get("review_candidates", 0) or 0)
        review = data.get("llm_second_pass") if isinstance(data.get("llm_second_pass"), dict) else {}
        decision = str(review.get("decision", "") or "").strip()
        if decision:
            return f"Second-pass boundary-overlay review finished: decision={decision}, reviewed_regions={reviewed}."
        return f"Second-pass boundary-overlay review prepared {reviewed} candidate region(s)."

    if tool_name == "cls.run":
        label = str(data.get("class_name", "") or data.get("label", "") or "unknown").strip()
        conf = data.get("confidence")
        if isinstance(conf, (int, float)):
            return f"Classification finished: {label} ({float(conf):.2f})."
        return f"Classification finished: {label}."

    if tool_name == "geo.background":
        terrain = data.get("terrain") if isinstance(data.get("terrain"), dict) else {}
        geology = data.get("geology") if isinstance(data.get("geology"), dict) else {}
        slope = terrain.get("slope_deg")
        lithology = str(geology.get("lithology", "") or geology.get("unit_name", "") or "").strip()
        try:
            slope_text = f"{float(slope):.1f}°" if slope is not None else "unknown"
        except Exception:
            slope_text = "unknown"
        if lithology:
            return f"Geologic background updated: slope {slope_text}, lithology {lithology}."
        return f"Geologic background updated: slope {slope_text}."

    if tool_name == "geo.nearby":
        count = int(data.get("count", 0) or 0)
        if data.get("source_status") == "cached":
            return f"Nearby context loaded from cache: {count} feature(s)."
        return f"Nearby context retrieved: {count} feature(s)."

    if tool_name == "fuse.decision":
        if data.get("has_landslide") is True:
            decision = "landslide indicated"
        elif data.get("has_landslide") is False:
            decision = "landslide not indicated"
        else:
            decision = "decision unavailable"
        severity = str(data.get("severity", "n/a") or "n/a")
        return f"Final fused decision completed: {decision}, severity={severity}."

    if tool_name == "report.write":
        report_path = str(data.get("report_path", "") or "").strip()
        return f"Final report written to {report_path}." if report_path else "Final report written."

    return f"{tool_name} completed."


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


def _unavailable_geo_nearby(reason: str = "observation coordinates were not provided") -> dict[str, Any]:
    return {
        "observation_point": None,
        "radius_m": 0,
        "count": 0,
        "features": [],
        "warnings": [f"OSM nearby context is unavailable: {reason}."],
        "source": "",
        "source_status": "unavailable",
    }


def _unavailable_geo_background(reason: str = "observation coordinates were not provided") -> dict[str, Any]:
    return {
        "observation_point": None,
        "address": {},
        "terrain": {
            "elevation_m": None,
            "slope_deg": None,
            "aspect_deg": None,
            "dem_source": "",
        },
        "geology": {
            "description": "not available",
            "source": "not_available",
        },
        "warnings": [f"Geographic background is unavailable: {reason}."],
        "source_status": "unavailable",
    }


def _resolve_qwen_model_class(model_path: str):
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration, Qwen3_5ForConditionalGeneration

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = str(getattr(config, "model_type", "") or "").strip().lower()
    architectures = [str(name) for name in (getattr(config, "architectures", None) or [])]

    if "Qwen3VLForConditionalGeneration" in architectures or model_type in {"qwen3_vl", "qwen3vl"}:
        return Qwen3VLForConditionalGeneration, "Qwen3-VL"
    if "Qwen3_5ForConditionalGeneration" in architectures or model_type in {"qwen3_5", "qwen3.5"}:
        return Qwen3_5ForConditionalGeneration, "Qwen3.5"

    raise RuntimeError(
        f"Unsupported model config for {model_path}: model_type={model_type!r}, architectures={architectures!r}"
    )


def _lora_matches_model(lora_path: str, model_path: str) -> tuple[bool, str]:
    adapter_config_path = Path(lora_path) / "adapter_config.json"
    if not adapter_config_path.exists():
        return True, ""

    try:
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"failed to parse {adapter_config_path}: {exc}"

    base_model = str(adapter_config.get("base_model_name_or_path", "") or "").strip()
    if not base_model:
        return True, ""

    active_name = Path(model_path).resolve().name
    adapter_name = Path(base_model).name
    if adapter_name and adapter_name == active_name:
        return True, ""

    return False, f"adapter targets {base_model!r}, but active model directory is {model_path!r}"

# Serve static files and index.html
@app.get("/")
def read_root():
    return FileResponse(
        str(PROJECT_ROOT / "static" / "index.html"),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )

@app.get("/health")
def health_check():
    with model_state_lock:
        status = model_status
        error_text = model_error
    return {
        "status": "ok",
        "mock_mode": mock_mode,
        "model_status": status,
        "model_ready": bool(mock_mode or model is not None),
        "model_error": error_text if status == "error" else "",
        "model_path": MODEL_PATH,
        "lora_path": LORA_PATH if lora_loaded else "",
        "lora_loaded": lora_loaded,
    }

app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "static")), name="static")
(PROJECT_ROOT / "outputs").mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(PROJECT_ROOT / "outputs")), name="outputs")


def _health_ok(url: str) -> bool:
    try:
        with request.urlopen(url, timeout=2.0) as resp:
            return resp.status == 200
    except Exception:
        return False


def _cls_fallback_ready() -> bool:
    cls_env_python = os.getenv("CLS_ENV_PYTHON", "python")
    mmpretrain_root = os.getenv("MMPRETRAIN_ROOT", "")
    config_path = os.getenv("CLS_CONFIG_PATH", "")
    checkpoint_path = os.getenv("CLS_CHECKPOINT_PATH", "")
    class_mapping_path = os.getenv("CLS_CLASS_MAPPING_PATH", "")
    required_ok = all(
        Path(p).exists()
        for p in (cls_env_python, mmpretrain_root, config_path, checkpoint_path)
    )
    if not required_ok:
        return False
    if class_mapping_path and (not Path(class_mapping_path).exists()):
        return False
    return True


def _start_if_needed(service_url: str, python_path: str, app_target: str, port: int, log_name: str) -> str:
    if _health_ok(f"{service_url.rstrip('/')}/health"):
        return "already_running"
    if not os.path.exists(python_path):
        return f"missing_python:{python_path}"

    log_path = LOG_DIR / log_name
    with open(log_path, "ab") as logf:
        subprocess.Popen(
            [python_path, "-m", "uvicorn", app_target, "--host", "0.0.0.0", "--port", str(port)],
            cwd=str(PROJECT_ROOT),
            stdout=logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    for _ in range(8):
        if _health_ok(f"{service_url.rstrip('/')}/health"):
            return "started"
        time.sleep(0.5)
    return f"start_failed:check_{log_path}"


def _post_json(url: str, payload: dict, timeout: float = 30.0) -> dict:
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _extract_latest_image_path(messages: list["ChatMessage"]) -> str:
    for msg in reversed(messages):
        content = msg.content
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image":
                    image_path = part.get("image_path") or part.get("image")
                    if image_path:
                        return str(image_path)
    return ""


def _latest_user_has_image(messages: list["ChatMessage"]) -> bool:
    for msg in reversed(messages):
        if msg.role != "user":
            continue
        content = msg.content
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image":
                    return True
        return False
    return False


def _should_require_report_write(messages: list["ChatMessage"]) -> bool:
    return _report_write_enabled() and _latest_user_has_image(messages)


def _extract_latest_user_image_path(messages: list["ChatMessage"]) -> str:
    for msg in reversed(messages):
        if msg.role != "user":
            continue
        content = msg.content
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image":
                    image_path = part.get("image_path") or part.get("image")
                    if image_path:
                        return str(image_path)
        return ""
    return ""


def _safe_resolve_image(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    else:
        p = p.resolve()
    allowed_root = PROJECT_ROOT.resolve()
    if not str(p).startswith(str(allowed_root)):
        raise HTTPException(status_code=400, detail="image path not allowed")
    if not p.exists():
        raise HTTPException(status_code=404, detail="image not found")
    return p


@app.get("/media")
def media(path: str = Query(..., description="absolute or project-relative image path")):
    p = _safe_resolve_image(path)
    return FileResponse(str(p))


if multipart_available:
    @app.post("/v1/media/upload")
    async def media_upload(file: UploadFile = File(...)):
        filename = str(file.filename or "upload.bin")
        suffix = Path(filename).suffix.lower()
        allowed = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
        if suffix and suffix not in allowed:
            raise HTTPException(status_code=400, detail="unsupported file type")

        upload_dir = PROJECT_ROOT / "outputs" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        safe_suffix = suffix if suffix else ".png"
        out_path = upload_dir / f"{uuid4().hex}{safe_suffix}"

        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="empty file")
        max_bytes = int(os.getenv("UPLOAD_MAX_BYTES", str(50 * 1024 * 1024)))
        if len(data) > max_bytes:
            raise HTTPException(status_code=413, detail="file too large")

        out_path.write_bytes(data)
        return {
            "path": str(out_path),
            "name": filename,
            "size": len(data),
            "media_url": _to_media_url(str(out_path)),
        }
else:
    @app.post("/v1/media/upload")
    async def media_upload_unavailable():
        raise HTTPException(
            status_code=503,
            detail='File upload support requires the optional dependency "python-multipart".',
        )


def _to_media_url(path: str) -> str:
    version = ""
    try:
        version = f"&v={int(Path(path).stat().st_mtime_ns)}"
    except Exception:
        version = ""
    return f"/media?path={quote(path, safe='/')}{version}"


def _build_artifacts_payload(
    *,
    include_images: bool,
    image_path: str,
    seg_overlay_path: str,
    seg_mask_path: str,
    seg_refine_overlay_path: str,
) -> dict[str, str]:
    if not include_images:
        return {}
    return {
        "original": _to_media_url(image_path) if image_path else "",
        "seg_mask": _to_media_url(seg_overlay_path)
        if seg_overlay_path
        else (_to_media_url(seg_mask_path) if seg_mask_path else ""),
        "seg_refine_overlay": _to_media_url(seg_refine_overlay_path) if seg_refine_overlay_path else "",
    }


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


def _mandatory_seg_llm_review_area_ratio() -> float:
    resolved = _resolve_seg_llm_second_pass_max_area_ratio()
    if resolved is None:
        return 0.20
    try:
        value = float(resolved)
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


def _is_successful_tool_output(value: Any) -> bool:
    return isinstance(value, dict) and ("error" not in value)


def _ensure_seg_refine_overlay(
    *,
    image_path: str,
    refinement_output: dict[str, Any] | None,
    seg_mask_path: str = "",
) -> str:
    if not image_path or not os.path.exists(image_path):
        return ""
    if refinement_output is None:
        refinement_output = {}
    overlay_path = str(refinement_output.get("overlay_path", "") or "")
    if overlay_path and os.path.exists(overlay_path):
        return overlay_path

    candidate_masks: list[str] = []
    direct_mask_path = str(refinement_output.get("mask_path", "") or "").strip()
    if direct_mask_path:
        candidate_masks.append(direct_mask_path)
    segmentation = refinement_output.get("segmentation")
    if isinstance(segmentation, dict):
        nested_mask_path = str(segmentation.get("mask_path", "") or "").strip()
        if nested_mask_path:
            candidate_masks.append(nested_mask_path)
    seg_mask_path = str(seg_mask_path or "").strip()
    if seg_mask_path:
        candidate_masks.append(seg_mask_path)

    try:
        from src.pipelines.stage4_segmentation_refine import _render_segmentation_boundary_overlay
    except Exception:
        _render_segmentation_boundary_overlay = None  # type: ignore[assignment]

    if _render_segmentation_boundary_overlay is None:
        return ""

    for mask_path in candidate_masks:
        if not mask_path or not os.path.exists(mask_path):
            continue
        try:
            rendered = str(_render_segmentation_boundary_overlay(image_path, mask_path) or "")
        except Exception:
            rendered = ""
        if rendered and os.path.exists(rendered):
            return rendered

    return ""


def _set_model_state(status: str, error: str = "") -> None:
    global model_status, model_error
    with model_state_lock:
        model_status = status
        model_error = error


def _load_model_impl() -> None:
    global model, tokenizer, processor, lora_loaded
    if not (os.path.exists(MODEL_PATH) and os.listdir(MODEL_PATH)):
        raise RuntimeError(f"Model path {MODEL_PATH} is empty or does not exist.")
    logging.info("Loading model from %s...", MODEL_PATH)
    from transformers import AutoProcessor, AutoTokenizer
    import torch

    model_class, detected_model_name = _resolve_qwen_model_class(MODEL_PATH)
    logging.info("Detected model architecture: %s", detected_model_name)

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = model_class.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).eval()
    if LORA_PATH and os.path.exists(LORA_PATH):
        lora_ok, lora_reason = _lora_matches_model(LORA_PATH, MODEL_PATH)
        if not lora_ok:
            logging.warning("Skipping LoRA adapter at %s because %s", LORA_PATH, lora_reason)
        else:
            from peft import PeftModel

            logging.info("Loading LoRA adapter from %s...", LORA_PATH)
            model = PeftModel.from_pretrained(model, LORA_PATH, is_trainable=False)
            model = model.eval()
            lora_loaded = True
            logging.info("LoRA adapter loaded.")
    elif LORA_PATH:
        logging.warning("Configured LoRA path does not exist: %s", LORA_PATH)
    logging.info("%s model loaded.", detected_model_name)


def _model_loader_worker() -> None:
    global model, tokenizer, processor, lora_loaded
    try:
        lora_loaded = False
        _load_model_impl()
        _set_model_state("ready")
    except Exception as exc:
        model = None
        tokenizer = None
        processor = None
        lora_loaded = False
        _set_model_state("error", f"{type(exc).__name__}: {exc}")
        logging.exception("LLM model load failed.")


def _start_model_loader_if_needed() -> str:
    global model_loader_thread, model_status, model_error
    with model_state_lock:
        status = model_status
        alive = model_loader_thread is not None and model_loader_thread.is_alive()
        if status == "ready":
            return "ready"
        if status == "loading" and alive:
            return "loading"
        if status == "error":
            return "error"
        model_status = "loading"
        model_error = ""
        thread = threading.Thread(target=_model_loader_worker, daemon=True, name="llm-model-loader")
        model_loader_thread = thread
        thread.start()
        return "loading"

@app.on_event("startup")
def load_model():
    global model, tokenizer, processor, mock_mode, lora_loaded, model_status, model_error
    mock_env = os.getenv("LLM_MOCK", "0")
    mock_mode = mock_env in ("1", "true", "True")
    lora_loaded = False
    model_error = ""
    if mock_mode:
        logging.info("LLM_MOCK enabled, skipping model loading and using mock responses.")
        model = None
        tokenizer = None
        processor = None
        model_status = "ready"
        return
    model_status = "not_started"
    state = _start_model_loader_if_needed()
    logging.info("Model loader state: %s", state)

class ChatMessage(BaseModel):
    role: str
    content: Any
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: float = 0.2
    max_tokens: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: str = "auto"
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    context_summary: Optional[str] = None
    enable_seg_llm_second_pass: bool = False


def _resolve_max_new_tokens(req: ChatRequest) -> int:
    if req.max_tokens is not None:
        try:
            explicit = int(req.max_tokens)
        except Exception:
            explicit = 0
        if explicit > 0:
            return explicit

    env_key = "LLM_TOOL_MAX_NEW_TOKENS" if req.tools else "LLM_MAX_NEW_TOKENS"
    default_value = 768 if req.tools else 1024
    try:
        resolved = int(os.getenv(env_key, str(default_value)))
    except Exception:
        resolved = default_value
    return max(64, resolved)


class StartServicesRequest(BaseModel):
    start_seg: bool = True
    start_cls: bool = True


class StopServicesRequest(BaseModel):
    stop_seg: bool = True
    stop_cls: bool = True
    stop_llm: bool = True


def _to_dict_message(m: ChatMessage) -> dict[str, Any]:
    if hasattr(m, "model_dump"):
        return m.model_dump(exclude_none=True)
    return m.dict(exclude_none=True)


def _schema_type_label(schema: Any) -> str:
    if isinstance(schema, dict):
        t = schema.get("type")
    else:
        t = None
    if isinstance(t, list):
        return "|".join(str(x) for x in t if x is not None) or "any"
    if isinstance(t, str) and t.strip():
        return t.strip()
    return "any"


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return str(content)


_TOOL_CATALOG_BEGIN = "<<TOOL_CATALOG_BEGIN>>"
_TOOL_CATALOG_END = "<<TOOL_CATALOG_END>>"


def _build_tool_catalog_text(tools: list[dict[str, Any]] | None) -> str:
    if not isinstance(tools, list) or not tools:
        return ""

    lines: list[str] = [
        "Tool Catalog (authoritative):",
        "Use tool schemas below as contract for arguments.",
    ]
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        name = str(function.get("name", "") or "").strip()
        if not name:
            continue
        description = str(function.get("description", "") or "").strip()
        params = function.get("parameters") if isinstance(function.get("parameters"), dict) else {}
        required = params.get("required") if isinstance(params.get("required"), list) else []
        required_set = {str(x) for x in required}
        properties = params.get("properties") if isinstance(params.get("properties"), dict) else {}

        arg_chunks: list[str] = []
        for key, value in properties.items():
            key_name = str(key)
            schema = value if isinstance(value, dict) else {}
            type_label = _schema_type_label(schema)
            required_text = "required" if key_name in required_set else "optional"
            default_text = f", default={schema.get('default')}" if "default" in schema else ""
            arg_chunks.append(f"{key_name}:{type_label} ({required_text}{default_text})")
        arg_text = "; ".join(arg_chunks) if arg_chunks else "no explicit properties"

        lines.append(f"- {name}: {description}")
        lines.append(f"  Inputs: {arg_text}")

    if len(lines) <= 2:
        return ""
    lines.append(
        "If native structured tool-calls are unavailable, output one JSON object with keys `name` and `arguments`."
    )
    payload = "\n".join(lines).strip()
    return f"{_TOOL_CATALOG_BEGIN}\n{payload}\n{_TOOL_CATALOG_END}"


def _inject_tool_catalog_system_message(
    messages: list[ChatMessage],
    tools: list[dict[str, Any]] | None,
) -> list[ChatMessage]:
    tool_catalog = _build_tool_catalog_text(tools)
    if not tool_catalog:
        return list(messages)

    merged = list(messages)
    if merged and merged[0].role == "system":
        merged_head = _content_to_text(merged[0].content).strip()
        if _TOOL_CATALOG_BEGIN in merged_head and _TOOL_CATALOG_END in merged_head:
            # Already injected in this conversation context: keep single catalog block.
            start = merged_head.find(_TOOL_CATALOG_BEGIN)
            end = merged_head.find(_TOOL_CATALOG_END, start)
            if start >= 0 and end >= 0:
                end += len(_TOOL_CATALOG_END)
                replaced = (merged_head[:start].rstrip() + "\n\n" + tool_catalog + "\n\n" + merged_head[end:].lstrip()).strip()
                merged[0] = ChatMessage(role="system", content=replaced)
                return merged
        merged[0] = ChatMessage(role="system", content=f"{merged_head}\n\n{tool_catalog}" if merged_head else tool_catalog)
        return merged

    return [ChatMessage(role="system", content=tool_catalog)] + merged


def _inject_forced_system_messages(messages: list[dict[str, Any]], req: ChatRequest) -> list[dict[str, Any]]:
    latest_image_path = _extract_latest_user_image_path(req.messages)
    coord_text = ""
    if req.latitude is not None and req.longitude is not None:
        coord_text = (
            f" Coordinates provided: lat={float(req.latitude):.8f}, lon={float(req.longitude):.8f}; reuse these exact values when collecting geographic evidence."
        )
    if latest_image_path:
        workflow_instruction = _analysis_workflow_instruction()
        fuse_instruction = _fuse_decision_required_call_instruction()
        forced_system = {
            "role": "system",
            "content": (
                "For any tool call that needs an image path, use this exact path: "
                f"{latest_image_path}. "
                + workflow_instruction
                + " "
                + fuse_instruction
                + coord_text
            ),
        }
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = f"{messages[0].get('content', '')}\n\n{forced_system['content']}"
        else:
            messages = [forced_system] + messages
    return messages


def _inject_followup_system_message(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    followup_system = {
        "role": "system",
        "content": (
            "This is a follow-up turn without a new image upload. "
            "Focus on answering questions about previous conclusions and evidence."
        ),
    }
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = f"{messages[0].get('content', '')}\n\n{followup_system['content']}"
        return messages
    return [followup_system] + messages


def _inject_context_summary_message(messages: list[dict[str, Any]], req: ChatRequest) -> list[dict[str, Any]]:
    summary = str(req.context_summary or "").strip()
    if not summary:
        return messages
    context_system = {
        "role": "system",
        "content": (
            "Use the following persisted session context as factual memory for this follow-up turn. "
            "Prefer these confirmed details when answering questions about prior geo/OSM findings, unless the user explicitly asks to rerun tools or replace the context.\n\n"
            f"{summary}"
        ),
    }
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = f"{messages[0].get('content', '')}\n\n{context_system['content']}"
        return messages
    return [context_system] + messages


def _message_contains_image(content: Any) -> bool:
    if not isinstance(content, list):
        return False
    for part in content:
        if isinstance(part, dict) and part.get("type") == "image":
            return True
    return False


def _trim_messages_to_latest_image_session(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Prevent cross-image context pollution:
    keep only the latest image analysis session when multiple images are present.
    """
    latest_user_image_idx = -1
    for idx, msg in enumerate(messages):
        if msg.get("role") == "user" and _message_contains_image(msg.get("content")):
            latest_user_image_idx = idx

    if latest_user_image_idx < 0:
        return messages

    preserved_system = [m for m in messages[:latest_user_image_idx] if m.get("role") == "system"]
    return preserved_system + messages[latest_user_image_idx:]


def _as_openai_tools_from_registry(registry) -> list[dict[str, Any]]:
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


def _build_geo_payload(req: ChatRequest) -> dict[str, Any]:
    if req.latitude is None or req.longitude is None:
        return {"observation_point": None}
    return {
        "observation_point": {
            "lat": float(req.latitude),
            "lon": float(req.longitude),
        }
    }


def _has_geo_inputs(req: ChatRequest) -> bool:
    return req.latitude is not None and req.longitude is not None


def _osm_ssl_context() -> ssl.SSLContext | None:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return None


@app.post("/v1/agent/analyze")
def agent_analyze(req: ChatRequest):
    from src.agent.default_server import create_default_server

    latest_user_has_image = _latest_user_has_image(req.messages)
    thresholds_path = str(PROJECT_ROOT / "configs" / "thresholds.json")
    server = create_default_server(thresholds_path, enable_seg_llm_second_pass=req.enable_seg_llm_second_pass)
    messages = [_to_dict_message(m) for m in req.messages]
    if latest_user_has_image:
        messages = _inject_forced_system_messages(messages, req)
    else:
        messages = _inject_followup_system_message(messages)
        messages = _inject_context_summary_message(messages, req)

    rpc_resp = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "chat",
            "params": {
                "messages": messages,
                "latitude": req.latitude,
                "longitude": req.longitude,
            },
        }
    )
    if "error" in rpc_resp:
        raise HTTPException(status_code=500, detail=f"agent chat failed: {rpc_resp['error']}")

    result = rpc_resp["result"]
    history = result.get("history", [])
    final_msg = result.get("message", {"role": "assistant", "content": ""})

    call_args_by_id: dict[str, Any] = {}
    for msg in history:
        if msg.get("role") != "assistant":
            continue
        for call in msg.get("tool_calls") or []:
            fn = call.get("function", {}) or {}
            raw_args = fn.get("arguments", "{}")
            try:
                parsed = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                parsed = {}
            call_args_by_id[call.get("id", "")] = parsed

    tool_trace: list[dict[str, Any]] = []
    image_path = _extract_latest_user_image_path(req.messages) if latest_user_has_image else ""
    seg_overlay_path = ""
    seg_mask_path = ""
    seg_refine_overlay_path = ""
    last_refine_output: dict[str, Any] | None = None

    for msg in history:
        if msg.get("role") != "tool":
            continue
        name = msg.get("name", "")
        raw_content = msg.get("content", "")
        try:
            output = json.loads(raw_content) if isinstance(raw_content, str) else raw_content
            status = "ok"
            error_text = ""
        except json.JSONDecodeError:
            output = {"raw_content": raw_content}
            status = "error"
            error_text = "invalid tool content json"

        trace_item = {
            "tool": name,
            "status": status,
            "input": call_args_by_id.get(msg.get("tool_call_id", ""), {}),
            "output": output,
        }
        if error_text:
            trace_item["error"] = error_text
        tool_trace.append(trace_item)

        if name == "tiff.info" and _is_successful_tool_output(output):
            image_path = output.get("image_path", image_path)
        if name == "seg.run" and _is_successful_tool_output(output):
            next_seg_overlay_path = str(output.get("overlay_path", "")).replace("\\", "/")
            next_seg_mask_path = str(output.get("mask_path", "")).replace("\\", "/")
            if next_seg_overlay_path:
                seg_overlay_path = next_seg_overlay_path
            if next_seg_mask_path:
                seg_mask_path = next_seg_mask_path
        if name == "seg.refine" and _is_successful_tool_output(output):
            last_refine_output = output
            next_refine_overlay_path = str(output.get("overlay_path", "")).replace("\\", "/")
            if next_refine_overlay_path:
                seg_refine_overlay_path = next_refine_overlay_path

    if not seg_refine_overlay_path:
        seg_refine_overlay_path = _ensure_seg_refine_overlay(
            image_path=image_path,
            refinement_output=last_refine_output,
            seg_mask_path=seg_mask_path,
        )

    now = int(time.time())
    final_resp = {
        "id": f"chatcmpl-{now}",
        "object": "chat.completion",
        "created": now,
        "model": LLM_API_MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": final_msg,
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "agent_trace": tool_trace,
        "artifacts": _build_artifacts_payload(
            include_images=bool(image_path),
            image_path=image_path,
            seg_overlay_path=seg_overlay_path,
            seg_mask_path=seg_mask_path,
            seg_refine_overlay_path=seg_refine_overlay_path,
        ),
        "geo": _build_geo_payload(req),
        "history": history,
    }
    return final_resp


@app.post("/v1/agent/analyze_stream")
def agent_analyze_stream(req: ChatRequest):
    from src.agent.default_server import create_default_server

    def _stream():
        latest_user_has_image = _latest_user_has_image(req.messages)
        thresholds_path = str(PROJECT_ROOT / "configs" / "thresholds.json")
        server = create_default_server(thresholds_path, enable_seg_llm_second_pass=req.enable_seg_llm_second_pass)
        messages = [_to_dict_message(m) for m in req.messages]
        if latest_user_has_image:
            messages = _inject_forced_system_messages(messages, req)
        else:
            messages = _inject_followup_system_message(messages)
            messages = _inject_context_summary_message(messages, req)
        tools = _as_openai_tools_from_registry(server.registry)
        report_write_required = _should_require_report_write(req.messages)
        history: list[dict[str, Any]] = list(messages)
        tool_trace: list[dict[str, Any]] = []
        image_path = _extract_latest_user_image_path(req.messages) if latest_user_has_image else ""
        latest_user_has_image = _latest_user_has_image(req.messages)
        seg_overlay_path = ""
        seg_mask_path = ""
        seg_refine_overlay_path = ""
        last_refine_output: dict[str, Any] | None = None
        tool_state: dict[str, Any] = {
            "outputs": {},
            "fuse_called": 0,
            "report_written": 0,
            "call_counts": {},
        }
        report_guard_retries = 0

        def _guarded_call_tool(name: str, raw_args: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            args = dict(raw_args or {})
            outputs = tool_state["outputs"]
            call_counts = tool_state["call_counts"]
            current_calls = int(call_counts.get(name, 0))

            if name == "tiff.info":
                current_image_path = str(image_path or _extract_latest_user_image_path(req.messages) or "").strip()
                tool_image_path = str(args.get("image_path", "") or "").strip()
                if not _is_existing_file(tool_image_path):
                    if _is_existing_file(current_image_path):
                        args["image_path"] = current_image_path
                    elif tool_image_path and Path(tool_image_path).exists() and Path(tool_image_path).is_dir():
                        raise ValueError(f"tiff.info requires an image file path, got directory: {tool_image_path}")
                    else:
                        raise ValueError(
                            "tiff.info requires a valid image_path file; no usable image path was found in tool arguments or latest user image."
                        )
            elif name in ("geo.nearby", "geo.background"):
                if ("lat" not in args or "lon" not in args) and req.latitude is not None and req.longitude is not None:
                    args["lat"] = float(req.latitude)
                    args["lon"] = float(req.longitude)
                    if name == "geo.nearby" and "radius" not in args:
                        args["radius"] = 300
                if "lat" not in args or "lon" not in args:
                    result = (
                        _unavailable_geo_nearby()
                        if name == "geo.nearby"
                        else _unavailable_geo_background()
                    )
                    outputs[name] = result
                    call_counts[name] = current_calls + 1
                    return result, args
            elif name == "image.tile":
                if "image_info" not in args and "tiff.info" in outputs:
                    args["image_info"] = outputs["tiff.info"]
                if "image_info" not in args:
                    raise ValueError("image.tile requires image_info (call tiff.info first).")
            elif name in ("llm.first_pass", "seg.run"):
                if "tiff.info" not in outputs:
                    raise ValueError(f"{name} requires tiff.info first for initial cross-check.")
                args["image_info"] = outputs["tiff.info"]
            elif name == "cls.run":
                if "image_info" not in args and "tiff.info" in outputs:
                    args["image_info"] = outputs["tiff.info"]
                if "image_info" not in args:
                    raise ValueError("cls.run requires image_info.")
            elif name == "seg.refine":
                if "image_info" not in args and ("tiff.info" in outputs):
                    args["image_info"] = outputs["tiff.info"]
                if "segmentation" not in args and "seg.run" in outputs:
                    args["segmentation"] = outputs["seg.run"]
                if "image_info" not in args:
                    raise ValueError("seg.refine requires image_info (call tiff.info first).")
                if not isinstance(args.get("segmentation"), dict):
                    raise ValueError("seg.refine requires segmentation output (call seg.run first).")
                if "regions" in args and not all(_looks_like_region_item(d) for d in (args.get("regions") or [])):
                    args.pop("regions", None)
            elif name == "seg.llm_review":
                if "refinement" not in args and "seg.refine" in outputs:
                    args["refinement"] = outputs["seg.refine"]
                if "refinement" not in args and "tiff.info" in outputs and "seg.run" in outputs:
                    args["refinement"] = server.registry.call_tool(
                        "seg.refine",
                        {"image_info": outputs["tiff.info"], "segmentation": outputs["seg.run"]},
                    )
                if "refinement" not in args:
                    raise ValueError("seg.llm_review requires segmentation-guided refinement context.")
                if "stage1" not in args and "llm.first_pass" in outputs:
                    args["stage1"] = outputs["llm.first_pass"]
                if "image_info" not in args and "tiff.info" in outputs:
                    args["image_info"] = outputs["tiff.info"]
                if "image_info" not in args:
                    raise ValueError("seg.llm_review requires image_info (call tiff.info first).")
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
                if "refinement" not in args and "tiff.info" in outputs and "segmentation" in args:
                    args["refinement"] = server.registry.call_tool(
                        "seg.refine",
                        {"image_info": outputs["tiff.info"], "segmentation": args["segmentation"]},
                    )
                if "segmentation" not in args and isinstance(args.get("refinement"), dict):
                    derived_seg = args["refinement"].get("segmentation")
                    if isinstance(derived_seg, dict):
                        args["segmentation"] = derived_seg
                mandatory_review_threshold = _mandatory_seg_llm_review_area_ratio()
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
                    args["llm_second_pass"] = (outputs.get("seg.llm_review") or {}).get("llm_second_pass")
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
                    current_image_path = ""
                    if isinstance(outputs.get("tiff.info"), dict):
                        current_image_path = str(outputs["tiff.info"].get("image_path", "") or "").strip()
                    if not current_image_path:
                        current_image_path = str(image_path or _extract_latest_user_image_path(req.messages) or "").strip()
                    args["out_path"] = _default_report_out_path(current_image_path)
                if not isinstance(args.get("report"), dict):
                    raise ValueError("report.write requires a report object; call fuse.decision first.")
                if tool_state["report_written"] >= 1:
                    raise ValueError("report.write should only be called once.")

            result = server.registry.call_tool(name, args)
            outputs[name] = result
            call_counts[name] = current_calls + 1
            if name == "fuse.decision":
                tool_state["fuse_called"] += 1
            if name == "report.write":
                tool_state["report_written"] += 1
            return result, args

        try:
            while True:
                chat_req = ChatRequest(
                    messages=[ChatMessage(**m) for m in history],
                    temperature=req.temperature,
                    max_tokens=req.max_tokens,
                    tools=tools,
                    tool_choice="auto",
                    latitude=req.latitude,
                    longitude=req.longitude,
                    context_summary=req.context_summary,
                    enable_seg_llm_second_pass=req.enable_seg_llm_second_pass,
                )
                llm_resp = chat_completions(chat_req)
                assistant_msg = llm_resp["choices"][0]["message"]
                tool_calls = assistant_msg.get("tool_calls") or []
                stream_content = assistant_msg.get("content", "")
                raw_model_output = str(llm_resp.get("raw_response", "") or "")
                if not tool_calls:
                    planned_msg = _mock_tool_response(chat_req)
                    planned_calls = (planned_msg or {}).get("tool_calls") if isinstance(planned_msg, dict) else None
                    if planned_calls:
                        assistant_msg = planned_msg
                        tool_calls = planned_calls
                        stream_content = str(planned_msg.get("content", "") or "")
                if (
                    not tool_calls
                    and report_write_required
                    and tool_state["report_written"] < 1
                ):
                    report_guard_retries += 1
                    if report_guard_retries > 2:
                        yield json.dumps(
                            {
                                "type": "error",
                                "error": "Mandatory finalization failed: report.write was not completed.",
                            },
                            ensure_ascii=False,
                        ) + "\n"
                        return
                    fuse_error_text = _latest_fuse_tool_error_text(tool_trace)
                    fuse_retry_instruction = str(
                        _fuse_retry_system_instruction_from_error(fuse_error_text) or ""
                    ).strip()
                    history.append(
                        {
                            "role": "system",
                            "content": (
                                "Do not finish yet. Mandatory finalization is incomplete: "
                                "call fuse.decision if needed, then call report.write with a valid local out_path."
                                + ((" " + fuse_retry_instruction) if fuse_retry_instruction else "")
                            ).strip(),
                        }
                    )
                    yield json.dumps(
                        {
                            "type": "assistant",
                            "content": "Waiting for mandatory report.write completion.",
                            "tool_calls": [],
                        },
                        ensure_ascii=False,
                    ) + "\n"
                    continue
                if not tool_calls:
                    fuse_output = tool_state["outputs"].get("fuse.decision")
                    structured_final_text = ""
                    if isinstance(fuse_output, dict):
                        structured_final_text = str(fuse_output.get("final_description", "") or "").strip()
                    if structured_final_text:
                        assistant_msg = {"role": "assistant", "content": structured_final_text}
                        stream_content = structured_final_text
                history.append(assistant_msg)
                if raw_model_output.strip():
                    yield json.dumps(
                        {
                            "type": "model_raw",
                            "content": raw_model_output,
                        },
                        ensure_ascii=False,
                    ) + "\n"
                yield json.dumps(
                    {
                        "type": "assistant",
                        "content": stream_content,
                        "tool_calls": tool_calls,
                    },
                    ensure_ascii=False,
                ) + "\n"

                if not tool_calls:
                    if not seg_refine_overlay_path:
                        seg_refine_overlay_path = _ensure_seg_refine_overlay(
                            image_path=image_path,
                            refinement_output=last_refine_output,
                            seg_mask_path=seg_mask_path,
                        )
                    now = int(time.time())
                    final_resp = {
                        "id": f"chatcmpl-{now}",
                        "object": "chat.completion",
                        "created": now,
                        "model": LLM_API_MODEL_NAME,
                        "choices": [{"index": 0, "message": assistant_msg, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                        "agent_trace": tool_trace,
                        "artifacts": _build_artifacts_payload(
                            include_images=bool(image_path),
                            image_path=image_path,
                            seg_overlay_path=seg_overlay_path,
                            seg_mask_path=seg_mask_path,
                            seg_refine_overlay_path=seg_refine_overlay_path,
                        ),
                        "geo": _build_geo_payload(req),
                        "history": history,
                    }
                    yield json.dumps({"type": "final", "data": final_resp}, ensure_ascii=False) + "\n"
                    return

                for call in tool_calls:
                    fn = call.get("function", {}) or {}
                    name = fn.get("name", "")
                    raw_args = fn.get("arguments", "{}")
                    try:
                        parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except json.JSONDecodeError:
                        parsed_args = {}
                    t0 = time.time()
                    captured_model_events: list[dict[str, str]] = []
                    current_model_events: list[dict[str, str]] = []
                    try:
                        with capture_model_raw_events() as current_model_events:
                            output, used_args = _guarded_call_tool(name, parsed_args)
                        captured_model_events = list(current_model_events)
                        status = "ok"
                    except Exception as exc:
                        captured_model_events = list(current_model_events)
                        output = {"error": str(exc)}
                        used_args = parsed_args
                        status = "error"
                    retry_system_instruction = ""
                    if name == "fuse.decision" and status == "error":
                        error_text = str((output or {}).get("error", "")) if isinstance(output, dict) else ""
                        retry_system_instruction = str(_fuse_retry_system_instruction_from_error(error_text) or "")
                    yield json.dumps(
                        {"type": "tool_call", "name": name, "arguments": used_args},
                        ensure_ascii=False,
                    ) + "\n"
                    cost_ms = int((time.time() - t0) * 1000)
                    trace_item = {
                        "tool": name,
                        "status": status,
                        "cost_ms": cost_ms,
                        "input": used_args,
                        "output": output,
                    }
                    for model_event in captured_model_events:
                        raw_content = str((model_event or {}).get("content", "") or "").strip()
                        if not raw_content:
                            continue
                        yield json.dumps(
                            {
                                "type": "model_raw",
                                "source": str((model_event or {}).get("source", "") or f"tool.{name}"),
                                "content": raw_content,
                            },
                            ensure_ascii=False,
                        ) + "\n"
                    trace_item["summary"] = _deterministic_tool_summary(name, status, output)
                    tool_trace.append(trace_item)
                    if name == "tiff.info" and status == "ok" and _is_successful_tool_output(output):
                        image_path = output.get("image_path", image_path)
                    if name == "seg.run" and status == "ok" and _is_successful_tool_output(output):
                        next_seg_overlay_path = str(output.get("overlay_path", "")).replace("\\", "/")
                        next_seg_mask_path = str(output.get("mask_path", "")).replace("\\", "/")
                        if next_seg_overlay_path:
                            seg_overlay_path = next_seg_overlay_path
                        if next_seg_mask_path:
                            seg_mask_path = next_seg_mask_path
                    if name == "seg.refine" and status == "ok" and _is_successful_tool_output(output):
                        last_refine_output = output
                        next_refine_overlay_path = str(output.get("overlay_path", "")).replace("\\", "/")
                        if next_refine_overlay_path:
                            seg_refine_overlay_path = next_refine_overlay_path

                    history.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.get("id", ""),
                            "name": name,
                            "content": json.dumps(output, ensure_ascii=False),
                        }
                    )
                    if retry_system_instruction:
                        history.append(
                            {
                                "role": "system",
                                "content": retry_system_instruction,
                            }
                        )
                    yield json.dumps({"type": "tool_result", "data": trace_item}, ensure_ascii=False) + "\n"

                    if name == "report.write" and status == "ok":
                        if not seg_refine_overlay_path:
                            seg_refine_overlay_path = _ensure_seg_refine_overlay(
                                image_path=image_path,
                                refinement_output=last_refine_output,
                                seg_mask_path=seg_mask_path,
                            )
                        fuse_output = tool_state["outputs"].get("fuse.decision")
                        structured_final_text = ""
                        if isinstance(fuse_output, dict):
                            structured_final_text = str(fuse_output.get("final_description", "") or "").strip()
                        final_content = structured_final_text or "Report has been written successfully."
                        now = int(time.time())
                        final_resp = {
                            "id": f"chatcmpl-{now}",
                            "object": "chat.completion",
                            "created": now,
                            "model": LLM_API_MODEL_NAME,
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": final_content,
                                    },
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                            "agent_trace": tool_trace,
                            "artifacts": _build_artifacts_payload(
                                include_images=bool(image_path),
                                image_path=image_path,
                                seg_overlay_path=seg_overlay_path,
                                seg_mask_path=seg_mask_path,
                                seg_refine_overlay_path=seg_refine_overlay_path,
                            ),
                            "geo": _build_geo_payload(req),
                            "history": history,
                        }
                        yield json.dumps({"type": "final", "data": final_resp}, ensure_ascii=False) + "\n"
                        return
        except Exception as exc:
            logging.exception("agent_analyze_stream failed")
            yield json.dumps({"type": "error", "error": str(exc)}, ensure_ascii=False) + "\n"

    return StreamingResponse(_stream(), media_type="application/x-ndjson")


@app.get("/v1/geo/nearby")
def geo_nearby(
    lat: float = Query(..., description="observation latitude"),
    lon: float = Query(..., description="observation longitude"),
    radius: int = Query(300, ge=100, le=10000, description="search radius in meters"),
):
    return query_osm_nearby_safe(lat, lon, radius)


@app.get("/admin/services")
def admin_services():
    seg_url = os.getenv("SEG_SERVICE_URL", "http://127.0.0.1:8002")
    cls_url = os.getenv("CLS_SERVICE_URL", "http://127.0.0.1:8004")
    llm_url = os.getenv("LLM_SERVICE_URL", "http://127.0.0.1:8003")
    llm_service_online = _health_ok(f"{llm_url.rstrip('/')}/health")
    seg_service_online = _health_ok(f"{seg_url.rstrip('/')}/health")
    cls_service_online = _health_ok(f"{cls_url.rstrip('/')}/health")
    cls_fallback_ready = _cls_fallback_ready()
    cls_available = cls_service_online or cls_fallback_ready
    return {
        "llm": llm_service_online,
        "seg": seg_service_online,
        # cls.run supports local fallback (cls_cli_predict) when cls service is unreachable.
        "cls": cls_available,
        "llm_service_online": llm_service_online,
        "seg_service_online": seg_service_online,
        "cls_service_online": cls_service_online,
        "cls_fallback_ready": cls_fallback_ready,
        "cls_available": cls_available,
    }


@app.post("/admin/start_services")
def admin_start_services(req: StartServicesRequest):
    seg_url = os.getenv("SEG_SERVICE_URL", "http://127.0.0.1:8002")
    cls_url = os.getenv("CLS_SERVICE_URL", "http://127.0.0.1:8004")
    seg_python = os.getenv("SEG_ENV_PYTHON", "python")
    cls_python = os.getenv("CLS_ENV_PYTHON", "python")

    result = {"llm": "already_running", "seg": "skipped", "cls": "skipped"}
    if req.start_seg:
        result["seg"] = _start_if_needed(
            service_url=seg_url,
            python_path=seg_python,
            app_target="scripts.seg_service:app",
            port=8002,
            log_name="seg_service.log",
        )
    if req.start_cls:
        result["cls"] = _start_if_needed(
            service_url=cls_url,
            python_path=cls_python,
            app_target="scripts.cls_service:app",
            port=8004,
            log_name="cls_service.log",
        )
    return result


@app.post("/admin/stop_services")
def admin_stop_services(req: StopServicesRequest):
    result = {"seg": "skipped", "cls": "skipped", "llm": "skipped"}
    if req.stop_seg:
        rc = subprocess.run(["pkill", "-f", "uvicorn scripts.seg_service:app"], check=False).returncode
        result["seg"] = "stopped_or_not_running" if rc in (0, 1) else f"pkill_error:{rc}"
    if req.stop_cls:
        rc = subprocess.run(["pkill", "-f", "uvicorn scripts.cls_service:app"], check=False).returncode
        result["cls"] = "stopped_or_not_running" if rc in (0, 1) else f"pkill_error:{rc}"
    if req.stop_llm:
        result["llm"] = "stopping"

        def _delayed_exit():
            time.sleep(0.8)
            os._exit(0)

        threading.Thread(target=_delayed_exit, daemon=True).start()

    return result

@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    if not model and not mock_mode:
        state = _start_model_loader_if_needed()
        if state == "loading":
            raise HTTPException(status_code=503, detail="Model is loading, please retry in a few seconds.")
        if state == "error":
            with model_state_lock:
                err = model_error
            raise HTTPException(status_code=503, detail=f"Model load failed: {err or 'unknown error'}")
        raise HTTPException(status_code=503, detail="Model not loaded")
    logging.info(f"Processing chat request with {len(req.messages)} messages")
    
    if mock_mode:
        tool_resp = _mock_tool_response(req)
        if tool_resp is not None:
            now = int(time.time())
            mock_raw_response = ""
            if tool_resp.get("tool_calls"):
                mock_raw_response = json.dumps({"tool_calls": tool_resp.get("tool_calls", [])}, ensure_ascii=False)
            else:
                mock_raw_response = str(tool_resp.get("content", "") or "")
            return {
                "id": f"chatcmpl-{now}",
                "object": "chat.completion",
                "created": now,
                "model": LLM_API_MODEL_NAME,
                "choices": [
                    {
                        "index": 0,
                        "message": tool_resp,
                        "finish_reason": "tool_calls" if tool_resp.get("tool_calls") else "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 10, "total_tokens": 10},
                "raw_response": mock_raw_response,
            }
        content = "This is a response from Qwen3-VL (mock mode). Your request has been received."
        now = int(time.time())
        return {
            "id": f"chatcmpl-{now}",
            "object": "chat.completion",
            "created": now,
            "model": LLM_API_MODEL_NAME,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 10, "total_tokens": 10},
                "raw_response": content,
            }

    # Real model inference logic
    effective_messages = _inject_tool_catalog_system_message(req.messages, req.tools)
    normalized = []
    images = []
    for msg in effective_messages:
        content = msg.content
        if isinstance(content, list):
            text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
            image_parts = [p for p in content if isinstance(p, dict) and p.get("type") == "image"]
            content_items = []
            for part in image_parts:
                image_path = part.get("image_path") or part.get("image")
                if image_path and os.path.exists(image_path):
                    image = Image.open(image_path).convert("RGB")
                    images.append(image)
                    content_items.append({"type": "image", "image": image})
                    # Keep explicit file path in textual context so the model can pass exact path to tools.
                    content_items.append({"type": "text", "text": f"[image_path]{image_path}[/image_path]"})
            for text in text_parts:
                if text:
                    content_items.append({"type": "text", "text": text})
            normalized.append({"role": msg.role, "content": content_items})
        else:
            normalized.append({"role": msg.role, "content": content})

    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    if req.tools:
        template_kwargs["tools"] = req.tools
    try:
        prompt = processor.apply_chat_template(normalized, **template_kwargs)
    except TypeError as exc:
        # Backward compatibility for processor versions without tools support.
        # Tool catalog is already injected as system text so tool descriptions remain visible.
        logging.warning("apply_chat_template(tools=...) unsupported, fallback to text-only template: %s", exc)
        prompt = processor.apply_chat_template(
            normalized,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    inputs = processor(text=prompt, images=images if images else None, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    generation_kwargs: dict[str, Any] = {
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "max_new_tokens": _resolve_max_new_tokens(req),
        "repetition_penalty": 1.12,
    }
    if req.tools:
        # Tool mode needs stable, non-rambling outputs more than creativity.
        generation_kwargs["do_sample"] = False
    else:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = max(req.temperature, 0.01)
        generation_kwargs["top_p"] = 0.8
        generation_kwargs["top_k"] = 20

    import torch
    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)
    
    input_ids = inputs["input_ids"]
    response_ids = output_ids[0][input_ids.shape[1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)

    content, tool_calls = _extract_tool_calls(response)
    content = _dedupe_repeated_lines(content)
    finish_reason = "tool_calls" if tool_calls else "stop"
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    now = int(time.time())
    return {
        "id": f"chatcmpl-{now}",
        "object": "chat.completion",
        "created": now,
        "model": LLM_API_MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": len(response_ids), "total_tokens": len(response_ids)},
        "raw_response": response,
    }


def _extract_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    pattern = re.compile(r"<tool_call>\s*<function=([^>\n]+)>\s*(.*?)</function>\s*</tool_call>", flags=re.DOTALL)
    param_pattern = re.compile(r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>", flags=re.DOTALL)
    tool_calls: list[dict[str, Any]] = []
    matched_spans: list[tuple[int, int]] = []
    for idx, match in enumerate(pattern.finditer(text)):
        name = match.group(1).strip()
        body = match.group(2)
        if not name:
            continue
        matched_spans.append((match.start(), match.end()))
        arguments: dict[str, Any] = {}
        for param_match in param_pattern.finditer(body):
            param_name = param_match.group(1).strip()
            raw_value = param_match.group(2).strip()
            if not param_name:
                continue
            try:
                value = json.loads(raw_value)
            except json.JSONDecodeError:
                value = raw_value
            arguments[param_name] = value
        tool_calls.append(
            {
                "id": f"call_{uuid4().hex[:12]}_{idx}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        )
    clean_content = pattern.sub("", text)

    if not tool_calls:
        # Fallback: some model outputs use JSON-style function calls instead of <tool_call> tags.
        json_call_pattern = re.compile(
            r"\{\s*\"name\"\s*:\s*\"([^\"]+)\"\s*,\s*\"arguments\"\s*:\s*(\{[\s\S]*?)\}\s*\}",
            flags=re.DOTALL,
        )
        fallback_spans: list[tuple[int, int]] = []
        for idx, match in enumerate(json_call_pattern.finditer(text)):
            name = str(match.group(1) or "").strip()
            raw_args_block = str(match.group(2) or "").strip()
            if not name:
                continue

            parsed_args: dict[str, Any] = {}
            try:
                maybe_args = json.loads(raw_args_block + "}")
                if isinstance(maybe_args, dict):
                    parsed_args = maybe_args
            except Exception:
                # Tolerate truncated JSON, e.g. {"image_path":"/root/
                image_path_match = re.search(r'"image_path"\s*:\s*"([^"]*)', raw_args_block)
                if image_path_match:
                    parsed_args["image_path"] = image_path_match.group(1)
            tool_calls.append(
                {
                    "id": f"call_{uuid4().hex[:12]}_json_{idx}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(parsed_args, ensure_ascii=False),
                    },
                }
            )
            fallback_spans.append((match.start(), match.end()))

        if not tool_calls:
            # Last-resort fallback for highly truncated outputs:
            # {"name":"tiff.info","arguments":{"image_path":"/root/
            truncated_name = re.search(r'"name"\s*:\s*"([^"]+)"', text)
            truncated_image_path = re.search(r'"image_path"\s*:\s*"([^"]*)', text)
            if truncated_name:
                name = str(truncated_name.group(1) or "").strip()
                if name:
                    parsed_args: dict[str, Any] = {}
                    if truncated_image_path:
                        parsed_args["image_path"] = str(truncated_image_path.group(1) or "")
                    tool_calls.append(
                        {
                            "id": f"call_{uuid4().hex[:12]}_trunc",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(parsed_args, ensure_ascii=False),
                            },
                        }
                    )
                    clean_content = ""
            else:
                clean_content = text
        else:
            # Remove matched JSON call blocks from assistant text content.
            pieces: list[str] = []
            cursor = 0
            for start, end in fallback_spans:
                if start > cursor:
                    pieces.append(text[cursor:start])
                cursor = max(cursor, end)
            if cursor < len(text):
                pieces.append(text[cursor:])
            clean_content = "".join(pieces)

    clean_content = re.sub(r"<think>\s*</think>\s*", "", clean_content, flags=re.DOTALL).strip()
    return clean_content, tool_calls


def _dedupe_repeated_lines(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return text.strip()
    deduped: list[str] = []
    prev = None
    for line in lines:
        if line == prev:
            continue
        deduped.append(line)
        prev = line
    return "\n".join(deduped).strip()


def _mock_tool_response(req: ChatRequest) -> Optional[dict[str, Any]]:
    if not req.tools:
        return None

    if not _latest_user_has_image(req.messages):
        return None

    available = {
        t.get("function", {}).get("name")
        for t in req.tools
        if isinstance(t, dict) and isinstance(t.get("function"), dict)
    }
    executed: dict[str, Any] = {}
    attempted_counts: dict[str, int] = {}
    for msg in req.messages:
        if msg.role != "tool" or not msg.name:
            continue
        try:
            parsed = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
        except Exception:
            parsed = {}
        attempted_counts[msg.name] = int(attempted_counts.get(msg.name, 0)) + 1
        if _is_successful_tool_output(parsed):
            executed[msg.name] = parsed

    latest_image = _extract_latest_image_path(req.messages)

    def _call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": f"call_mock_{uuid4().hex[:10]}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
                }
            ],
        }

    def _under_attempt_limit(name: str, limit: int = 3) -> bool:
        return int(attempted_counts.get(name, 0)) < limit

    def _geo_arguments(include_radius: bool = False) -> dict[str, Any]:
        if req.latitude is None or req.longitude is None:
            return {}
        args = {"lat": float(req.latitude), "lon": float(req.longitude)}
        if include_radius:
            args["radius"] = 300
        return args

    stage1_output = executed.get("llm.first_pass") if isinstance(executed.get("llm.first_pass"), dict) else None
    segmentation_output = executed.get("seg.run") if isinstance(executed.get("seg.run"), dict) else None
    refinement_output = executed.get("seg.refine") if isinstance(executed.get("seg.refine"), dict) else None
    screening_positive = _cross_check_positive(stage1_output, segmentation_output)
    second_pass_area_ratio_limit = _resolve_seg_llm_second_pass_max_area_ratio()
    mandatory_review_threshold = _mandatory_seg_llm_review_area_ratio()
    refinement_area_ratio = _resolve_refinement_area_ratio(refinement_output, segmentation_output)
    review_is_mandatory = bool(
        screening_positive
        and isinstance(refinement_output, dict)
        and _region_count(refinement_output) > 0
        and refinement_area_ratio is not None
        and refinement_area_ratio < mandatory_review_threshold
    )
    review_is_requested = bool(
        screening_positive
        and req.enable_seg_llm_second_pass
        and isinstance(refinement_output, dict)
        and _region_count(refinement_output) > 0
        and (
            second_pass_area_ratio_limit is None
            or (
                refinement_area_ratio is not None
                and refinement_area_ratio <= second_pass_area_ratio_limit
            )
        )
    )
    should_run_review = review_is_mandatory or review_is_requested

    if "tiff.info" in available and "tiff.info" not in executed and latest_image and _under_attempt_limit("tiff.info"):
        return _call("tiff.info", {"image_path": latest_image})
    if "llm.first_pass" in available and "llm.first_pass" not in executed and "tiff.info" in executed and _under_attempt_limit("llm.first_pass"):
        return _call("llm.first_pass", {"image_info": executed["tiff.info"]})
    if "seg.run" in available and "seg.run" not in executed and "tiff.info" in executed and _under_attempt_limit("seg.run"):
        return _call("seg.run", {"image_info": executed["tiff.info"]})
    if "cls.run" in available and "cls.run" not in executed and "tiff.info" in executed and _under_attempt_limit("cls.run"):
        return _call("cls.run", {"image_info": executed["tiff.info"]})
    if "geo.nearby" in available and "geo.nearby" not in executed and _under_attempt_limit("geo.nearby"):
        return _call("geo.nearby", _geo_arguments(include_radius=True))
    if "geo.background" in available and "geo.background" not in executed and _under_attempt_limit("geo.background"):
        return _call("geo.background", _geo_arguments())
    if (
        "seg.refine" in available
        and "seg.refine" not in executed
        and "tiff.info" in executed
        and "seg.run" in executed
        and _under_attempt_limit("seg.refine")
    ):
        return _call(
            "seg.refine",
            {"image_info": executed["tiff.info"], "segmentation": executed["seg.run"]},
        )
    if (
        "seg.llm_review" in available
        and screening_positive
        and should_run_review
        and "seg.llm_review" not in executed
        and "tiff.info" in executed
        and "seg.run" in executed
        and _under_attempt_limit("seg.llm_review")
    ):
        if "seg.refine" in available and "seg.refine" not in executed and _under_attempt_limit("seg.refine"):
            return _call(
                "seg.refine",
                {"image_info": executed["tiff.info"], "segmentation": executed["seg.run"]},
            )
        if isinstance(refinement_output, dict):
            return _call(
                "seg.llm_review",
                {
                    "refinement": refinement_output,
                    "stage1": executed.get("llm.first_pass", {}),
                    "image_info": executed["tiff.info"],
                },
            )
    fuse_prereqs = ("llm.first_pass", "seg.run", "seg.refine", "cls.run", "geo.nearby", "geo.background")
    if (
        "fuse.decision" in available
        and screening_positive is not None
        and "fuse.decision" not in executed
        and all(k in executed for k in fuse_prereqs)
        and (not review_is_mandatory or "seg.llm_review" in executed)
        and _under_attempt_limit("fuse.decision", limit=4)
    ):
        args = {
            "stage1": executed["llm.first_pass"],
            "segmentation": executed["seg.run"],
            "refinement": executed["seg.refine"],
            "classification": executed["cls.run"],
            "geo_context": {
                "nearby": executed["geo.nearby"],
                "background": executed["geo.background"],
            },
        }
        if "seg.llm_review" in executed:
            args["llm_second_pass"] = (executed.get("seg.llm_review") or {}).get("llm_second_pass")
        return _call(
            "fuse.decision",
            args,
        )
    if "report.write" in available and "fuse.decision" in executed and "report.write" not in executed:
        return _call("report.write", {})

    summary = executed.get("fuse.decision") or {}
    final_text = summary.get("final_description") if isinstance(summary, dict) else ""
    if not final_text:
        final_text = "Mock agent completed with available tool outputs."
    return {"role": "assistant", "content": final_text}
