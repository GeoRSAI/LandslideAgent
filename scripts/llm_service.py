import importlib.util
from dataclasses import replace

from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
import os
import time
import subprocess
import threading
from contextlib import contextmanager
from PIL import Image
import json
import re
import ssl
from pathlib import Path
from urllib import request
from urllib import error as urlerror
from urllib.parse import quote
from uuid import uuid4

from src.agent.controller import get_policy, format_rule_violations
from src.tools.osm_tool import query_osm_nearby_safe
from src.models.llm_client import capture_model_raw_events

DEFAULT_LLM_API_MODEL_NAME = "qwen3-vl-8b-instruct"
DEFAULT_MODEL_PATH = "models/Qwen3-VL-8B-Instruct"
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

# Serializes ALL forward/generate calls against the single resident `model`,
# including LoRA adapter switches (dual-head classification <-> sft
# tool-calling). Reentrant so a locked call path can call another locked
# helper on the same thread (e.g. chat_completions -> _generate_continuation_text)
# without deadlocking. Must be held for the *entire* span during which a
# non-default adapter is active, so no other request's generate() call can
# observe the wrong adapter mid-sequence.
INFERENCE_LOCK = threading.RLock()

SFT_ADAPTER_NAME = "default"
DUAL_HEAD_ADAPTER_NAME = "dualhead"
dual_head_available = False
dual_head_labels: list[str] = []
classification_head = None  # torch.nn.Module, set once dual-head assets load

MODEL_PATH = os.getenv("LLM_MODEL_PATH", DEFAULT_MODEL_PATH)
LORA_PATH = os.getenv("LLM_LORA_PATH", "")
DUAL_HEAD_PATH = os.getenv(
    "LLM_DUAL_HEAD_PATH",
    "models/landslide_qwen3vl_dual_head_continuous_10ep_best",
)
# The boot script enables the dual-head adapter for classification by default.
# Per-request adapter scopes still select base or dualhead for generation.
DUAL_HEAD_ADAPTER_ALWAYS_ON = os.getenv("LLM_DUAL_HEAD_ADAPTER_ALWAYS_ON", "0") in ("1", "true", "True")
DEFAULT_DUAL_HEAD_PROMPT = (
    "Analyze this satellite/aerial image for landslide evidence. Respond in English with exactly "
    "these three lines:\n"
    "Landslide presence: yes or no\n"
    "Classification: <one of No landslide, Debris flow, Mud flow, Mudslide, Earth slide, Earthflow, "
    "Rock fall, Rock slide>\n"
    "Landslide type: <same label as Classification>"
)
DUAL_HEAD_PROMPT = os.getenv("LLM_DUAL_HEAD_PROMPT", DEFAULT_DUAL_HEAD_PROMPT)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
(PROJECT_ROOT / "outputs").mkdir(parents=True, exist_ok=True)
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

def _resolve_seg_llm_second_pass_max_area_ratio() -> float | None:
    return get_policy().tiny_area_review_threshold

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
    return get_policy().fusion_argument_hint(missing)

def _fuse_decision_required_call_instruction() -> str:
    return get_policy().fusion_required_call_instruction()

def _seg_llm_review_required_call_instruction() -> str:
    return get_policy().second_pass_required_instruction()

def _report_write_enabled() -> bool:
    return os.getenv("AGENT_ENABLE_REPORT_WRITE", "1") in ("1", "true", "True")


def _analysis_workflow_instruction() -> str:
    return get_policy().workflow_instruction()

def _fuse_retry_system_instruction_from_error(error_text: str) -> str | None:
    return get_policy().retry_instruction_from_error(error_text)

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


def _select_tool_summary_text(raw_response: Any, content: Any) -> str:
    raw_text = str(raw_response or "").strip()
    if raw_text:
        return raw_text
    return str(content or "").strip()


def _summarize_tool_result_with_llm(
    tool_name: str,
    status: str,
    input_args: dict[str, Any],
    output: Any,
) -> tuple[str, str]:
    if mock_mode or model is None:
        return "", ""
    try:
        summary_req = ChatRequest(
            messages=[
                ChatMessage(
                    role="system",
                    content=(
                        "You write concise UI updates for a landslide-analysis operator. "
                        "Read one tool execution result and summarize it in exactly one short, complete English sentence. "
                        "The sentence must be grammatically complete and end with final punctuation. "
                        "Use only facts present in the tool output. "
                        "If the tool failed, state the main failure reason. "
                        "No markdown, no bullets, no JSON, and no extra commentary."
                    ),
                ),
                ChatMessage(
                    role="user",
                    content=(
                        f"Tool: {tool_name}\n"
                        f"Status: {status}\n"
                        f"Input: {_serialize_summary_payload(input_args)}\n"
                        f"Output: {_serialize_summary_payload(output)}\n"
                        "Return only the summary sentence."
                    ),
                ),
            ],
            temperature=0.1,
        )
        response = chat_completions(summary_req)
        raw_response = str(response.get("raw_response", "") or "").strip()
        message = ((response or {}).get("choices") or [{}])[0].get("message", {}) or {}
        content = str(message.get("content", "") or "").strip()
        summary_text = _select_tool_summary_text(raw_response, content)
        return summary_text, raw_response
    except Exception:
        logging.exception("tool result summarization failed for %s", tool_name)
        return "", ""


def _validate_fuse_required_arguments(args: dict[str, Any]) -> None:
    policy = get_policy()
    missing = policy.missing_fusion_requirements(args)
    if missing:
        raise ValueError(policy.fusion_argument_hint(missing))

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
        "dual_head_path": DUAL_HEAD_PATH if dual_head_available else "",
        "dual_head_loaded": dual_head_available,
        "dual_head_labels": dual_head_labels if dual_head_available else [],
        "dual_head_adapter_always_on": DUAL_HEAD_ADAPTER_ALWAYS_ON,
    }

class DualHeadClassifyRequest(BaseModel):
    image_path: str
    topk: int = 5


def _dual_head_classify_impl(image_path: str, topk: int = 5) -> dict[str, Any]:
    """Run the dual-head classification head once: switch to the dual-head
    LoRA, one forward pass (no generate(), no sampling -- classification-only
    so it cannot leak into any generated tokens), read the head off the last
    prompt-token hidden state, switch back to the sft adapter. Must be called
    with INFERENCE_LOCK held so no concurrent request observes a mid-switch
    adapter state or interleaves a generate() call under the wrong adapter."""
    import torch
    from peft import PeftModel

    image_path = str(image_path or "").strip()
    if not image_path:
        raise ValueError("image_path is required")
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)
    if not (dual_head_available and classification_head is not None):
        raise RuntimeError("dual-head classifier is not loaded")

    image = Image.open(image_path).convert("RGB")
    prompt_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": DUAL_HEAD_PROMPT},
            ],
        }
    ]
    prompt_text = processor.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=prompt_text, images=[image], return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    is_peft = isinstance(model, PeftModel)
    prev_adapter = None
    if is_peft:
        try:
            prev_adapter = model.active_adapters()[0]
        except Exception:
            prev_adapter = getattr(model, "active_adapter", None)
    try:
        if is_peft and not DUAL_HEAD_ADAPTER_ALWAYS_ON:
            model.set_adapter(DUAL_HEAD_ADAPTER_NAME)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        hidden = outputs.hidden_states[-1][:, -1, :].float()
        logits = classification_head(hidden)
        probabilities = torch.softmax(logits.float(), dim=-1)[0].detach().cpu().tolist()
    finally:
        if is_peft and prev_adapter and not DUAL_HEAD_ADAPTER_ALWAYS_ON:
            model.set_adapter(prev_adapter)

    order = sorted(range(len(dual_head_labels)), key=lambda i: probabilities[i], reverse=True)
    class_id = int(order[0])
    class_name = dual_head_labels[class_id]
    confidence = float(probabilities[class_id])
    topk = max(1, min(int(topk or 5), len(order)))
    topk_list = [
        {"class_id": int(i), "class_name": dual_head_labels[i], "score": float(probabilities[i])}
        for i in order[:topk]
    ]
    no_landslide_prob = 0.0
    if "No landslide" in dual_head_labels:
        no_landslide_prob = float(probabilities[dual_head_labels.index("No landslide")])
    return {
        "class_id": class_id,
        "class_name": class_name,
        "confidence": confidence,
        "topk": topk_list,
        "num_classes": len(dual_head_labels),
        "has_landslide": class_name != "No landslide",
        "score": 1.0 - no_landslide_prob,
    }


@contextmanager
def _request_adapter_scope(adapter_mode: str):
    """Select an adapter only for one serialized generation request.

    ``base`` disables all LoRA deltas, keeping agent planning/parameters on
    the original Qwen. ``dualhead`` is used only for an explicit visual
    evidence generation pass. The caller already holds ``INFERENCE_LOCK``.
    """
    try:
        from peft import PeftModel
    except Exception:  # pragma: no cover - base-model fallback
        yield
        return
    if not isinstance(model, PeftModel):
        yield
        return

    mode = str(adapter_mode or "base").strip().lower()
    if mode == "base":
        with model.disable_adapter():
            yield
        return
    if mode == "dualhead":
        if not dual_head_available:
            raise RuntimeError("dualhead adapter requested but is not loaded")
        try:
            previous = model.active_adapters()[0]
        except Exception:
            previous = getattr(model, "active_adapter", None)
        try:
            model.set_adapter(DUAL_HEAD_ADAPTER_NAME)
            yield
        finally:
            if previous:
                model.set_adapter(previous)
        return
    if mode == "sft":
        # The description LoRA fine-tuned on the 10-field landslide answer
        # format (presence, type, position, morphology, material, movement,
        # environment, impact, reason, causation).
        if not lora_loaded:
            raise RuntimeError("sft adapter requested but is not loaded")
        try:
            previous = model.active_adapters()[0]
        except Exception:
            previous = getattr(model, "active_adapter", None)
        try:
            model.set_adapter(SFT_ADAPTER_NAME)
            yield
        finally:
            if previous:
                model.set_adapter(previous)
        return
    raise ValueError(f"unsupported adapter_mode: {adapter_mode!r}")


@app.post("/v1/dual_head_classify")
def dual_head_classify_endpoint(req: DualHeadClassifyRequest):
    if mock_mode:
        return {
            "class_id": 1,
            "class_name": "Debris flow",
            "confidence": 0.5,
            "topk": [{"class_id": 1, "class_name": "Debris flow", "score": 0.5}],
            "num_classes": 8,
            "has_landslide": True,
            "score": 0.5,
        }
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not dual_head_available:
        raise HTTPException(status_code=503, detail="Dual-head classifier not available")
    try:
        with INFERENCE_LOCK:
            return _dual_head_classify_impl(req.image_path, req.topk)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"image not found: {exc}")
    except Exception as exc:
        logging.exception("dual-head classification failed")
        raise HTTPException(status_code=500, detail=str(exc))


app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "static")), name="static")
app.mount("/outputs", StaticFiles(directory=str(PROJECT_ROOT / "outputs")), name="outputs")


def _health_ok(url: str) -> bool:
    try:
        with request.urlopen(url, timeout=2.0) as resp:
            return resp.status == 200
    except Exception:
        return False


def _cls_fallback_ready() -> bool:
    cls_env_python = os.getenv("CLS_ENV_PYTHON", __import__("sys").executable)
    mmpretrain_root = os.getenv("MMPRETRAIN_ROOT", "models/mmpretrain-main")
    config_path = os.getenv(
        "CLS_CONFIG_PATH",
        "models/mmpretrain-main/work_dirs/_debug_convnext_start/convnext-tiny_1xb16_landslide-50e.py",
    )
    checkpoint_path = os.getenv(
        "CLS_CHECKPOINT_PATH",
        "models/mmpretrain-main/work_dirs/_debug_convnext_start/best_accuracy_top1_epoch_45.pth",
    )
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
    allowed_root = Path(os.getenv("IMAGE_ALLOWED_ROOT", str(PROJECT_ROOT))).resolve()
    if not p.is_relative_to(allowed_root):
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


def _artifact_paths_from_outputs(outputs: dict[str, Any] | None) -> tuple[str, str, str, str]:
    """Recover artifact paths from the complete agent ledger, including retry loops."""
    outputs = outputs if isinstance(outputs, dict) else {}
    image_path = ""
    seg_overlay = ""
    seg_mask = ""
    refine_overlay = ""
    tiff = outputs.get("tiff.info")
    if isinstance(tiff, dict):
        image_path = str(tiff.get("image_path", "") or "")
    seg = outputs.get("seg.run")
    if isinstance(seg, dict):
        seg_overlay = str(seg.get("overlay_path", "") or "")
        seg_mask = str(seg.get("mask_path", "") or "")
    refine = outputs.get("seg.refine")
    if isinstance(refine, dict):
        refine_overlay = str(refine.get("overlay_path", "") or "")
        if not seg_mask:
            seg_mask = str(refine.get("mask_path", "") or "")
    return image_path, seg_overlay, seg_mask, refine_overlay


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


class _DualHeadClassificationHead:
    """Lazily built so torch.nn is only imported once the model is loading."""

    @staticmethod
    def build(input_size: int, num_labels: int, hidden_size: int, dropout: float):
        import torch.nn as nn

        class ClassificationHead(nn.Module):
            """Mirrors modeling_dual_head.ClassificationHead from the dual-head
            checkpoint: LayerNorm -> Linear -> GELU -> Dropout -> Linear, run on
            the last hidden state of the prompt (read-only; does not feed back
            into generation)."""

            def __init__(self) -> None:
                super().__init__()
                self.norm = nn.LayerNorm(input_size)
                self.dense = nn.Linear(input_size, hidden_size)
                self.activation = nn.GELU()
                self.dropout = nn.Dropout(dropout)
                self.out_proj = nn.Linear(hidden_size, num_labels)

            def forward(self, hidden):
                hidden = self.norm(hidden.float())
                hidden = self.dense(hidden)
                hidden = self.activation(hidden)
                hidden = self.dropout(hidden)
                return self.out_proj(hidden)

        return ClassificationHead()


def _load_dual_head_impl(base_model) -> Any:
    """Load the dual-head adapter and classification head on the resident model."""
    global dual_head_available, dual_head_labels, classification_head
    dual_head_available = False
    dual_head_labels = []
    classification_head = None

    if not DUAL_HEAD_PATH or not os.path.isdir(DUAL_HEAD_PATH):
        logging.info("Dual-head path not configured/found (%s); skipping.", DUAL_HEAD_PATH)
        return base_model

    checkpoint = Path(DUAL_HEAD_PATH)
    config_path = checkpoint / "dual_head_config.json"
    adapter_dir = checkpoint / "adapter"
    head_path = checkpoint / "classification_head.pt"
    if not (config_path.exists() and adapter_dir.exists() and head_path.exists()):
        logging.warning("Dual-head assets incomplete under %s; skipping.", DUAL_HEAD_PATH)
        return base_model

    try:
        import torch
        from peft import PeftModel

        metadata = json.loads(config_path.read_text(encoding="utf-8"))
        labels = list(metadata.get("labels") or [])
        if not labels:
            logging.warning("Dual-head config missing labels; skipping.")
            return base_model

        lora_ok, lora_reason = _lora_matches_model(str(adapter_dir), MODEL_PATH)
        if not lora_ok:
            logging.warning("Skipping dual-head adapter because %s", lora_reason)
            return base_model

        logging.info("Loading dual-head LoRA adapter from %s...", adapter_dir)
        if isinstance(base_model, PeftModel):
            base_model.load_adapter(str(adapter_dir), adapter_name=DUAL_HEAD_ADAPTER_NAME)
            if SFT_ADAPTER_NAME in getattr(base_model, "peft_config", {}):
                base_model.set_adapter(SFT_ADAPTER_NAME)
        else:
            base_model = PeftModel.from_pretrained(
                base_model, str(adapter_dir), adapter_name=DUAL_HEAD_ADAPTER_NAME, is_trainable=False
            )
        base_model = base_model.eval()

        head_config = metadata.get("classification_head", {}) or {}
        text_config = getattr(base_model.config, "text_config", None)
        hidden_size = int(getattr(text_config, "hidden_size", None) or getattr(base_model.config, "hidden_size"))
        head = _DualHeadClassificationHead.build(
            input_size=hidden_size,
            num_labels=len(labels),
            hidden_size=int(head_config.get("hidden_size", 2048)),
            dropout=float(head_config.get("dropout", 0.1)),
        )
        head.load_state_dict(torch.load(str(head_path), map_location="cpu", weights_only=True))
        head.to(next(base_model.parameters()).device)
        head = head.eval()

        classification_head = head
        dual_head_labels = labels
        dual_head_available = True
        logging.info("Dual-head classifier ready (%d classes).", len(labels))
    except Exception:
        logging.exception("Dual-head classifier load failed; continuing without it.")
        dual_head_available = False
        dual_head_labels = []
        classification_head = None
    return base_model


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
            model = PeftModel.from_pretrained(model, LORA_PATH, adapter_name=SFT_ADAPTER_NAME, is_trainable=False)
            model = model.eval()
            lora_loaded = True
            logging.info("LoRA adapter loaded.")
    elif LORA_PATH:
        logging.warning("Configured LoRA path does not exist: %s", LORA_PATH)

    model = _load_dual_head_impl(model)
    if DUAL_HEAD_ADAPTER_ALWAYS_ON and dual_head_available:
        from peft import PeftModel

        if isinstance(model, PeftModel):
            model.set_adapter(DUAL_HEAD_ADAPTER_NAME)
            logging.warning(
                "LLM_DUAL_HEAD_ADAPTER_ALWAYS_ON=1: dual-head LoRA stays active for tool-calling too. "
                "This adapter was not trained on the <tool_call> protocol; expect degraded/absent tool calls."
            )
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
    nearby_radius: Optional[int] = None
    review_threshold: Optional[float] = Field(default=None, gt=0, le=1)
    context_summary: Optional[str] = None
    # Full tool ledger is carried across UI pause/resume calls; it is kept separate from model messages.
    agent_trace: List[Dict[str, Any]] = Field(default_factory=list)
    # Model calls already spent by this image analysis before a UI pause.
    agent_turns_used: int = Field(default=0, ge=0)
    enable_seg_llm_second_pass: bool = False
    # Shared model-call budget for rule and free evaluation arms.
    max_turns: int = Field(default=30, ge=1)
    # "graph" -> deterministic LangGraph pipeline (default, back-compatible).
    # "agent" -> LLM-driven tool orchestration under the domain contract.
    # "free"  -> the same agent with the contract switched off (ablation arm).
    agent_mode: str = "graph"
    # ``base`` keeps all adapters disabled for ordinary agent planning and
    # report generation. ``dualhead`` is reserved for an explicitly requested
    # visual-evidence pass; it is restored before the next request.
    adapter_mode: str = "base"


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


_INCOMPLETE_RESPONSE_SUFFIXES = (
    "，",
    ",",
    "、",
    "：",
    ":",
    "；",
    ";",
    "的",
    "和",
    "或",
    "以及",
    "包括",
    "例如",
    "如下",
)


def _looks_like_incomplete_response(text: str) -> bool:
    value = str(text or "").rstrip()
    if not value:
        return False
    if re.search(r"(?:\*\*|__|`|\[|\(|\{|[，,、：:；;])$", value):
        return True
    if re.search(r"(?:^|\n)\s*(?:\d+[.)、]|[-*])\s*(?:\*\*)?$", value):
        return True
    if re.search(r"(?:列表|工具|包括|如下)\s*[：:]\s*\d*$", value):
        return True
    if value.endswith(_INCOMPLETE_RESPONSE_SUFFIXES):
        return True
    return bool(
        re.search(
            r"(?:\b(?:and|or|but|because|including|such as|with|to|of)\s*)$",
            value.lower(),
        )
    )


def _merge_response_continuation(previous: str, continuation: str) -> str:
    left = str(previous or "").rstrip()
    right = str(continuation or "").lstrip()
    if not right:
        return left
    if not left:
        return right
    if right.startswith(left):
        return right
    if left.endswith(right):
        return left

    max_overlap = min(len(left), len(right), 160)
    for overlap in range(max_overlap, 0, -1):
        if left[-overlap:] == right[:overlap]:
            return left + right[overlap:]
    return left + right


def _ensure_terminal_punctuation(text: str) -> str:
    value = str(text or "").rstrip()
    if not value or value.endswith((".", "。", "!", "！", "?", "？", ";", "；", ":", "：", ")", "）", "]", "】", "}", "》", "'", "\"")):
        return value
    if value.endswith(("...", "……")):
        return value
    if re.search(r"[\u4e00-\u9fff]$", value):
        return value + "。"
    if " " in value and re.search(r"[A-Za-z0-9]$", value):
        return value + "."
    return value


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
    if latest_image_path:
        if _agent_mode_enabled(req):
            policy = get_policy()
            if req.review_threshold is not None:
                policy = replace(policy, tiny_area_review_threshold=req.review_threshold)
            instruction = policy.autonomous_task_contract()
        else:
            instruction = _analysis_workflow_instruction() + " " + _fuse_decision_required_call_instruction()
        if req.review_threshold is not None:
            instruction += f" Review area-ratio threshold for this run: {req.review_threshold:.2f}."
        forced_system = {
            "role": "system",
            "content": (
                _immutable_run_inputs_message(
                    req,
                    image_path=latest_image_path,
                    nearby_radius=(req.nearby_radius if _agent_mode_enabled(req) else (req.nearby_radius or 300)),
                )
                + "\n\n"
                + instruction
            ),
        }
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = f"{messages[0].get('content', '')}\n\n{forced_system['content']}"
        else:
            messages = [forced_system] + messages
    return messages



_CURRENT_REPORT_TITLES = (
    "Landslide presence",
    "Landslide type",
    "Image relative position within the image frame",
    "Morphological characteristics",
    "Material composition and surface cover",
    "Movement and deformation features",
    "Surrounding environmental context",
    "Impact on human infrastructure",
    "Reason for landslide classification",
    "Landslide causation inference",
)
_FUSION_REPORT_TEXT_KEYS = {
    "final_description",
    "summary",
    "report",
    "report_text",
    "narrative",
    "recommendations",
    "visual_description",
    "spatial_distribution",
    "tool_interpretation",
    "uncertainty",
    "classification_reference_note",
    "second_pass_note",
    "whole_image_overview",
}


def _current_report_sections(text: str) -> list[tuple[str, str]]:
    raw = str(text or "")
    labels = "|".join(re.escape(title) for title in _CURRENT_REPORT_TITLES)
    marker = re.compile(
        rf"(?im)^[ \t]*\*{{0,2}}(?P<label>{labels}):\*{{0,2}}[ \t]*"
    )
    matches = list(marker.finditer(raw))
    if len(matches) < 7:
        return []

    found: dict[str, str] = {}
    titles = {title.casefold(): title for title in _CURRENT_REPORT_TITLES}
    for index, match in enumerate(matches):
        label = titles.get(str(match.group("label") or "").casefold())
        if not label:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
        body = raw[match.end():end]
        body = re.split(r"(?m)^[ \t]*#{1,6}[ \t]+", body, maxsplit=1)[0]
        body = " ".join(body.split())
        if body:
            found[label] = body
    return [(title, found[title]) for title in _CURRENT_REPORT_TITLES if title in found]


def _summarize_report_text_for_context(text: str, *, limit: int = 1800) -> str:
    sections = _current_report_sections(text)
    if not sections:
        return ""
    summary = "\n".join(f"**{title}:** {body}" for title, body in sections)
    return summary[:limit].rstrip() + ("..." if len(summary) > limit else "")


def _looks_like_legacy_report_text(text: str) -> bool:
    raw = str(text or "")
    if re.search(r"###\s*Final Decision Report", raw, flags=re.IGNORECASE):
        return True
    headings = re.findall(r"###\s+", raw)
    return len(headings) >= 6 and bool(
        re.search(r"###\s*Final Determination", raw, flags=re.IGNORECASE)
    )


def _looks_like_full_report_text(text: str) -> bool:
    return bool(_current_report_sections(text)) or _looks_like_legacy_report_text(text)


def _strip_legacy_report_values(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if str(key).strip().lower() in _FUSION_REPORT_TEXT_KEYS:
                continue
            cleaned[key] = _strip_legacy_report_values(item)
        return cleaned
    if isinstance(value, list):
        return [_strip_legacy_report_values(item) for item in value]
    if isinstance(value, str) and _looks_like_legacy_report_text(value):
        return "[previous report omitted from model context]"
    return value


def _compact_fuse_decision_message(message: dict[str, Any]) -> dict[str, Any]:
    raw = message.get("content", "")
    try:
        payload = raw if isinstance(raw, dict) else json.loads(str(raw or ""))
    except Exception:
        payload = None
    if not isinstance(payload, dict):
        if _looks_like_legacy_report_text(str(raw or "")):
            compacted = dict(message)
            compacted["content"] = "[previous report omitted from model context]"
            return compacted
        return message

    cleaned = _strip_legacy_report_values(payload)
    bounded = _compact_tool_output("fuse.decision", cleaned, string_limit=1200, array_limit=40)
    if len(json.dumps(bounded, ensure_ascii=False)) > 6000:
        bounded = _compact_tool_output("fuse.decision", cleaned, string_limit=300, array_limit=10, key_limit=40)
    compacted = dict(message)
    compacted["content"] = json.dumps(bounded, ensure_ascii=False, separators=(",", ":"))
    return compacted


def _compress_followup_report_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project session history onto current report fields and compact tool context."""
    compressed: list[dict[str, Any]] = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            compressed.append(msg)
            continue

        role = msg.get("role")
        content = str(msg.get("content", "") or "")
        if role == "assistant":
            summary = _summarize_report_text_for_context(content)
            if summary:
                next_msg = dict(msg)
                next_msg["content"] = f"[Previous structured-2 report summary]\n{summary}"
                compressed.append(next_msg)
            elif _looks_like_legacy_report_text(content):
                next_msg = dict(msg)
                next_msg["content"] = "[Previous report omitted from model context]"
                compressed.append(next_msg)
            else:
                compressed.append(msg)
        elif role == "tool" and msg.get("name") == "fuse.decision":
            compressed.append(_compact_fuse_decision_message(msg))
        elif role == "tool" and _looks_like_legacy_report_text(content):
            next_msg = dict(msg)
            next_msg["content"] = "[Previous report omitted from model context]"
            compressed.append(next_msg)
        elif role == "tool":
            try:
                payload = msg.get("content") if isinstance(msg.get("content"), dict) else json.loads(content)
            except Exception:
                payload = None
            if isinstance(payload, dict):
                tool_name = str(msg.get("name") or "")
                bounded = _compact_tool_output(tool_name, payload, string_limit=1200, array_limit=40)
                if len(json.dumps(bounded, ensure_ascii=False)) > 6000:
                    bounded = _compact_tool_output(tool_name, payload, string_limit=300, array_limit=10, key_limit=40)
                next_msg = dict(msg)
                next_msg["content"] = json.dumps(bounded, ensure_ascii=False, separators=(",", ":"))
                compressed.append(next_msg)
            else:
                compressed.append(msg)
        else:
            compressed.append(msg)
    return compressed

def _history_tool_trace(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recover tool evidence from older UI histories that lack a separate trace ledger."""
    trace: list[dict[str, Any]] = []
    for message in messages or []:
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        name = str(message.get("name") or "").strip()
        if not name:
            continue
        raw = message.get("content", "")
        if isinstance(raw, dict):
            output = raw
        else:
            text = str(raw or "").strip()
            try:
                output = json.loads(text)
            except Exception:
                output = None
                for marker in ("{", "["):
                    pos = text.find(marker)
                    if pos < 0:
                        continue
                    try:
                        output = json.JSONDecoder().raw_decode(text[pos:])[0]
                        break
                    except Exception:
                        pass
        if not isinstance(output, dict):
            continue
        verification = output.get("verification")
        refused = isinstance(verification, dict) and verification.get("status") == "refused"
        deferred = bool(output.get("not_executed"))
        output = {k: v for k, v in output.items() if k != "verification"}
        failed = bool(output.get("error"))
        execution = "deferred" if deferred else "refused" if refused else "failed" if failed else "completed"
        trace.append({
            "tool": name,
            "status": "error" if failed else "ok",
            "execution_state": execution,
            "cached": False,
            "cost_ms": 0,
            "input": {},
            "output": output,
        })
    return trace


def _compact_trace_value(value: Any, *, string_limit: int = 1200, array_limit: int = 40, key_limit: int = 80, depth: int = 0) -> Any:
    if isinstance(value, str):
        return value if len(value) <= string_limit else value[:string_limit] + "…[truncated]"
    if depth >= 8:
        return value if value is None or isinstance(value, (str, int, float, bool)) else "[nested value omitted]"
    if isinstance(value, list):
        return [_compact_trace_value(item, string_limit=string_limit, array_limit=array_limit,
                                     key_limit=key_limit, depth=depth + 1)
                for item in value[:array_limit]]
    if isinstance(value, dict):
        return {str(key): _compact_trace_value(item, string_limit=string_limit, array_limit=array_limit,
                                               key_limit=key_limit, depth=depth + 1)
                for key, item in list(value.items())[:key_limit]}
    return value


def _compact_tool_output(tool: str, value: Any, *, string_limit: int = 1200, array_limit: int = 40, key_limit: int = 80) -> Any:
    compacted = _compact_trace_value(value or {}, string_limit=string_limit,
                                     array_limit=array_limit, key_limit=key_limit)
    if tool == "geo.nearby" and isinstance(value, dict) and isinstance(value.get("features"), list):
        counts: dict[tuple[str, str], int] = {}
        for feature in value["features"]:
            if not isinstance(feature, dict):
                continue
            key = (str(feature.get("type") or "unknown"), str(feature.get("subtype") or ""))
            try:
                count = max(1, int(feature.get("_count") or 1))
            except (TypeError, ValueError):
                count = 1
            counts[key] = counts.get(key, 0) + count
        if isinstance(compacted, dict):
            compacted["features"] = [
                {"type": kind, "subtype": subtype, "_count": count}
                for (kind, subtype), count in counts.items()
            ]
    return compacted


def _compact_agent_trace(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for item in (trace or [])[-120:]:
        if not isinstance(item, dict):
            continue
        tool = str(item.get("tool") or "").strip()
        raw_output = item.get("output") or {}
        if tool == "fuse.decision":
            raw_output = _strip_legacy_report_values(raw_output)
        output = _compact_tool_output(tool, raw_output, string_limit=1200, array_limit=40)
        record = {
            "tool": tool,
            "status": str(item.get("status") or ""),
            "execution_state": str(item.get("execution_state") or ""),
            "cached": bool(item.get("cached")),
            "cost_ms": int(item.get("cost_ms") or 0),
            "input": _compact_trace_value(item.get("input") or {}, string_limit=1200, array_limit=40),
            "output": output,
        }
        if len(json.dumps(record, ensure_ascii=False)) > 12000:
            record["input"] = _compact_trace_value(item.get("input") or {}, string_limit=400, array_limit=12, key_limit=48)
            raw_output = item.get("output") or {}
            if tool == "fuse.decision":
                raw_output = _strip_legacy_report_values(raw_output)
            record["output"] = _compact_tool_output(tool, raw_output, string_limit=400, array_limit=12, key_limit=48)
        compacted.append(record)
    while compacted and len(json.dumps(compacted, ensure_ascii=False)) > 200000:
        compacted.pop(0)
    return compacted


def _resume_agent_trace(req: ChatRequest, *, image_path: str, nearby_radius: int | None) -> list[dict[str, Any]]:
    """Invalidate known foreign request evidence for both runtime and report."""
    supplied = [dict(item) for item in (req.agent_trace or []) if isinstance(item, dict)]
    invalid: set[str] = set()
    foreign_image = False
    for item in supplied:
        if item.get("execution_state") not in {"completed", "reused", "failed", "degraded", "declared_unavailable"}:
            continue
        name, args = str(item.get("tool") or ""), item.get("input") or {}
        output = item.get("output") or {}
        info = args.get("image_info") if isinstance(args, dict) else None
        reference = (output.get("image_path") if name == "tiff.info" and isinstance(output, dict) else None)
        if isinstance(args, dict):
            reference = reference or args.get("image_path") or (info.get("image_path") if isinstance(info, dict) else None)
        if reference and image_path and Path(str(reference)).resolve() != Path(image_path).resolve():
            foreign_image = True
        if name not in ("geo.background", "geo.nearby") or not isinstance(args, dict):
            continue
        constants = {"lat": req.latitude, "lon": req.longitude}
        if name == "geo.nearby":
            constants["radius"] = nearby_radius
        for key, current in constants.items():
            if key not in args:
                continue  # older traces may lack provenance
            try:
                matches = current is not None and float(args[key]) == float(current)
            except (TypeError, ValueError):
                matches = False
            if not matches:
                invalid.update((name, "fuse.decision", "report.write"))
    for item in supplied:
        if foreign_image or str(item.get("tool") or "") in invalid:
            # Preserve the original observation for audit; it cannot seed the
            # runtime or be labelled verified in the current report.
            item.update(execution_state="invalidated", status="error",
                        invalidation_reason="request image or geo constants changed")
    return supplied


def _initial_agent_trace(req: ChatRequest, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    supplied = _resume_agent_trace(req, image_path=_extract_latest_user_image_path(req.messages),
                                  nearby_radius=req.nearby_radius)
    return _compact_agent_trace(supplied or _history_tool_trace(messages))


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


def _immutable_run_inputs_message(
    req: ChatRequest, *, image_path: str, nearby_radius: int | None
) -> str:
    """Facts supplied once at the start of an agent run.

    This is shared task context, not a workflow instruction: either agent may
    choose whether to use a tool, but it must not invent values for a field the
    request already supplied.
    """
    facts = ["[immutable run inputs]"]
    if image_path:
        facts.append(f"image_path={image_path}")
    if req.latitude is not None and req.longitude is not None:
        facts.append(
            f"latitude={float(req.latitude):.8f}; longitude={float(req.longitude):.8f}"
        )
    if nearby_radius is not None:
        facts.append(f"nearby_radius_m={int(nearby_radius)}")
    facts.append(
        "If you call a tool with image_path, lat, lon, or radius, copy the corresponding "
        "run input exactly. These are supplied observation metadata; never infer, replace, "
        "or fabricate them from the image."
    )
    return "\n".join(facts)


def _osm_ssl_context() -> ssl.SSLContext | None:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return None


def _agent_mode(req: ChatRequest) -> str:
    return str(getattr(req, "agent_mode", "graph") or "graph").strip().lower()


def _agent_mode_enabled(req: ChatRequest) -> bool:
    """True when the request opts into LLM-driven agent orchestration."""
    return _agent_mode(req) in ("agent", "free")


def _contract_enforced(req: ChatRequest) -> bool:
    """False only for the ablation arm, which runs the agent without the contract.

    Both arms share the task brief, the ReAct preamble, the tool catalogue, the
    tool implementations and the loop-hygiene settings; this flag is the single
    experimental variable between them.
    """
    return _agent_mode(req) != "free"



def _build_agent_runner(
    *,
    req: ChatRequest,
    thresholds_path: str,
    image_path: str,
    report_write_required: bool,
    lazy_radius: bool = False,
    lazy_review_threshold: bool = False,
):
    """Construct the single Agent-mode runner used by sync and stream APIs."""
    from src.orchestration.agent_runner import AgentRunner
    from src.orchestration.evidence import record_output

    effective_radius = req.nearby_radius if lazy_radius else (req.nearby_radius or 300)

    def _model_fn(msgs, tools):
        chat_req = ChatRequest(
            messages=[ChatMessage(**m) for m in msgs],
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            tools=tools,
            tool_choice="auto",
        )
        resp = chat_completions(chat_req)
        return {
            "message": resp["choices"][0]["message"],
            "raw": str(resp.get("raw_response", "") or ""),
        }

    # Resume from the separate evidence ledger, not truncated model text.
    initial_outputs: dict[str, Any] = {}
    initial_tool_inputs: dict[str, Any] = {}
    initial_tool_failures: dict[str, int] = {}
    if image_path:
        for item in _resume_agent_trace(req, image_path=image_path, nearby_radius=effective_radius):
            name = str(item.get("tool") or "")
            output = item.get("output")
            args = item.get("input") or {}
            if not name or not isinstance(output, dict):
                continue
            # A trace for another uploaded image must not seed this analysis.
            reference = (output.get("image_path") if name == "tiff.info" else None)
            if isinstance(args, dict):
                info = args.get("image_info") or {}
                reference = reference or args.get("image_path") or (info.get("image_path") if isinstance(info, dict) else None)
            if reference and Path(str(reference)).resolve() != Path(image_path).resolve():
                continue
            execution = item.get("execution_state")
            if execution in ("failed", "degraded") and not item.get("cached"):
                initial_tool_failures[name] = initial_tool_failures.get(name, 0) + 1
            elif execution == "completed":
                initial_tool_failures.pop(name, None)
            if execution in ("completed", "reused", "degraded", "declared_unavailable", "failed"):
                record_output(initial_outputs, name, output)
                if isinstance(args, dict):
                    initial_tool_inputs[name] = args

    initial_outputs = {name: output for name, output in initial_outputs.items() if not output.get("error")}

    return AgentRunner(
        model_fn=_model_fn,
        thresholds_path=thresholds_path,
        enable_second_pass=req.enable_seg_llm_second_pass,
        report_write_required=report_write_required,
        image_path=image_path,
        latitude=req.latitude,
        longitude=req.longitude,
        nearby_radius=effective_radius,
        review_threshold=req.review_threshold,
        lazy_review_threshold=lazy_review_threshold,
        max_turns=req.max_turns,
        initial_turns=req.agent_turns_used,
        initial_outputs=initial_outputs if req.agent_trace else None,
        initial_tool_inputs=initial_tool_inputs,
        initial_tool_failures=initial_tool_failures,
        enforce_contract=_contract_enforced(req),
        react_mode=False,
        # Held constant across both arms on purpose - see RuleGuidedAgent.
        loop_hygiene=True,
    )



def _persist_canonical_report_record(
    response: dict[str, Any],
    report: dict[str, Any],
    text: str,
    composer: str,
    tool_trace: list[dict[str, Any]],
) -> None:
    """Make the saved report match the current structured report shown to users."""
    report_path = str(response.get("report_path") or "").strip()
    if not report_path:
        return
    try:
        from src.pipelines.report_writer import parse_answer
        from src.pipelines.stage6_report import run_stage6

        answer = response.get("composed_answer")
        if not isinstance(answer, dict):
            answer = parse_answer(text)

        record = dict(report)
        record["final_description"] = text
        record["report_text"] = text
        record["report_composer"] = composer
        record["report_path"] = report_path
        record["composed_answer"] = answer

        if answer.get("has_landslide") is not None:
            record["has_landslide"] = answer["has_landslide"]
        if answer.get("landslide_type"):
            record["landslide_type"] = answer["landslide_type"]

        for item in reversed(tool_trace or []):
            if item.get("tool") != "fuse.decision":
                continue
            output = item.get("output")
            if isinstance(output, dict):
                for key in ("classification_confidence", "severity"):
                    if key in output:
                        record[key] = output[key]
            break

        run_stage6(record, report_path)
    except Exception as exc:
        logging.exception("canonical report persistence failed")
        response["report_persistence_error"] = str(exc)


def _attach_structured_report(
    response: dict[str, Any], tool_trace: list[dict[str, Any]], req: "ChatRequest", *, enabled: bool,
    failure_reason: str = "",
) -> dict[str, Any]:
    """Attach and persist the current ten-field, provenance-tracked report."""
    if not enabled:
        return response
    try:
        from src.pipelines.structured_report import build_structured_report, render_structured_report
        from src.pipelines.report_writer import compose_report, parse_answer

        report = build_structured_report(tool_trace, latitude=req.latitude, longitude=req.longitude)
        choice = dict((response.get("choices") or [{}])[0])

        if os.getenv("FUSE_LENIENT", "0") in ("1", "true", "True"):
            own = str((((choice.get("message") or {}).get("content")) or "")).strip()
            if _current_report_sections(own):
                text = own
                composer = "agent"
            else:
                text = render_structured_report(report)
                composer = "template"
            response["composed_answer"] = parse_answer(text)
        else:
            text = None
            if os.getenv("REPORT_COMPOSER", "llm") == "llm":
                from src.models.llm_client import _openai_chat_completion

                text = compose_report(report, _openai_chat_completion, tool_trace)
            composer = "llm" if text else "template"
            if not text:
                text = render_structured_report(report)
            response["composed_answer"] = parse_answer(text)

        if failure_reason:
            text = f"> 本次运行未通过最终校验：{failure_reason}\n\n" + text

        choice["message"] = {"role": "assistant", "content": text}
        response["choices"] = [choice]
        response["report_composer"] = composer
        response["structured_report"] = report
        response.pop("legacy_message", None)
        _persist_canonical_report_record(response, report, text, composer, tool_trace)
        if isinstance(response.get("agent_trace"), list):
            response["agent_trace"] = _compact_agent_trace(response["agent_trace"])
    except Exception as exc:  # never lose the run because of the report view
        logging.exception("structured report failed")
        response["structured_report_error"] = str(exc)
        response.pop("legacy_message", None)
        choices = list(response.get("choices") or [])
        if choices:
            choice = dict(choices[0])
            message = dict(choice.get("message") or {})
            if _looks_like_legacy_report_text(str(message.get("content") or "")):
                message["content"] = "The current structured report could not be composed from the recorded evidence."
                choice["message"] = message
                choices[0] = choice
                response["choices"] = choices
    return response


@app.post("/v1/agent/analyze")
def agent_analyze(req: ChatRequest):
    latest_user_has_image = _latest_user_has_image(req.messages)
    if latest_user_has_image and not _agent_mode_enabled(req):
        return graph_analyze(req)

    thresholds_path = str(PROJECT_ROOT / "configs" / "thresholds.json")
    image_path = (
        _extract_latest_user_image_path(req.messages)
        if latest_user_has_image
        else ""
    )
    messages = [_to_dict_message(m) for m in req.messages]
    messages = _compress_followup_report_messages(messages)
    if latest_user_has_image:
        messages = _trim_messages_to_latest_image_session(messages)
        messages = _inject_forced_system_messages(messages, req)
    else:
        messages = _inject_followup_system_message(messages)
        messages = _inject_context_summary_message(messages, req)

    policy = get_policy(thresholds_path)
    report_write_required = bool(latest_user_has_image and policy.require_report_write)
    agent = _build_agent_runner(
        req=req,
        thresholds_path=thresholds_path,
        image_path=image_path,
        report_write_required=report_write_required,
    )

    tool_trace: list[dict[str, Any]] = (
        _initial_agent_trace(req, messages) if latest_user_has_image else []
    )
    final_data: dict[str, Any] = {}
    fallback_reason = ""
    fallback_outputs: dict[str, Any] = {}
    fallback_turns_used = req.agent_turns_used
    for event in agent.stream(messages):
        event_type = event.get("type")
        if event_type == "tool_result":
            tool_trace.append(dict(event.get("data") or {}))
        elif event_type == "final_core":
            final_data = dict(event.get("data") or {})
        elif event_type == "fallback":
            fallback_reason = str(event.get("reason") or "task rules not satisfied")
            fallback_outputs = dict(event.get("outputs") or {})
            fallback_turns_used = int(event.get("turns_used", req.agent_turns_used) or 0)

    if fallback_reason and image_path and _is_existing_file(image_path):
        if os.getenv("AGENT_FALLBACK_TO_GRAPH", "0") in ("1", "true", "True"):
            graph_result = graph_analyze(req)
            graph_report = graph_result.get("final_report") or {}
            content = str(
                graph_result.get("report_text")
                or graph_report.get("final_description")
                or graph_report.get("summary")
                or "Deterministic workflow completed after Agent fallback."
            ).strip()
            now = int(time.time())
            return {
                "id": f"chatcmpl-{now}",
                "object": "chat.completion",
                "created": now,
                "model": LLM_API_MODEL_NAME,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "agent_trace": _compact_agent_trace(tool_trace + list(graph_result.get("trace") or [])),
                "artifacts": graph_result.get("artifacts") or {},
                "geo": _build_geo_payload(req),
                "history": [{"role": "assistant", "content": content}],
                "mode": "graph",
                "critic": {"degraded": True, "unmet_rules": [fallback_reason]},
                "errors": graph_result.get("errors") or [],
            }

    # A failed run keeps whatever evidence it did gather, so it can still be
    # scored against the deliverable contract offline.
    outputs = final_data.get("outputs") or fallback_outputs
    if fallback_reason:
        message = _failed_run_message(
            fallback_reason, outputs, include_report=latest_user_has_image
        )
    else:
        message = final_data.get("message") or {
            "role": "assistant",
            "content": "Agent mode returned no final result.",
        }
        # Only swap in the structured report on the initial analysis turn; a
        # follow-up must not re-dump the whole report over the model's answer.
        if latest_user_has_image:
            message = _prefer_structured_final_message(message, outputs)
    ledger_image, seg_overlay, seg_mask, refine_overlay = _artifact_paths_from_outputs(outputs)
    resolved_image = image_path or ledger_image
    report_write = outputs.get("report.write")
    report_path = str(report_write.get("report_path", "") or "") if isinstance(report_write, dict) else ""
    now = int(time.time())
    return _attach_structured_report({
        "id": f"chatcmpl-{now}",
        "object": "chat.completion",
        "created": now,
        "model": LLM_API_MODEL_NAME,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "agent_trace": tool_trace,
        "agent_turns_used": int(final_data.get("turns_used", fallback_turns_used) or 0),
        "report_path": report_path,
        "artifacts": _build_artifacts_payload(
            include_images=bool(resolved_image),
            image_path=resolved_image,
            seg_overlay_path=seg_overlay,
            seg_mask_path=seg_mask,
            seg_refine_overlay_path=refine_overlay,
        ),
        "geo": _build_geo_payload(req),
        "history": final_data.get("history") or [],
        "mode": "agent",
        "critic": final_data.get("critic")
        or ({"degraded": True, "unmet_rules": [fallback_reason]} if fallback_reason else {}),
    }, tool_trace, req, enabled=bool(latest_user_has_image), failure_reason=fallback_reason)


@app.post("/v1/graph/analyze")
def graph_analyze(req: ChatRequest):
    """Run the explicit LangGraph workflow without changing the legacy agent API."""
    from fastapi import HTTPException
    from src.graph.landslide_graph import invoke_landslide_graph

    image_path = _extract_latest_user_image_path(req.messages)
    if not image_path or not _is_existing_file(image_path):
        raise HTTPException(status_code=400, detail="A valid image attachment/path is required")

    thresholds_path = str(PROJECT_ROOT / "configs" / "thresholds.json")
    policy = get_policy(thresholds_path)
    report_out_path = _default_report_out_path(image_path) if policy.require_report_write else ""
    result = invoke_landslide_graph(
        image_path=image_path,
        latitude=req.latitude,
        longitude=req.longitude,
        nearby_radius=req.nearby_radius or 300,
        report_out_path=report_out_path,
        enable_second_pass=req.enable_seg_llm_second_pass,
        thread_id=f"graph-{uuid4().hex}",
    )
    errors = list(result.get("errors") or [])
    report_trace = _graph_report_trace(result)
    report_response = _attach_structured_report(
        {
            "choices": [{"message": {"role": "assistant", "content": str((result.get("final_report") or {}).get("final_description") or "")}}],
            "report_path": str(result.get("report_path") or ""),
        },
        report_trace, req, enabled=True,
    )
    return {
        "status": "completed" if result.get("final_report") and not errors else "degraded",
        "final_report": result.get("final_report"),
        "report_text": report_response["choices"][0]["message"]["content"],
        "report_composer": report_response.get("report_composer"),
        "structured_report": report_response.get("structured_report"),
        "report_path": result.get("report_path", ""),
        "artifacts": {
            "image_path": result.get("image_info", {}).get("image_path", ""),
            "segmentation": result.get("segmentation", {}),
            "refinement": result.get("refinement", {}),
            "region_location": result.get("region_location", {}),
        },
        "trace": result.get("trace", []),
        "errors": errors,
        "state": result,
    }


def _graph_report_trace(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Map graph node outputs to the same report evidence used by Agent mode."""
    nodes = (
        ("tiff.info", "image_info"), ("llm.first_pass", "stage1"),
        ("seg.run", "segmentation"), ("seg.refine", "refinement"),
        ("region.locate", "region_location"), ("seg.llm_review", "llm_second_pass"),
        ("cls.run", "classification"),
    )
    trace = [
        {"tool": tool, "execution_state": "completed", "input": {}, "output": state[key]}
        for tool, key in nodes if isinstance(state.get(key), dict) and state[key]
    ]
    geo = state.get("geo_context") or {}
    for tool, key in (("geo.background", "background"), ("geo.nearby", "nearby")):
        if isinstance(geo.get(key), dict):
            trace.append({"tool": tool, "execution_state": "completed", "input": {}, "output": geo[key]})
    fusion = state.get("final_report")
    if isinstance(fusion, dict):
        trace.append({"tool": "fuse.decision", "execution_state": "completed", "input": {
            "stage1": state.get("stage1") or {},
            "segmentation": state.get("segmentation") or {},
            "refinement": state.get("refinement") or {},
            "classification": state.get("classification") or {},
            "geo_context": geo,
            "llm_second_pass": state.get("llm_second_pass") or {},
        }, "output": fusion})
    return trace


def _stream_langgraph_analysis(
    req: ChatRequest,
    image_path: str,
    thresholds_path: str,
):
    from src.graph.landslide_graph import route_second_pass, stream_landslide_graph

    planned_nodes = [
        "input",
        "first_pass",
        "segmentation",
        "refinement",
        "region_locate",
        "second_pass_review",
        "classification",
        "geo_context",
        "fusion",
        "report",
    ]
    yield json.dumps(
        {
            "type": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "langgraph-plan",
                    "type": "function",
                    "function": {
                        "name": "langgraph.analysis",
                        "arguments": json.dumps({"nodes": planned_nodes}, ensure_ascii=False),
                    },
                }
            ],
        },
        ensure_ascii=False,
    ) + "\n"

    node_tools = {
        "input": "tiff.info",
        "first_pass": "llm.first_pass",
        "segmentation": "seg.run",
        "refinement": "seg.refine",
        "region_locate": "region.locate",
        "second_pass_review": "seg.llm_review",
        "classification": "cls.run",
        "fusion": "fuse.decision",
        "report": "report.write",
    }
    next_nodes = {
        "input": "first_pass",
        "first_pass": "segmentation",
        "segmentation": "refinement",
        "refinement": "region_locate",
        "region_locate": None,
        "second_pass_review": "classification",
        "classification": "geo_context",
        "geo_context": "fusion",
        "fusion": "report",
        "report": None,
    }

    def _line(payload: dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False) + "\n"

    def _tool_call(tool_name: str) -> str:
        return _line({"type": "tool_call", "name": tool_name, "arguments": {}})

    def _tool_result(
        tool_name: str,
        trace_item: dict[str, Any],
        output: Any,
    ) -> str:
        status = str(trace_item.get("status", "ok") or "ok")
        cost_ms = int(float(trace_item.get("duration_ms", 0) or 0))
        return _line(
            {
                "type": "tool_result",
                "data": {
                    "tool": tool_name,
                    "status": status,
                    "cost_ms": cost_ms,
                    "input": {},
                    "output": output if isinstance(output, dict) else {"value": output},
                },
            }
        )

    state: dict[str, Any] = {
        "errors": [],
        "trace": [],
        "enable_second_pass": req.enable_seg_llm_second_pass,
    }

    try:
        policy = get_policy(thresholds_path)
        review_threshold = req.review_threshold or policy.tiny_area_review_threshold
        state["second_pass_area_ratio"] = review_threshold
        report_out_path = _default_report_out_path(image_path) if policy.require_report_write else ""
        graph_stream = iter(
            stream_landslide_graph(
                image_path=image_path,
                latitude=req.latitude,
                longitude=req.longitude,
                nearby_radius=req.nearby_radius or 300,
                report_out_path=report_out_path,
                enable_second_pass=req.enable_seg_llm_second_pass,
                second_pass_area_ratio=review_threshold,
                thread_id=f"graph-{uuid4().hex}",
            )
        )
        pending_node: str | None = "input"
        while pending_node:
            pending_tool = node_tools.get(pending_node)
            if pending_node == "geo_context":
                yield _tool_call("geo.background")
                yield _tool_call("geo.nearby")
            elif pending_tool:
                # This is emitted before next(graph_stream), so the UI shows the
                # node as running while its model/service call is in progress.
                yield _tool_call(pending_tool)

            streamed = next(graph_stream)
            node = str((streamed or {}).get("node", "") or pending_node)
            update = dict((streamed or {}).get("update") or {})
            for key, value in update.items():
                if key in {"errors", "trace"}:
                    state.setdefault(key, []).extend(value or [])
                else:
                    state[key] = value
            trace_items = update.get("trace") or []
            trace_item = dict(trace_items[-1] or {}) if trace_items else {"node": node}

            if node == "geo_context":
                geo_context = state.get("geo_context") or {}
                yield _tool_result("geo.background", trace_item, geo_context.get("background") or {})
                yield _tool_result("geo.nearby", trace_item, geo_context.get("nearby") or {})
            else:
                tool_name = node_tools.get(node)
                if tool_name:
                    output_keys = {
                        "input": "image_info",
                        "first_pass": "stage1",
                        "segmentation": "segmentation",
                        "refinement": "refinement",
                        "region_locate": "region_location",
                        "second_pass_review": "llm_second_pass",
                        "classification": "classification",
                        "fusion": "final_report",
                    }
                    output = state.get(output_keys.get(node, ""), {})
                    if trace_item.get("status") != "ok" and not isinstance(output, dict):
                        output = {"error": trace_item.get("error", "graph node failed")}
                    if node == "report":
                        output = {"report_path": state.get("report_path", "")}
                    yield _tool_result(tool_name, trace_item, output or {})

            if node == "region_locate":
                pending_node = (
                    "second_pass_review"
                    if route_second_pass(state) == "review"
                    else "classification"
                )
            else:
                pending_node = next_nodes.get(node)
    except StopIteration:
        pass
    except Exception as exc:
        logging.exception("LangGraph analysis failed")
        yield _line({"type": "error", "error": f"LangGraph analysis failed: {exc}"})
        return

    result = state

    final_report = result.get("final_report") or {}
    final_content = str(
        final_report.get("final_description")
        or final_report.get("summary")
        or "LangGraph analysis completed, but no final report description was returned."
    ).strip()
    segmentation = result.get("segmentation") or {}
    refinement = result.get("refinement") or {}
    artifacts = _build_artifacts_payload(
        include_images=True,
        image_path=image_path,
        seg_overlay_path=str(segmentation.get("overlay_path", "") or ""),
        seg_mask_path=str(segmentation.get("mask_path", "") or ""),
        seg_refine_overlay_path=str(refinement.get("overlay_path", "") or ""),
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
                "message": {"role": "assistant", "content": final_content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "report_path": str(state.get("report_path") or ""),
        "agent_trace": result.get("trace", []),
        "artifacts": artifacts,
        "geo": _build_geo_payload(req),
        "history": [
            {"role": "assistant", "content": final_content},
        ],
        "graph": {
            "name": "landslide_graph",
            "errors": result.get("errors", []),
        },
    }
    final_resp = _attach_structured_report(final_resp, _graph_report_trace(result), req, enabled=True)
    final_resp["agent_trace"] = _compact_agent_trace(_graph_report_trace(result))
    final_resp["history"] = [final_resp["choices"][0]["message"]]
    yield json.dumps({"type": "final", "data": final_resp}, ensure_ascii=False) + "\n"



def _failed_run_message(
    reason: str,
    agent_outputs: dict[str, Any] | None,
    *,
    include_report: bool,
) -> dict[str, Any]:
    """Closing message for an agent run that did not satisfy the task contract.

    The failure is always stated first. A current structured fused assessment
    may follow, labelled as not having passed final verification. Retired report
    formats are never revived in the closing message.
    """
    notice = f"**Analysis incomplete.** The agent did not satisfy the task rules: {reason}."
    if include_report:
        structured = str(
            _prefer_structured_final_message(
                {"role": "assistant", "content": ""}, agent_outputs
            ).get("content")
            or ""
        ).strip()
        if structured:
            return {
                "role": "assistant",
                "content": (
                    notice
                    + "\n\nThe fused assessment recorded before the run stopped is shown "
                    "below. It has not passed final verification.\n\n"
                    + structured
                ),
            }
    return {"role": "assistant", "content": notice}



def _looks_sectioned_report(text: str) -> bool:
    return bool(_current_report_sections(text))


def _prefer_structured_final_message(
    message: dict[str, Any] | None,
    agent_outputs: dict[str, Any] | None,
) -> dict[str, Any]:
    """Prefer a current ten-field report; never surface the retired template."""
    msg = dict(message or {"role": "assistant", "content": ""})
    outputs = agent_outputs if isinstance(agent_outputs, dict) else {}
    fusion = outputs.get("fuse.decision")
    if not isinstance(fusion, dict):
        if _looks_like_legacy_report_text(str(msg.get("content") or "")):
            msg["content"] = ""
        return msg

    structured = str(fusion.get("final_description") or "").strip()
    if not structured:
        structured = str(fusion.get("summary") or "").strip()

    existing = str(msg.get("content") or "").strip()
    if _looks_sectioned_report(structured):
        msg["content"] = structured
        return msg
    if _looks_sectioned_report(existing):
        return msg
    if _looks_like_legacy_report_text(existing):
        msg["content"] = ""
    if _looks_like_legacy_report_text(structured):
        return msg
    if structured:
        msg["content"] = structured
    return msg


@app.post("/v1/agent/analyze_stream")
def agent_analyze_stream(req: ChatRequest):
    def _stream():
        latest_user_has_image = _latest_user_has_image(req.messages)
        thresholds_path = str(PROJECT_ROOT / "configs" / "thresholds.json")
        image_path = _extract_latest_user_image_path(req.messages) if latest_user_has_image else ""
        agent_mode = _agent_mode_enabled(req)
        if latest_user_has_image and not agent_mode:
            yield from _stream_langgraph_analysis(req, image_path, thresholds_path)
            return
        messages = [_to_dict_message(m) for m in req.messages]
        messages = _compress_followup_report_messages(messages)
        if latest_user_has_image:
            messages = _trim_messages_to_latest_image_session(messages)
            messages = _inject_forced_system_messages(messages, req)
        else:
            messages = _inject_followup_system_message(messages)
            messages = _inject_context_summary_message(messages, req)

        policy = get_policy(thresholds_path)
        report_write_required = bool(latest_user_has_image and policy.require_report_write)

        agent = _build_agent_runner(
            req=req,
            thresholds_path=thresholds_path,
            image_path=image_path,
            report_write_required=report_write_required,
            lazy_radius=True,
            lazy_review_threshold=True,
        )

        tool_trace: list[dict[str, Any]] = (
            _initial_agent_trace(req, messages) if latest_user_has_image else []
        )
        seg_overlay_path = ""
        seg_mask_path = ""
        seg_refine_overlay_path = ""
        last_refine_output: dict[str, Any] | None = None
        resolved_image_path = image_path

        def _degraded_final(reason: str, ledger: dict[str, Any] | None = None, turns_used: int = 0):
            ledger = ledger if isinstance(ledger, dict) else {}
            report_write = ledger.get("report.write") if isinstance(ledger.get("report.write"), dict) else {}
            report_path = str(report_write.get("report_path") or "")
            l_image, l_overlay, l_mask, l_refine = _artifact_paths_from_outputs(ledger)
            message = _failed_run_message(
                reason, ledger, include_report=latest_user_has_image
            )
            now = int(time.time())
            yield json.dumps(
                {
                    "type": "final",
                    "data": {
                        "id": f"chatcmpl-{now}",
                        "object": "chat.completion",
                        "created": now,
                        "model": LLM_API_MODEL_NAME,
                        "choices": [
                            {
                                "index": 0,
                                "message": message,
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                        "agent_trace": _compact_agent_trace(tool_trace),
                        "report_path": report_path,
                        "agent_turns_used": int(turns_used or 0),
                        "artifacts": _build_artifacts_payload(
                            include_images=bool(resolved_image_path or l_image),
                            image_path=resolved_image_path or l_image,
                            seg_overlay_path=seg_overlay_path or l_overlay,
                            seg_mask_path=seg_mask_path or l_mask,
                            seg_refine_overlay_path=seg_refine_overlay_path or l_refine,
                        ),
                        "geo": _build_geo_payload(req),
                        "history": [],
                        "mode": "agent",
                        "critic": {"degraded": True, "unmet_rules": [reason]},
                    } if not latest_user_has_image else _attach_structured_report({
                        "id": f"chatcmpl-{now}",
                        "object": "chat.completion",
                        "created": now,
                        "model": LLM_API_MODEL_NAME,
                        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                        "agent_trace": _compact_agent_trace(tool_trace),
                        "report_path": report_path,
                        "agent_turns_used": int(turns_used or 0),
                        "artifacts": _build_artifacts_payload(
                            include_images=bool(resolved_image_path or l_image),
                            image_path=resolved_image_path or l_image,
                            seg_overlay_path=seg_overlay_path or l_overlay,
                            seg_mask_path=seg_mask_path or l_mask,
                            seg_refine_overlay_path=seg_refine_overlay_path or l_refine,
                        ),
                        "geo": _build_geo_payload(req),
                        "history": [],
                        "mode": "agent",
                        "critic": {"degraded": True, "unmet_rules": [reason]},
                    }, tool_trace, req, enabled=True, failure_reason=reason),
                },
                ensure_ascii=False,
            ) + "\n"

        try:
            for ev in agent.stream(messages):
                etype = ev.get("type")

                if etype == "tool_result":
                    d = ev.get("data") or {}
                    name = d.get("tool", "")
                    status = d.get("status", "")
                    output = d.get("output") if isinstance(d.get("output"), dict) else {}
                    tool_trace.append(d)
                    if status == "ok":
                        if name == "tiff.info" and output.get("image_path"):
                            resolved_image_path = output["image_path"]
                        elif name == "seg.run":
                            seg_overlay_path = str(
                                output.get("overlay_path", "") or seg_overlay_path
                            ).replace("\\", "/")
                            seg_mask_path = str(
                                output.get("mask_path", "") or seg_mask_path
                            ).replace("\\", "/")
                        elif name == "seg.refine":
                            last_refine_output = output
                            seg_refine_overlay_path = str(
                                output.get("overlay_path", "") or seg_refine_overlay_path
                            ).replace("\\", "/")
                    yield json.dumps(ev, ensure_ascii=False) + "\n"
                    continue

                if etype in ("assistant", "tool_call", "model_raw"):
                    yield json.dumps(ev, ensure_ascii=False) + "\n"
                    continue

                if etype == "need_nearby_radius":
                    yield json.dumps({"type": "need_nearby_radius", "agent_turns_used": int(ev.get("turns_used", req.agent_turns_used) or 0)}, ensure_ascii=False) + "\n"
                    return

                if etype == "need_review_threshold":
                    yield json.dumps({"type": "need_review_threshold", "agent_turns_used": int(ev.get("turns_used", req.agent_turns_used) or 0)}, ensure_ascii=False) + "\n"
                    return

                if etype == "fallback":
                    reason = str(ev.get("reason") or "task rules not satisfied")
                    use_graph = (
                        os.getenv("AGENT_FALLBACK_TO_GRAPH", "0") in ("1", "true", "True")
                        and resolved_image_path
                        and _is_existing_file(resolved_image_path)
                    )
                    yield json.dumps(
                        {
                            "type": "assistant",
                            "tool_calls": [],
                            "content": (
                                f"Agent mode could not satisfy the task rules ({reason}). "
                                + (
                                    "Falling back to the deterministic LangGraph pipeline.\n"
                                    if use_graph
                                    else "The run stops here.\n"
                                )
                            ),
                        },
                        ensure_ascii=False,
                    ) + "\n"
                    if use_graph:
                        yield from _stream_langgraph_analysis(req, resolved_image_path, thresholds_path)
                    else:
                        yield from _degraded_final(reason, ev.get("outputs"), int(ev.get("turns_used", req.agent_turns_used) or 0))
                    return

                if etype == "final_core":
                    d = ev.get("data") or {}
                    if not seg_refine_overlay_path:
                        seg_refine_overlay_path = _ensure_seg_refine_overlay(
                            image_path=resolved_image_path,
                            refinement_output=last_refine_output,
                            seg_mask_path=seg_mask_path,
                        )
                    msg = d.get("message") or {"role": "assistant", "content": ""}
                    now = int(time.time())
                    _agent_outputs = d.get("outputs") or {}
                    # The final event may follow a fuse retry/new graph cycle; recover
                    # artifacts from the authoritative ledger, not only current events.
                    ledger_image, ledger_seg_overlay, ledger_seg_mask, ledger_refine = _artifact_paths_from_outputs(_agent_outputs)
                    resolved_image_path = resolved_image_path or ledger_image
                    seg_overlay_path = seg_overlay_path or ledger_seg_overlay
                    seg_mask_path = seg_mask_path or ledger_seg_mask
                    seg_refine_overlay_path = seg_refine_overlay_path or ledger_refine
                    _report_write = _agent_outputs.get("report.write")
                    report_path = ""
                    if isinstance(_report_write, dict):
                        report_path = str(_report_write.get("report_path", "") or "")
                    # The model's own closing turn is usually a bare acknowledgement
                    # ("report saved to ..."), not the report.  fuse.decision already
                    # built the full sectioned narrative, so present that instead --
                    # same contract the LangGraph path yields.
                    if latest_user_has_image:
                        msg = _prefer_structured_final_message(msg, _agent_outputs)
                    final_resp = {
                        "id": f"chatcmpl-{now}",
                        "object": "chat.completion",
                        "created": now,
                        "model": LLM_API_MODEL_NAME,
                        "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                        "agent_trace": _compact_agent_trace(tool_trace),
                        "agent_turns_used": int(d.get("turns_used", req.agent_turns_used) or 0),
                        "report_path": report_path,
                        "artifacts": _build_artifacts_payload(
                            include_images=bool(resolved_image_path),
                            image_path=resolved_image_path,
                            seg_overlay_path=seg_overlay_path,
                            seg_mask_path=seg_mask_path,
                            seg_refine_overlay_path=seg_refine_overlay_path,
                        ),
                        "geo": _build_geo_payload(req),
                        "history": d.get("history") or [],
                        "mode": d.get("mode", "agent"),
                        "critic": d.get("critic") or {},
                    }
                    final_resp = _attach_structured_report(
                        final_resp, tool_trace, req, enabled=bool(latest_user_has_image)
                    )
                    yield json.dumps({"type": "final", "data": final_resp}, ensure_ascii=False) + "\n"
                    return

                yield json.dumps(ev, ensure_ascii=False) + "\n"
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
    seg_python = os.getenv("SEG_ENV_PYTHON", __import__("sys").executable)
    cls_python = os.getenv("CLS_ENV_PYTHON", __import__("sys").executable)

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


def _max_new_tokens(requested: int | None = None) -> int:
    def _read_positive(name: str, fallback: int) -> int:
        try:
            value = int(os.getenv(name, str(fallback)))
        except (TypeError, ValueError):
            value = fallback
        return value if value > 0 else fallback

    default_value = _read_positive("LLM_DEFAULT_MAX_TOKENS", 1024)
    cap_value = _read_positive("LLM_MAX_TOKENS_CAP", 4096)
    requested_value = requested if isinstance(requested, int) and requested > 0 else default_value
    return max(1, min(requested_value, cap_value))


def _generate_continuation_text(messages: list[ChatMessage], req: ChatRequest) -> str:
    """Generate a text-only continuation for a response that ended mid-thought."""
    effective_messages = list(messages)
    normalized: list[dict[str, Any]] = []
    images: list[Image.Image] = []
    for message in effective_messages:
        content = message.content
        if isinstance(content, list):
            content_items: list[dict[str, Any]] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "image":
                    image_path = part.get("image_path") or part.get("image")
                    if image_path and os.path.exists(image_path):
                        image = Image.open(image_path).convert("RGB")
                        images.append(image)
                        content_items.append({"type": "image", "image": image})
                        content_items.append(
                            {
                                "type": "text",
                                "text": f"[image_path]{image_path}[/image_path]",
                            }
                        )
                elif part.get("type") == "text" and part.get("text"):
                    content_items.append({"type": "text", "text": part["text"]})
            normalized.append({"role": message.role, "content": content_items})
        else:
            normalized.append({"role": message.role, "content": content})

    prompt = processor.apply_chat_template(
        normalized,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = processor(text=prompt, images=images if images else None, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    generation_kwargs: dict[str, Any] = {
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.12,
        "max_new_tokens": _max_new_tokens(req.max_tokens),
        "do_sample": False,
    }
    import torch

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)
    response_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(response_ids, skip_special_tokens=True).strip()


def _repair_incomplete_response(req: ChatRequest, content: str) -> str:
    repaired = str(content or "").strip()
    if not _looks_like_incomplete_response(repaired):
        return _ensure_terminal_punctuation(repaired)

    for _ in range(2):
        if not _looks_like_incomplete_response(repaired):
            break
        continuation_messages = list(req.messages)
        continuation_messages.append(ChatMessage(role="assistant", content=repaired))
        continuation_messages.append(
            ChatMessage(
                role="user",
                content=(
                    "Continue exactly from the last incomplete character. "
                    "Return only the missing continuation, do not repeat any previous text, "
                    "and finish with complete natural sentences and punctuation."
                ),
            )
        )
        try:
            raw_continuation = _generate_continuation_text(continuation_messages, req)
        except Exception:
            logging.exception("incomplete response continuation failed")
            break
        continuation, continuation_tools = _extract_tool_calls(raw_continuation)
        if continuation_tools or not continuation.strip():
            break
        merged = _merge_response_continuation(repaired, continuation)
        if merged == repaired:
            break
        repaired = merged.strip()
    return _ensure_terminal_punctuation(repaired)


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    # Serialize against dual-head classification (adapter switches) and any
    # other request touching the single resident `model`. RLock so nested
    # calls within this same request (e.g. the continuation-repair path)
    # don't deadlock.
    with INFERENCE_LOCK:
        with _request_adapter_scope(req.adapter_mode):
            return _chat_completions_impl(req)


def _chat_completions_impl(req: ChatRequest):
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
            if msg.role == "assistant" and getattr(msg, "tool_calls", None):
                _parts: list[str] = []
                _base = content if isinstance(content, str) else ""
                if _base.strip():
                    _parts.append(_base.strip())
                for _call in msg.tool_calls:
                    _fn = (_call or {}).get("function", {}) or {}
                    _name = _fn.get("name", "")
                    _raw = _fn.get("arguments", "{}")
                    try:
                        _args = json.loads(_raw) if isinstance(_raw, str) else (_raw or {})
                    except (json.JSONDecodeError, TypeError):
                        _args = {}
                    _parts.append(
                        "<tool_call>\n"
                        + json.dumps({"name": _name, "arguments": _args}, ensure_ascii=False)
                        + "\n</tool_call>"
                    )
                normalized.append({"role": "assistant", "content": "\n".join(_parts)})
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
        "repetition_penalty": 1.12,
        "max_new_tokens": _max_new_tokens(req.max_tokens),
    }
    if req.tools:
        # Tool mode needs stable, non-rambling outputs more than creativity.
        generation_kwargs["do_sample"] = False
        # Stop right after a tool call so the model cannot ramble past it and
        # confuse the parser / burn tokens.
        generation_kwargs["stop_strings"] = ["</tool_call>"]
        generation_kwargs["tokenizer"] = tokenizer
    else:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = max(req.temperature, 0.01)
        generation_kwargs["top_p"] = 0.8
        generation_kwargs["top_k"] = 20

    import torch
    with torch.no_grad():
        try:
            output_ids = model.generate(**inputs, **generation_kwargs)
        except (TypeError, ValueError) as exc:
            # Older/newer generate() signatures may reject stop_strings/tokenizer.
            logging.warning("generate() rejected stop_strings, retrying without: %s", exc)
            generation_kwargs.pop("stop_strings", None)
            generation_kwargs.pop("tokenizer", None)
            output_ids = model.generate(**inputs, **generation_kwargs)

    input_ids = inputs["input_ids"]
    response_ids = output_ids[0][input_ids.shape[1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    if "<tool_call>" in response and "</tool_call>" not in response:
        # stop_strings trimmed the closing tag (or the model omitted it).
        response = response + "\n</tool_call>"

    content, tool_calls = _extract_tool_calls(response)
    tool_calls = _filter_tool_calls_to_available(tool_calls, req.tools)
    content = _dedupe_repeated_lines(content)
    if not tool_calls:
        content = _repair_incomplete_response(req, content)
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


_TOOLCALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", flags=re.DOTALL)
_XML_FUNC_RE = re.compile(r"<function=([^>\n]+)>\s*(.*?)\s*</function>", flags=re.DOTALL)
_XML_PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>", flags=re.DOTALL)
_BARE_JSON_CALL_RE = re.compile(
    r'\{\s*"name"\s*:\s*"([A-Za-z0-9_.\-]+)"\s*,\s*"(?:arguments|parameters)"\s*:\s*(\{.*?\})\s*\}',
    flags=re.DOTALL,
)
_FENCE_RE = re.compile(r"```(?:json|tool_code|python)?\s*(.*?)```", flags=re.DOTALL)


def _coerce_tool_args(raw: Any) -> dict[str, Any]:
    """Best-effort coercion of a tool 'arguments' value into a dict."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    s = raw.strip()
    if not s:
        return {}
    for candidate in (s, s + "}", s + "}}"):
        try:
            value = json.loads(candidate)
        except Exception:
            continue
        if isinstance(value, dict):
            return value
    m = re.search(r'"image_path"\s*:\s*"([^"]*)', s)
    return {"image_path": m.group(1)} if m else {}


def _parse_tool_call_body(body: str) -> tuple[str, Any] | None:
    """Parse one tool-call body (XML <function=> or a JSON object) -> (name, args)."""
    body = (body or "").strip()
    if not body:
        return None
    xm = _XML_FUNC_RE.search(body)
    if xm and xm.group(1).strip():
        params: dict[str, Any] = {}
        for pm in _XML_PARAM_RE.finditer(xm.group(2)):
            pn = pm.group(1).strip()
            if not pn:
                continue
            rv = pm.group(2).strip()
            try:
                params[pn] = json.loads(rv)
            except Exception:
                params[pn] = rv
        return xm.group(1).strip(), params
    for cand in (_FENCE_RE.findall(body) or [body]):
        try:
            obj = json.loads(cand.strip())
        except Exception:
            obj = None
        if isinstance(obj, dict) and str(obj.get("name") or "").strip():
            return str(obj["name"]).strip(), obj.get("arguments", obj.get("parameters", {}))
    jm = _BARE_JSON_CALL_RE.search(body)
    if jm:
        return jm.group(1).strip(), _coerce_tool_args(jm.group(2))
    return None


def _available_tool_names(tools: list[dict[str, Any]] | None) -> set[str]:
    names: set[str] = set()
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        if not isinstance(fn, dict):
            continue
        name = str(fn.get("name", "") or "").strip()
        if name:
            names.add(name)
    return names


def _filter_tool_calls_to_available(
    tool_calls: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Accept only tool calls that were exposed in this model turn."""
    available = _available_tool_names(tools)
    if not available:
        if tool_calls:
            logging.info("Dropping %d tool call(s) because no tools are exposed in this turn", len(tool_calls))
        return []
    kept: list[dict[str, Any]] = []
    for call in tool_calls or []:
        fn = call.get("function") if isinstance(call, dict) else None
        name = str((fn or {}).get("name", "") or "").strip() if isinstance(fn, dict) else ""
        if name in available:
            kept.append(call)
        else:
            logging.info("Dropping hidden/unavailable tool call from model output: %s", name)
    return kept


def _extract_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Robustly pull tool calls out of a raw model completion.

    Handles: multiple ``<tool_call>`` blocks, ``<tool_call>{json}</tool_call>``,
    ``<tool_call><function=..>`` XML, bare / fenced JSON calls, ``arguments`` as a
    JSON string, ``parameters`` alias, and truncated output.  Empty / nameless
    blocks are ignored; duplicate calls within one completion are de-duplicated.
    """
    text = str(text or "")
    tool_calls: list[dict[str, Any]] = []
    seen: set[str] = set()
    consumed: list[tuple[int, int]] = []

    def _add(name: str, args: Any) -> None:
        name = str(name or "").strip()
        if not name:
            return
        arguments = json.dumps(_coerce_tool_args(args), ensure_ascii=False)
        key = f"{name}|{arguments}"
        if key in seen:
            return
        seen.add(key)
        tool_calls.append(
            {
                "id": f"call_{uuid4().hex[:12]}_{len(tool_calls)}",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )

    for m in _TOOLCALL_BLOCK_RE.finditer(text):
        parsed = _parse_tool_call_body(m.group(1))
        if parsed:
            _add(*parsed)
            consumed.append((m.start(), m.end()))

    if not tool_calls:
        for cand in _FENCE_RE.findall(text):
            try:
                obj = json.loads(cand.strip())
            except Exception:
                continue
            if isinstance(obj, dict) and str(obj.get("name") or "").strip():
                _add(str(obj["name"]), obj.get("arguments", obj.get("parameters", {})))
        for jm in _BARE_JSON_CALL_RE.finditer(text):
            _add(jm.group(1), _coerce_tool_args(jm.group(2)))
            consumed.append((jm.start(), jm.end()))

    if not tool_calls:
        tn = re.search(r'"name"\s*:\s*"([A-Za-z0-9_.\-]+)"', text)
        if tn and ("<tool_call>" in text or '"arguments"' in text):
            ip = re.search(r'"image_path"\s*:\s*"([^"]*)', text)
            _add(tn.group(1), {"image_path": ip.group(1)} if ip else {})

    clean_content = text
    for start, end in sorted(consumed, reverse=True):
        clean_content = clean_content[:start] + clean_content[end:]
    if tool_calls:
        # Otherwise the assistant turn saved to history teaches the model to
        # emit an empty <tool_call></tool_call> on every subsequent turn.
        clean_content = re.sub(
            r"</?tool_call>|</?function[^>]*>|</?parameter[^>]*>", "", clean_content, flags=re.DOTALL
        )
    clean_content = re.sub(r"<think>\s*</think>\s*", "", clean_content, flags=re.DOTALL)
    clean_content = re.sub(r"```(?:json|tool_code|python)?\s*```", "", clean_content).strip()
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
    # TEST DOUBLE ONLY (LLM_MOCK=1): a canned "model" that walks a minimal
    # tool sequence so the orchestration/plumbing can be exercised without a
    # real VLM. Its hardcoded ordering is NOT authoritative - LandslidePolicy
    # is. Do not port decision logic out of here.
    if not req.tools:
        return None

    available = {
        t.get("function", {}).get("name")
        for t in req.tools
        if isinstance(t, dict) and isinstance(t.get("function"), dict)
    }
    executed: dict[str, Any] = {}
    for msg in req.messages:
        if msg.role != "tool" or not msg.name:
            continue
        try:
            executed[msg.name] = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
        except Exception:
            executed[msg.name] = {}

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

    stage1_output = executed.get("llm.first_pass") if isinstance(executed.get("llm.first_pass"), dict) else None
    segmentation_output = executed.get("seg.run") if isinstance(executed.get("seg.run"), dict) else None
    refinement_output = executed.get("seg.refine") if isinstance(executed.get("seg.refine"), dict) else None
    screening_positive = _cross_check_positive(stage1_output, segmentation_output)
    second_pass_area_ratio_limit = _resolve_seg_llm_second_pass_max_area_ratio()

    if "tiff.info" in available and "tiff.info" not in executed and latest_image:
        return _call("tiff.info", {"image_path": latest_image})
    if "llm.first_pass" in available and "llm.first_pass" not in executed and "tiff.info" in executed:
        return _call("llm.first_pass", {"image_info": executed["tiff.info"]})
    if "seg.run" in available and "seg.run" not in executed and "tiff.info" in executed:
        return _call("seg.run", {"image_info": executed["tiff.info"]})
    if (
        "seg.llm_review" in available
        and screening_positive
        and req.enable_seg_llm_second_pass
        and "seg.llm_review" not in executed
        and "tiff.info" in executed
        and "seg.run" in executed
    ):
        if "seg.refine" in available and "seg.refine" not in executed:
            return _call(
                "seg.refine",
                {"image_info": executed["tiff.info"], "segmentation": executed["seg.run"]},
            )
        if isinstance(refinement_output, dict):
            area_ratio = float(refinement_output.get("area_ratio", 0.0) or 0.0)
            if _region_count(refinement_output) > 0 and (
                second_pass_area_ratio_limit is None or area_ratio <= second_pass_area_ratio_limit
            ):
                return _call(
                    "seg.llm_review",
                    {
                        "refinement": refinement_output,
                        "stage1": executed.get("llm.first_pass", {}),
                        "image_info": executed["tiff.info"],
                    },
                )
    if (
        "region.locate" in available
        and "region.locate" not in executed
        and isinstance(refinement_output, dict)
        and "seg.refine" in executed
    ):
        return _call("region.locate", {"refinement": refinement_output, "image_info": executed.get("tiff.info", {})})
    if "fuse.decision" in available and screening_positive is not None and "fuse.decision" not in executed and all(
        k in executed for k in ("llm.first_pass", "seg.run", "seg.refine", "region.locate", "cls.run", "geo.background", "geo.nearby")
    ):
        args = {
            "stage1": executed["llm.first_pass"],
            "segmentation": executed["seg.run"],
        }
        if isinstance(refinement_output, dict):
            args["refinement"] = refinement_output
        if "cls.run" in executed:
            args["classification"] = executed["cls.run"]
        if "seg.llm_review" in executed:
            args["llm_second_pass"] = (executed.get("seg.llm_review") or {}).get("llm_second_pass")
        if "geo.nearby" in executed and "geo.background" in executed:
            args["geo_context"] = {"nearby": executed["geo.nearby"], "background": executed["geo.background"]}
        elif "geo.nearby" in executed:
            args["geo_context"] = executed["geo.nearby"]
        elif "geo.background" in executed:
            args["geo_context"] = executed["geo.background"]
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
