from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any
from urllib import error, request

from src.models import dual_head_client
from src.models import cruden_varnes


def _parse_classifier_output(stdout_text: str) -> dict[str, Any]:
    text = str(stdout_text or "").strip()
    if not text:
        raise RuntimeError("classifier returned empty stdout")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    prefix = "CLS_RESULT_JSON\t"
    for line in reversed(lines):
        idx = line.find(prefix)
        if idx != -1:
            return json.loads(line[idx + len(prefix) :])

    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise RuntimeError(f"invalid classifier output: {text[:500]}")


def _fallback_classification(image_path: str, topk: int) -> dict[str, Any]:
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
    cls_device = os.getenv("CLS_DEVICE", "cpu")
    cli_path = Path(__file__).resolve().parents[2] / "scripts" / "cls_cli_predict.py"
    cmd = [
        cls_env_python,
        str(cli_path),
        "--image",
        image_path,
        "--mmpretrain-root",
        mmpretrain_root,
        "--config",
        config_path,
        "--checkpoint",
        checkpoint_path,
        "--device",
        cls_device,
        "--topk",
        str(topk),
    ]
    if class_mapping_path:
        cmd.extend(["--class-mapping", class_mapping_path])
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=float(os.getenv("CLS_SERVICE_TIMEOUT", "180")),
        check=True,
        env={**os.environ, "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS", "1")},
    )
    return _parse_classifier_output(proc.stdout)


def _convnext_classification(image_path: str, topk: int) -> dict[str, Any]:
    """The pre-dual-head classifier: ConvNeXt-tiny via cls_service, subprocess
    fallback if the service is down. Kept as CLS_BACKEND=convnext / an
    automatic fallback if the dual-head classifier is unavailable."""
    service_url = os.getenv("CLS_SERVICE_URL", "http://127.0.0.1:8004").rstrip("/")
    timeout = float(os.getenv("CLS_SERVICE_TIMEOUT", "180.0"))
    req = request.Request(
        f"{service_url}/predict",
        data=json.dumps({"image_path": image_path, "topk": topk}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (error.URLError, TimeoutError, ConnectionError):
        return _fallback_classification(image_path, topk)


def run_classification(image_info: dict[str, Any]) -> dict[str, Any]:
    image_path = str(image_info.get("image_path", "")).strip()
    empty = {
        "class_id": -1,
        "class_name": "",
        "confidence": 0.0,
        "topk": [],
    }
    if not image_path:
        return empty
    topk = int(os.getenv("CLS_TOPK", "5"))
    backend = os.getenv("CLS_BACKEND", "dualhead").strip().lower()
    second_opinion = os.getenv("CLS_SECOND_OPINION", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    if backend == "dualhead":
        # Reuses the same classification stage1 (llm.first_pass) already
        # triggered for this image -- one dual-head forward pass serves both
        # tools, see src.models.dual_head_client.
        result = dual_head_client.classify(image_path, topk=topk)
        if int(result.get("class_id", -1)) < 0:
            # Dual-head unavailable/failed for this call -- fall back to ConvNeXt
            # rather than silently returning an empty classification.
            return _convnext_classification(image_path, topk)

        llm_result = {
            "class_id": result["class_id"],
            "class_name": result["class_name"],
            "confidence": result["confidence"],
            "topk": result.get("topk", []),
        }
        if not second_opinion:
            return llm_result

        # Image-classifier (ConvNeXt) second opinion. The VLM opinion comes
        # from the dual-head classifier; when the two disagree on the subtype,
        # fall back to the shared Cruden & Varnes parent (movement/material),
        # and flag a conflict when even the parent disagrees. A second-opinion
        # failure degrades gracefully to the plain VLM result.
        try:
            img_result = _convnext_classification(image_path, topk)
        except Exception as exc:
            logging.warning("Cruden & Varnes second opinion unavailable (%s); using VLM result only.", exc)
            return llm_result

        resolution = cruden_varnes.resolve_disagreement(
            llm_result.get("class_name", ""),
            img_result.get("class_name", ""),
        )
        merged = dict(llm_result)
        merged.update(
            {
                "resolution": resolution["resolution"],
                "parent_class": resolution["parent_class"],
                "conflict": bool(resolution["conflict"]),
                "classification_note": resolution["note"],
                "sources": {
                    "vlm": llm_result,
                    "image_classifier": img_result,
                },
            }
        )
        # On parent-level fallback the resolved name is the parent (e.g.
        # "Flow" / "Rock"), which no longer maps to one of the 8 class ids.
        if resolution["resolution"] in ("movement_parent_fallback", "material_parent_fallback"):
            merged["class_name"] = resolution["class_name"] or llm_result["class_name"]
            merged["class_id"] = -1
        return merged

    return _convnext_classification(image_path, topk)
