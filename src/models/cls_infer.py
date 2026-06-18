from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any
from urllib import error, request


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
    cls_env_python = os.getenv("CLS_ENV_PYTHON", "python")
    mmpretrain_root = os.getenv("MMPRETRAIN_ROOT", "")
    config_path = os.getenv("CLS_CONFIG_PATH", "")
    checkpoint_path = os.getenv("CLS_CHECKPOINT_PATH", "")
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


def run_classification(image_info: dict[str, Any]) -> dict[str, Any]:
    image_path = str(image_info.get("image_path", "")).strip()
    if not image_path:
        return {
            "class_id": -1,
            "class_name": "",
            "confidence": 0.0,
            "topk": [],
        }
    topk = int(os.getenv("CLS_TOPK", "5"))
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
