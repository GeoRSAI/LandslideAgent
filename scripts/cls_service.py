from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Landslide Classification Service")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ClassificationRequest(BaseModel):
    image_path: str
    topk: int = 5


def _parse_classifier_output(stdout_text: str) -> dict:
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


def _service_config() -> dict[str, str]:
    return {
        "env_python": os.getenv("CLS_ENV_PYTHON", "python"),
        "mmpretrain_root": os.getenv("MMPRETRAIN_ROOT", ""),
        "config_path": os.getenv("CLS_CONFIG_PATH", ""),
        "checkpoint_path": os.getenv("CLS_CHECKPOINT_PATH", ""),
        "class_mapping_path": os.getenv("CLS_CLASS_MAPPING_PATH", ""),
        "device": os.getenv("CLS_DEVICE", "cpu"),
    }


def _run_classification(image_path: str, topk: int) -> dict:
    cfg = _service_config()
    cli_path = PROJECT_ROOT / "scripts" / "cls_cli_predict.py"
    cmd = [
        cfg["env_python"],
        str(cli_path),
        "--image",
        image_path,
        "--mmpretrain-root",
        cfg["mmpretrain_root"],
        "--config",
        cfg["config_path"],
        "--checkpoint",
        cfg["checkpoint_path"],
        "--device",
        cfg["device"],
        "--topk",
        str(topk),
    ]
    if cfg["class_mapping_path"]:
        cmd.extend(["--class-mapping", cfg["class_mapping_path"]])
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=float(os.getenv("CLS_SERVICE_TIMEOUT", "180")),
        check=False,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"classification exited with code {proc.returncode}")
    return _parse_classifier_output(proc.stdout)


@app.post("/predict")
def predict(req: ClassificationRequest):
    try:
        return _run_classification(req.image_path, req.topk)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Classification failed: {exc}") from exc


@app.get("/health")
def health():
    cfg = _service_config()
    problems = []
    for key in ("env_python", "mmpretrain_root", "config_path", "checkpoint_path"):
        if not Path(cfg[key]).exists():
            problems.append(f"missing_{key}:{cfg[key]}")
    if cfg["class_mapping_path"] and (not Path(cfg["class_mapping_path"]).exists()):
        problems.append(f"missing_class_mapping_path:{cfg['class_mapping_path']}")
    return {
        "status": "ok" if not problems else "error",
        "model_loaded": not problems,
        "startup_error": "; ".join(problems),
        "env_python": cfg["env_python"],
        "mmpretrain_root": cfg["mmpretrain_root"],
        "config_path": cfg["config_path"],
        "checkpoint_path": cfg["checkpoint_path"],
        "class_mapping_path": cfg["class_mapping_path"],
        "device": cfg["device"],
    }
