from __future__ import annotations

from src.models.cls_infer import run_classification


def run_stage3(image_info: dict) -> dict:
    return run_classification(image_info)
