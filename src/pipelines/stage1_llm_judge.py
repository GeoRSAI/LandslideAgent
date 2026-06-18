from __future__ import annotations

from src.models.llm_client import llm_judge_has_landslide


def run_stage1(image_info: dict) -> dict:
    return llm_judge_has_landslide({"image_info": image_info})
