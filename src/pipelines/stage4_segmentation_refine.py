from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.models.llm_client import llm_second_pass_on_boxed_image
from src.pipelines.stage2_segmentation import run_stage2
from src.utils import build_artifact_path


def _stage1_assessment_label(stage1: dict | None) -> str:
    stage1 = stage1 or {}
    label = " ".join(str(stage1.get("assessment_label", "") or "").strip().lower().split())
    if label in {"likely", "unlikely", "uncertain", "error"}:
        return label

    evidence = " ".join(str(stage1.get("evidence", "") or "").strip().lower().split())
    for candidate in ("likely", "unlikely", "uncertain"):
        if evidence.startswith(candidate):
            return candidate
    return ""


def _stage1_scene_description(stage1: dict | None) -> str:
    stage1 = stage1 or {}
    explicit = " ".join(str(stage1.get("scene_description", "") or "").strip().split())
    if explicit:
        return explicit

    evidence = " ".join(str(stage1.get("evidence", "") or "").strip().split())
    if not evidence:
        return ""

    for separator in ("|", ":", "-"):
        if separator not in evidence:
            continue
        head, tail = evidence.split(separator, 1)
        normalized_head = " ".join(head.strip().lower().split())
        if normalized_head in {"likely", "unlikely", "uncertain"}:
            return tail.strip() or evidence

    lower = evidence.lower()
    if lower.startswith("likely"):
        return evidence[len("likely"):].lstrip(" |:-") or evidence
    if lower.startswith("unlikely"):
        return evidence[len("unlikely"):].lstrip(" |:-") or evidence
    if lower.startswith("uncertain"):
        return evidence[len("uncertain"):].lstrip(" |:-") or evidence
    return evidence


def _stage1_positive(stage1: dict | None) -> bool:
    stage1 = stage1 or {}
    if bool(stage1.get("has_landslide", False)):
        return True
    return _stage1_assessment_label(stage1) == "likely"


def _annotate_review_input(
    review_input: dict,
    *,
    stage1: dict | None,
    regions: list[dict],
) -> dict:
    if not review_input:
        return review_input

    stage1_is_positive = _stage1_positive(stage1)
    review_input["review_purpose"] = (
        "description_only" if stage1_is_positive and bool(regions) else "verification"
    )
    review_input["stage1_positive"] = stage1_is_positive

    stage1_label = _stage1_assessment_label(stage1)
    if stage1_label:
        review_input["stage1_assessment_label"] = stage1_label

    stage1_scene_description = _stage1_scene_description(stage1)
    if stage1_scene_description:
        review_input["stage1_scene_description"] = stage1_scene_description

    return review_input


def _render_region_overlay(image_path: str, regions: list[dict]) -> str:
    if not image_path or not regions:
        return ""
    try:
        base = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        h, w = base.shape[:2]
        if h <= 0 or w <= 0:
            return ""
        region_mask = np.zeros((h, w), dtype=bool)
        for region in regions:
            bbox = region.get("bbox") or []
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in bbox]
            ix1 = max(0, min(w - 1, int(np.floor(x1))))
            iy1 = max(0, min(h - 1, int(np.floor(y1))))
            ix2 = max(ix1 + 1, min(w, int(np.ceil(x2))))
            iy2 = max(iy1 + 1, min(h, int(np.ceil(y2))))
            region_mask[iy1:iy2, ix1:ix2] = True

        if not region_mask.any():
            return ""

        edge = _thicken_edge(_mask_boundary(region_mask))
        tint = np.array([255, 140, 140], dtype=np.uint8)
        blended = (base[region_mask].astype(np.float32) * 0.78 + tint.astype(np.float32) * 0.22).clip(0, 255)
        base[region_mask] = blended.astype(np.uint8)
        base[edge] = np.array([255, 64, 64], dtype=np.uint8)
        overlay = build_artifact_path("outputs/seg_refine", image_path, "seg_refine_overlay.png")
        Image.fromarray(base).save(overlay)
        return str(overlay)
    except Exception:
        return ""


def _load_binary_mask(mask_path: str) -> np.ndarray:
    if not mask_path or not Path(mask_path).exists():
        return np.zeros((0, 0), dtype=bool)
    try:
        mask = Image.open(mask_path)
        arr = np.array(mask)
        if arr.ndim == 3:
            arr = arr[..., 0]
        return arr != 0
    except Exception:
        return np.zeros((0, 0), dtype=bool)


def _mask_bbox_regions(binary_mask: np.ndarray, area_ratio: float) -> list[dict]:
    if binary_mask.size == 0:
        return []
    ys, xs = np.where(binary_mask)
    if ys.size == 0 or xs.size == 0:
        return []
    x1 = float(xs.min())
    y1 = float(ys.min())
    x2 = float(xs.max() + 1)
    y2 = float(ys.max() + 1)
    return [
        {
            "tile_id": 0,
            "bbox": [x1, y1, x2, y2],
            "class_id": 0,
            "source": "segmentation_mask",
            "area_ratio": float(area_ratio),
        }
    ]


def _mask_boundary(binary_mask: np.ndarray) -> np.ndarray:
    if binary_mask.size == 0:
        return np.zeros((0, 0), dtype=bool)
    h, w = binary_mask.shape
    if h < 3 or w < 3:
        return binary_mask.copy()

    interior = np.zeros_like(binary_mask, dtype=bool)
    interior[1:-1, 1:-1] = (
        binary_mask[1:-1, 1:-1]
        & binary_mask[:-2, 1:-1]
        & binary_mask[2:, 1:-1]
        & binary_mask[1:-1, :-2]
        & binary_mask[1:-1, 2:]
    )
    edge = binary_mask & (~interior)
    return edge


def _thicken_edge(edge: np.ndarray) -> np.ndarray:
    if edge.size == 0:
        return edge
    h, w = edge.shape
    padded = np.pad(edge, 1, mode="constant", constant_values=False)
    thick = np.zeros_like(edge, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            thick |= padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
    return thick


def _render_segmentation_boundary_overlay(
    image_path: str,
    mask_path: str,
) -> str:
    if not image_path or not mask_path or not Path(image_path).exists():
        return ""
    binary_mask = _load_binary_mask(mask_path)
    if binary_mask.size == 0:
        return ""
    try:
        base = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        if base.shape[0] != binary_mask.shape[0] or base.shape[1] != binary_mask.shape[1]:
            resized = Image.fromarray(binary_mask.astype(np.uint8) * 255).resize(
                (base.shape[1], base.shape[0]),
                resample=Image.NEAREST,
            )
            binary_mask = np.array(resized, dtype=np.uint8) > 0

        edge = _mask_boundary(binary_mask)
        edge = _thicken_edge(edge)
        base[edge] = np.array([255, 64, 64], dtype=np.uint8)
        overlay = build_artifact_path("outputs/review_patches", image_path, "seg_boundary_review.png")
        Image.fromarray(base).save(overlay)
        return str(overlay)
    except Exception:
        return ""


def _normalize_external_regions(raw: object) -> list[dict] | None:
    if not isinstance(raw, list):
        return None
    normalized: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            normalized.append(
                {
                    "tile_id": int(item.get("tile_id", 0) or 0),
                    "bbox": [float(v) for v in bbox],
                    "class_id": int(item.get("class_id", 0) or 0),
                }
            )
            if "score" in item:
                normalized[-1]["legacy_score_ignored"] = True
        except Exception:
            continue
    if not normalized:
        return None
    return normalized


def _build_boundary_review_input(
    image_path: str,
    regions: list[dict],
    *,
    overlay_path: str,
    mask_path: str = "",
) -> dict:
    if not image_path or not regions:
        return {}
    resolved_overlay_path = str(overlay_path or "").strip()
    if not resolved_overlay_path:
        resolved_overlay_path = _render_region_overlay(image_path, regions)
    if not resolved_overlay_path:
        return {}
    normalized_regions = []
    for region in regions:
        bbox = region.get("bbox") or []
        if len(bbox) != 4:
            continue
        normalized_regions.append(
            {
                "tile_id": int(region.get("tile_id", 0) or 0),
                "bbox": [float(v) for v in bbox],
                "class_id": int(region.get("class_id", 0) or 0),
            }
        )
        if "score" in region:
            normalized_regions[-1]["legacy_score_ignored"] = True
    if not normalized_regions:
        return {}
    return {
        "review_mode": "seg_boundary_whole_image",
        "overlay_source": "segmentation_mask_boundary",
        "image_path": image_path,
        "review_image_path": resolved_overlay_path,
        "overlay_path": resolved_overlay_path,
        "mask_path": str(mask_path or ""),
        "regions": normalized_regions,
        "reviewed_regions": len(normalized_regions),
    }


def _resolve_image_info(image_info: dict | None, tiles: list[dict]) -> dict:
    resolved = dict(image_info or {})
    image_path = str(resolved.get("image_path", "") or "").strip()
    if image_path:
        if "width" in resolved and "height" in resolved:
            return resolved
        try:
            with Image.open(image_path) as img:
                resolved.setdefault("width", int(img.width))
                resolved.setdefault("height", int(img.height))
        except Exception:
            pass
        return resolved

    if tiles:
        first = tiles[0] if isinstance(tiles[0], dict) else {}
        tile_path = str(first.get("image_path", "") or "").strip()
        if tile_path:
            resolved["image_path"] = tile_path
            try:
                with Image.open(tile_path) as img:
                    resolved.setdefault("width", int(img.width))
                    resolved.setdefault("height", int(img.height))
            except Exception:
                pass
    return resolved


def _normalize_max_area_ratio(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        return None
    if ratio <= 0.0:
        return None
    return ratio


def run_stage4(
    tiles: list[dict],
    *,
    image_info: dict | None = None,
    stage1: dict | None = None,
    regions: list[dict] | None = None,
    segmentation: dict | None = None,
    run_llm_second_pass: bool = False,
    llm_second_pass_max_area_ratio: float | None = None,
) -> dict:
    max_area_ratio = _normalize_max_area_ratio(llm_second_pass_max_area_ratio)
    resolved_tiles = list(tiles or [])
    resolved_image_info = _resolve_image_info(image_info, resolved_tiles)
    image_path = str(resolved_image_info.get("image_path", "") or "").strip()
    width = int(resolved_image_info.get("width", 0) or 0)
    height = int(resolved_image_info.get("height", 0) or 0)

    if not resolved_tiles and image_path and width > 0 and height > 0:
        resolved_tiles = [
            {
                "tile_id": 0,
                "x": 0,
                "y": 0,
                "w": width,
                "h": height,
                "image_path": image_path,
            }
        ]

    external_regions = _normalize_external_regions(regions)
    if isinstance(segmentation, dict):
        resolved_segmentation = dict(segmentation)
    elif external_regions is not None:
        resolved_segmentation = {
            "mask_path": "",
            "overlay_path": "",
            "landslide_pixels": 0,
            "area_ratio": 0.0,
            "polygon_count": 1 if external_regions else 0,
        }
    else:
        resolved_segmentation = run_stage2(resolved_image_info)
    mask_path = str(resolved_segmentation.get("mask_path", "") or "").strip()
    area_ratio = float(resolved_segmentation.get("area_ratio", 0.0) or 0.0)
    selected_regions = (
        list(external_regions)
        if external_regions is not None
        else _mask_bbox_regions(_load_binary_mask(mask_path), area_ratio)
    )

    if area_ratio <= 0.0 and selected_regions and width > 0 and height > 0:
        region_area = 0.0
        for region_item in selected_regions:
            bbox = region_item.get("bbox") or []
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in bbox]
            region_area += max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_ratio = region_area / float(width * height)
        resolved_segmentation["area_ratio"] = area_ratio

    candidate_tiles = [
        t for t in resolved_tiles if any(r.get("tile_id") == t.get("tile_id") for r in selected_regions)
    ]

    if not candidate_tiles and width > 0 and height > 0 and image_path:
        candidate_tiles = [
            {
                "tile_id": 0,
                "x": 0,
                "y": 0,
                "w": width,
                "h": height,
                "image_path": image_path,
            }
        ]

    overlay_path = str(resolved_segmentation.get("boundary_overlay_path", "") or "").strip()
    if not overlay_path:
        overlay_path = _render_segmentation_boundary_overlay(image_path, mask_path)
    if not overlay_path:
        overlay_path = _render_region_overlay(image_path, selected_regions)

    review_image = image_path
    review_input = _build_boundary_review_input(
        review_image,
        selected_regions,
        overlay_path=overlay_path,
        mask_path=mask_path,
    )
    review_input = _annotate_review_input(
        review_input,
        stage1=stage1,
        regions=selected_regions,
    )
    skip_for_large_area = bool(max_area_ratio is not None and area_ratio > max_area_ratio)
    llm2 = (
        llm_second_pass_on_boxed_image(review_input)
        if run_llm_second_pass and review_input and (not skip_for_large_area)
        else None
    )
    review_inputs = [review_input] if review_input else []

    return {
        "regions": selected_regions,
        "candidate_tiles": candidate_tiles,
        "llm_review_input": review_input,
        "llm_review_tiles": review_inputs,
        "overlay_path": overlay_path,
        "llm_second_pass": llm2,
        "area_ratio": area_ratio,
        "llm_second_pass_skipped_for_large_area": skip_for_large_area,
        "source": "segmentation_mask",
        "mask_path": mask_path,
        "review_candidates": len(selected_regions),
        "segmentation": resolved_segmentation,
    }


def run_stage4_llm_review(
    refinement: dict | None,
    *,
    image_info: dict | None = None,
    stage1: dict | None = None,
    llm_second_pass_max_area_ratio: float | None = None,
) -> dict:
    max_area_ratio = _normalize_max_area_ratio(llm_second_pass_max_area_ratio)
    refinement = refinement or {}
    review_input = refinement.get("llm_review_input") if isinstance(refinement.get("llm_review_input"), dict) else {}
    review_inputs = refinement.get("llm_review_tiles")
    if not review_input and isinstance(review_inputs, list) and review_inputs and isinstance(review_inputs[0], dict):
        review_input = review_inputs[0]
    area_ratio = float(refinement.get("area_ratio", 0.0) or 0.0)

    selected_regions = refinement.get("regions", []) if isinstance(refinement.get("regions"), list) else []
    review_input = _annotate_review_input(
        review_input,
        stage1=stage1,
        regions=selected_regions,
    )

    if not review_input:
        resolved_info = _resolve_image_info(image_info, refinement.get("candidate_tiles", []) or [])
        review_image = str(resolved_info.get("image_path", "")).strip()
        segmentation = refinement.get("segmentation") if isinstance(refinement.get("segmentation"), dict) else {}
        mask_path = str(refinement.get("mask_path", "") or segmentation.get("mask_path", "") or "").strip()
        overlay_path = str(refinement.get("overlay_path", "") or "").strip()
        if not overlay_path:
            overlay_path = _render_segmentation_boundary_overlay(review_image, mask_path)
        if not overlay_path:
            overlay_path = _render_region_overlay(review_image, selected_regions)
        review_input = _build_boundary_review_input(
            review_image,
            selected_regions,
            overlay_path=overlay_path,
            mask_path=mask_path,
        )
        review_input = _annotate_review_input(
            review_input,
            stage1=stage1,
            regions=selected_regions,
        )
        if not review_input and image_info:
            rebuilt = run_stage4(
                [],
                image_info=resolved_info,
                stage1=stage1,
                regions=refinement.get("regions"),
                segmentation=segmentation,
                run_llm_second_pass=False,
                llm_second_pass_max_area_ratio=llm_second_pass_max_area_ratio,
            )
            review_input = rebuilt.get("llm_review_input") if isinstance(rebuilt.get("llm_review_input"), dict) else {}
            area_ratio = float(rebuilt.get("area_ratio", area_ratio) or area_ratio or 0.0)

    skip_for_large_area = bool(max_area_ratio is not None and area_ratio > max_area_ratio)
    llm2 = llm_second_pass_on_boxed_image(review_input) if review_input and (not skip_for_large_area) else None
    review_inputs = [review_input] if review_input else []
    review_candidates = len(review_input.get("regions", [])) if review_input else int(len(selected_regions or []))
    return {
        "llm_review_input": review_input,
        "llm_review_tiles": review_inputs,
        "llm_second_pass": llm2,
        "area_ratio": area_ratio,
        "review_candidates": review_candidates,
        "llm_second_pass_skipped_for_large_area": skip_for_large_area,
    }
