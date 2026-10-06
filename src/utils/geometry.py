"""Frame-relative geometry helpers shared by the fusion and LLM-client layers.

Previously duplicated verbatim in ``src/pipelines/stage5_fusion.py`` and
``src/models/llm_client.py``. The frame is split into vertical/horizontal thirds
purely for a human-readable position phrase - this is presentation, not a
decision rule.
"""
from __future__ import annotations

import os
from typing import Any

_LOW_THIRD = 1.0 / 3.0
_HIGH_THIRD = 2.0 / 3.0


def first_valid_bbox(refinement: dict[str, Any] | None) -> list[float] | None:
    regions = (refinement or {}).get("regions", [])
    if not isinstance(regions, list):
        return None
    for region in regions:
        if not isinstance(region, dict):
            continue
        bbox = region.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            try:
                return [float(v) for v in bbox]
            except (TypeError, ValueError):
                continue
    return None


def bbox_text(bbox: list[float] | None) -> str:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return "not available"
    return "[" + ", ".join(f"{float(v):.1f}" for v in bbox) + "]"


def estimate_frame_extent(refinement: dict[str, Any] | None) -> tuple[float, float]:
    candidate_tiles = (refinement or {}).get("candidate_tiles", [])
    if not isinstance(candidate_tiles, list):
        return 0.0, 0.0
    max_x = 0.0
    max_y = 0.0
    for tile in candidate_tiles:
        if not isinstance(tile, dict):
            continue
        try:
            tile_x = float(tile.get("x", 0.0) or 0.0)
            tile_y = float(tile.get("y", 0.0) or 0.0)
            tile_w = float(tile.get("w", 0.0) or 0.0)
            tile_h = float(tile.get("h", 0.0) or 0.0)
        except Exception:
            continue
        max_x = max(max_x, tile_x + max(0.0, tile_w))
        max_y = max(max_y, tile_y + max(0.0, tile_h))
    return max_x, max_y


def _frame_size(
    refinement: dict[str, Any] | None, image_info: dict[str, Any] | None
) -> tuple[float, float]:
    """Frame extent in pixels: prefer explicit image_info, fall back to the
    bounding box of the refinement's candidate tiles."""
    info = image_info if isinstance(image_info, dict) else {}
    try:
        w = float(info.get("width") or 0.0)
        h = float(info.get("height") or 0.0)
    except (TypeError, ValueError):
        w = h = 0.0
    if w > 0.0 and h > 0.0:
        return w, h
    return estimate_frame_extent(refinement)


def locate_primary_candidate(
    refinement: dict[str, Any] | None,
    image_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Structured frame-relative position of the primary landslide candidate.

    The frame is read as a 3x3 grid (vertical/horizontal thirds). Purely
    descriptive - it plays no role in the landslide decision.
    """
    bbox = first_valid_bbox(refinement)
    if bbox is None:
        return {
            "available": False,
            "position": "unknown",
            "description": "No retained candidate region was available for frame-relative positioning.",
        }

    frame_w, frame_h = _frame_size(refinement, image_info)
    if frame_w <= 0.0 or frame_h <= 0.0:
        return {
            "available": False,
            "position": "unknown",
            "bbox": bbox,
            "description": (
                f"Primary candidate region bbox={bbox_text(bbox)} pixels; "
                "normalized frame position is unavailable (frame size unknown)."
            ),
        }

    x1, y1, x2, y2 = bbox
    # Prefer the centroid of the segmentation mask: for large or irregular
    # bodies the bbox spans most of the frame and its centre collapses to
    # "middle-center" even when the landslide mass sits near an edge.
    centroid = _mask_centroid(refinement)
    if centroid is not None:
        rel_x, rel_y = centroid
        position_source = "mask_centroid"
    else:
        rel_x = ((x1 + x2) / 2.0) / frame_w
        rel_y = ((y1 + y2) / 2.0) / frame_h
        position_source = "bbox_center"
    horiz = "left" if rel_x < _LOW_THIRD else ("right" if rel_x > _HIGH_THIRD else "center")
    vert = "upper" if rel_y < _LOW_THIRD else ("lower" if rel_y > _HIGH_THIRD else "middle")
    position = f"{vert}-{horiz}"
    return {
        "available": True,
        "position": position,
        "vertical": vert,
        "horizontal": horiz,
        "rel_center": [round(rel_x, 4), round(rel_y, 4)],
        "position_source": position_source,
        "bbox": bbox,
        "frame_size": [round(frame_w, 1), round(frame_h, 1)],
        "description": (
            f"Primary candidate region lies in the {position} part of the frame; "
            f"bbox={bbox_text(bbox)} pixels."
        ),
    }


def _mask_centroid(refinement: dict[str, Any] | None) -> tuple[float, float] | None:
    """Relative (x, y) centroid of the refinement mask, or None if unavailable."""
    ref = refinement or {}
    review = ref.get("llm_review_input") if isinstance(ref.get("llm_review_input"), dict) else {}
    mask_path = str(ref.get("mask_path") or review.get("mask_path") or "").strip()
    if not mask_path or not os.path.isfile(mask_path):
        return None
    try:
        import numpy as np
        from PIL import Image

        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
    except Exception:
        return None
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    h, w = mask.shape
    return float(xs.mean()) / float(w), float(ys.mean()) / float(h)


def describe_primary_candidate_position(refinement: dict[str, Any] | None) -> str:
    return str(locate_primary_candidate(refinement)["description"])
