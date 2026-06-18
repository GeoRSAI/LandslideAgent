from __future__ import annotations

import re

from src.models.llm_client import llm_generate_final_report


REQUIRED_SECTIONS = [
    "Final Decision Report",
    "Conclusion",
    "Evidence Summary",
    "Image and Spatial Interpretation",
    "Landslide Typology (Reference Only)",
    "Geographic and Exposure Context",
    "Reliability and Uncertainty",
    "Final Determination",
]


_INCOMPLETE_TAIL_PATTERNS = (
    " a",
    " an",
    " the",
    " of",
    " to",
    " near",
    " near a",
    " adjacent to",
    " adjacent to a",
    " with",
    " by",
    " in",
    " on",
    " at",
    " from",
)


def _drop_incomplete_tail_sentence(text: str) -> str:
    cleaned = " ".join(str(text or "").split()).strip()
    if not cleaned:
        return ""
    stripped = cleaned.rstrip()
    lowered = stripped.rstrip(".!?").lower()
    has_bad_short_tail = bool(
        re.search(
            r"\b(?:near|adjacent to|beside|next to|with|of|to|from|in|on|at|by|for)\s+(?:a|an|the)\s+[a-z]{1,3}$",
            lowered,
        )
    )
    if has_bad_short_tail or any(lowered.endswith(pattern) for pattern in _INCOMPLETE_TAIL_PATTERNS):
        last_boundary = max(stripped.rfind("."), stripped.rfind("!"), stripped.rfind("?"))
        if last_boundary >= 0:
            return stripped[: last_boundary + 1].strip()
        return ""
    return stripped


def _clean_text(value: object, default: str = "Not available.") -> str:
    text = " ".join(str(value or "").split()).strip()
    text = text.replace("…", ".")
    while "..." in text:
        text = text.replace("...", ".")
    while ".." in text:
        text = text.replace("..", ".")
    text = text.replace(" .", ".")
    text = _drop_incomplete_tail_sentence(text)
    return text or default


def _stage1_scene_description(stage1: dict | None) -> str:
    stage1 = stage1 or {}
    explicit_description = _clean_text(stage1.get("scene_description", ""), "")
    if explicit_description:
        return explicit_description

    evidence = _clean_text(stage1.get("evidence", ""), "")
    if not evidence:
        return "Whole-image VLM scene description is unavailable."

    lower_evidence = evidence.lower()
    for separator in ("|", ":"):
        if separator not in evidence:
            continue
        head, tail = evidence.split(separator, 1)
        normalized_head = " ".join(head.strip().lower().split())
        if normalized_head in {"likely", "unlikely", "uncertain"}:
            extracted = _clean_text(tail, "")
            if extracted:
                return extracted

    if lower_evidence.startswith("unlikely"):
        return evidence[len("unlikely"):].lstrip(" |:-") or evidence
    if lower_evidence.startswith("likely"):
        return evidence[len("likely"):].lstrip(" |:-") or evidence
    if lower_evidence.startswith("uncertain"):
        return evidence[len("uncertain"):].lstrip(" |:-") or evidence
    return evidence


def _stage1_assessment_label(stage1: dict | None) -> str:
    stage1 = stage1 or {}
    label = _clean_text(stage1.get("assessment_label", ""), "").lower()
    if label in {"likely", "unlikely", "uncertain", "error"}:
        return label

    evidence = _clean_text(stage1.get("evidence", ""), "").lower()
    for candidate in ("unlikely", "likely", "uncertain"):
        if evidence.startswith(candidate):
            return candidate
    return ""


def _strip_leading_assessment_token(text: object) -> str:
    cleaned = _clean_text(text, "")
    if not cleaned:
        return ""

    for separator in ("|", ":"):
        if separator not in cleaned:
            continue
        head, tail = cleaned.split(separator, 1)
        normalized_head = " ".join(head.strip().lower().split())
        if normalized_head in {
            "likely",
            "unlikely",
            "uncertain",
            "support",
            "oppose",
            "positive",
            "negative",
            "describe",
            "descriptive",
        }:
            stripped = _clean_text(tail, "")
            if stripped:
                return stripped

    lowered = cleaned.lower()
    for prefix in ("likely", "unlikely", "uncertain", "support", "describe"):
        if lowered.startswith(prefix):
            stripped = cleaned[len(prefix) :].lstrip(" |:-")
            if stripped:
                return _clean_text(stripped, "")

    return cleaned


def _strip_inline_assessment_tokens(text: object) -> str:
    cleaned = _clean_text(text, "")
    if not cleaned:
        return ""

    replacements = (
        "Likely | ",
        "likely | ",
        "Unlikely | ",
        "unlikely | ",
        "Uncertain | ",
        "uncertain | ",
        "Support | ",
        "support | ",
        "Describe | ",
        "describe | ",
    )
    for marker in replacements:
        cleaned = cleaned.replace(marker, "")

    return _clean_text(cleaned, "")


def _ensure_sentence(text: str) -> str:
    cleaned = _clean_text(text, "")
    if not cleaned:
        return ""
    return cleaned if cleaned.endswith((".", "!", "?")) else f"{cleaned}."


def _second_pass_reviewed_regions(llm_second_pass: dict | None) -> int:
    llm_second_pass = llm_second_pass or {}
    reviewed_regions = int(llm_second_pass.get("reviewed_regions", 0) or 0)
    if reviewed_regions > 0:
        return reviewed_regions
    reviewed_tiles = llm_second_pass.get("reviewed_tiles")
    if isinstance(reviewed_tiles, list):
        return len(reviewed_tiles)
    return 0


def _second_pass_purpose(llm_second_pass: dict | None) -> str:
    llm_second_pass = llm_second_pass or {}
    purpose = _clean_text(llm_second_pass.get("review_purpose", ""), "").lower()
    if purpose in {"verification", "description_only"}:
        return purpose
    decision = _clean_text(llm_second_pass.get("decision", ""), "").lower()
    if decision in {"descriptive", "description_only"}:
        return "description_only"
    return "verification"


def _second_pass_decision(llm_second_pass: dict | None) -> str:
    llm_second_pass = llm_second_pass or {}
    if _second_pass_purpose(llm_second_pass) == "description_only":
        return "descriptive"
    decision = _clean_text(llm_second_pass.get("decision", ""), "").lower()
    if decision in {"positive", "negative", "uncertain", "error", "unavailable", "descriptive"}:
        return decision
    supports = llm_second_pass.get("supports_landslide")
    if supports is True:
        return "positive"
    if supports is False:
        return "negative"
    positive_tiles = llm_second_pass.get("positive_tiles")
    if isinstance(positive_tiles, list) and positive_tiles:
        return "positive"
    if _second_pass_reviewed_regions(llm_second_pass) > 0:
        return "uncertain"
    return ""


def _second_pass_workflow_text(
    refinement: dict | None,
    llm_second_pass: dict | None,
    region_area_ratio: float,
) -> str:
    refinement = refinement or {}
    llm_second_pass = llm_second_pass or {}
    reviewed_regions = _second_pass_reviewed_regions(llm_second_pass)
    decision = _second_pass_decision(llm_second_pass)
    purpose = _second_pass_purpose(llm_second_pass)
    evidence = _strip_leading_assessment_token(llm_second_pass.get("evidence", ""))
    review_mode = " ".join(str(llm_second_pass.get("review_mode", "") or "").strip().lower().split())
    refinement_source = " ".join(str(refinement.get("source", "") or "").strip().lower().split())
    boundary_review = review_mode.startswith("seg_") or ("seg" in refinement_source) or ("mask" in refinement_source)
    review_object = "segmentation boundaries" if boundary_review else "candidate boxes"

    if reviewed_regions > 0 or evidence:
        workflow = f"This workflow performed a second-pass VLM review on the full image with {review_object} overlaid"
        if purpose == "description_only":
            if reviewed_regions > 0:
                workflow += f", explicitly supplementing the spatial description of {reviewed_regions} highlighted region(s)"
            workflow += ", after first-pass screening and segmentation-guided refinement already indicated landslide presence."
        else:
            if reviewed_regions > 0:
                workflow += f", explicitly reconsidering {reviewed_regions} highlighted region(s)"
            if decision == "positive":
                workflow += ", and the whole-image second-pass review supported the original-image interpretation."
            elif decision == "negative":
                workflow += ", and the whole-image second-pass review raised counter-evidence against the initial interpretation."
            elif decision == "uncertain":
                workflow += ", but the whole-image second-pass review remained inconclusive."
            else:
                workflow += "."
        if evidence:
            workflow += f" Review note: {evidence}"
        return workflow

    if refinement.get("llm_second_pass_skipped_for_large_area"):
        return (
            "A second-pass whole-image review was not run for this analysis, even though the candidate region "
            f"covered about {region_area_ratio:.4f} of the frame."
        )

    return "No second-pass whole-image review was used in this workflow."


def _screening_decision(stage1: dict | None, refinement: dict | None) -> dict[str, object]:
    stage1 = stage1 or {}
    refinement = refinement or {}
    regions = refinement.get("regions", [])
    region_count = len(regions) if isinstance(regions, list) else 0
    stage1_positive = bool(stage1.get("has_landslide", False))
    refinement_positive = region_count > 0
    return {
        "stage1_positive": stage1_positive,
        "refinement_positive": refinement_positive,
        "has_positive_screening": bool(stage1_positive or refinement_positive),
        "region_count": region_count,
    }


def screening_requires_full_analysis(stage1: dict | None, refinement: dict | None) -> bool:
    return bool(_screening_decision(stage1, refinement)["has_positive_screening"])


def _negative_scene_summary(stage1: dict | None, refinement: dict | None) -> str:
    stage1 = stage1 or {}
    evidence = str(stage1.get("evidence", "") or "").strip()
    region_count = int(_screening_decision(stage1, refinement)["region_count"])
    if evidence and region_count == 0:
        return (
            "No landslide indicated after initial screening. "
            f"Whole-image VLM note: {_stage1_scene_description(stage1)}"
        )
    return "No landslide indicated after initial LLM and segmentation-guided screening."



def _refinement_source(refinement: dict | None) -> str:
    refinement = refinement or {}
    source = " ".join(str(refinement.get("source", "") or "").strip().lower().split())
    if source:
        return source

    regions = refinement.get("regions", [])
    if isinstance(regions, list) and regions:
        first = regions[0]
        if isinstance(first, dict):
            item_source = " ".join(str(first.get("source", "") or "").strip().lower().split())
            if item_source:
                return item_source
    return ""


def _segmentation_support(segmentation: dict | None) -> tuple[bool, float, int]:
    segmentation = segmentation or {}
    seg_ratio = float(segmentation.get("area_ratio", 0.0) or 0.0)
    seg_pixels = int(segmentation.get("landslide_pixels", 0) or 0)
    positive = seg_ratio >= 0.01 or (seg_ratio >= 0.005 and seg_pixels >= 512)
    return positive, seg_ratio, seg_pixels


def _classification_confidence(classification: dict | None) -> float:
    try:
        return min(max(float((classification or {}).get("confidence", 0.0) or 0.0), 0.0), 1.0)
    except Exception:
        return 0.0


def _derive_fused_decision(
    *,
    stage1: dict,
    refinement: dict,
    segmentation: dict | None,
    classification: dict | None,
    llm_second_pass: dict | None,
    llm_second_pass_threshold: float,
    min_region_score: float,
    gate: dict | None,
) -> dict[str, object]:
    stage1_label = _stage1_assessment_label(stage1)
    _ = min_region_score  # Kept for backward-compatible calls; region score thresholds are no longer used.
    stage1_positive = bool((stage1 or {}).get("has_landslide", False) or stage1_label == "likely")

    region_count = int(_screening_decision(stage1, refinement)["region_count"])
    refinement_source = _refinement_source(refinement)
    segmentation_guided_refinement = ("seg" in refinement_source) or ("mask" in refinement_source)
    reliable_region_signal = (
        (not segmentation_guided_refinement)
        and region_count > 0
    )

    seg_positive, seg_ratio, seg_pixels = _segmentation_support(segmentation)

    second_pass_decision = _second_pass_decision(llm_second_pass)
    second_pass_purpose = _second_pass_purpose(llm_second_pass)
    second_pass_positive = (
        second_pass_purpose != "description_only"
        and second_pass_decision == "positive"
    )
    second_pass_negative = second_pass_purpose != "description_only" and second_pass_decision == "negative"
    second_pass_descriptive = second_pass_purpose == "description_only"

    positive_votes = sum(
        int(flag)
        for flag in (
            stage1_positive,
            reliable_region_signal,
            seg_positive,
            second_pass_positive,
        )
    )
    has_landslide = positive_votes >= 2

    region_area_ratio = float((gate or {}).get("area_ratio", refinement.get("area_ratio", 0.0) or 0.0) or 0.0)
    if not has_landslide:
        severity = "none"
    elif max(seg_ratio, region_area_ratio) >= 0.15:
        severity = "high"
    elif max(seg_ratio, region_area_ratio) >= 0.05:
        severity = "medium"
    else:
        severity = "low"

    return {
        "has_landslide": has_landslide,
        "severity": severity,
        "confidence": None,
        "confidence_source": "not_computed",
        "support_summary": {
            "stage1_positive": stage1_positive,
            "reliable_region_signal": reliable_region_signal,
            "segmentation_positive": seg_positive,
            "second_pass_positive": second_pass_positive,
            "second_pass_negative": second_pass_negative,
            "second_pass_descriptive": second_pass_descriptive,
            "llm_scores_ignored": True,
            "heuristic_scores_removed": True,
            "positive_votes": positive_votes,
            "region_count": region_count,
            "refinement_source": refinement_source or "unknown",
            "segmentation_area_ratio": round(seg_ratio, 6),
        },
    }


def _first_valid_bbox(refinement: dict | None) -> list[float] | None:
    regions = (refinement or {}).get("regions", [])
    if not isinstance(regions, list):
        return None
    for det in regions:
        bbox = det.get("bbox") if isinstance(det, dict) else None
        if isinstance(bbox, list) and len(bbox) == 4:
            try:
                return [float(v) for v in bbox]
            except Exception:
                return None
    return None


def _bbox_text(bbox: list[float] | None) -> str:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return "not available"
    return "[" + ", ".join(f"{float(v):.1f}" for v in bbox) + "]"


def _estimate_frame_extent(refinement: dict | None) -> tuple[float, float]:
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


def _describe_frame_position(refinement: dict | None) -> str:
    bbox = _first_valid_bbox(refinement)
    if bbox is None:
        return "No retained candidate bbox is available for frame-relative positioning."

    frame_w, frame_h = _estimate_frame_extent(refinement)
    if frame_w <= 0.0 or frame_h <= 0.0:
        return (
            f"Primary candidate bbox={_bbox_text(bbox)} pixels; normalized frame position is unavailable "
            "because full image dimensions were not propagated to fusion."
        )

    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    rel_x = cx / frame_w
    rel_y = cy / frame_h

    if rel_x < 0.33:
        horiz = "left"
    elif rel_x > 0.67:
        horiz = "right"
    else:
        horiz = "center"

    if rel_y < 0.33:
        vert = "upper"
    elif rel_y > 0.67:
        vert = "lower"
    else:
        vert = "middle"

    return f"Primary candidate region lies in the {vert}-{horiz} part of the frame; bbox={_bbox_text(bbox)} pixels."


def _format_geo_context(geo_context: dict | None) -> tuple[dict, dict, dict, int, int]:
    geo_context = geo_context or {}
    background = geo_context.get("background", {}) if isinstance(geo_context.get("background"), dict) else {}
    terrain = background.get("terrain", {}) if isinstance(background.get("terrain"), dict) else {}
    geology = background.get("geology", {}) if isinstance(background.get("geology"), dict) else {}
    nearby = geo_context.get("nearby", geo_context if isinstance(geo_context, dict) else {})
    if not isinstance(nearby, dict):
        nearby = {}
    nearby_count = int(nearby.get("count", 0) or 0)
    radius_m = int(nearby.get("radius_m", 0) or 0)
    return terrain, geology, nearby, nearby_count, radius_m



def _normalize_for_dedupe(text: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in text).split())


def _append_unique_sentence(parts: list[str], text: str) -> None:
    sentence = _ensure_sentence(text)
    if not sentence:
        return
    normalized = _normalize_for_dedupe(sentence)
    if not normalized:
        return
    for existing in parts:
        existing_normalized = _normalize_for_dedupe(existing)
        if not existing_normalized:
            continue
        if (
            normalized == existing_normalized
            or normalized in existing_normalized
            or existing_normalized in normalized
        ):
            return
    parts.append(sentence)


def _text_has_frame_position_hint(text: str) -> bool:
    normalized = _normalize_for_dedupe(text)
    if not normalized:
        return False

    if "bbox" in normalized:
        return True

    for phrase in (
        "frame position",
        "normalized frame",
        "upper left",
        "upper right",
        "lower left",
        "lower right",
    ):
        if phrase in normalized:
            return True

    tokens = set(normalized.split())
    hint_tokens = {"frame", "quadrant", "upper", "lower", "left", "right", "middle", "center", "position"}
    return bool(tokens & hint_tokens)


def _aspect_direction(aspect_deg: float | None) -> str:
    if aspect_deg is None:
        return "n/a"
    try:
        degree = float(aspect_deg) % 360.0
    except Exception:
        return "n/a"
    labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return labels[int((degree + 22.5) // 45) % 8]


def _format_slope_aspect_line(slope: object, aspect: object) -> str:
    slope_text = "n/a"
    if slope is not None:
        try:
            slope_text = f"{float(slope):.2f}°"
        except Exception:
            slope_text = str(slope)

    aspect_text = "n/a"
    if aspect is not None:
        try:
            aspect_deg = float(aspect) % 360.0
            aspect_text = f"{aspect_deg:.2f}° ({_aspect_direction(aspect_deg)})"
        except Exception:
            aspect_text = str(aspect)

    return f"Terrain slope/aspect: slope={slope_text}, aspect={aspect_text}."


def _summarize_osm_poi(nearby: dict | None, max_items: int = 6) -> str:
    nearby = nearby if isinstance(nearby, dict) else {}
    features = nearby.get("features") if isinstance(nearby.get("features"), list) else []
    radius_m = int(nearby.get("radius_m", 0) or 0)

    if not features:
        if radius_m > 0:
            return f"Nearby OSM POI: none reported within {radius_m} m."
        return "Nearby OSM POI: none reported."

    rendered: list[str] = []
    seen: set[str] = set()
    for feature in features:
        if not isinstance(feature, dict):
            continue
        ftype = str(feature.get("type", "other") or "other").strip() or "other"
        subtype = str(feature.get("subtype", "") or "").strip()
        name = str(feature.get("name", "") or "").strip()

        if name and subtype:
            label = f"{name} ({ftype}:{subtype})"
        elif name:
            label = f"{name} ({ftype})"
        elif subtype:
            label = f"{ftype}:{subtype}"
        else:
            label = ftype

        normalized = _normalize_for_dedupe(label)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        rendered.append(label)
        if len(rendered) >= max_items:
            break

    count = int(nearby.get("count", len(features)) or len(features))
    if not rendered:
        return f"Nearby OSM POI: {count} feature(s) available, but name/type details were missing."

    sample_text = "; ".join(rendered)
    more_suffix = f" (+{count - len(rendered)} more)" if count > len(rendered) else ""
    if radius_m > 0:
        return f"Nearby OSM POI within {radius_m} m: {sample_text}{more_suffix}."
    return f"Nearby OSM POI: {sample_text}{more_suffix}."


def _to_bullet_block(lines: list[str]) -> str:
    rendered: list[str] = []
    for line in lines:
        cleaned = _clean_text(line, "")
        if not cleaned:
            continue
        rendered.append(f"- {_ensure_sentence(cleaned)}")
    return "\n".join(rendered) if rendered else "- Not available."


def _format_recommendations_text(report: dict, fallback: str) -> str:
    raw = report.get("recommendations")
    items: list[str] = []
    if isinstance(raw, list):
        for item in raw:
            cleaned = _clean_text(item, "")
            if cleaned:
                items.append(cleaned)
    elif raw:
        cleaned = _clean_text(raw, "")
        if cleaned:
            items.append(cleaned)

    if not items:
        items = [_clean_text(fallback, "Manual review is recommended based on current evidence.")]

    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = _normalize_for_dedupe(item)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(item)

    if not deduped:
        deduped = ["Manual review is recommended based on current evidence."]

    return "\n".join(f"{idx}. {_ensure_sentence(text)}" for idx, text in enumerate(deduped, start=1))


def _format_structured_final_description(
    *,
    report: dict,
    stage1: dict,
    refinement: dict,
    classification: dict | None,
    geo_context: dict | None,
    gate: dict | None,
    segmentation: dict | None,
    llm_second_pass: dict | None,
) -> str:
    stage1 = stage1 or {}
    refinement = refinement or {}
    classification = classification or {}
    segmentation = segmentation or {}
    llm_second_pass = llm_second_pass or {}
    report = report or {}

    regions = refinement.get("regions", []) if isinstance(refinement.get("regions"), list) else []
    region_count = len(regions)
    seg_ratio = float(segmentation.get("area_ratio", 0.0) or 0.0)
    seg_pixels = int(segmentation.get("landslide_pixels", 0) or 0)
    polygon_count = int(segmentation.get("polygon_count", 0) or 0)
    region_area_ratio = float((gate or {}).get("area_ratio", refinement.get("area_ratio", 0.0) or 0.0) or 0.0)

    scene_description = _stage1_scene_description(stage1)
    whole_image_overview = _clean_text(report.get("whole_image_overview", ""), scene_description) or scene_description
    report_summary = _clean_text(report.get("summary", ""), "")
    lower_summary = report_summary.lower()
    if (
        lower_summary.startswith("error connecting to llm service")
        or lower_summary.startswith("{")
        or lower_summary.startswith("[")
        or "\"summary\"" in lower_summary
        or "\"visual_description\"" in lower_summary
    ):
        report_summary = ""
    report_visual_description = _clean_text(report.get("visual_description", ""), scene_description)
    if _normalize_for_dedupe(report_visual_description) == _normalize_for_dedupe(whole_image_overview):
        report_visual_description = ""
    report_spatial_distribution = _clean_text(report.get("spatial_distribution", ""), "")
    stage1_evidence = _strip_leading_assessment_token(stage1.get("evidence", "")) or "No scene-level evidence was provided."
    stage1_label = _stage1_assessment_label(stage1)

    report_evidence = report.get("evidence") if isinstance(report.get("evidence"), dict) else {}
    region_evidence = _clean_text(report_evidence.get("refinement", ""), f"regions={region_count}")
    geo_evidence = _clean_text(report_evidence.get("geo_context", ""), "")

    second_pass_workflow = _second_pass_workflow_text(refinement, llm_second_pass, region_area_ratio)
    second_pass_decision = _second_pass_decision(llm_second_pass)
    second_pass_description_only = _second_pass_purpose(llm_second_pass) == "description_only"
    reviewed_regions = _second_pass_reviewed_regions(llm_second_pass)
    second_pass_model_note = _strip_leading_assessment_token(llm_second_pass.get("evidence", ""))
    has_second_pass = bool(reviewed_regions > 0 or second_pass_model_note)

    terrain, geology, nearby, nearby_count, radius_m = _format_geo_context(geo_context)
    slope = terrain.get("slope_deg")
    aspect = terrain.get("aspect_deg")
    lithology = str(geology.get("lithology", "") or geology.get("unit_name", "") or "").strip()

    cls_name = str(classification.get("class_name", "") or report.get("landslide_type", "") or "unknown").strip() or "unknown"
    cls_conf = classification.get("confidence")
    cls_text = cls_name if cls_conf is None else f"{cls_name} ({float(cls_conf):.2f})"
    classification_note = _clean_text(report.get("classification_reference_note", ""), "")
    if not classification_note:
        if cls_name != "unknown":
            classification_note = (
                f"The classifier provides a reference-only subtype cue pointing to {cls_text}. "
                "It helps semantic interpretation but does not override the final yes/no decision."
            )
        else:
            classification_note = "No reliable classification reference was available for subtype interpretation."

    severity = str(report.get("severity", "none") or "none").lower()
    has_landslide = bool(report.get("has_landslide", False))

    # Conclusion
    if has_landslide:
        conclusion = _ensure_sentence(
            "Landslide presence is confirmed by cross-stage evidence."
        )
    else:
        conclusion = _ensure_sentence(
            "Landslide presence is not confirmed under current evidence."
        )
    if report_summary:
        conclusion = f"{conclusion} {_ensure_sentence(report_summary)}"

    # Evidence Summary
    if stage1_label == "likely":
        stage1_status = "Confirmed landslide presence"
    elif stage1_label == "unlikely":
        stage1_status = "Did not support landslide presence"
    elif stage1_label == "uncertain":
        stage1_status = "Remained uncertain on landslide presence"
    else:
        stage1_status = "Provided a scene-level screening note"

    evidence_lines = [
        _ensure_sentence(
            f"Initial Screening: {stage1_status}, citing {stage1_evidence}"
        ),
        _ensure_sentence(
            f"Segmentation Results: {polygon_count} landslide polygon(s) identified, covering {seg_ratio * 100:.1f}% of image area ({seg_pixels} pixels)."
        ),
        _ensure_sentence(
            f"Region Refinement: {region_count} retained candidate region(s), summarized as {region_evidence}"
        ),
    ]
    if cls_name != "unknown":
        evidence_lines.append(
            _ensure_sentence(
                f"Classification Reference: Tentatively labeled \"{cls_name}\""
                + (f" (confidence: {float(cls_conf) * 100:.0f}%)." if cls_conf is not None else ".")
                + " This label is used only for contextual support."
            )
        )
    else:
        evidence_lines.append(
            _ensure_sentence("Classification Reference: No reliable subtype label was available; classification is not used as a decision vote.")
        )

    if has_second_pass:
        if second_pass_description_only:
            consistency = "Consistency Check: Multi-stage outputs are spatially aligned, and the second-pass boundary-overlay review was used as descriptive support only."
        elif second_pass_decision == "positive":
            consistency = "Consistency Check: Multi-stage outputs are consistent, and the second-pass boundary-overlay review further supports the observed morphology."
        elif second_pass_decision == "negative":
            consistency = "Consistency Check: Core stages indicate landslide signals, but second-pass boundary-overlay review raised counter-evidence that warrants careful manual review."
        else:
            consistency = "Consistency Check: Core stages are largely consistent, while second-pass boundary-overlay review remained inconclusive."
    else:
        consistency = "Consistency Check: Stage-1 screening and segmentation-driven outputs are mutually consistent without conflicting stage signals."
    evidence_lines.append(_ensure_sentence(consistency))
    if has_second_pass:
        evidence_lines.append(_ensure_sentence(f"Second-pass Review: {second_pass_workflow}"))
    evidence_summary = " ".join(line for line in evidence_lines if line)

    # Image and Spatial Interpretation
    frame_position_text = _describe_frame_position(refinement)
    if report_spatial_distribution:
        spatial_distribution = _ensure_sentence(report_spatial_distribution)
        if (
            _normalize_for_dedupe(frame_position_text) not in _normalize_for_dedupe(spatial_distribution)
            and not _text_has_frame_position_hint(spatial_distribution)
        ):
            spatial_distribution = f"{spatial_distribution} {_ensure_sentence(frame_position_text)}"
    else:
        spatial_distribution = _ensure_sentence(frame_position_text)
    spatial_distribution = (
        f"{spatial_distribution} "
        f"{_ensure_sentence(f'Segmentation footprint covers {seg_ratio * 100:.1f}% of the frame, with candidate-area ratio {region_area_ratio:.4f}.')}"
    )

    overall_scene_line = _ensure_sentence(f"Overall scene description: {whole_image_overview}")
    image_spatial_interpretation = f"{overall_scene_line} {spatial_distribution}"
    if report_visual_description:
        image_spatial_interpretation = (
            f"{image_spatial_interpretation}\n"
            f"{_ensure_sentence(f'Detailed image interpretation: {report_visual_description}')}"
        )

    # Landslide Typology
    if cls_name != "unknown":
        reference_classification = (
            f"Reference classification: {cls_name} ({float(cls_conf):.2f})."
            if cls_conf is not None
            else f"Reference classification: {cls_name}."
        )
        typology = _ensure_sentence(f"{reference_classification} {classification_note}")
    else:
        typology = _ensure_sentence("No robust subtype classification was available; typology remains open and reference-only.")

    # Geographic and Exposure Context
    if has_landslide and nearby_count > 0:
        environmental_impact = _ensure_sentence(
            f"Potential environmental and infrastructure exposure exists around {nearby_count} mapped nearby feature(s)"
            + (f" within {radius_m} m." if radius_m > 0 else ".")
        )
    elif has_landslide:
        environmental_impact = _ensure_sentence(
            "Potential environmental impact is plausible from detected slope-failure morphology, though mapped nearby assets are limited."
        )
    else:
        environmental_impact = _ensure_sentence(
            "No clear downstream environmental impact is inferred from the current negative determination."
        )

    causal_parts: list[str] = []
    if slope is not None:
        causal_parts.append(f"steep topography ({float(slope):.2f}° slope) may predispose instability")
    if aspect is not None:
        causal_parts.append(f"slope aspect is {float(aspect):.2f}° ({_aspect_direction(float(aspect))})")
    if lithology:
        causal_parts.append(f"material context indicates {lithology}")
    if has_landslide and causal_parts:
        causal_inference = _ensure_sentence("Potential causal contributors include " + "; ".join(causal_parts) + ".")
    elif has_landslide:
        causal_inference = _ensure_sentence("Potential trigger mechanisms remain uncertain due to limited external forcing data.")
    else:
        causal_inference = _ensure_sentence("Current evidence does not justify a positive causal inference for landslide occurrence.")

    geo_parts = [
        _ensure_sentence(_format_slope_aspect_line(slope, aspect)),
        _ensure_sentence(_summarize_osm_poi(nearby)),
    ]
    if nearby_count > 0 and radius_m > 0:
        geo_parts.append(_ensure_sentence(f"Nearby OSM feature count: {nearby_count} within {radius_m} m."))
    if lithology:
        geo_parts.append(_ensure_sentence(f"Geologic background: {lithology}."))
    if geo_evidence:
        compact_geo = geo_evidence.replace(" ", "").lower()
        if "nearby_features=" not in compact_geo:
            geo_parts.append(_ensure_sentence(f"Additional geographic note: {geo_evidence}"))
    geographic_exposure_context = " ".join([environmental_impact, causal_inference] + geo_parts)

    # Reliability and Uncertainty
    reliability_parts = [
        "Decision reliability is described qualitatively from cross-stage agreement; no calibrated overall confidence score is reported."
    ]
    if has_second_pass:
        if second_pass_description_only:
            reliability_parts.append(
                "The second-pass boundary-overlay review was descriptive and did not add an extra yes/no vote."
            )
        elif second_pass_decision == "negative":
            reliability_parts.append("Counter-evidence from second-pass review increases interpretation uncertainty.")
        elif second_pass_decision == "uncertain":
            reliability_parts.append("Second-pass review remained inconclusive.")
        else:
            reliability_parts.append("Second-pass boundary-overlay review was incorporated as supporting context.")
    else:
        reliability_parts.append("No whole-image second-pass boundary-overlay review was used.")
    uncertainty = _clean_text(report.get("uncertainty", ""), "No explicit uncertainty statement was provided.")
    reliability_parts.append(uncertainty)
    reliability_uncertainty = " ".join(_ensure_sentence(part) for part in reliability_parts if part)

    # Final Determination
    if has_landslide:
        final_determination = _ensure_sentence(
            f"Landslide presence is affirmed based on multi-stage evidence convergence (severity={severity})."
        )
    else:
        final_determination = _ensure_sentence(
            "No landslide is indicated under the current multi-stage evidence."
        )

    return "\n".join(
        [
            "### Final Decision Report",
            "",
            "### Conclusion",
            conclusion,
            "",
            "### Evidence Summary",
            evidence_summary,
            "",
            "### Image and Spatial Interpretation",
            image_spatial_interpretation,
            "",
            "### Landslide Typology (Reference Only)",
            typology,
            "",
            "### Geographic and Exposure Context",
            geographic_exposure_context,
            "",
            "### Reliability and Uncertainty",
            reliability_uncertainty,
            "",
            "### Final Determination",
            final_determination,
        ]
    ).strip()


def _build_early_negative_report(
    *,
    stage1: dict,
    refinement: dict,
    classification: dict | None,
    geo_context: dict | None,
) -> dict:
    screening = _screening_decision(stage1, refinement)
    summary = _negative_scene_summary(stage1, refinement)
    geo_count = int((geo_context or {}).get("count", 0) or 0)
    cls_name = str((classification or {}).get("class_name", "") or "").strip()
    report = {
        "report_version": "1.0",
        "summary": summary,
        "whole_image_overview": _stage1_scene_description(stage1),
        "has_landslide": False,
        "confidence": None,
        "confidence_source": "not_computed",
        "severity": "none",
        "landslide_type": "unknown",
        "key_metrics": {
            "regions_count": int(screening["region_count"]),
            "seg_area_ratio": 0.0,
            "landslide_pixels": 0,
        },
        "evidence": {
            "stage1": str(stage1.get("evidence", "") or "").strip(),
            "classification": cls_name or "reference_only",
            "refinement": "regions=0",
            "segmentation": "skipped_after_negative_screening",
            "geo_context": f"nearby_features={geo_count}" if geo_count else "skipped_after_negative_screening",
        },
        "uncertainty": "Low, because both initial LLM screening and segmentation-guided refinement found no landslide evidence.",
        "recommendations": [
            "No further model stages were run because both initial screening steps were negative."
        ],
        "report_source": "screening_early_stop",
    }
    report["final_description"] = _format_structured_final_description(
        report=report,
        stage1=stage1,
        refinement=refinement,
        classification=classification,
        geo_context=geo_context,
        gate={"area_ratio": float(refinement.get("area_ratio", 0.0) or 0.0)},
        segmentation=None,
        llm_second_pass=None,
    )
    return report


def _fallback_description(
    *,
    stage1: dict,
    refinement: dict,
    classification: dict | None,
    geo_context: dict | None,
    gate: dict | None,
    segmentation: dict | None,
    llm_second_pass: dict | None,
) -> str:
    stage1 = stage1 or {}
    refinement = refinement or {}
    classification = classification or {}
    segmentation = segmentation or {}
    screening = _screening_decision(stage1, refinement)
    region_count = int(screening["region_count"])
    seg_ratio = float(segmentation.get("area_ratio", 0.0) or 0.0)
    seg_pixels = int(segmentation.get("landslide_pixels", 0) or 0)
    geo_count = int((geo_context or {}).get("count", 0) or 0)
    cls_name = str(classification.get("class_name", "") or "").strip()
    has_landslide = bool(stage1.get("has_landslide", False) or region_count > 0 or seg_ratio >= 0.01)
    severity = "none"
    if has_landslide:
        severity = "high" if seg_ratio >= 0.15 else ("medium" if seg_ratio >= 0.05 else "low")

    report = {
        "report_version": "1.0",
        "summary": _negative_scene_summary(stage1, refinement) if not has_landslide else _stage1_scene_description(stage1),
        "whole_image_overview": _stage1_scene_description(stage1),
        "has_landslide": has_landslide,
        "confidence": None,
        "confidence_source": "not_computed",
        "severity": severity,
        "landslide_type": cls_name if has_landslide and cls_name else "unknown",
        "key_metrics": {
            "regions_count": region_count,
            "seg_area_ratio": round(seg_ratio, 6),
            "landslide_pixels": seg_pixels,
        },
        "evidence": {
            "stage1": str(stage1.get("evidence", "") or "").strip(),
            "classification": cls_name or "reference_only",
            "refinement": f"regions={region_count}",
            "segmentation": f"ratio={seg_ratio:.4f}, pixels={seg_pixels}",
            "geo_context": f"nearby_features={geo_count}",
        },
        "uncertainty": "Fallback narrative was generated because structured LLM narration was unavailable.",
        "recommendations": ["Manual review is recommended based on cross-stage evidence."],
        "report_source": "fallback",
    }
    return _format_structured_final_description(
        report=report,
        stage1=stage1,
        refinement=refinement,
        classification=classification,
        geo_context=geo_context,
        gate=gate,
        segmentation=segmentation,
        llm_second_pass=llm_second_pass,
    )


def run_stage5(
    stage1: dict,
    refinement: dict,
    classification: dict | None,
    geo_context: dict | None,
    gate: dict | None,
    segmentation: dict | None,
    llm_second_pass: dict | None,
    llm_second_pass_threshold: float,
    min_region_score: float = 0.45,
) -> dict:
    segmentation = segmentation or {
        "mask_path": "",
        "overlay_path": "",
        "landslide_pixels": 0,
        "area_ratio": 0.0,
        "polygon_count": 0,
    }
    screening = _screening_decision(stage1, refinement)
    if not screening["has_positive_screening"]:
        return _build_early_negative_report(
            stage1=stage1,
            refinement=refinement,
            classification=classification,
            geo_context=geo_context,
        )

    fused_decision = _derive_fused_decision(
        stage1=stage1,
        refinement=refinement,
        segmentation=segmentation,
        classification=classification,
        llm_second_pass=llm_second_pass,
        llm_second_pass_threshold=llm_second_pass_threshold,
        min_region_score=min_region_score,
        gate=gate,
    )

    llm_report = llm_generate_final_report(
        stage1=stage1,
        refinement=refinement,
        segmentation=segmentation,
        classification=classification,
        geo_context=geo_context,
        gate=gate,
        llm_second_pass=llm_second_pass,
        fused_decision=fused_decision,
    )
    llm_report["whole_image_overview"] = _stage1_scene_description(stage1)
    final_description = str(llm_report.get("final_description", "") or "").strip()
    if not final_description:
        final_description = _fallback_description(
            stage1=stage1,
            refinement=refinement,
            classification=classification,
            geo_context=geo_context,
            gate=gate,
            segmentation=segmentation,
            llm_second_pass=llm_second_pass,
        )
        llm_report["summary"] = llm_report.get("summary") or final_description
        llm_report["final_description"] = final_description
        llm_report["report_source"] = llm_report.get("report_source") or "fallback"

    llm_report["has_landslide"] = bool(fused_decision["has_landslide"])
    llm_report["confidence"] = None
    llm_report["confidence_source"] = str(fused_decision.get("confidence_source", "not_computed"))
    llm_report["severity"] = str(fused_decision["severity"])
    llm_report["landslide_type"] = (
        str(
            llm_report.get("landslide_type")
            or (classification or {}).get("class_name")
            or "unknown"
        )
        if llm_report["has_landslide"]
        else "unknown"
    )

    recommendations = llm_report.get("recommendations")
    if not isinstance(recommendations, list) or not recommendations:
        if llm_report["has_landslide"]:
            llm_report["recommendations"] = ["Manual review is recommended for the detected landslide signals."]
        else:
            llm_report["recommendations"] = ["No positive landslide determination is supported after full-analysis cross-checking."]

    llm_report["final_description"] = _format_structured_final_description(
        report=llm_report,
        stage1=stage1,
        refinement=refinement,
        classification=classification,
        geo_context=geo_context,
        gate=gate,
        segmentation=segmentation,
        llm_second_pass=llm_second_pass,
    )
    llm_report["decision_support"] = fused_decision["support_summary"]
    return llm_report
