from __future__ import annotations

from src.agent.controller import get_policy
from src.models.llm_client import llm_generate_final_report
from src.utils.geometry import describe_primary_candidate_position
from src.utils.landslide_taxonomy import reconcile_landslide_types


REQUIRED_SECTIONS = [
    "Final Decision Report",
    "Conclusion",
    "Evidence Summary",
    "Spatial Distribution",
    "Image Feature Description",
    "Landslide Typology (Reference Only)",
    "Rationale for Landslide Presence",
    "Image Quality Assessment",
    "Relative Position Within Image Frame",
    "Environmental Impact",
    "Confidence Level",
    "Uncertainty Analysis",
    "Causal Inference",
    "Geographic Context",
    "Final Determination",
]


def _clean_text(value: object, default: str = "Not available.") -> str:
    text = " ".join(str(value or "").split()).strip()
    text = text.replace("…", ".")
    while "..." in text:
        text = text.replace("...", ".")
    while ".." in text:
        text = text.replace("..", ".")
    text = text.replace(" .", ".")
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


def _screening_decision(
    stage1: dict | None, refinement: dict | None, segmentation: dict | None = None
) -> dict[str, object]:
    stage1 = stage1 or {}
    refinement = refinement or {}
    regions = refinement.get("regions", [])
    region_count = len(regions) if isinstance(regions, list) else 0
    policy = get_policy()
    stage1_positive = policy.stage1_is_positive(stage1) is True
    refinement_positive = region_count > 0
    # A material segmentation signal must not be silently dropped by the
    # early-stop gate: it is a primary vote in the fusion decision.
    segmentation_positive = (
        policy.seg_is_positive(segmentation) if segmentation is not None else False
    )
    return {
        "stage1_positive": stage1_positive,
        "refinement_positive": refinement_positive,
        "segmentation_positive": segmentation_positive,
        "has_positive_screening": bool(
            stage1_positive or refinement_positive or segmentation_positive
        ),
        "region_count": region_count,
    }


def screening_requires_full_analysis(
    stage1: dict | None, refinement: dict | None, segmentation: dict | None = None
) -> bool:
    return bool(_screening_decision(stage1, refinement, segmentation)["has_positive_screening"])


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



def _top_region_score(refinement: dict | None) -> float:
    regions = (refinement or {}).get("regions", [])
    if not isinstance(regions, list):
        return 0.0
    return max(
        [float(item.get("score", 0.0) or 0.0) for item in regions if isinstance(item, dict)],
        default=0.0,
    )


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


def _derive_fused_decision(
    *,
    stage1: dict,
    refinement: dict,
    segmentation: dict | None,
    llm_second_pass: dict | None,
    gate: dict | None,
) -> dict[str, object]:
    """Thin adapter over the single decision rule in ``LandslidePolicy``.

    The landslide yes/no verdict and its confidence (a real model score, never
    synthesised) are decided by
    ``policy.fuse_decision`` (whole-image VLM screening cross-validated against
    segmentation, arbitrated by the boundary re-check's verdict label on
    disagreement). This function only keeps the ``support_summary`` shape the
    report renderer wants.
    """
    # ``gate.area_ratio`` (the refinement extent seen by the caller) overrides
    # the raw refinement ratio used by the fusion rule when supplied.
    extent_ref = dict(refinement or {})
    if isinstance(gate, dict) and "area_ratio" in gate:
        extent_ref["area_ratio"] = gate.get("area_ratio")

    decision = get_policy().fuse_decision(
        stage1=stage1 or {},
        segmentation=segmentation,
        refinement=extent_ref,
        llm_second_pass=llm_second_pass,
    )
    modalities = decision["modalities"]

    region_count = int(_screening_decision(stage1, refinement)["region_count"])
    top_region_score = _top_region_score(refinement)

    return {
        "has_landslide": decision["has_landslide"],
        "confidence": decision["confidence"],
        "confidence_source": decision["confidence_source"],
        "severity": decision["severity"],
        "support_summary": {
            "decision_basis": decision["decision_basis"],
            "stage1_positive": modalities["vlm_positive"],
            "vlm_verdict": modalities["vlm_verdict"],
            "segmentation_positive": modalities["segmentation_positive"],
            "segmentation_area_ratio": modalities["segmentation_area_ratio"],
            "modality_agreement": modalities["agreement"],
            "second_pass_role": modalities["second_pass_role"],
            "second_pass_positive": modalities["second_pass_positive"],
            "second_pass_negative": modalities["second_pass_negative"],
            "second_pass_descriptive": modalities["second_pass_descriptive"],
            "region_count": region_count,
            "refinement_source": _refinement_source(refinement) or "unknown",
        },
    }


_describe_frame_position = describe_primary_candidate_position


_QWEN_NAME = "Qwen classification head"
_CONVNEXT_NAME = "ConvNeXt image classifier"


def _classifier_reconciliation(classification: dict | None) -> dict[str, object]:
    """Cross-check the two image classifiers inside the classification tool.

    Both opinions come from models that actually looked at the image: the Qwen
    classification head (``sources.vlm``) and the ConvNeXt image classifier
    (``sources.image_classifier``). Their Cruden-Varnes resolution is already
    computed by the tool; this only restates it with unambiguous names.
    """
    cls = classification or {}
    sources = cls.get("sources") if isinstance(cls.get("sources"), dict) else {}
    qwen = str(((sources.get("vlm") or {}).get("class_name")) or "")
    convnext = str(((sources.get("image_classifier") or {}).get("class_name")) or "")
    resolved = str(cls.get("class_name", "") or "")
    resolution = str(cls.get("resolution", "") or "")
    if qwen and convnext:
        status = {
            "subclass_agreement": "agreement",
            "movement_parent_fallback": "parent_fallback",
            "material_parent_fallback": "parent_fallback",
            "conflict": "conflict",
        }.get(resolution, "conflict" if cls.get("conflict") else "agreement")
    elif resolved:
        status = "single_source"
    else:
        status = "unavailable"
    return {
        "qwen_head_label": qwen,
        "convnext_label": convnext,
        "resolved_label": resolved,
        "parent_class": str(cls.get("parent_class", "") or ""),
        "status": status,
        "conflict": status == "conflict",
    }


def _classification_reconciliation_report_note(reconciliation: dict[str, object]) -> str:
    status = str(reconciliation.get("status", "unavailable") or "unavailable")
    qwen = str(reconciliation.get("qwen_head_label", "") or "unknown")
    convnext = str(reconciliation.get("convnext_label", "") or "unknown")
    resolved = str(reconciliation.get("resolved_label", "") or "unknown")
    if status == "agreement":
        return f"Inference: The {_QWEN_NAME} and the {_CONVNEXT_NAME} agree on the Cruden-Varnes subtype ({resolved})."
    if status == "parent_fallback":
        return (
            f"Inference: The {_QWEN_NAME} ({qwen}) and the {_CONVNEXT_NAME} ({convnext}) differ on the subtype "
            f"but share a Cruden-Varnes parent class; the report falls back to that parent class ({resolved})."
        )
    if status == "conflict":
        return (
            f"Inference: Classification conflict. The {_QWEN_NAME} proposed {qwen}, while the {_CONVNEXT_NAME} proposed "
            f"{convnext}; they disagree even at the Cruden-Varnes parent level, so no subtype was forced."
        )
    if status == "single_source":
        return f"Inference: Only one classifier opinion was available ({resolved}); cross-classifier agreement was not evaluated."
    return "Inference: No usable landslide subtype was returned by the classification tool."


def _confidence_phrase(confidence: float | None, source: str = "") -> str:
    """Report the subtype classifier's confidence.

    ``confidence`` is the classification model's score (or ``None``). The
    detection verdict itself is a rule-based cross-modality decision and carries
    no separate numeric confidence, so only the classifier's score is reported.
    """
    if confidence is None:
        return "the subtype classifier stated no numeric confidence"
    return f"the subtype classifier stated a confidence score of {confidence:.2f}"


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


def _looks_like_llm_final_report(text: str) -> bool:
    cleaned = str(text or "").strip()
    if not cleaned:
        return False

    lowered = cleaned.lower()
    return all(section.lower() in lowered for section in REQUIRED_SECTIONS)


def _match_section_heading(line: str, section: str) -> str | None:
    stripped = line.strip()
    if not stripped:
        return None
    stripped = stripped.lstrip("#").strip()
    if not stripped.lower().startswith(section.lower()):
        return None
    tail = stripped[len(section) :].strip()
    if not tail:
        return ""
    if tail.startswith(":") or tail.startswith("-"):
        return tail[1:].strip()
    return None


def _normalize_llm_sectioned_report(
    text: str, overrides: dict[str, str] | None = None
) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""

    lines = cleaned.splitlines()
    seen: set[str] = set()
    content_by_section: dict[str, list[str]] = {section: [] for section in REQUIRED_SECTIONS}
    current_section: str | None = None

    for line in lines:
        matched_section = None
        inline_content = ""
        for section in REQUIRED_SECTIONS:
            match = _match_section_heading(line, section)
            if match is not None:
                matched_section = section
                inline_content = match
                break
        if matched_section:
            seen.add(matched_section)
            current_section = matched_section
            if inline_content:
                content_by_section[current_section].append(inline_content)
            continue
        if current_section:
            content_by_section[current_section].append(line)

    if not all(section in seen for section in REQUIRED_SECTIONS):
        return ""

    overrides = overrides or {}
    blocks: list[str] = []
    for section in REQUIRED_SECTIONS:
        body = str(overrides.get(section, "") or "").strip()
        if not body:
            body = "\n".join(content_by_section.get(section, [])).strip()
        blocks.append(f"### {section}")
        blocks.append(body)
        blocks.append("")
    return "\n".join(blocks).strip()


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

    screening = _screening_decision(stage1, refinement)
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
    stage1_score = float(stage1.get("score", 0.0) or 0.0)

    report_evidence = report.get("evidence") if isinstance(report.get("evidence"), dict) else {}
    region_evidence = _clean_text(report_evidence.get("refinement", ""), f"regions={region_count}")
    seg_evidence = _clean_text(
        report_evidence.get("segmentation", ""),
        f"ratio={seg_ratio:.4f}, pixels={seg_pixels}",
    )
    geo_evidence = _clean_text(report_evidence.get("geo_context", ""), "")

    tool_interpretation = _strip_inline_assessment_tokens(report.get("tool_interpretation", ""))
    if not tool_interpretation:
        tool_interpretation = (
            f"Segmentation-guided refinement retained {region_count} candidate region(s); segmentation reported {seg_evidence}; "
            f"classification evidence was {_clean_text(report_evidence.get('classification', ''), 'not emphasized')}"
        )

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

    _raw_conf = classification.get("confidence")
    confidence = float(_raw_conf) if isinstance(_raw_conf, (int, float)) else None
    confidence_phrase = _confidence_phrase(confidence)
    has_landslide = bool(report.get("has_landslide", False))

    top_region_score = max(
        [float(det.get("score", 0.0) or 0.0) for det in regions if isinstance(det, dict)],
        default=0.0,
    )

    # Conclusion
    if has_landslide:
        conclusion = _ensure_sentence(
            f"Landslide presence is confirmed; {confidence_phrase}."
        )
    else:
        conclusion = _ensure_sentence(
            f"Landslide presence is not confirmed under current evidence; {confidence_phrase}."
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

    # Spatial Distribution
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

    # Image Feature Description (overall description listed separately)
    overall_scene_line = _ensure_sentence(f"Overall scene description: {whole_image_overview}")
    image_feature_description = overall_scene_line
    if report_visual_description:
        image_feature_description = f"{image_feature_description}\n{_ensure_sentence(f'Detailed image interpretation: {report_visual_description}') }"

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

    # Rationale
    if has_landslide:
        rationale_parts = []
        if screening["stage1_positive"]:
            rationale_parts.append("positive whole-scene screening")
        if region_count > 0:
            rationale_parts.append(f"retained segmentation-guided candidates ({region_count})")
        if seg_ratio > 0.0:
            rationale_parts.append(f"non-trivial segmented footprint ({seg_ratio * 100:.1f}% area)")
        if has_second_pass and second_pass_decision == "positive":
            rationale_parts.append("supportive second-pass boundary-overlay review")
        rationale = _ensure_sentence(
            "Landslide presence is supported by " + ", ".join(rationale_parts) + "."
            if rationale_parts
            else "Landslide presence is supported by cross-stage evidence convergence."
        )
    else:
        rationale = _ensure_sentence(
            "Landslide presence is not supported because screening, segmentation-guided region evidence, and corroborative signals do not jointly meet positive-evidence criteria."
        )
    rationale = f"{rationale} {_ensure_sentence(f'Integrated interpretation: {tool_interpretation}') }"

    # Image Quality Assessment
    quality_assessment = "Image quality supports stable terrain-feature interpretation with sufficient contrast for morphology reading."
    if has_second_pass:
        quality_assessment += " A second-pass whole-image boundary-overlay review added an extra consistency check."
    elif region_count > 0:
        quality_assessment += " Segmentation-driven region evidence is available, though no second-pass boundary-overlay review was used."
    quality_assessment = _ensure_sentence(quality_assessment)

    # Relative Position Within Image Frame
    relative_position = _ensure_sentence(frame_position_text)

    # Environmental Impact
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

    # Confidence Level
    confidence_level = _ensure_sentence(
        f"For confidence, {confidence_phrase}; the verdict itself rests on agreement between "
        "whole-image VLM screening and segmentation."
    )

    # Uncertainty Analysis
    uncertainty = _clean_text(report.get("uncertainty", ""), "No explicit uncertainty statement was provided.")
    if has_second_pass:
        if second_pass_description_only:
            uncertainty += " The second-pass boundary-overlay review was descriptive and did not add an extra yes/no vote."
        elif second_pass_decision == "negative":
            uncertainty += " Counter-evidence from second-pass review increases interpretation uncertainty."
        elif second_pass_decision == "uncertain":
            uncertainty += " Second-pass review remained inconclusive."
    else:
        uncertainty += " No whole-image second-pass boundary-overlay review was used."
    uncertainty_analysis = _ensure_sentence(uncertainty)

    # Causal Inference
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

    # Geographic Context (must include slope/aspect and OSM POI)
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
    geographic_context = " ".join(geo_parts)

    # Final Determination
    if has_landslide:
        final_determination = _ensure_sentence(
            f"Landslide presence is affirmed by whole-image screening / segmentation agreement "
            f"({confidence_phrase})."
        )
    else:
        final_determination = _ensure_sentence(
            f"No landslide is indicated under the current cross-modality evidence; {confidence_phrase}."
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
            "### Spatial Distribution",
            spatial_distribution,
            "",
            "### Image Feature Description",
            image_feature_description,
            "",
            "### Landslide Typology (Reference Only)",
            typology,
            "",
            "### Rationale for Landslide Presence",
            rationale,
            "",
            "### Image Quality Assessment",
            quality_assessment,
            "",
            "### Relative Position Within Image Frame",
            relative_position,
            "",
            "### Environmental Impact",
            environmental_impact,
            "",
            "### Confidence Level",
            confidence_level,
            "",
            "### Uncertainty Analysis",
            uncertainty_analysis,
            "",
            "### Causal Inference",
            causal_inference,
            "",
            "### Geographic Context",
            geographic_context,
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
    # Same decision rule as the full path: negative here, confidence = the
    # screening model's own score (or None), never a hand-rolled formula.
    _decision = get_policy().fuse_decision(stage1=stage1 or {}, segmentation=None, refinement=refinement)
    _cls_conf = (classification or {}).get("confidence")
    report = {
        "report_version": "1.0",
        "summary": summary,
        "whole_image_overview": _stage1_scene_description(stage1),
        "has_landslide": bool(_decision["has_landslide"]),
        "classification_confidence": round(float(_cls_conf), 4) if isinstance(_cls_conf, (int, float)) else None,
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
    fused_decision: dict | None = None,
) -> str:
    stage1 = stage1 or {}
    refinement = refinement or {}
    classification = classification or {}
    segmentation = segmentation or {}
    screening = _screening_decision(stage1, refinement, segmentation)
    region_count = int(screening["region_count"])
    seg_ratio = float(segmentation.get("area_ratio", 0.0) or 0.0)
    seg_pixels = int(segmentation.get("landslide_pixels", 0) or 0)
    geo_count = int((geo_context or {}).get("count", 0) or 0)
    cls_name = str(classification.get("class_name", "") or "").strip()
    # Never re-derive the verdict here - carry the one decision from the policy.
    decision = fused_decision or _derive_fused_decision(
        stage1=stage1, refinement=refinement, segmentation=segmentation,
        llm_second_pass=llm_second_pass,
        gate=gate,
    )
    has_landslide = bool(decision["has_landslide"])
    _conf = decision["confidence"]

    report = {
        "report_version": "1.0",
        "summary": _negative_scene_summary(stage1, refinement) if not has_landslide else _stage1_scene_description(stage1),
        "whole_image_overview": _stage1_scene_description(stage1),
        "has_landslide": has_landslide,
        "confidence": round(float(_conf), 4) if _conf is not None else None,
        "confidence_source": decision.get("confidence_source", "unavailable"),
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
    unavailable_evidence: list | None = None,
) -> dict:
    segmentation = segmentation or {
        "mask_path": "",
        "overlay_path": "",
        "landslide_pixels": 0,
        "area_ratio": 0.0,
        "polygon_count": 0,
    }
    screening = _screening_decision(stage1, refinement, segmentation)
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
        llm_second_pass=llm_second_pass,
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
        unavailable_evidence=unavailable_evidence,
    )
    llm_report["whole_image_overview"] = _stage1_scene_description(stage1)
    llm_report["has_landslide"] = bool(fused_decision["has_landslide"])
    # Only the subtype classifier carries a genuine (softmax) confidence; the
    # detection verdict is a rule-based cross-modality decision, so no VLM-derived
    # detection confidence is surfaced in the report.
    _cls_conf = (classification or {}).get("confidence")
    llm_report["classification_confidence"] = (
        round(float(_cls_conf), 4) if isinstance(_cls_conf, (int, float)) else None
    )
    llm_report.pop("confidence", None)
    llm_report.pop("confidence_source", None)
    # Severity/hazard cannot be inferred from a single image and its model
    # outputs. Keep only directly traceable detection evidence in the report.
    llm_report.pop("severity", None)

    # The narrative model never sees the image, so it contributes no subtype of
    # its own; the subtype and its consistency come from the two image
    # classifiers inside the classification tool.
    llm_report.pop("llm_landslide_type", None)
    reconciliation = _classifier_reconciliation(classification)
    reconciliation["report_note"] = _classification_reconciliation_report_note(reconciliation)
    llm_report["classification_reconciliation"] = reconciliation
    reference_note = str(llm_report.get("classification_reference_note", "") or "").strip()
    report_note = str(reconciliation["report_note"] or "").strip()
    llm_report["classification_reference_note"] = " ".join(
        part for part in (reference_note, report_note) if part
    )

    resolved_label = str(reconciliation.get("resolved_label", "") or "").strip()
    if llm_report["has_landslide"]:
        if reconciliation.get("status") == "conflict":
            llm_report["landslide_type"] = "classification conflict"
        else:
            llm_report["landslide_type"] = (
                resolved_label
                or str((classification or {}).get("class_name", "") or "")
                or "unknown"
            )
    else:
        llm_report["landslide_type"] = "unknown"

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
            fused_decision=fused_decision,
        )
        llm_report["summary"] = llm_report.get("summary") or final_description
        llm_report["final_description"] = final_description
        llm_report["report_source"] = llm_report.get("report_source") or "fallback"

    recommendations = llm_report.get("recommendations")
    if not isinstance(recommendations, list) or not recommendations:
        if llm_report["has_landslide"]:
            llm_report["recommendations"] = ["Manual review is recommended for the detected landslide signals."]
        else:
            llm_report["recommendations"] = ["No positive landslide determination is supported after full-analysis cross-checking."]

    report_source = str(llm_report.get("report_source", "") or "").strip().lower()
    direct_final_description = str(llm_report.get("final_description", "") or "").strip()

    if report_source.startswith("llm") and _looks_like_llm_final_report(direct_final_description):
        normalized_final = _normalize_llm_sectioned_report(
            direct_final_description,
            overrides={
                # Computed from the refined mask -- the same value region.locate
                # returns. The narrative model routinely contradicts it ("lower
                # right quadrant" for a middle-center candidate), so for this
                # section the computed value wins.
                "Relative Position Within Image Frame": _ensure_sentence(
                    _describe_frame_position(refinement)
                ),
                # Make the cross-model resolution visible even when the LLM
                # returned a complete sectioned narrative of its own.
                "Landslide Typology (Reference Only)": _ensure_sentence(
                    str(llm_report.get("classification_reference_note", "") or "")
                ),
            },
        )
        if normalized_final:
            llm_report["final_description"] = normalized_final
        else:
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
    else:
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
    gaps = [g for g in (unavailable_evidence or []) if isinstance(g, dict) and g.get("tool")]
    if gaps:
        llm_report["unavailable_evidence"] = gaps
        llm_report["final_description"] = _state_unavailable_evidence(
            str(llm_report.get("final_description", "") or ""), gaps
        )
    return llm_report


_UNAVAILABLE_LABELS = {
    "geo.background": "terrain and geological background",
    "geo.nearby": "nearby mapped facilities",
    "cls.run": "landslide subtype classification",
    "seg.llm_review": "vision-language boundary re-check",
}


def _state_unavailable_evidence(text: str, gaps: list[dict]) -> str:
    """Guarantee the report states which evidence could not be obtained."""
    items = "; ".join(
        f"{_UNAVAILABLE_LABELS.get(g['tool'], g['tool'])} ({g['tool']}): {g.get('reason') or 'not obtained'}"
        for g in gaps
    )
    sentence = (
        f"Observed: The following evidence was not available for this analysis and was not used: {items}."
    )
    marker = "### Uncertainty Analysis"
    if marker in text:
        head, tail = text.split(marker, 1)
        return head + marker + "\n" + sentence + "\n" + tail.lstrip("\n")
    return (text.rstrip() + "\n\n" + marker + "\n" + sentence).strip()
