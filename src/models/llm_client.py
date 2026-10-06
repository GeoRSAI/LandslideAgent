from __future__ import annotations

import contextvars
import json
import os
import re
from contextlib import contextmanager
from typing import Any, Callable
from urllib import request, error

from src.models import dual_head_client
from src.utils.geometry import (
    describe_primary_candidate_position as _describe_primary_candidate_position,
)


_MODEL_RAW_EVENT_SINK: contextvars.ContextVar[list[dict[str, str]] | None] = contextvars.ContextVar(
    "llm_client_model_raw_event_sink",
    default=None,
)


@contextmanager
def capture_model_raw_events():
    parent_sink = _MODEL_RAW_EVENT_SINK.get()
    bucket: list[dict[str, str]] = []
    token = _MODEL_RAW_EVENT_SINK.set(bucket)
    try:
        yield bucket
    finally:
        _MODEL_RAW_EVENT_SINK.reset(token)
        if parent_sink is not None and bucket:
            parent_sink.extend(bucket)


def _record_model_raw_event(raw_text: Any, source: str) -> None:
    sink = _MODEL_RAW_EVENT_SINK.get()
    if sink is None:
        return
    content = str(raw_text or "").strip()
    if not content:
        return
    sink.append({"source": str(source or "llm"), "content": content})


def _compact_text(text: str, max_chars: int = 96) -> str:
    compact = " ".join(str(text or "").strip().split())
    compact = compact.replace("…", ".")
    while "..." in compact:
        compact = compact.replace("...", ".")
    while ".." in compact:
        compact = compact.replace("..", ".")
    compact = compact.replace(" .", ".")
    return compact


def _parse_stage1_scene_assessment(text: str) -> tuple[str, str]:
    compact = _compact_text(text, max_chars=280)
    if not compact:
        return "uncertain", ""

    lower = compact.lower()
    label = "uncertain"
    scene_description = compact

    for separator in ("|", ":", "-"):
        if separator not in compact:
            continue
        head, tail = compact.split(separator, 1)
        normalized_head = " ".join(head.strip().lower().split())
        if normalized_head in {"likely", "unlikely", "uncertain"}:
            label = normalized_head
            scene_description = tail.strip() or compact
            break

    if label == "uncertain":
        if lower.startswith("unlikely"):
            label = "unlikely"
            scene_description = compact[len("unlikely"):].lstrip(" |:-") or compact
        elif lower.startswith("likely"):
            label = "likely"
            scene_description = compact[len("likely"):].lstrip(" |:-") or compact
        elif lower.startswith("uncertain"):
            label = "uncertain"
            scene_description = compact[len("uncertain"):].lstrip(" |:-") or compact

    return label, _compact_text(scene_description, max_chars=220)




def llm_judge_has_landslide(context: dict) -> dict:
    image_path = context.get("image_info", {}).get("image_path")
    if image_path:
        user_content = [
            {"type": "image", "image_path": image_path},
            {"type": "text", "text": f"Analyze this image and determine if a landslide is likely. Context: {json.dumps(context)}"},
        ]
    else:
        user_content = f"Analyze this image metadata and determine if a landslide is likely. Context: {json.dumps(context)}"
    messages = [
        {
            "role": "system",
            "content": (
                "You are a landslide analysis expert reviewing the whole image. "
                "Respond in English on one line using exactly one leading label: "
                "Likely, Unlikely, or Uncertain. "
                "After the label, add ' | ' and then an information-dense whole-scene visual description with as much detail as needed. "
                "Prioritize direct image observations over abstract judgement text. Describe slope condition, scarps, exposed material texture/color, debris/runout direction, vegetation or drainage disruption, and how the suspicious area is positioned relative to the broader scene."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    try:
        response = _openai_chat_completion(messages, temperature=0.1, trace_label="tool.llm.first_pass")
        content = _compact_text(response.get("content", ""), max_chars=220)
        label, scene_description = _parse_stage1_scene_assessment(content)
        # assessment_label stays a text-parsed likely/unlikely/uncertain verdict --
        # LandslidePolicy.should_run_second_pass gates on this string (e.g. an
        # "uncertain" screening always forces a second pass) regardless of what
        # has_landslide/score end up being below.
        if label == "likely":
            has_landslide = True
            score = 0.8
        elif label == "unlikely":
            has_landslide = False
            score = 0.2
        else:
            has_landslide = "yes" in content.lower() or (
                "likely" in content.lower() and "unlikely" not in content.lower()
            )
            score = 0.5 if content else 0.0
        score_source = "vlm_text_label"

        classification: dict[str, Any] | None = None
        if image_path:
            # Real evidence source: the dual-head classifier's own softmax
            # probability, replacing the hardcoded 0.8/0.2/0.5 above whenever
            # it is available. stage1_is_positive() (controller.py) prefers
            # has_landslide when it is a bool, so this takes priority there;
            # assessment_label is untouched so the uncertain/error gate still
            # works even when the classifier disagrees or is unavailable.
            dual_head_result = dual_head_client.classify(image_path)
            if int(dual_head_result.get("class_id", -1)) >= 0:
                has_landslide = bool(dual_head_result["has_landslide"])
                score = float(dual_head_result["score"])
                score_source = "dual_head_classifier"
                classification = {
                    "class_id": dual_head_result["class_id"],
                    "class_name": dual_head_result["class_name"],
                    "confidence": dual_head_result["confidence"],
                    "topk": dual_head_result.get("topk", []),
                }

        result = {
            "has_landslide": has_landslide,
            "score": score,
            "score_source": score_source,
            "evidence": content,
            "assessment_label": label,
            "scene_description": scene_description,
        }
        if classification is not None:
            result["classification"] = classification
        return result
    except Exception as e:
        return {
            "has_landslide": False,
            "score": 0.0,
            "evidence": f"LLM service error: {str(e)}",
            "assessment_label": "error",
            "scene_description": "",
        }


# The exact instruction the description LoRA was fine-tuned on.
DESCRIBE_PROMPT = (
    "Please examine the remote sensing image and identify whether it contains a landslide. "
    "If so, classify the landslide type and explain the visual evidence supporting your decision."
)
DESCRIBE_FIELDS = {
    "landslide presence": "presence",
    "landslide type": "type",
    "image relative position within the image frame": "position",
    "morphological characteristics": "morphology",
    "material composition and surface cover": "material",
    "movement and deformation features": "movement",
    "surrounding environmental context": "environment",
    "impact on human infrastructure": "impact",
    "reason for landslide classification": "reason",
    "landslide causation inference": "causation",
}


def parse_scene_description(text: str) -> dict[str, str]:
    """Split the fine-tuned 10-field answer into its fields ("Label:" then value)."""
    fields: dict[str, str] = {}
    current = None
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        label, sep, rest = stripped.partition(":")
        key = DESCRIBE_FIELDS.get(label.strip().lower()) if sep else None
        if key:
            current = key
            fields[key] = rest.strip()
        elif current:
            fields[current] = (fields[current] + " " + stripped).strip()
    return fields


def llm_describe_scene(image_path: str) -> dict[str, Any]:
    """Whole-image description using the dual-head adapter."""
    if not str(image_path or "").strip():
        return {"error": "vlm.describe needs an image_path"}
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image_path": image_path},
            {"type": "text", "text": DESCRIBE_PROMPT},
        ],
    }]
    response = _openai_chat_completion(
        messages, temperature=0.0, trace_label="tool.vlm.describe", timeout=180.0, max_tokens=768,
    )
    text = str(response.get("content", "") or "")
    if text.startswith("Error connecting to LLM service"):
        return {"error": text}
    fields = parse_scene_description(text)
    if len(fields) < 3:
        return {"error": "description did not follow the expected field format", "raw_text": text}
    return {"fields": fields, "raw_text": text, "adapter": os.getenv("LLM_DESCRIBE_ADAPTER", "dualhead")}


def llm_second_pass_on_boxed_image(review: dict[str, Any] | None) -> dict:
    review = review or {}
    review_image_path = str(review.get("review_image_path", "") or review.get("overlay_path", "") or "").strip()
    original_image_path = str(review.get("image_path", "") or "").strip()
    review_mode = " ".join(str(review.get("review_mode", "boxed_whole_image") or "boxed_whole_image").strip().split())
    review_purpose = " ".join(str(review.get("review_purpose", "verification") or "verification").strip().lower().split())
    if review_purpose not in {"verification", "description_only"}:
        review_purpose = "verification"
    overlay_source = " ".join(
        str(review.get("overlay_source", "refinement_regions") or "refinement_regions").strip().lower().split()
    )
    use_segmentation_boundary = (
        "seg" in overlay_source
        or "mask" in overlay_source
        or review_mode.startswith("seg_")
    )
    stage1_scene_description = _compact_text(str(review.get("stage1_scene_description", "") or ""), max_chars=220)
    regions = review.get("regions", []) if isinstance(review.get("regions"), list) else []
    reviewed_regions = len(regions)
    region_summary = [
        {
            "bbox": det.get("bbox"),
            "score": float(det.get("score", 0.0) or 0.0),
            "class_id": int(det.get("class_id", 0) or 0),
        }
        for det in regions[:6]
        if isinstance(det, dict)
    ]
    marker_name = "segmentation-mask boundary" if use_segmentation_boundary else "refinement boxes"
    marker_metadata_key = "regions"
    if not review_image_path:
        return {
            "review_mode": review_mode,
            "review_purpose": review_purpose,
            "review_image_path": "",
            "original_image_path": original_image_path,
            "reviewed_regions": reviewed_regions,
            "decision": "unavailable",
            "supports_landslide": None,
            "score": 0.0,
            "evidence": "Second-pass review could not run because no whole-image overlay was available.",
        }

    if review_purpose == "description_only":
        guidance_text = (
            f"This is the same full scene with {marker_name} already overlaid. "
            "The initial whole-image screening and segmentation-guided region extraction already indicate a landslide, so this second pass must not re-decide whether a landslide exists. "
            f"Use the {marker_name} only as spatial guidance to supplement the narrative description. "
            "Describe where the suspected landslide sits in the whole scene and in the image frame, and how the highlighted region corresponds to visible landslide morphology. "
        )
        if stage1_scene_description:
            guidance_text += f"Initial whole-image scene description: {stage1_scene_description}. "
        guidance_text += (
            "Region metadata: "
            + json.dumps({"reviewed_regions": reviewed_regions, marker_metadata_key: region_summary}, ensure_ascii=False)
        )
        user_content: Any = [
            {"type": "image", "image_path": review_image_path},
            {"type": "text", "text": guidance_text},
        ]
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a landslide interpretation expert performing a descriptive second-pass review on a full-scene image with {marker_name} overlaid. "
                    "Landslide presence is already treated as supported by the first-pass screening and segmentation-guided region output. "
                    "This review is only for narrative enrichment and spatial localization, not for another yes/no decision. "
                    "Respond in English on one line. "
                    "Start with exactly one label: Describe. "
                    "Then add ' | ' and a detailed scene-grounded explanation with as much detail as needed. Prioritize image morphology and spatial anchors: where the boxed region sits in the frame, how it relates to slope breaks/runout paths, and which visible cues support that reading."
                ),
            },
            {"role": "user", "content": user_content},
        ]
        try:
            response = _openai_chat_completion(messages, temperature=0.1, trace_label="tool.seg.llm_review")
            content = _compact_text(response.get("content", ""), max_chars=280)
            return {
                "review_mode": review_mode,
                "review_purpose": review_purpose,
                "review_image_path": review_image_path,
                "original_image_path": original_image_path,
                "reviewed_regions": reviewed_regions,
                "decision": "descriptive",
                "supports_landslide": None,
                "score": 0.5,
                "evidence": content,
            }
        except Exception as e:
            return {
                "review_mode": review_mode,
                "review_purpose": review_purpose,
                "review_image_path": review_image_path,
                "original_image_path": original_image_path,
                "reviewed_regions": reviewed_regions,
                "decision": "error",
                "supports_landslide": None,
                "score": 0.0,
                "evidence": f"LLM service error: {str(e)}",
            }

    user_content = [
        {"type": "image", "image_path": review_image_path},
        {
            "type": "text",
            "text": (
                f"This is the same full scene with {marker_name} already overlaid. "
                f"The {marker_name} mark suspected landslide location(s) inside the full image and should be used as spatial guidance, not as cropped patches. "
                "Explain where the suspected landslide sits in the whole scene, judge whether the highlighted regions align with landslide morphology, "
                "and summarize how this whole-image second-pass review confirms, weakens, or leaves uncertain the original full-scene interpretation. "
                + "Region metadata: "
                + json.dumps({"reviewed_regions": reviewed_regions, marker_metadata_key: region_summary}, ensure_ascii=False)
            ),
        },
    ]
    messages = [
        {
            "role": "system",
            "content": (
                f"You are a landslide interpretation expert performing a second-pass review on a full-scene image with {marker_name} overlaid. "
                f"The {marker_name} indicate suspected landslide location(s) within the same scene that was first judged from the original image. "
                "Respond in English on one line. "
                "Start with exactly one label: Support, NotSupport, or Uncertain. "
                "Then add ' | ' and a scene-grounded explanation with as much detail as needed. Prioritize direct visual evidence: frame-relative position, slope morphology, material/texture contrast, runout continuity, and what the boxed whole-image review confirms, weakens, or leaves uncertain."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    try:
        response = _openai_chat_completion(messages, temperature=0.1, trace_label="tool.seg.llm_review")
        content = _compact_text(response.get("content", ""), max_chars=280)
        verdict = content.lower()
        if verdict.startswith("support"):
            decision = "positive"
            score = 0.78
            supports_landslide: bool | None = True
        elif verdict.startswith("notsupport") or verdict.startswith("not support"):
            decision = "negative"
            score = 0.28
            supports_landslide = False
        else:
            decision = "uncertain"
            score = 0.5
            supports_landslide = None
        return {
            "review_mode": review_mode,
            "review_purpose": review_purpose,
            "review_image_path": review_image_path,
            "original_image_path": original_image_path,
            "reviewed_regions": reviewed_regions,
            "decision": decision,
            "supports_landslide": supports_landslide,
            "score": score,
            "evidence": content,
        }
    except Exception as e:
        return {
            "review_mode": review_mode,
            "review_purpose": review_purpose,
            "review_image_path": review_image_path,
            "original_image_path": original_image_path,
            "reviewed_regions": reviewed_regions,
            "decision": "error",
            "supports_landslide": None,
            "score": 0.0,
            "evidence": f"LLM service error: {str(e)}",
        }


def llm_generate_final_report(
    *,
    stage1: dict,
    refinement: dict,
    segmentation: dict,
    classification: dict | None = None,
    geo_context: dict | None = None,
    gate: dict | None = None,
    llm_second_pass: dict | None = None,
    fused_decision: dict[str, Any] | None = None,
    unavailable_evidence: list | None = None,
) -> dict:
    payload = {
        "unavailable_evidence": unavailable_evidence or [],
        "stage1": stage1,
        "refinement": refinement,
        "segmentation": segmentation,
        "classification": classification,
        "geo_context": geo_context,
        "gate": gate,
        "llm_second_pass": llm_second_pass,
        # Severity is not an image-grounded hazard assessment. Keep the
        # internal fusion value out of the narrative-model context.
        "final_decision": {
            key: value for key, value in (fused_decision or {}).items()
            if key != "severity"
        },
    }

    def _parse_json_object(text: str) -> dict[str, Any] | None:
        if not text:
            return None
        text = text.strip()
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    def _stage1_scene_description() -> str:
        explicit = _compact_text(str((stage1 or {}).get("scene_description", "") or ""), max_chars=220)
        if explicit:
            return explicit
        _, parsed = _parse_stage1_scene_assessment(str((stage1 or {}).get("evidence", "") or ""))
        if parsed:
            return parsed
        return _compact_text(
            str((stage1 or {}).get("evidence", "") or "No whole-image scene description available."),
            max_chars=220,
        )

    def _second_pass_default_note() -> str:
        review = llm_second_pass if isinstance(llm_second_pass, dict) else {}
        evidence = _compact_text(str(review.get("evidence", "") or ""), max_chars=220)
        decision = str(review.get("decision", "") or "").lower()
        review_purpose = " ".join(str(review.get("review_purpose", "") or "").strip().lower().split())
        reviewed_regions = int(review.get("reviewed_regions", 0) or 0)
        if evidence:
            if review_purpose == "description_only" or decision == "descriptive":
                prefix = "A second-pass region-overlay whole-image review was used only to supplement spatial description after first-pass screening and refinement already indicated landslide"
            elif decision == "positive":
                prefix = "A second-pass region-overlay whole-image review supported the landslide interpretation"
            elif decision == "negative":
                prefix = "A second-pass region-overlay whole-image review questioned the landslide interpretation"
            elif decision == "uncertain":
                prefix = "A second-pass region-overlay whole-image review remained inconclusive"
            else:
                prefix = "A second-pass region-overlay whole-image review was available"
            if reviewed_regions > 0:
                prefix += f" after examining {reviewed_regions} highlighted candidate region(s) that marked suspected landslide location(s)"
            return _compact_text(f"{prefix}. {evidence}", max_chars=260)
        return "No second-pass region-overlay whole-image review was used; the report relies on the original-image reading plus tool outputs."

    def _classification_reference_default() -> str:
        cls_name = str((classification or {}).get("class_name", "") or "").strip()
        cls_conf = float((classification or {}).get("confidence", 0.0) or 0.0)
        topk = (classification or {}).get("topk", [])
        if cls_name:
            detail = f"The classifier ranks this scene as {cls_name} with confidence {cls_conf:.3f}."
            if isinstance(topk, list) and topk:
                labels = []
                for item in topk[:3]:
                    if not isinstance(item, dict):
                        continue
                    label = str(item.get("class_name", "") or item.get("label", "") or "").strip()
                    score = item.get("confidence")
                    if not label:
                        continue
                    if score is None:
                        labels.append(label)
                    else:
                        labels.append(f"{label} ({float(score):.3f})")
                if labels:
                    detail += " Top-k reference classes: " + ", ".join(labels) + "."
            detail += " This classification is reference-only and helps explain likely landslide type, not the final yes/no decision."
            return _compact_text(detail, max_chars=260)
        return "No reliable classification reference was available; subtype interpretation remains open."

    def _spatial_distribution_default(region_count: int, seg_ratio: float, seg_pixels: int) -> str:
        position_text = _describe_primary_candidate_position(refinement)
        region_area_ratio = float((refinement or {}).get("area_ratio", 0.0) or 0.0)
        if region_count > 0:
            return _compact_text(
                f"{position_text} Refinement retained {region_count} candidate region(s) covering about {region_area_ratio:.4f} of the frame; segmentation estimated {seg_ratio:.4f} ({seg_pixels} pixels).",
                max_chars=300,
            )
        if seg_ratio > 0.0 or seg_pixels > 0:
            return _compact_text(
                f"No retained refinement box localized the feature, but segmentation estimated an affected area ratio of {seg_ratio:.4f} ({seg_pixels} pixels) from the full scene.",
                max_chars=300,
            )
        return "No retained refinement box or segmentation extent was available for spatial localization."

    def _tool_interpretation_default(region_count: int, seg_ratio: float, seg_pixels: int, geo_count: int) -> str:
        parts = [
            f"Scene-level model reading: {_stage1_scene_description()}",
            f"refinement retained {region_count} candidate region(s)",
            f"segmentation estimated an affected area ratio of {seg_ratio:.4f} ({seg_pixels} pixels)",
        ]
        cls_name = str((classification or {}).get("class_name", "") or "").strip()
        cls_conf = float((classification or {}).get("confidence", 0.0) or 0.0)
        if cls_name:
            parts.append(f"classification reference suggests {cls_name} ({cls_conf:.3f}) but is not used to override the decision")
        if geo_count > 0:
            parts.append(f"geographic context contributed {geo_count} nearby mapped feature(s)")
        parts.append(_second_pass_default_note())
        return _compact_text("; ".join(parts) + ".", max_chars=420)

    def _build_default_report(summary_text: str, source: str) -> dict[str, Any]:
        regions = refinement.get("regions", []) if isinstance(refinement, dict) else []
        region_count = len(regions) if isinstance(regions, list) else 0
        seg_ratio = float((segmentation or {}).get("area_ratio", 0.0) or 0.0)
        seg_pixels = int((segmentation or {}).get("landslide_pixels", 0) or 0)
        stage1_has = bool((stage1 or {}).get("has_landslide", False))
        cls_name = str((classification or {}).get("class_name", "") or "").strip()
        cls_conf = float((classification or {}).get("confidence", 0.0) or 0.0)
        geo_count = int((geo_context or {}).get("count", 0) or 0)
        # The verdict and confidence come from the single fusion rule
        # (LandslidePolicy.fuse_decision) - this narrative layer never re-derives
        # them. confidence is a real model score or None (never synthesised).
        decision = fused_decision if isinstance(fused_decision, dict) else {}
        has_landslide = bool(decision.get("has_landslide", False))
        _conf = decision.get("confidence")
        confidence = float(_conf) if isinstance(_conf, (int, float)) else None

        scene_description = _stage1_scene_description()
        if summary_text:
            summary = _compact_text(summary_text, max_chars=280)
        elif has_landslide:
            summary = _compact_text(
                f"The fused analysis supports landslide presence after cross-checking the whole-scene reading with downstream evidence. Primary scene description: {scene_description}",
                max_chars=280,
            )
        else:
            summary = _compact_text(
                f"The fused analysis does not support landslide presence after cross-checking the initial screening result with downstream evidence. Primary scene description: {scene_description}",
                max_chars=280,
            )

        recommendation = "Routine monitoring."
        if has_landslide:
            recommendation = "Manual review is recommended if the site is operationally sensitive."

        report = {
            "report_version": "1.0",
            "summary": summary,
            "has_landslide": has_landslide,
            "classification_confidence": round(cls_conf, 4) if cls_name else None,
            "landslide_type": cls_name if has_landslide and cls_name else "unknown",
            # Independent visual opinion from the narrative model. The fusion
            # layer reconciles this against the image classifier later.
            "llm_landslide_type": "",
            "key_metrics": {
                "regions_count": region_count,
                "seg_area_ratio": round(seg_ratio, 6),
                "landslide_pixels": seg_pixels,
            },
            "evidence": {
                "stage1": _compact_text((stage1 or {}).get("evidence", ""), 180),
                "classification": _compact_text(
                    f"{cls_name} ({cls_conf:.3f})" if cls_name else "",
                    120,
                ),
                "refinement": f"regions={region_count}",
                "segmentation": f"ratio={seg_ratio:.6f}, pixels={seg_pixels}",
                "geo_context": f"nearby_features={geo_count}",
            },
            "uncertainty": (
                "Evidence remains mixed and should be interpreted cautiously."
                if has_landslide
                else "Corroboration remains insufficient for a positive landslide determination."
            ),
            "recommendations": [recommendation],
            "visual_description": scene_description,
            "spatial_distribution": _spatial_distribution_default(region_count, seg_ratio, seg_pixels),
            "tool_interpretation": _tool_interpretation_default(region_count, seg_ratio, seg_pixels, geo_count),
            "classification_reference_note": _classification_reference_default(),
            "second_pass_note": _second_pass_default_note(),
            "report_source": source,
        }
        report["final_description"] = report["summary"]
        return report

    requested_fields = [
        "summary",
        "visual_description",
        "tool_interpretation",
        "uncertainty",
        "recommendations",
        "final_description",
    ]
    accepted_fields = requested_fields + [
        "spatial_distribution",
        "classification_reference_note",
        "second_pass_note",
    ]
    messages = [
        {
            "role": "system",
            "content": (
                "You are a geospatial landslide analyst writing the English narrative portion of a final report. "
                "The final decision and numeric metrics are already computed in final_decision and must not be overturned. "
                "If final_decision.has_landslide is false, explain that screening triggered further analysis but corroboration remained insufficient. "
                "If final_decision.has_landslide is true, explain the corroborated evidence without overstating certainty. "
                "Do not assign, imply, or discuss low/medium/high severity, hazard, risk, impact magnitude, or urgency: the available image evidence does not support those assessments. "
                "Return only one JSON object with no markdown and no extra commentary. "
                "Required fields: "
                + ", ".join(requested_fields)
                + ". "
                "Optional fields you may also include when useful: spatial_distribution, classification_reference_note, second_pass_note. "
                "You do not see the image: do not propose, rank, or change the landslide subtype yourself. The subtype and its cross-classifier consistency come only from the classification tool output (classification.class_name, classification.resolution, classification.sources). "
                "In segmentation/refinement outputs, regions[].class_id and tile_id are region indices, not landslide classes; never read them as a landslide type. "
                "unavailable_evidence lists evidence that could not be obtained (and any tool output with evidence_unavailable=true): state explicitly that it was not available and why, and never describe, estimate or infer its content. "
                "The required field final_description must be a complete standalone report using exactly these section headings in this order: Final Decision Report; Conclusion; Evidence Summary; Spatial Distribution; Image Feature Description; Landslide Typology (Reference Only); Rationale for Landslide Presence; Image Quality Assessment; Relative Position Within Image Frame; Environmental Impact; Confidence Level; Uncertainty Analysis; Causal Inference; Geographic Context; Final Determination. "
                "Each sentence in final_description must be complete natural prose with no dangling fragments. "
                "Keep the writing faithful to the provided tool outputs, avoid boilerplate, and avoid repeating raw numbers unless they materially improve interpretation. "
                "Every report must visibly separate fact from interpretation: prefix direct image observations and explicit tool-returned measurements with 'Observed:', and prefix logical, causal, predictive, or cross-stage reasoning with 'Inference:'. "
                "Inside final_description, use these labels in the relevant paragraphs and include explicit 'Observed evidence:' and 'Logical inference:' labels in the Evidence Summary when both are present. "
                "Never rewrite an inference as an observation; use 'Not observed' or 'Not available' for missing evidence. Keep recommendations and causal explanations under Inference, while image morphology and tool measurements stay under Observed. "
                "Image-grounded description is the highest priority in this task. In summary/visual_description/tool_interpretation, allocate at least half of the narrative to concrete scene observations (terrain shape, scarp edges, exposed material, runout traces, vegetation/drainage disruption, and frame-relative layout) before decision-level interpretation. Write visual_description as a rich multi-sentence whole-scene account rather than a short phrase. Prefer detailed whole-scene narration with concrete morphology and layout cues when evidence allows. Each string field should be natural prose, and recommendations may include as many items as needed."
            ),
        },
        {
            "role": "user",
            "content": (
                "Generate the narrative JSON for the final landslide assessment report from these tool outputs: "
                f"{json.dumps(payload, ensure_ascii=False)}"
            ),
        },
    ]
    try:
        _report_budget = int(os.getenv("LLM_REPORT_MAX_TOKENS", "3500") or "3500")
        response = _openai_chat_completion(
            messages,
            temperature=0.1,
            trace_label="tool.fuse.decision.report",
            max_tokens=_report_budget,
            timeout=180.0,
        )
        content = _compact_text(response.get("content", ""), max_chars=4000)
        parsed = _parse_json_object(content)
        base = _build_default_report("", "llm_narrative")
        if parsed:
            for key in accepted_fields:
                if key in parsed:
                    base[key] = parsed[key]
            base["summary"] = _compact_text(str(base.get("summary", "")), 280)
            base["visual_description"] = _compact_text(str(base.get("visual_description", "")), 900)
            base["spatial_distribution"] = _compact_text(str(base.get("spatial_distribution", "")), 520)
            base["tool_interpretation"] = _compact_text(str(base.get("tool_interpretation", "")), 420)
            base["llm_landslide_type"] = _compact_text(str(base.get("llm_landslide_type", "")), 120)
            base["classification_reference_note"] = _compact_text(str(base.get("classification_reference_note", "")), 280)
            base["second_pass_note"] = _compact_text(str(base.get("second_pass_note", "")), 280)
            recs = base.get("recommendations")
            if not isinstance(recs, list):
                recs = [str(recs)] if recs else []
            base["recommendations"] = [str(x) for x in recs]
            if not base["recommendations"]:
                base["recommendations"] = _build_default_report("", "llm_narrative")["recommendations"]
            base["uncertainty"] = _compact_text(str(base.get("uncertainty", "")), 200)
            final_description = str(base.get("final_description", "") or "").strip()
            if not final_description:
                final_description = str(base.get("summary", "") or "").strip()
            base["report_source"] = "llm_narrative"
            base["final_description"] = final_description
            return base
        if content:
            return _build_default_report(content, "llm_text_fallback")
    except Exception:
        pass
    return _build_default_report("", "fallback")


def chat_with_tools(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tool_executor: Callable[[str, dict[str, Any]], dict[str, Any]],
    *,
    max_turns: int | None = None,
    temperature: float = 0.2,
) -> dict[str, Any]:
    history = list(messages)
    turns = 0
    while True:
        if isinstance(max_turns, int) and max_turns > 0 and turns >= max_turns:
            return {
                "message": {
                    "role": "assistant",
                    "content": "Tool loop limit reached before a final response.",
                },
                "history": history,
            }
        turns += 1
        assistant_msg = _openai_chat_completion(
            history,
            tools=tools,
            tool_choice="auto",
            temperature=temperature,
            trace_label="agent.chat_with_tools",
        )
        tool_calls = assistant_msg.get("tool_calls") or []
        history.append(assistant_msg)

        if not tool_calls:
            return {"message": assistant_msg, "history": history}

        for call in tool_calls:
            fn = call.get("function", {}) or {}
            name = fn.get("name", "")
            raw_args = fn.get("arguments", "{}")
            try:
                parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                parsed_args = {}
            try:
                result = tool_executor(name, parsed_args)
            except Exception as exc:
                result = {"error": str(exc)}
            history.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "name": name,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )
            # fuse.decision is the authoritative terminal result. Do not send
            # it back to the LLM for an unnecessary review/planning turn, and
            # ignore any calls that were emitted after fusion in the same batch.
            if name == "fuse.decision" and isinstance(result, dict) and not result.get("error"):
                return {"message": assistant_msg, "history": history}


def _openai_chat_completion(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str = "auto",
    temperature: float = 0.2,
    timeout: float = 60.0,
    trace_label: str | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    mock_mode = os.getenv("LLM_MOCK", "0") in ("1", "true", "True")
    if mock_mode and not tools:
        response = {
            "role": "assistant",
            "content": "Yes, landslide likely. (mock)",
        }
        _record_model_raw_event(response.get("content", ""), trace_label or "llm.mock")
        return response

    base_url = os.getenv("LLM_SERVICE_URL", "http://localhost:8003/v1")
    model = os.getenv("LLM_API_MODEL_NAME", "qwen3-vl-8b-instruct")

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    # Keep planner/report generation on the base model by default. A dedicated
    # visual-evidence pass can opt into the dual-head adapter without making
    # its classification-oriented LoRA affect tool arguments or later prose.
    if trace_label == "tool.llm.first_pass":
        payload["adapter_mode"] = os.getenv("LLM_FIRST_PASS_ADAPTER", "dualhead")
    elif trace_label == "tool.seg.llm_review":
        payload["adapter_mode"] = os.getenv("LLM_VISUAL_EVIDENCE_ADAPTER", "base")
    elif trace_label == "tool.vlm.describe":
        payload["adapter_mode"] = os.getenv("LLM_DESCRIBE_ADAPTER", "dualhead")
    if max_tokens is not None and int(max_tokens) > 0:
        payload["max_tokens"] = int(max_tokens)
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice

    url = f"{base_url.rstrip('/')}/chat/completions"
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    req = request.Request(
        url,
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        _record_model_raw_event(data.get("raw_response", ""), trace_label or "llm")
        return data["choices"][0]["message"]
    except error.URLError as e:
        fallback = {
            "role": "assistant",
            "content": f"Error connecting to LLM service at {base_url}: {e}"
        }
        _record_model_raw_event(fallback.get("content", ""), trace_label or "llm.error")
        return fallback
