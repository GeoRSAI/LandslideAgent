from src.pipelines.stage5_fusion import run_stage5, screening_requires_full_analysis


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


def _assert_structured_sections(report: dict) -> None:
    final_description = report["final_description"]
    for section in REQUIRED_SECTIONS:
        assert f"### {section}\n" in final_description


def test_negative_screening_stops_full_analysis():
    stage1 = {"has_landslide": False, "score": 0.2, "evidence": "Stable vegetated slope."}
    refinement = {"regions": [], "area_ratio": 0.0, "source": "segmentation_mask"}

    assert screening_requires_full_analysis(stage1, refinement) is False

    report = run_stage5(
        stage1=stage1,
        refinement=refinement,
        classification={"class_name": "non-landslide", "confidence": 0.9},
        geo_context=None,
        gate={"area_ratio": 0.0},
        segmentation=None,
        llm_second_pass=None,
        llm_second_pass_threshold=0.6,
    )

    assert report["has_landslide"] is False
    assert report["report_source"] == "screening_early_stop"
    assert report["severity"] == "none"
    assert report["evidence"]["segmentation"] == "skipped_after_negative_screening"
    _assert_structured_sections(report)


def test_positive_screening_does_not_force_positive_final_decision():
    stage1 = {"has_landslide": False, "score": 0.2, "evidence": "Scene-level evidence weak."}
    refinement = {
        "regions": [{"bbox": [1, 2, 3, 4], "score": 0.8, "class_id": 0}],
        "area_ratio": 0.01,
        "source": "segmentation_mask",
    }

    assert screening_requires_full_analysis(stage1, refinement) is True

    report = run_stage5(
        stage1=stage1,
        refinement=refinement,
        classification={"class_name": "debris flow", "confidence": 0.7},
        geo_context=None,
        gate={"area_ratio": 0.01},
        segmentation={"area_ratio": 0.0, "landslide_pixels": 0, "polygon_count": 0},
        llm_second_pass=None,
        llm_second_pass_threshold=0.6,
    )

    assert report["has_landslide"] is False
    assert report["severity"] == "none"
    _assert_structured_sections(report)


def test_corroborated_positive_evidence_yields_positive_final_decision():
    stage1 = {"has_landslide": True, "score": 0.8, "evidence": "Likely | broad exposed scar with downslope debris trail."}
    refinement = {
        "regions": [{"bbox": [1, 2, 30, 40], "score": 0.8, "class_id": 0}],
        "area_ratio": 0.01,
        "source": "segmentation_mask",
    }

    report = run_stage5(
        stage1=stage1,
        refinement=refinement,
        classification={"class_name": "debris flow", "confidence": 0.7},
        geo_context=None,
        gate={"area_ratio": 0.01},
        segmentation={"area_ratio": 0.03, "landslide_pixels": 1200, "polygon_count": 1},
        llm_second_pass=None,
        llm_second_pass_threshold=0.6,
    )

    assert report["has_landslide"] is True
    assert report["severity"] in {"low", "medium", "high"}
    _assert_structured_sections(report)


def test_report_foregrounds_whole_image_context_and_second_pass_review():
    stage1 = {
        "has_landslide": True,
        "score": 0.8,
        "evidence": "Likely | broad exposed scar across a steep hillside with downslope debris.",
        "scene_description": "broad exposed scar across a steep hillside with downslope debris.",
    }
    refinement = {
        "regions": [{"bbox": [10, 20, 110, 160], "score": 0.88, "class_id": 0}],
        "area_ratio": 0.04,
        "source": "segmentation_mask",
    }
    segmentation = {"area_ratio": 0.03, "landslide_pixels": 1500, "polygon_count": 1}
    llm_second_pass = {
        "review_mode": "seg_boundary_whole_image",
        "reviewed_regions": 1,
        "decision": "positive",
        "score": 0.78,
        "evidence": "Support | the boxed region aligns with the exposed scar and downslope runout visible in the full scene.",
    }

    report = run_stage5(
        stage1=stage1,
        refinement=refinement,
        classification={"class_name": "debris flow", "confidence": 0.72},
        geo_context=None,
        gate={"area_ratio": 0.04},
        segmentation=segmentation,
        llm_second_pass=llm_second_pass,
        llm_second_pass_threshold=0.6,
    )

    final_description = report["final_description"]
    assert "spatial_distribution" in report
    assert "Primary candidate region" in report["spatial_distribution"]
    assert "Overall scene description:" in final_description
    assert "broad exposed scar across a steep hillside with downslope debris." in final_description
    assert "second-pass VLM review on the full image with segmentation boundaries overlaid" in final_description
    assert "whole-image second-pass review" in final_description
    assert "Reference classification: debris flow (0.72)." in final_description
    assert "reference-only" in final_description


def test_description_only_second_pass_supplements_description_without_becoming_a_vote():
    stage1 = {
        "has_landslide": True,
        "score": 0.82,
        "assessment_label": "likely",
        "evidence": "Likely | exposed scar on the upper-left hillside with short downslope debris.",
        "scene_description": "exposed scar on the upper-left hillside with short downslope debris.",
    }
    refinement = {
        "regions": [{"bbox": [10, 20, 90, 120], "score": 0.88, "class_id": 0}],
        "area_ratio": 0.03,
        "source": "segmentation_mask",
    }
    llm_second_pass = {
        "review_mode": "seg_boundary_whole_image",
        "review_purpose": "description_only",
        "reviewed_regions": 1,
        "decision": "descriptive",
        "score": 0.5,
        "evidence": "Describe | the boxed landslide sits in the upper-left part of the frame and follows the exposed slope break.",
    }

    report = run_stage5(
        stage1=stage1,
        refinement=refinement,
        classification={"class_name": "debris flow", "confidence": 0.72},
        geo_context=None,
        gate={"area_ratio": 0.03},
        segmentation={"area_ratio": 0.03, "landslide_pixels": 1200, "polygon_count": 1},
        llm_second_pass=llm_second_pass,
        llm_second_pass_threshold=0.6,
    )

    assert report["has_landslide"] is True
    assert report["decision_support"]["second_pass_positive"] is False
    assert report["decision_support"]["second_pass_negative"] is False
    assert report["decision_support"]["second_pass_descriptive"] is True
    assert "descriptive support only" in report["final_description"] or "descriptive and did not add an extra yes/no vote" in report["final_description"]
    _assert_structured_sections(report)


def test_llm_and_region_scores_are_ignored_in_final_report_confidence():
    stage1 = {
        "has_landslide": True,
        "score": 0.01,
        "assessment_label": "likely",
        "evidence": "Likely | exposed scar on a steep hillside with downslope debris.",
        "scene_description": "exposed scar on a steep hillside with downslope debris.",
    }
    refinement = {
        "regions": [{"bbox": [10, 20, 90, 120], "score": 0.8, "class_id": 0}],
        "area_ratio": 0.03,
        "source": "segmentation_mask",
    }
    segmentation = {"area_ratio": 0.03, "landslide_pixels": 1200, "polygon_count": 1}

    report = run_stage5(
        stage1=stage1,
        refinement=refinement,
        classification={"class_name": "debris flow", "confidence": 0.7},
        geo_context=None,
        gate={"area_ratio": 0.03},
        segmentation=segmentation,
        llm_second_pass={
            "review_mode": "seg_boundary_whole_image",
            "review_purpose": "verification",
            "reviewed_regions": 1,
            "decision": "positive",
            "score": 0.99,
            "evidence": "Support | highlighted region aligns with the exposed scar.",
        },
        llm_second_pass_threshold=0.6,
    )

    assert report["has_landslide"] is True
    assert report["confidence"] is None
    assert report["confidence_source"] == "not_computed"
    assert report["decision_support"]["llm_scores_ignored"] is True
    assert report["decision_support"]["heuristic_scores_removed"] is True
    assert "Initial Screening: Confirmed landslide presence with score" not in report["final_description"]
    assert "top confidence" not in report["final_description"]
    assert "confidence=0." not in report["final_description"]
    _assert_structured_sections(report)


def test_final_report_drops_obviously_incomplete_tail_sentences():
    stage1 = {
        "has_landslide": True,
        "assessment_label": "likely",
        "evidence": "Likely | A complete sentence about an exposed scar. The landslide is near a me",
        "scene_description": "A complete sentence about an exposed scar. The landslide is near a me",
    }
    refinement = {
        "regions": [{"bbox": [10, 20, 90, 120], "class_id": 0}],
        "area_ratio": 0.03,
        "source": "segmentation_mask",
    }
    report = run_stage5(
        stage1=stage1,
        refinement=refinement,
        classification={"class_name": "debris flow", "confidence": 0.7},
        geo_context=None,
        gate={"area_ratio": 0.03},
        segmentation={"area_ratio": 0.03, "landslide_pixels": 1200, "polygon_count": 1},
        llm_second_pass={
            "review_mode": "seg_boundary_whole_image",
            "review_purpose": "description_only",
            "reviewed_regions": 1,
            "decision": "descriptive",
            "evidence": "Describe | The highlighted region follows the exposed slope break. It is adjacent to a me",
        },
        llm_second_pass_threshold=0.6,
    )

    assert "near a." not in report["final_description"]
    assert "near a me." not in report["final_description"]
    assert "adjacent to a." not in report["final_description"]
    assert "adjacent to a me." not in report["final_description"]
    assert "A complete sentence about an exposed scar." in report["final_description"]
    _assert_structured_sections(report)


def test_final_report_strips_ellipsis_from_report_text():
    stage1 = {
        "has_landslide": True,
        "score": 0.8,
        "assessment_label": "likely",
        "evidence": "Likely | upper-left exposed scar... with short downslope debris... and stripped vegetation.",
        "scene_description": "upper-left exposed scar... with short downslope debris... and stripped vegetation.",
    }
    refinement = {
        "regions": [{"bbox": [10, 20, 90, 120], "score": 0.88, "class_id": 0}],
        "area_ratio": 0.03,
        "source": "segmentation_mask",
    }
    segmentation = {"area_ratio": 0.03, "landslide_pixels": 1500, "polygon_count": 1}
    llm_second_pass = {
        "review_mode": "seg_boundary_whole_image",
        "review_purpose": "description_only",
        "reviewed_regions": 1,
        "decision": "descriptive",
        "score": 0.5,
        "evidence": "Describe | the boxed region sits in the upper-left frame... and follows the exposed slope break...",
    }

    report = run_stage5(
        stage1=stage1,
        refinement=refinement,
        classification={"class_name": "debris flow", "confidence": 0.72},
        geo_context=None,
        gate={"area_ratio": 0.03},
        segmentation=segmentation,
        llm_second_pass=llm_second_pass,
        llm_second_pass_threshold=0.6,
    )

    assert "..." not in report["summary"]
    assert "..." not in report["final_description"]
    assert "…" not in report["final_description"]
    _assert_structured_sections(report)


def test_report_includes_slope_aspect_and_osm_poi_details_when_geo_context_available():
    stage1 = {
        "has_landslide": True,
        "score": 0.84,
        "assessment_label": "likely",
        "evidence": "Likely | broad scar with downslope debris fan.",
        "scene_description": "broad scar with downslope debris fan.",
    }
    refinement = {
        "regions": [{"bbox": [120, 90, 360, 310], "score": 0.9, "class_id": 0}],
        "area_ratio": 0.08,
        "source": "segmentation_mask",
        "candidate_tiles": [{"tile_id": 0, "x": 0, "y": 0, "w": 512, "h": 512}],
    }
    geo_context = {
        "nearby": {
            "count": 3,
            "radius_m": 300,
            "features": [
                {"type": "amenity", "subtype": "school", "name": "Xiangshan School"},
                {"type": "road", "subtype": "primary", "name": "S201"},
                {"type": "settlement", "subtype": "village", "name": "Shanbei"},
            ],
        },
        "background": {
            "terrain": {"slope_deg": 32.6, "aspect_deg": 145.2},
            "geology": {"lithology": "colluvium"},
        },
    }

    report = run_stage5(
        stage1=stage1,
        refinement=refinement,
        classification={"class_name": "debris flow", "confidence": 0.78},
        geo_context=geo_context,
        gate={"area_ratio": 0.08},
        segmentation={"area_ratio": 0.04, "landslide_pixels": 2200, "polygon_count": 1},
        llm_second_pass=None,
        llm_second_pass_threshold=0.6,
    )

    final_description = report["final_description"]
    assert "Terrain slope/aspect:" in final_description
    assert "aspect=145.20°" in final_description
    assert "Nearby OSM POI" in final_description
    assert "Xiangshan School" in final_description
    assert "S201" in final_description
