"""The landslide yes/no rule: LandslidePolicy.fuse_decision.

Decision = cross-validation of the two independent detection modalities
(whole-image VLM screening + semantic segmentation). Sub-type classification
and geospatial context are not inputs. On disagreement the VLM boundary
re-check (seg.llm_review) is the arbiter; without it the outcome is a
conservative negative.
"""
from src.agent.controller import LandslidePolicy

P = LandslidePolicy()

VLM_YES = {"has_landslide": True, "assessment_label": "likely", "score": 0.8}
VLM_NO = {"has_landslide": False, "assessment_label": "unlikely", "score": 0.15}
SEG_YES = {"area_ratio": 0.08, "landslide_pixels": 5000}
SEG_NO = {"area_ratio": 0.0, "landslide_pixels": 0}


def test_both_modalities_agree_positive():
    d = P.fuse_decision(stage1=VLM_YES, segmentation=SEG_YES)
    assert d["has_landslide"] is True
    assert d["modalities"]["agreement"] == "positive"
    # confidence is the model's own first-pass score, verbatim
    assert d["confidence"] == 0.8
    assert d["confidence_source"] == "vlm_first_pass"


def test_both_modalities_agree_negative():
    d = P.fuse_decision(stage1=VLM_NO, segmentation=SEG_NO)
    assert d["has_landslide"] is False
    assert d["severity"] == "none"
    assert d["modalities"]["agreement"] == "negative"


def test_disagreement_without_review_is_conservative_negative():
    d = P.fuse_decision(stage1=VLM_NO, segmentation=SEG_YES)
    assert d["has_landslide"] is False
    assert d["modalities"]["agreement"] == "disagreement"
    assert "conservative" in d["decision_basis"]


def test_disagreement_resolved_positive_by_review():
    review = {"decision": "positive", "score": 0.82, "review_purpose": "verification"}
    d = P.fuse_decision(stage1=VLM_NO, segmentation=SEG_YES, llm_second_pass=review)
    assert d["has_landslide"] is True
    assert d["modalities"]["second_pass_role"] == "arbiter"
    # The review's numeric score is a label placeholder, not a calibrated model
    # score, so confidence keeps the first-pass screening score.
    assert d["confidence"] == 0.15
    assert d["confidence_source"] == "vlm_first_pass"


def test_disagreement_resolved_negative_by_review():
    review = {"decision": "negative", "score": 0.7, "review_purpose": "verification"}
    d = P.fuse_decision(stage1=VLM_YES, segmentation=SEG_NO, llm_second_pass=review)
    assert d["has_landslide"] is False


def test_descriptive_second_pass_is_not_an_arbiter():
    review = {"decision": "descriptive", "score": 0.5, "review_purpose": "description_only"}
    d = P.fuse_decision(stage1=VLM_NO, segmentation=SEG_YES, llm_second_pass=review)
    assert d["has_landslide"] is False
    assert d["modalities"]["second_pass_role"] == "descriptive"
    assert d["modalities"]["second_pass_descriptive"] is True


def test_agreement_gate_can_be_disabled():
    p = LandslidePolicy(require_modality_agreement=False)
    assert p.fuse_decision(stage1=VLM_NO, segmentation=SEG_YES)["has_landslide"] is True
    assert p.fuse_decision(stage1=VLM_YES, segmentation=SEG_NO)["has_landslide"] is True


def test_severity_tracks_extent():
    assert P.fuse_decision(stage1=VLM_YES, segmentation={"area_ratio": 0.20})["severity"] == "high"
    assert P.fuse_decision(stage1=VLM_YES, segmentation={"area_ratio": 0.08})["severity"] == "medium"
    assert P.fuse_decision(
        stage1=VLM_YES, segmentation={"area_ratio": 0.02, "landslide_pixels": 5000}
    )["severity"] == "low"


def test_confidence_is_only_a_real_model_score_never_synthesised():
    # no score anywhere -> confidence is None, not a fabricated number
    d = P.fuse_decision(
        stage1={"assessment_label": "likely"},
        segmentation={"area_ratio": 0.10, "landslide_pixels": 8000},
    )
    assert d["has_landslide"] is True
    assert d["confidence"] is None
    assert d["confidence_source"] == "unavailable"

    # first-pass score is passed straight through, unclamped
    d2 = P.fuse_decision(stage1={"has_landslide": True, "score": 0.91}, segmentation=SEG_YES)
    assert d2["confidence"] == 0.91
