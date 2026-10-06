import numpy as np
from PIL import Image

from src.pipelines.stage5_fusion import _classification_reconciliation_report_note, _classifier_reconciliation
from src.utils.geometry import locate_primary_candidate


def _cls(qwen, convnext, resolution, class_name, conflict=False):
    return {
        "class_name": class_name,
        "resolution": resolution,
        "conflict": conflict,
        "sources": {"vlm": {"class_name": qwen}, "image_classifier": {"class_name": convnext}},
    }


def test_conflict_names_both_image_classifiers():
    rec = _classifier_reconciliation(_cls("Earth slide", "Debris flow", "conflict", "Earth slide", True))
    note = _classification_reconciliation_report_note(rec)
    assert rec["status"] == "conflict"
    assert "Qwen classification head proposed Earth slide" in note
    assert "ConvNeXt image classifier proposed Debris flow" in note
    assert "LLM" not in note


def test_agreement_and_single_source():
    assert _classifier_reconciliation(_cls("Mud flow", "Mud flow", "subclass_agreement", "Mud flow"))["status"] == "agreement"
    assert _classifier_reconciliation({"class_name": "Rock fall"})["status"] == "single_source"
    assert _classifier_reconciliation({})["status"] == "unavailable"


def test_position_uses_mask_centroid(tmp_path):
    mask = np.zeros((300, 300), dtype=np.uint8)
    mask[0:300, 0:60] = 255      # thin strip on the left edge
    mask[0:20, 0:300] = 255      # thin strip on the top edge -> bbox spans whole frame
    path = tmp_path / "m.png"
    Image.fromarray(mask).save(path)
    ref = {"regions": [{"bbox": [0, 0, 300, 300]}], "mask_path": str(path)}
    loc = locate_primary_candidate(ref, {"width": 300, "height": 300})
    assert loc["position_source"] == "mask_centroid"
    assert loc["position"] != "middle-center"


def test_position_falls_back_to_bbox_without_mask():
    loc = locate_primary_candidate({"regions": [{"bbox": [0, 0, 100, 100]}]}, {"width": 300, "height": 300})
    assert loc["position_source"] == "bbox_center"
    assert loc["position"] == "upper-left"
