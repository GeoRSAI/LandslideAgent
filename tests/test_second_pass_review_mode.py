from pathlib import Path

from PIL import Image

from src.pipelines.stage4_segmentation_refine import run_stage4


def test_second_pass_uses_boxed_whole_image_input(tmp_path):
    image_path = tmp_path / "scene.png"
    Image.new("RGB", (128, 96), color=(240, 240, 240)).save(image_path)

    region = {
        "bbox": [12, 18, 80, 70],
        "score": 0.91,
        "class_id": 0,
        "tile_id": 0,
    }
    result = run_stage4(
        [],
        image_info={"image_path": str(image_path), "width": 128, "height": 96},
        regions=[region],
        run_llm_second_pass=False,
        llm_second_pass_max_area_ratio=0.1,
    )

    review_input = result["llm_review_input"]
    assert review_input["review_mode"] == "seg_boundary_whole_image"
    assert review_input["overlay_source"] == "segmentation_mask_boundary"
    assert review_input["review_purpose"] == "verification"
    assert review_input["image_path"] == str(image_path)
    assert review_input["review_image_path"] == result["overlay_path"]
    assert Path(review_input["review_image_path"]).exists()
    assert review_input["reviewed_regions"] == 1
    assert review_input["regions"][0]["bbox"] == [12.0, 18.0, 80.0, 70.0]
    assert "crop_bbox" not in review_input
    assert result["llm_review_tiles"] == [review_input]
    assert result["llm_second_pass_skipped_for_large_area"] is False


def test_second_pass_becomes_description_only_when_stage1_and_region_refinement_agree(tmp_path):
    image_path = tmp_path / "scene.png"
    Image.new("RGB", (128, 96), color=(240, 240, 240)).save(image_path)

    region = {
        "bbox": [12, 18, 80, 70],
        "score": 0.91,
        "class_id": 0,
        "tile_id": 0,
    }
    stage1 = {
        "has_landslide": True,
        "assessment_label": "likely",
        "scene_description": "broad exposed scar on the upper-left slope.",
    }

    result = run_stage4(
        [],
        image_info={"image_path": str(image_path), "width": 128, "height": 96},
        stage1=stage1,
        regions=[region],
        run_llm_second_pass=False,
        llm_second_pass_max_area_ratio=0.1,
    )

    review_input = result["llm_review_input"]
    assert review_input["review_purpose"] == "description_only"
    assert review_input["stage1_positive"] is True
    assert review_input["stage1_assessment_label"] == "likely"
    assert review_input["stage1_scene_description"] == "broad exposed scar on the upper-left slope."
