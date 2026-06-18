from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image

from src.models import seg_infer


class _FakePred:
    def __init__(self, arr: np.ndarray):
        self.data = arr


class _FakeResult:
    def __init__(self, arr: np.ndarray):
        self.pred_sem_seg = _FakePred(arr)


def _install_fake_mmseg(monkeypatch, pred_2d: np.ndarray) -> None:
    def fake_init_model(config_path: str, checkpoint: str | None = None, device: str = "cpu"):
        return {
            "config_path": config_path,
            "checkpoint": checkpoint,
            "device": device,
        }

    def fake_inference_model(model, image_path: str):
        _ = model, image_path
        # mmseg often uses shape [1, H, W]
        return _FakeResult(pred_2d[None, ...].astype(np.int64))

    mmseg_mod = types.ModuleType("mmseg")
    apis_mod = types.ModuleType("mmseg.apis")
    apis_mod.init_model = fake_init_model
    apis_mod.inference_model = fake_inference_model
    mmseg_mod.apis = apis_mod

    monkeypatch.setitem(sys.modules, "mmseg", mmseg_mod)
    monkeypatch.setitem(sys.modules, "mmseg.apis", apis_mod)


def _prepare_mmseg_env(monkeypatch, tmp_path: Path) -> None:
    config = tmp_path / "mmseg_config.py"
    checkpoint = tmp_path / "mmseg_checkpoint.pth"
    config.write_text("# fake config\n", encoding="utf-8")
    checkpoint.write_bytes(b"fake")

    monkeypatch.setenv("SEG_BACKEND", "mmseg")
    monkeypatch.setenv("MMSEG_CONFIG_PATH", str(config))
    monkeypatch.setenv("MMSEG_CHECKPOINT_PATH", str(checkpoint))
    monkeypatch.setenv("MMSEG_DEVICE", "cpu")


def _reset_cached_model(monkeypatch) -> None:
    monkeypatch.setattr(seg_infer, "_MODEL", None)


def test_mmseg_backend_uses_target_class_index(monkeypatch, tmp_path):
    _reset_cached_model(monkeypatch)
    _prepare_mmseg_env(monkeypatch, tmp_path)
    monkeypatch.setenv("MMSEG_LANDSLIDE_CLASS_INDEX", "2")
    monkeypatch.setenv("MMSEG_FOREGROUND_NONZERO", "0")
    _install_fake_mmseg(monkeypatch, np.array([[0, 2], [1, 2]], dtype=np.int64))

    image_path = tmp_path / "scene.png"
    Image.new("RGB", (2, 2), color=(200, 200, 200)).save(image_path)

    result = seg_infer.run_segmentation({"image_path": str(image_path)})
    assert result["landslide_pixels"] == 2
    assert abs(result["area_ratio"] - 0.5) < 1e-9
    assert Path(result["mask_path"]).exists()
    assert Path(result["overlay_path"]).exists()


def test_mmseg_backend_supports_nonzero_foreground_mode(monkeypatch, tmp_path):
    _reset_cached_model(monkeypatch)
    _prepare_mmseg_env(monkeypatch, tmp_path)
    monkeypatch.setenv("MMSEG_FOREGROUND_NONZERO", "1")
    _install_fake_mmseg(monkeypatch, np.array([[0, 2], [1, 2]], dtype=np.int64))

    image_path = tmp_path / "scene2.png"
    Image.new("RGB", (2, 2), color=(200, 200, 200)).save(image_path)

    result = seg_infer.run_segmentation({"image_path": str(image_path)})
    assert result["landslide_pixels"] == 3
    assert abs(result["area_ratio"] - 0.75) < 1e-9
