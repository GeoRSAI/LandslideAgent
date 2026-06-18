from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from src.utils import build_artifact_path

_MODEL = None
_MODEL_BACKEND = ""
DEFAULT_MMSEG_CONFIG_PATH = ""
DEFAULT_MMSEG_CHECKPOINT_PATH = ""


def _truthy_env(name: str, default: str = "0") -> bool:
    raw = str(os.getenv(name, default) or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _resolve_backend() -> str:
    """
    Backend choices:
    - mmseg: OpenMMLab mmsegmentation model loader (default)
    - legacy: existing deeplabv3-plus-pytorch loader
    - auto: use mmseg when config is available, otherwise legacy
    """
    requested = str(
        os.getenv("SEG_BACKEND", os.getenv("DEEPLAB_BACKEND", "mmseg")) or "mmseg"
    ).strip().lower()
    if requested in {"", "auto"}:
        has_cfg = bool(str(os.getenv("MMSEG_CONFIG_PATH", "") or "").strip())
        if not has_cfg:
            has_cfg = Path(DEFAULT_MMSEG_CONFIG_PATH).exists()
        return "mmseg" if has_cfg else "legacy"
    if requested not in {"legacy", "mmseg"}:
        raise ValueError(
            f"Unsupported SEG_BACKEND/DEEPLAB_BACKEND={requested!r}. "
            "Use mmseg, legacy, or auto."
        )
    return requested


def _load_legacy_model():
    repo_path = os.getenv("DEEPLAB_REPO_PATH", "")
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    from deeplab import DeeplabV3

    model_path = os.getenv("DEEPLAB_MODEL_PATH", "")
    num_classes = int(os.getenv("DEEPLAB_NUM_CLASSES", "2"))
    backbone = os.getenv("DEEPLAB_BACKBONE", "mobilenet")
    input_shape_str = os.getenv("DEEPLAB_INPUT_SHAPE", "512,512")
    input_shape_parts = [p.strip() for p in input_shape_str.split(",") if p.strip()]
    input_shape = [int(p) for p in input_shape_parts] if len(input_shape_parts) == 2 else [512, 512]
    return DeeplabV3(
        model_path=model_path,
        num_classes=num_classes,
        backbone=backbone,
        input_shape=input_shape,
        cuda=torch.cuda.is_available(),
    )


def _load_mmseg_model():
    config_path = str(os.getenv("MMSEG_CONFIG_PATH", DEFAULT_MMSEG_CONFIG_PATH) or "").strip()
    checkpoint_path = str(
        os.getenv(
            "MMSEG_CHECKPOINT_PATH",
            os.getenv("DEEPLAB_MODEL_PATH", DEFAULT_MMSEG_CHECKPOINT_PATH),
        )
        or ""
    ).strip()
    device = str(
        os.getenv("MMSEG_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu") or "cpu"
    ).strip()

    if not config_path:
        raise RuntimeError(
            "MMSEG backend selected but MMSEG_CONFIG_PATH is empty. "
            "Please provide your OpenMMLab config .py path."
        )
    if not Path(config_path).exists():
        raise RuntimeError(f"MMSEG config not found: {config_path}")
    if checkpoint_path and not Path(checkpoint_path).exists():
        raise RuntimeError(f"MMSEG checkpoint not found: {checkpoint_path}")

    # PyTorch >=2.6 defaults torch.load(weights_only=True), while many
    # OpenMMLab checkpoints store additional python objects in metadata.
    if _truthy_env("MMSEG_FORCE_NO_WEIGHTS_ONLY_LOAD", "1"):
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    try:
        from mmseg.apis import init_model
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "MMSEG backend selected but mmseg/mmcv/mmengine is unavailable in current env. "
            "Install mmsegmentation in SEG env, then retry."
        ) from exc

    return init_model(config_path, checkpoint=checkpoint_path or None, device=device)


def _load_model():
    global _MODEL, _MODEL_BACKEND
    backend = _resolve_backend()
    if _MODEL is not None and _MODEL_BACKEND == backend:
        return _MODEL
    _MODEL = _load_mmseg_model() if backend == "mmseg" else _load_legacy_model()
    _MODEL_BACKEND = backend
    return _MODEL


def _extract_mmseg_pred_label(result: Any) -> np.ndarray:
    obj = result
    if isinstance(obj, (list, tuple)) and obj:
        obj = obj[0]

    pred = None
    if isinstance(obj, np.ndarray):
        pred = obj
    elif isinstance(obj, dict):
        pred = obj.get("pred_sem_seg")
    elif hasattr(obj, "pred_sem_seg"):
        pred = getattr(obj, "pred_sem_seg")

    if pred is None:
        raise RuntimeError("mmseg inference result does not contain pred_sem_seg.")

    if hasattr(pred, "data") and not isinstance(pred, np.ndarray):
        pred = pred.data
    if hasattr(pred, "detach"):
        pred = pred.detach().cpu().numpy()
    else:
        pred = np.asarray(pred)

    if pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred[0]
    elif pred.ndim == 3 and pred.shape[-1] == 1:
        pred = pred[..., 0]

    if pred.ndim != 2:
        raise RuntimeError(f"Unexpected mmseg prediction shape: {pred.shape}")
    return pred.astype(np.int64, copy=False)


def _predict_binary_mask(model: Any, image: Image.Image, image_path: str) -> np.ndarray:
    backend = _resolve_backend()
    if backend == "legacy":
        mask = model.get_miou_png(image)
        return np.array(mask) != 0

    try:
        from mmseg.apis import inference_model
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("MMSEG backend runtime import failed.") from exc

    result = inference_model(model, image_path)
    pred_label = _extract_mmseg_pred_label(result)
    if _truthy_env("MMSEG_FOREGROUND_NONZERO", "0"):
        return pred_label != 0
    target_class = int(os.getenv("MMSEG_LANDSLIDE_CLASS_INDEX", "1"))
    return pred_label == target_class


def run_segmentation(image_info: dict) -> dict:
    image_path = image_info.get("image_path")
    if not image_path:
        return {
            "mask_path": "",
            "landslide_pixels": 0,
            "area_ratio": 0.0,
            "polygon_count": 0,
        }
    image = Image.open(image_path).convert("RGB")
    model = _load_model()
    binary_mask = _predict_binary_mask(model, image, str(image_path))
    landslide_pixels = int(binary_mask.sum())
    total = int(binary_mask.size)
    ratio = landslide_pixels / total if total else 0.0
    mask_path = build_artifact_path("outputs/masks", str(image_path), "mask.png")
    overlay_path = build_artifact_path("outputs/masks", str(image_path), "overlay.png")
    mask_img = Image.fromarray((binary_mask.astype(np.uint8) * 255), mode="L")
    mask_img.save(mask_path)

    # Build semi-transparent red overlay for frontend display.
    base = np.array(image.convert("RGB"), dtype=np.uint8)
    color = np.array([255, 64, 64], dtype=np.uint8)
    alpha = 0.4
    base[binary_mask] = ((1 - alpha) * base[binary_mask] + alpha * color).astype(np.uint8)
    Image.fromarray(base).save(overlay_path)

    return {
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
        "landslide_pixels": landslide_pixels,
        "area_ratio": ratio,
        "polygon_count": 1 if landslide_pixels > 0 else 0,
    }
