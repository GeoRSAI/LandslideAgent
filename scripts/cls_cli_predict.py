from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

RESULT_PREFIX = "CLS_RESULT_JSON\t"


def _load_class_mapping(path: str) -> dict[int, str]:
    mapping: dict[int, str] = {}
    path = str(path or "").strip()
    if not path:
        return mapping
    mapping_path = Path(path)
    if mapping_path.is_dir():
        return mapping
    if not mapping_path.exists():
        return mapping
    for line in mapping_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            parts = line.split("\t", 1)
        else:
            parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        try:
            mapping[int(parts[0].strip())] = parts[1].strip()
        except ValueError:
            continue
    return mapping


def _truthy_env(name: str, default: str = "0") -> bool:
    raw = str(os.getenv(name, default) or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _install_runtime_shims() -> None:
    # mmpretrain currently imports these optional modules at package import
    # time through broad __init__ imports. Provide lightweight shims so
    # ConvNeXt inference can run even when those extras are absent.
    try:
        import importlib_metadata  # type: ignore # noqa: F401
    except Exception:
        import importlib.metadata as std_meta

        shim = types.ModuleType("importlib_metadata")
        shim.PackageNotFoundError = std_meta.PackageNotFoundError
        shim.distribution = std_meta.distribution
        sys.modules["importlib_metadata"] = shim

    try:
        import einops  # type: ignore # noqa: F401
    except Exception:
        shim = types.ModuleType("einops")

        def _not_installed(*_args: Any, **_kwargs: Any) -> Any:
            raise NotImplementedError(
                "einops is not installed; this path should not be used for ConvNeXt inference."
            )

        shim.rearrange = _not_installed
        shim.repeat = _not_installed
        sys.modules["einops"] = shim

    try:
        import mat4py  # type: ignore # noqa: F401
    except Exception:
        shim = types.ModuleType("mat4py")

        def _not_installed(*_args: Any, **_kwargs: Any) -> Any:
            raise NotImplementedError(
                "mat4py is not installed; this path should not be used for ConvNeXt inference."
            )

        shim.loadmat = _not_installed
        sys.modules["mat4py"] = shim


def _resolve_class_name(
    class_id: int,
    classes: list[str],
    class_mapping: dict[int, str],
) -> str:
    if class_id in class_mapping and class_mapping[class_id]:
        return class_mapping[class_id]
    if 0 <= class_id < len(classes) and classes[class_id]:
        return classes[class_id]
    return f"class_{class_id}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run ConvNeXt single-image classification with MMPreTrain"
    )
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--mmpretrain-root",
        default=os.getenv("MMPRETRAIN_ROOT", ""),
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--class-mapping", default=os.getenv("CLS_CLASS_MAPPING_PATH", ""))
    parser.add_argument("--device", default=os.getenv("CLS_DEVICE", "cpu"))
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    mmpretrain_root = Path(args.mmpretrain_root).resolve()
    if not mmpretrain_root.exists():
        raise FileNotFoundError(f"mmpretrain root not found: {mmpretrain_root}")
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    image_path = Path(args.image).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")

    root_str = str(mmpretrain_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    if _truthy_env("CLS_FORCE_NO_WEIGHTS_ONLY_LOAD", "1"):
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    os.environ.setdefault("RICH_DISABLE", "1")
    _install_runtime_shims()

    from mmpretrain import ImageClassificationInferencer

    inferencer = ImageClassificationInferencer(
        model=str(config_path),
        pretrained=str(checkpoint_path),
        device=str(args.device),
    )

    output = inferencer(str(image_path))[0]
    scores_obj = output.get("pred_scores", [])
    if hasattr(scores_obj, "tolist"):
        scores = [float(x) for x in scores_obj.tolist()]
    else:
        scores = [float(x) for x in scores_obj]

    if not scores:
        raise RuntimeError("Classifier returned empty pred_scores.")

    classes = [str(c) for c in (getattr(inferencer, "classes", None) or [])]
    class_mapping = _load_class_mapping(args.class_mapping)
    sorted_items = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
    topk = max(1, min(int(args.topk), len(sorted_items)))
    top_items = sorted_items[:topk]
    class_id = int(top_items[0][0])

    result = {
        "image_path": str(image_path),
        "class_id": class_id,
        "class_name": _resolve_class_name(class_id, classes, class_mapping),
        "confidence": float(top_items[0][1]),
        "num_classes": len(scores),
        "topk": [
            {
                "class_id": int(idx),
                "class_name": _resolve_class_name(int(idx), classes, class_mapping),
                "score": float(score),
            }
            for idx, score in top_items
        ],
    }
    # Emit one machine-readable line for service-side parsing. Other runtime
    # warnings may still appear on stdout in some environments.
    print(f"{RESULT_PREFIX}{json.dumps(result, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
