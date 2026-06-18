from __future__ import annotations

from pathlib import Path
from time import time_ns


def build_artifact_path(base_dir: str | Path, image_path: str, suffix: str) -> Path:
    out_dir = Path(base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem or "artifact"
    return out_dir / f"{stem}_{time_ns()}_{suffix}"
