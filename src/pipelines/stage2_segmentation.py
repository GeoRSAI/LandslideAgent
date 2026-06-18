from __future__ import annotations

import json
import os
from urllib import request, error

from src.models.seg_infer import run_segmentation


def _run_stage2_via_service(image_info: dict) -> dict:
    image_path = str(image_info.get("image_path", ""))
    if not image_path:
        return {
            "mask_path": "",
            "landslide_pixels": 0,
            "area_ratio": 0.0,
            "polygon_count": 0,
        }
    service_url = os.getenv("SEG_SERVICE_URL", "http://127.0.0.1:8002").rstrip("/")
    timeout = float(os.getenv("SEG_SERVICE_TIMEOUT", "90.0"))
    req = request.Request(
        f"{service_url}/predict",
        data=json.dumps({"image_path": image_path}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_stage2(image_info: dict) -> dict:
    prefer_service = os.getenv("SEG_PREFER_SERVICE", "1") not in ("0", "false", "False")
    service_err = None
    if prefer_service:
        try:
            return _run_stage2_via_service(image_info)
        except (error.URLError, TimeoutError, ConnectionError) as exc:
            service_err = f"seg service unreachable: {exc}"
        except Exception as exc:
            service_err = f"seg service failed: {exc}"
    try:
        return run_segmentation(image_info)
    except Exception as exc:
        if service_err:
            raise RuntimeError(f"{service_err}; local fallback failed: {exc}") from exc
        raise
