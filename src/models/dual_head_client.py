from __future__ import annotations

import json
import os
from typing import Any
from urllib import error, request

# Process-local cache: stage1 (llm.first_pass) and stage3 (cls.run) both need
# the dual-head classification for the same image within one analysis. The
# classifier is a single forward pass keyed only by the image content /
# prompt, so it is safe (and avoids a second GPU forward + adapter switch) to
# compute it once and let cls.run read the cached result back.
_CACHE: dict[str, dict[str, Any]] = {}


def _service_url() -> str:
    # Mirrors llm_client._openai_chat_completion's LLM_SERVICE_URL, which
    # already includes the /v1 suffix.
    base = os.getenv("LLM_SERVICE_URL", "http://localhost:8003/v1")
    return base.rstrip("/")


def classify(image_path: str, topk: int = 5, use_cache: bool = True) -> dict[str, Any]:
    """Call llm_service's dual-head classifier. Returns the same contract as
    cls_infer.run_classification: {class_id, class_name, confidence, topk},
    plus has_landslide/score for stage1's use. Never raises: callers get a
    negative/empty result on any failure so the pipeline degrades gracefully
    (matches the fallback philosophy already used for cls_service/geo tools)."""
    image_path = str(image_path or "").strip()
    empty = {
        "class_id": -1,
        "class_name": "",
        "confidence": 0.0,
        "topk": [],
        "has_landslide": False,
        "score": 0.0,
    }
    if not image_path:
        return empty
    if use_cache and image_path in _CACHE:
        return _CACHE[image_path]

    url = f"{_service_url()}/dual_head_classify"
    payload = json.dumps({"image_path": image_path, "topk": topk}).encode("utf-8")
    req = request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    timeout = float(os.getenv("LLM_SERVICE_TIMEOUT", "120.0"))
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except (error.URLError, TimeoutError, ConnectionError, OSError):
        return empty
    except Exception:
        return empty

    if use_cache:
        _CACHE[image_path] = result
    return result


def clear_cache(image_path: str | None = None) -> None:
    if image_path is None:
        _CACHE.clear()
    else:
        _CACHE.pop(image_path, None)
