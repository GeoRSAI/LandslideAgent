from __future__ import annotations

import re
from typing import Any


# Cruden-Varnes style decomposition: movement form is the parent class and
# material composition identifies the more specific subtype.
_CLASS_METADATA: dict[str, dict[str, str]] = {
    "No landslide": {"parent": "No landslide", "movement": "none", "material": "none"},
    "Debris flow": {"parent": "Flow", "movement": "flow", "material": "debris"},
    "Mud flow": {"parent": "Flow", "movement": "flow", "material": "earth"},
    "Earthflow": {"parent": "Flow", "movement": "flow", "material": "earth"},
    "Mudslide": {"parent": "Slide", "movement": "slide", "material": "earth"},
    "Earth slide": {"parent": "Slide", "movement": "slide", "material": "earth"},
    "Rock fall": {"parent": "Fall", "movement": "fall", "material": "rock"},
    "Rock slide": {"parent": "Slide", "movement": "slide", "material": "rock"},
}


def _compact(value: Any) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def normalize_landslide_label(value: Any) -> str:
    """Normalize model labels into the report taxonomy where possible."""
    text = _compact(value)
    if not text or text in {"unknown", "uncertain", "not available", "not observed"}:
        return ""
    if any(token in text for token in ("no landslide", "non landslide", "no slide", "none")):
        return "No landslide"
    if "debris" in text and "flow" in text:
        return "Debris flow"
    if "mud" in text and "flow" in text:
        return "Mud flow"
    if "earth" in text and "flow" in text:
        return "Earthflow"
    if "mudslide" in text or ("mud" in text and "slide" in text):
        return "Mudslide"
    if "earth" in text and "slide" in text:
        return "Earth slide"
    if "rock" in text and "fall" in text:
        return "Rock fall"
    if "rock" in text and "slide" in text:
        return "Rock slide"
    if text in {"flow", "flows"} or text.endswith(" flow"):
        return "Flow"
    if text in {"slide", "slides"} or text.endswith(" slide"):
        return "Slide"
    if text in {"fall", "falls"} or text.endswith(" fall"):
        return "Fall"
    return str(value or "").strip()


def describe_landslide_class(value: Any) -> dict[str, Any]:
    canonical = normalize_landslide_label(value)
    metadata = _CLASS_METADATA.get(canonical)
    if metadata is None and canonical in {"Flow", "Slide", "Fall"}:
        metadata = {"parent": canonical, "movement": canonical.lower(), "material": "unspecified"}
    if metadata is None:
        return {
            "raw": str(value or "").strip(),
            "canonical": canonical,
            "known": False,
            "parent": "",
            "movement": "",
            "material": "",
        }
    return {
        "raw": str(value or "").strip(),
        "canonical": canonical,
        "known": True,
        **metadata,
    }


def reconcile_landslide_types(llm_label: Any, image_label: Any) -> dict[str, Any]:
    """Reconcile an LLM subtype with the image classifier under Cruden-Varnes.

    Different subtypes sharing the same movement-form parent fall back to that
    parent. Different parents are never silently resolved and are reported as a
    classification conflict.
    """
    llm = describe_landslide_class(llm_label)
    image = describe_landslide_class(image_label)
    llm_known = bool(llm["known"] and llm["canonical"])
    image_known = bool(image["known"] and image["canonical"])

    base = {
        "llm_label": llm["raw"],
        "image_classifier_label": image["raw"],
        "llm_canonical": llm["canonical"],
        "image_classifier_canonical": image["canonical"],
        "llm_parent": llm["parent"],
        "image_classifier_parent": image["parent"],
        "movement_form": "",
        "material_composition": "",
        "resolved_label": "",
        "status": "unavailable",
        "conflict": False,
        "explanation": "",
    }
    if not llm_known and not image_known:
        base["explanation"] = "Neither source supplied a usable Cruden-Varnes subtype."
        return base
    if not llm_known or not image_known:
        chosen = llm if llm_known else image
        base.update(
            resolved_label=chosen["canonical"],
            movement_form=chosen["movement"],
            material_composition=chosen["material"],
            status="single_source",
            explanation="Only one source supplied a usable subtype; no cross-source conflict was evaluated.",
        )
        return base

    if llm["canonical"] == image["canonical"]:
        base.update(
            resolved_label=llm["canonical"],
            movement_form=llm["movement"],
            material_composition=llm["material"],
            status="agreement",
            explanation="The LLM and image classifier agree on the subtype.",
        )
        return base

    if llm["parent"] and llm["parent"] == image["parent"]:
        base.update(
            resolved_label=llm["parent"],
            movement_form=llm["movement"],
            material_composition="mixed or unresolved",
            status="parent_fallback",
            explanation=(
                f"Subtype disagreement ({llm['canonical']} vs {image['canonical']}) "
                f"but common movement-form parent '{llm['parent']}' is retained."
            ),
        )
        return base

    base.update(
        resolved_label="classification conflict",
        status="conflict",
        conflict=True,
        explanation=(
            f"The LLM ({llm['canonical'] or llm['raw']}) and image classifier "
            f"({image['canonical'] or image['raw']}) disagree at the movement-form parent level."
        ),
    )
    return base
