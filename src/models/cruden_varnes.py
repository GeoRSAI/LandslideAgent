"""Cruden & Varnes (1996) landslide taxonomy for subtype-disagreement
resolution between the VLM (dual-head) classifier and the image classifier
(ConvNeXt).

Classification combines movement type (运动形式: Fall / Slide / Flow) with
material composition (物质组成: Rock / Debris / Mud / Earth). When the two
classifier opinions give different subclasses, we fall back to the parent
class they still agree on; when even the parent (movement or material)
disagrees, the result is flagged as a conflict for the report to surface.
"""

from __future__ import annotations

import re
from typing import Any

# Canonical subclass labels shared by the dual-head VLM and ConvNeXt.
CANONICAL_LABELS: tuple[str, ...] = (
    "No landslide",
    "Debris flow",
    "Mud flow",
    "Mudslide",
    "Earth slide",
    "Earthflow",
    "Rock fall",
    "Rock slide",
)

# Cruden & Varnes: subclass -> (movement type, material composition)
_LABEL_PARENTS: dict[str, dict[str, str]] = {
    "No landslide": {"movement": "None", "material": "None"},
    "Rock fall": {"movement": "Fall", "material": "Rock"},
    "Rock slide": {"movement": "Slide", "material": "Rock"},
    "Debris flow": {"movement": "Flow", "material": "Debris"},
    "Mud flow": {"movement": "Flow", "material": "Mud"},
    "Mudslide": {"movement": "Slide", "material": "Mud"},
    "Earth slide": {"movement": "Slide", "material": "Earth"},
    "Earthflow": {"movement": "Flow", "material": "Earth"},
}

_ALIASES: dict[str, str] = {
    "non": "No landslide",
    "no landslide": "No landslide",
    "none": "No landslide",
    "debris flow": "Debris flow",
    "debrisflow": "Debris flow",
    "mud flow": "Mud flow",
    "mudflow": "Mud flow",
    "mud slide": "Mudslide",
    "mudslide": "Mudslide",
    "earth slide": "Earth slide",
    "earthslide": "Earth slide",
    "earth flow": "Earthflow",
    "earthflow": "Earthflow",
    "rock fall": "Rock fall",
    "rockfall": "Rock fall",
    "rock slide": "Rock slide",
    "rockslide": "Rock slide",
}


def normalize_label(name: Any) -> str | None:
    """Map any classifier output (aliases, underscores, case variants) to a
    canonical Cruden & Varnes subclass label, or None if unknown/empty."""
    if name is None:
        return None
    key = str(name).strip().strip("`*_[](){}:;,.，。").lower()
    key = key.replace("_", " ").replace("-", " ")
    key = re.sub(r"\s+", " ", key).strip()
    return _ALIASES.get(key)


def parent_of(label: Any) -> dict[str, str] | None:
    """Return {movement, material} for a canonical subclass, else None."""
    canonical = normalize_label(label)
    if not canonical:
        return None
    return _LABEL_PARENTS.get(canonical)


def resolve_disagreement(llm_name: Any, img_name: Any) -> dict[str, Any]:
    """Resolve a subtype disagreement between the VLM opinion (``llm_name``)
    and the image-classifier opinion (``img_name``).

    Returns a dict with:

    - ``resolution``: one of ``subclass_agreement``,
      ``movement_parent_fallback``, ``material_parent_fallback``,
      ``conflict``, ``unavailable``
    - ``class_name``: the chosen subclass, or the fallback parent name when
      resolved at parent level (empty when unavailable)
    - ``parent_class``: the fallback parent name, or ""
    - ``conflict``: bool, True only when the two opinions disagree even on
      both Cruden & Varnes parent axes (movement and material)
    - ``note``: human-readable explanation for the report
    """
    llm = normalize_label(llm_name)
    img = normalize_label(img_name)

    if not llm and not img:
        return {
            "resolution": "unavailable",
            "class_name": "",
            "parent_class": "",
            "conflict": False,
            "note": "No classifier opinion was available.",
        }
    if not llm:
        return {
            "resolution": "unavailable",
            "class_name": img,
            "parent_class": "",
            "conflict": False,
            "note": f"Only the image classifier returned a label ({img}); no VLM opinion to compare.",
        }
    if not img:
        return {
            "resolution": "unavailable",
            "class_name": llm,
            "parent_class": "",
            "conflict": False,
            "note": f"Only the VLM returned a label ({llm}); no image-classifier opinion to compare.",
        }

    if llm == img:
        return {
            "resolution": "subclass_agreement",
            "class_name": llm,
            "parent_class": "",
            "conflict": False,
            "note": f"Both the VLM and the image classifier agree on {llm}.",
        }

    llm_parent = _LABEL_PARENTS[llm]
    img_parent = _LABEL_PARENTS[img]

    if llm_parent["movement"] == img_parent["movement"] and llm_parent["material"] == img_parent["material"]:
        # Same Cruden & Varnes combo cannot map to two distinct subclasses,
        # so this branch is defensive only.
        return {
            "resolution": "subclass_agreement",
            "class_name": llm,
            "parent_class": "",
            "conflict": False,
            "note": f"VLM and image classifier opinions ({llm}, {img}) share both parents.",
        }

    if llm_parent["movement"] == img_parent["movement"]:
        parent = llm_parent["movement"]
        return {
            "resolution": "movement_parent_fallback",
            "class_name": parent,
            "parent_class": parent,
            "conflict": False,
            "note": (
                f"The VLM says {llm} while the image classifier says {img}; their subclasses differ, "
                f"but both share the Cruden & Varnes movement type {parent}, so the subtype is "
                f"reported at the parent level {parent}."
            ),
        }

    if llm_parent["material"] == img_parent["material"]:
        parent = llm_parent["material"]
        return {
            "resolution": "material_parent_fallback",
            "class_name": parent,
            "parent_class": parent,
            "conflict": False,
            "note": (
                f"The VLM says {llm} while the image classifier says {img}; their subclasses differ, "
                f"but both share the Cruden & Varnes material composition {parent}, so the subtype "
                f"is reported at the parent level {parent}."
            ),
        }

    return {
        "resolution": "conflict",
        "class_name": llm,
        "parent_class": "",
        "conflict": True,
        "note": (
            f"The VLM says {llm} while the image classifier says {img}; they disagree even at the "
            f"Cruden & Varnes parent level (movement {llm_parent['movement']} vs {img_parent['movement']}, "
            f"material {llm_parent['material']} vs {img_parent['material']}), so the subtype is conflicting."
        ),
    }
