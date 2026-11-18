"""Canonical labels and dataset parsing for the Qwen3-VL dual-head model."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


LABELS = [
    "No landslide",
    "Debris flow",
    "Mud flow",
    "Mudslide",
    "Earth slide",
    "Earthflow",
    "Rock fall",
    "Rock slide",
]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}

ALIASES = {
    "non": "No landslide",
    "no landslide": "No landslide",
    "no": "No landslide",
    "debris flow": "Debris flow",
    "mud flow": "Mud flow",
    "mudflow": "Mud flow",
    "mudslide": "Mudslide",
    "earth slide": "Earth slide",
    "earthflow": "Earthflow",
    "earth flow": "Earthflow",
    "rock fall": "Rock fall",
    "rockfall": "Rock fall",
    "rock slide": "Rock slide",
    "rockslide": "Rock slide",
}


def normalize(value: str) -> str:
    value = value.strip().strip("`*_[](){}:;,.，。")
    value = value.lower().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", value).strip()


def canonical_label(value: str) -> str | None:
    return ALIASES.get(normalize(value))


def label_from_answer(answer: str) -> str | None:
    """Extract the canonical class name used by the assistant description."""
    presence = re.search(
        r"\*{0,2}Landslide presence:\*{0,2}\s*([^\n\r]+)",
        answer,
        flags=re.IGNORECASE,
    )
    if presence and normalize(presence.group(1)) == "no":
        return "No landslide"

    class_line = re.search(
        r"\bClassification:\s*([^\n\r]+)",
        answer,
        flags=re.IGNORECASE,
    )
    if class_line:
        label = canonical_label(class_line.group(1).replace("&#x20;", " "))
        if label:
            return label

    type_line = re.search(
        r"\*{0,2}Landslide type:\*{0,2}\s*\n?\s*([^\n\r]+)",
        answer,
        flags=re.IGNORECASE,
    )
    if type_line:
        return canonical_label(type_line.group(1).replace("&#x20;", " "))
    return None


def label_from_image(image_path: str) -> str | None:
    """Extract the ground-truth label from names such as non215.png or
    debris flow25432_Level_16.png.
    """
    name = Path(image_path).name
    match = re.match(
        r"^(?P<label>.+?)(?P<number>\d+)(?:_Level_\d+)?\.(?:png|jpe?g|tiff?)$",
        name,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return canonical_label(match.group("label"))


def as_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ).strip()
    return str(content or "")


def role_of(message: dict[str, Any]) -> str:
    role = message.get("role", message.get("from", ""))
    return {"human": "user", "gpt": "assistant"}.get(role, role)


def parse_record(record: dict[str, Any]) -> dict[str, Any]:
    """Convert the original dataset record to prompt/answer/image/label fields."""
    prompt_messages: list[dict[str, Any]] = []
    answer = ""
    image_paths: list[str] = []

    for message in record.get("messages", []):
        role = role_of(message)
        content = message.get("content", message.get("value", ""))
        if role == "assistant":
            answer = as_text(content)
            break

        prompt_messages.append({"role": role, "content": content})
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    image = item.get("image")
                    if image:
                        image_paths.append(str(image))

    label = label_from_image(image_paths[0]) if image_paths else None
    answer_label = label_from_answer(answer) if answer else None
    return {
        "prompt_messages": prompt_messages,
        "answer": answer,
        "image_paths": image_paths,
        "label": label,
        "label_id": LABEL_TO_ID.get(label) if label else None,
        "answer_label": answer_label,
    }
