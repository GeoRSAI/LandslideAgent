"""Final report written by the agent's language model from the evidence record.

The structured report (``structured_report.py``) collects, per fine-tuning field,
what the image reading (vlm.describe) and each tool returned, with provenance and
disagreement flags. That record is evidence, not a report: listing the pieces side
by side leaves the reader to reconcile them. Here the model composes the final
report from that record - one coherent statement per field, with any
disagreement between image reading and tools discussed and weighed, not stacked.

The model only writes; it adds no evidence. Missing evidence stays missing.
The structured record remains attached to the response for auditing.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Callable

from src.pipelines.structured_report import FIELD_TITLES

_KIND = {"image": "image reading (vlm.describe)", "tool": "tool"}
_CJK = re.compile(r"[㐀-鿿＀-￯]+")


def _round_numbers(text: str) -> str:
    # Long float artefacts (e.g. 447.58123779296875) -> 1 decimal.
    return re.sub(r"(\d+\.\d)\d{3,}", r"\1", text)


def evidence_for_prompt(report: dict[str, Any], trace: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Compact, model-readable view of the structured record."""
    fields = report.get("fields") or {}
    out: dict[str, Any] = {}
    for key, title in FIELD_TITLES.items():
        f = fields.get(key) or {}
        items = []
        for p in f.get("parts") or []:
            src = str(p.get("source") or "").split("#")[0]
            item = {"from": _KIND.get(p.get("kind"), p.get("kind")) + (f" {src}" if src and p.get("kind") == "tool" else "")}
            if p.get("status") == "unavailable" or not p.get("text"):
                item["unavailable"] = _round_numbers(str(p.get("note") or "not obtained"))
            else:
                item["says"] = _round_numbers(str(p.get("text")))
                if p.get("note"):
                    item["detail"] = _round_numbers(str(p.get("note")))
                if p.get("status") == "unverified":
                    item["caution"] = "this value could not be traced to a recorded tool output"
            items.append(item)
        entry: dict[str, Any] = {"evidence": items}
        if f.get("flags"):
            entry["disagreements"] = list(f["flags"])
        out[title] = entry
    review = report.get("boundary_review") or {}
    if review:
        out["_boundary_review"] = {k: review[k] for k in ("status", "note") if k in review}
    return out


_SYSTEM = (
    "You are the landslide analysis agent writing its final assessment report. Tool results and "
    "your own image reading are both your evidence; the reader should get one coherent assessment, "
    "not a list of sources.\n"
    "Write exactly these ten fields, in this order, each as '**<Field>:** <prose>':\n"
    + "\n".join(f"- {t}" for t in FIELD_TITLES.values())
    + "\n\nRules:\n"
    "1. Integrate. For each field, merge the image reading and the tool measurements into one or a "
    "few natural sentences. Do not attach source tags or repeat the same fact twice.\n"
    "2. Reconcile disagreements explicitly. When sources disagree (see 'disagreements', or a "
    "classifier conflict in the details), state briefly what each says, weigh them - confidence "
    "scores, whether the visible morphology/material/movement fits each candidate under the "
    "Cruden-Varnes scheme, and what each method can actually observe (a mask centroid versus a "
    "whole-scene reading) - and give your reasoned conclusion. If the evidence does not favour "
    "either side, say the question remains open and why.\n"
    "3. Landslide type: when the classifiers agree, or share a Cruden-Varnes parent, report that. "
    "When they conflict, name the type best supported by the weighed evidence as 'most consistent "
    "with ...' and state that the classifiers disagreed; never hide the conflict.\n"
    "4. Never invent. Use only facts present in the evidence. If a piece of evidence is marked "
    "unavailable, say it was not available (and why, if given) and do not describe or estimate it. "
    "Items marked 'caution' must not be presented as established.\n"
    "4b. Explain a disagreement only with reasons present in the evidence (confidence values, "
    "agreement with the observed morphology). Do not speculate about how a tool works internally, "
    "and do not invent labels or categories that are not in the evidence.\n"
    "5. Numbers: keep only the ones that matter, rounded sensibly (elevation to the metre, angles to "
    "0.1 degree, confidences to two decimals).\n"
    "6. Causation is inference: present terrain and lithology as predisposing conditions, triggers as "
    "likely, not proven.\n"
    "Write in English only. Return only the report text starting with the first field; no "
    "preamble, no closing remarks."
)


def free_report_request() -> str:
    """Closing request for the free arm: the same writing requirements as the rule
    arm's composer, phrased for an agent writing from its own conversation (it sees
    the image and its tool results directly rather than an evidence record)."""
    text = _SYSTEM
    for old, new in (
        ("You are the landslide analysis agent writing its final assessment report. ",
         "Write your final landslide assessment report now. "),
        ("(see 'disagreements', or a classifier conflict in the details)",
         "(for example your image reading versus a tool result, or classifiers that disagree)"),
        ("If a piece of evidence is marked unavailable, say it was not available (and why, if given) "
         "and do not describe or estimate it. Items marked 'caution' must not be presented as established.",
         "If a tool was not run or failed, say that evidence was not available and do not describe or "
         "estimate it."),
    ):
        assert old in text, old[:40]
        text = text.replace(old, new)
    return text + (
        "\nBegin the Landslide presence field with Yes or No, and the Landslide type field with the "
        "type name (or Undetermined / Not applicable)."
    )


def parse_answer(text: str) -> dict[str, Any]:
    """Headline verdict stated in a report ('**Landslide presence:** Yes ...')."""
    def field(title: str) -> str:
        m = re.search(r"\*{0,2}" + re.escape(title) + r"\s*:\s*\*{0,2}\s*(.+)", text or "", re.I)
        return m.group(1).strip() if m else ""

    presence = field("Landslide presence")
    has = None
    if re.match(r"(yes|present|a landslide is (present|confirmed|visible))\b", presence, re.I):
        has = True
    elif re.match(r"(no|none|absent|not present)\b", presence, re.I):
        has = False
    kind = field("Landslide type")
    kind = re.split(r"[.;,(—–]| - ", kind, maxsplit=1)[0].strip().strip("*").strip()
    kind = re.sub(r"^(most consistent with|consistent with|likely|probably)\s+(an?\s+)?", "", kind, flags=re.I).strip()
    if has is False:
        kind = "no landslide"
    return {"has_landslide": has, "landslide_type": kind or None, "source": "report text"}


def _space_report_fields(text: str) -> str:
    """Give each of the ten fine-tuning fields its own Markdown paragraph."""
    field_starts = tuple(title.lower() + ":" for title in FIELD_TITLES.values())
    lines: list[str] = []
    for line in text.splitlines():
        label = line.strip().lstrip("*").lower()
        if label.startswith(field_starts) and lines and lines[-1].strip():
            lines.append("")
        lines.append(line)
    return "\n".join(lines).strip()


def compose_report(
    report: dict[str, Any],
    chat: Callable[..., dict[str, Any]],
    trace: list[dict[str, Any]] | None = None,
) -> str | None:
    """Ask the model to write the report; None when the result is unusable."""
    evidence = evidence_for_prompt(report, trace)
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": "Evidence record for this image:\n" + json.dumps(evidence, ensure_ascii=False, indent=1)},
    ]
    text = ""
    for attempt in range(2):
        try:
            resp = chat(
                messages,
                temperature=0.1,
                trace_label="report.compose",
                max_tokens=int(os.getenv("LLM_REPORT_MAX_TOKENS", "3500") or "3500"),
                timeout=240.0,
            )
        except Exception:  # never lose the run because of the report view
            logging.exception("report composition failed")
            return None
        text = str((resp or {}).get("content") or "").strip()
        text = re.sub(r"^```\w*\s*|\s*```$", "", text).strip()
        if not _CJK.search(text):
            break
        if attempt == 0:  # the report is English; one rewrite, then drop stray CJK
            messages = messages + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": "Rewrite the same report entirely in English, with no Chinese characters."},
            ]
    text = _CJK.sub("", text)
    missing = [t for t in FIELD_TITLES.values() if t.lower() not in text.lower()]
    if missing:
        logging.warning("composed report misses fields %s; falling back", missing)
        return None
    return "### Landslide Assessment Report\n\n" + _space_report_fields(text)
