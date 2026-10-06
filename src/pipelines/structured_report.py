"""Integrated, provenance-tracked landslide report.

The report follows the 10 fields the description model was fine-tuned on
(presence, type, position, morphology, material, movement, environment,
impact, reason, causation). Each field merges what the vision-language model
saw in the image (``vlm.describe``) with what the tools measured (fusion,
classifier, region/segmentation, geo tools), and every part names the tool call
it came from. The merge is a fixed template - nothing is generated or
paraphrased - so the report is a deterministic view of the tool-call trace:
two runs that made the same calls produce the same report, and a run that
skipped, failed or fabricated a step shows it.

Part status:
  verified    - taken from a successful tool call (and, where the fusion step
                consumed it, fusion received exactly that output)
  unavailable - never obtained (not called / failed / refused / recorded as
                unavailable / not applicable to the request)
  unverified  - fusion consumed a value that matches no recorded tool output
Field status: verified (all parts verified), partial (some parts unavailable),
unavailable (no part available), unverified (any part unverified).
"""
from __future__ import annotations

import os
import re
from typing import Any

from src.orchestration.evidence import record_output

VERIFIED, PARTIAL, UNAVAILABLE, UNVERIFIED = "verified", "partial", "unavailable", "unverified"
_OK_STATES = {"completed", "reused"}

# Same labels and order as the fine-tuning answers / ground-truth descriptions.
FIELD_TITLES = {
    "presence": "Landslide presence",
    "type": "Landslide type",
    "position": "Image relative position within the image frame",
    "morphology": "Morphological characteristics",
    "material": "Material composition and surface cover",
    "movement": "Movement and deformation features",
    "environment": "Surrounding environmental context",
    "impact": "Impact on human infrastructure",
    "reason": "Reason for landslide classification",
    "causation": "Landslide causation inference",
}
STATUS_TEXT = {VERIFIED: "已核实", PARTIAL: "部分可用", UNAVAILABLE: "不可用", UNVERIFIED: "未核实"}
_MOVE = {"debrisflow": "flow", "mudflow": "flow", "earthflow": "flow", "flow": "flow", "mudslide": "slide",
         "earthslide": "slide", "rockslide": "slide", "slide": "slide", "rockfall": "fall", "fall": "fall"}


# --------------------------------------------------------------------------- #
# trace helpers
# --------------------------------------------------------------------------- #
def _calls(trace: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    calls = [{**item, "_n": i + 1} for i, item in enumerate(trace or [])
             if isinstance(item, dict) and item.get("tool")]
    ledger: dict[str, Any] = {}
    current: dict[str, dict[str, Any]] = {}
    for call in calls:
        name, output = call["tool"], call.get("output")
        if call.get("execution_state") not in {"completed", "reused", "failed", "degraded", "declared_unavailable"}:
            continue  # refused/deferred calls never replace admitted evidence
        if not isinstance(output, dict):
            output = {"error": "invalid backend result"}
        for invalidated in record_output(ledger, name, output):
            prior = current.pop(invalidated, None)
            if prior is not None:
                prior["_current"] = False
        prior = current.get(name)
        if prior is not None:
            prior["_current"] = False
        call["_current"] = True
        current[name] = call
    return calls


def _is_placeholder(output: Any) -> bool:
    return isinstance(output, dict) and bool(output.get("evidence_unavailable"))


def _ok(call: dict[str, Any]) -> bool:
    out = call.get("output")
    return (call.get("_current", True) and call.get("execution_state") in _OK_STATES and isinstance(out, dict)
            and not out.get("error") and not _is_placeholder(out))


def _last_ok(calls, tool):
    for c in reversed(calls):
        if c["tool"] == tool and _ok(c):
            return c
    return None


def _ref(call) -> str:
    return f"{call['tool']}#{call['_n']}"


def _short(text: Any, n: int = 160) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _norm(s: Any) -> str:
    return re.sub(r"[^a-z]", "", str(s or "").lower())


def _missing_reason(calls, tool: str) -> str:
    mine = [c for c in calls if c["tool"] == tool]
    if not mine:
        return f"未调用 {tool}"
    last = mine[-1]
    out = last.get("output") if isinstance(last.get("output"), dict) else {}
    state = last.get("execution_state")
    if last.get("_current") is False:
        return f"{tool} 的旧证据已失效，需重新获取"
    if _is_placeholder(out) or state == "declared_unavailable":
        return f"{tool} 不可用：{_short(out.get('reason'), 120)}"
    if state == "refused":
        return f"{tool} 被规则拦截 {len(mine)} 次：{_short(out.get('error'), 120)}"
    if state == "failed" or out.get("error"):
        return f"{tool} 调用失败 {len(mine)} 次：{_short(out.get('error'), 120)}"
    return f"{tool} 未返回可用结果"


def _same(a, b, keys) -> bool:
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    for k in keys:
        va, vb = a.get(k), b.get(k)
        if isinstance(va, float) or isinstance(vb, float):
            try:
                if abs(float(va) - float(vb)) > 1e-6:
                    return False
            except (TypeError, ValueError):
                return False
        elif va != vb:
            return False
    return True


def _fused_input_source(calls, passed, tool, keys):
    for c in reversed(calls):
        if c["tool"] == tool and _ok(c) and _same(passed, c["output"], keys):
            return c
    return None


def _misplaced_source(calls, passed, expected_tool) -> str:
    if not isinstance(passed, dict):
        return "传入的不是工具输出对象"
    for c in reversed(calls):
        if c["tool"] == expected_tool or not _ok(c):
            continue
        shared = [k for k in passed if k in c["output"] and not isinstance(passed[k], (dict, list))]
        if len(shared) >= 2 and _same(passed, c["output"], tuple(shared)):
            return f"传入的实际是 {_ref(c)} 的输出（参数放错位置）"
    return f"与任何 {expected_tool} 输出都不一致"


def _part(kind: str, text: Any, status: str, source: str = "", note: str = "") -> dict[str, Any]:
    """One piece of evidence inside a field. kind: 'image' (model saw) or 'tool' (measured/decided)."""
    return {"kind": kind, "text": text, "status": status, "source": source, "note": note}


def _field(parts: list[dict[str, Any]], flags: list[str] | None = None) -> dict[str, Any]:
    st = [p["status"] for p in parts]
    if not parts or all(s == UNAVAILABLE for s in st):
        status = UNAVAILABLE
    elif UNVERIFIED in st:
        status = UNVERIFIED
    elif UNAVAILABLE in st or PARTIAL in st:
        status = PARTIAL
    else:
        status = VERIFIED
    return {"status": status, "parts": parts, "flags": flags or []}


def _review_area_threshold() -> float:
    try:
        return float(os.getenv("SEG_LLM_SECOND_PASS_MAX_AREA_RATIO", "0.20") or 0.20)
    except ValueError:
        return 0.20


_CELL_WORDS = {"upper": "top", "lower": "bottom", "middle": "center", "centre": "center"}


def _cells(text: str) -> set[str]:
    out = set()
    s = str(text or "").lower().replace("centre", "center")
    for m in re.finditer(r"\b(top|bottom|upper|lower|middle|center)[\s-]*(left|right|center)?\b|\b(left|right)\b", s):
        if m.group(3):
            out.add("center-" + m.group(3))
            continue
        out.add(_CELL_WORDS.get(m.group(1), m.group(1)) + "-" + (m.group(2) or "center"))
    return out


def _grid_words(position: str) -> str:
    v, _, h = str(position or "").partition("-")
    v = {"upper": "top", "lower": "bottom", "middle": "center"}.get(v, v)
    if v == "center" and h == "center":
        return "center"
    return f"{v} {h}".strip()


# --------------------------------------------------------------------------- #
# context: the pieces every field draws on
# --------------------------------------------------------------------------- #
class _Ctx:
    def __init__(self, calls, has_coords):
        self.calls = calls
        # Free arm: fuse.decision is only a report signal, so presence/type are
        # read from the tools that actually produced them.
        self.tools_only = os.getenv("FUSE_LENIENT", "0") in ("1", "true", "True")
        self.has_coords = has_coords
        self.fuse = _last_ok(calls, "fuse.decision")
        self.fuse_in = (self.fuse or {}).get("input") or {}
        self.fuse_out = (self.fuse or {}).get("output") or {}
        self.vlm = _last_ok(calls, "vlm.describe")
        self.vlm_fields = ((self.vlm or {}).get("output") or {}).get("fields") or {}
        self.negative = (not self.tools_only and self.fuse is not None
                         and self.fuse_out.get("has_landslide") is False)

    def image_part(self, key: str) -> dict[str, Any]:
        """What the fine-tuned model wrote for this field (verbatim)."""
        if self.vlm is None:
            return _part("image", None, UNAVAILABLE, note=_missing_reason(self.calls, "vlm.describe"))
        text = str(self.vlm_fields.get(key, "") or "").strip()
        if not text:
            return _part("image", None, UNAVAILABLE, _ref(self.vlm), "描述中没有这一项")
        return _part("image", text, VERIFIED, _ref(self.vlm))


# --------------------------------------------------------------------------- #
# fields
# --------------------------------------------------------------------------- #
def _presence_from_tools(c: _Ctx):
    parts, flags = [], []
    fp, seg = _last_ok(c.calls, "llm.first_pass"), _last_ok(c.calls, "seg.run")
    if fp is None:
        parts.append(_part("tool", None, UNAVAILABLE, note=_missing_reason(c.calls, "llm.first_pass")))
    else:
        o = fp["output"]
        note = f"first-pass screening: {o.get('assessment_label')}, score {_num(o.get('score'), 2)}"
        if seg is not None:
            note += f"; segmentation: {100 * float(seg['output'].get('area_ratio') or 0):.1f}% of the frame"
        parts.append(_part("tool", "Yes" if o.get("has_landslide") else "No", VERIFIED, _ref(fp), note))
    img = c.image_part("presence")
    parts.append(img)
    if parts[0]["text"] and img["text"] and _norm(img["text"])[:2] != _norm(parts[0]["text"])[:2]:
        flags.append(f"看图结论（{img['text']}）与初判（{parts[0]['text']}）不一致")
    return _field(parts, flags)


def _type_from_tools(c: _Ctx):
    parts, flags = [], []
    cls = _last_ok(c.calls, "cls.run")
    if cls is None:
        parts.append(_part("tool", None, UNAVAILABLE, note=_missing_reason(c.calls, "cls.run")))
    else:
        o = cls["output"]
        sources = o.get("sources") if isinstance(o.get("sources"), dict) else {}
        qwen = (sources.get("vlm") or {}).get("class_name")
        cnx = (sources.get("image_classifier") or {}).get("class_name")
        qc = (sources.get("vlm") or {}).get("confidence")
        cc = (sources.get("image_classifier") or {}).get("confidence")
        bits = []
        if qwen or cnx:
            bits.append(f"Qwen 分类头={qwen or '无'}（{_num(qc, 2)}），ConvNeXt={cnx or '无'}（{_num(cc, 2)}）")
        if o.get("conflict"):
            bits.append("一致性=conflict")
        parts.append(_part("tool", "classification conflict" if o.get("conflict") else o.get("class_name"),
                           PARTIAL if o.get("conflict") else VERIFIED, _ref(cls), "；".join(bits)))
    img = c.image_part("type")
    parts.append(img)
    tool_type = parts[0]["text"]
    if img["text"] and tool_type and _norm(img["text"]) != _norm(tool_type):
        flags.append(f"看图类型（{img['text']}）与分类工具（{tool_type}）不同")
    return _field(parts, flags)


def _presence(c: _Ctx):
    if c.tools_only:
        return _presence_from_tools(c)
    parts, flags = [], []
    if c.fuse is None:
        parts.append(_part("tool", None, UNAVAILABLE, note="未得出结论：" + _missing_reason(c.calls, "fuse.decision")))
    else:
        basis, bad = [], []
        for arg, tool, keys in (("stage1", "llm.first_pass", ("has_landslide", "score")),
                                ("segmentation", "seg.run", ("area_ratio", "landslide_pixels"))):
            if arg not in c.fuse_in:
                continue
            src = _fused_input_source(c.calls, c.fuse_in[arg], tool, keys)
            if src:
                basis.append(_ref(src))
            else:
                bad.append(f"{arg}：{_misplaced_source(c.calls, c.fuse_in[arg], tool)}")
        decided = "Yes" if c.fuse_out.get("has_landslide") else "No"
        note = "依据 " + ("、".join(basis) or "无") + ("；未核实：" + "；".join(bad) if bad else "")
        parts.append(_part("tool", decided, UNVERIFIED if bad else VERIFIED, _ref(c.fuse), note))
    img = c.image_part("presence")
    parts.append(img)
    if parts[0]["text"] and img["text"] and _norm(img["text"])[:2] != _norm(parts[0]["text"])[:2]:
        flags.append(f"看图结论（{img['text']}）与最终判定（{parts[0]['text']}）不一致")
    return _field(parts, flags)


def _type(c: _Ctx):
    if c.tools_only:
        return _type_from_tools(c)
    parts, flags = [], []
    if c.negative:
        parts.append(_part("tool", "Not applicable (no landslide)", VERIFIED, _ref(c.fuse)))
    elif c.fuse is None:
        parts.append(_part("tool", None, UNAVAILABLE, note="未得出结论：" + _missing_reason(c.calls, "fuse.decision")))
    else:
        passed = c.fuse_in.get("classification")
        if _is_placeholder(passed) or (passed is None and _last_ok(c.calls, "cls.run") is None):
            parts.append(_part("tool", None, UNAVAILABLE, note=_missing_reason(c.calls, "cls.run")))
        else:
            src = _fused_input_source(c.calls, passed, "cls.run", ("class_name", "confidence"))
            bits = []
            if isinstance(passed, dict):
                sources = passed.get("sources") if isinstance(passed.get("sources"), dict) else {}
                qwen = (sources.get("vlm") or {}).get("class_name")
                cnx = (sources.get("image_classifier") or {}).get("class_name")
                if qwen or cnx:
                    bits.append(f"Qwen 分类头={qwen or '无'}，ConvNeXt={cnx or '无'}")
                if passed.get("confidence") is not None:
                    bits.append(f"置信度 {float(passed['confidence']):.3f}")
            rec = c.fuse_out.get("classification_reconciliation") or {}
            if rec.get("status"):
                bits.append(f"一致性={rec['status']}")
            if src is None:
                bits.insert(0, "融合使用的分类结果与任何 cls.run 输出都不一致")
            parts.append(_part("tool", c.fuse_out.get("landslide_type"), VERIFIED if src else UNVERIFIED,
                               _ref(src) if src else _ref(c.fuse), "；".join(bits)))
            if rec.get("status") == "conflict" and src:
                parts[-1]["status"] = PARTIAL  # the tools returned no agreed subtype
    img = c.image_part("type")
    parts.append(img)
    tool_type = parts[0]["text"]
    if img["text"] and tool_type and not c.negative:
        a, b = _MOVE.get(_norm(img["text"])), _MOVE.get(_norm(tool_type))
        if _norm(img["text"]) != _norm(tool_type):
            flags.append(f"看图类型（{img['text']}）与分类工具（{tool_type}）不同"
                         + ("，且运动方式大类也不同" if a and b and a != b else ""))
    return _field(parts, flags)


def _position(c: _Ctx):
    parts, flags = [], []
    loc = _last_ok(c.calls, "region.locate")
    seg = _last_ok(c.calls, "seg.run")
    tool_pos = None
    if c.negative:
        parts.append(_part("tool", "Not applicable (no landslide)", VERIFIED, _ref(c.fuse)))
    elif loc and loc["output"].get("available"):
        o = loc["output"]
        tool_pos = o.get("position")
        how = "掩膜质心" if o.get("position_source") == "mask_centroid" else "候选框中心"
        parts.append(_part("tool", f"The affected area is located in the {_grid_words(tool_pos)}.", VERIFIED,
                           _ref(loc), f"{how}，bbox={o.get('bbox')}"))
    else:
        refine = _last_ok(c.calls, "seg.refine")
        o = {"available": False}
        if refine:
            try:
                from src.utils.geometry import locate_primary_candidate
                o = locate_primary_candidate(refine["output"])
            except Exception:  # pragma: no cover
                pass
        if o.get("available"):
            tool_pos = o.get("position")
            parts.append(_part("tool", f"The affected area is located in the {_grid_words(tool_pos)}.", VERIFIED,
                               _ref(refine), "未调用 region.locate，由 seg.refine 结果推算"))
        else:
            reason = _missing_reason(c.calls, "region.locate") + ("" if refine else "；" + _missing_reason(c.calls, "seg.refine"))
            parts.append(_part("tool", None, UNAVAILABLE, note=reason))
    if seg and not c.negative:
        so = seg["output"]
        parts.append(_part("tool", f"Segmented extent: {float(so.get('area_ratio') or 0) * 100:.1f}% of the frame "
                                   f"({so.get('polygon_count')} polygon(s)).", VERIFIED, _ref(seg)))
    img = c.image_part("position")
    parts.append(img)
    if tool_pos and img["text"]:
        v, _, h = tool_pos.partition("-")
        cell = ({"upper": "top", "lower": "bottom", "middle": "center"}.get(v, v)) + "-" + h
        if _cells(img["text"]) and cell not in _cells(img["text"]):
            flags.append(f"看图描述的位置与工具定位（{_grid_words(tool_pos)}）不一致")
    return _field(parts, flags)


def _image_only(key):
    def build(c: _Ctx):
        return _field([c.image_part(key)])
    return build


def _num(v, nd):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return v


def _environment(c: _Ctx):
    parts = [c.image_part("environment")]
    bg = _last_ok(c.calls, "geo.background")
    if bg is None:
        reason = _missing_reason(c.calls, "geo.background")
        parts.append(_part("tool", None, UNAVAILABLE, note=("请求未提供经纬度；" if not c.has_coords else "") + reason))
    else:
        o = bg["output"]
        t, g = o.get("terrain") or {}, o.get("geology") or {}
        text = (f"Measured terrain: elevation {_num(t.get('elevation_m'), 0)} m, slope {t.get('slope_deg')}°, aspect {t.get('aspect_deg')}°; "
                f"geology: {g.get('lithology') or g.get('unit_name') or 'unknown'}."
                + (f" (DEM: {t.get('dem_source')}, slope/aspect over {_num(t.get('sample_distance_m'), 0)} m spacing)"
                   if t.get('dem_source') else ""))
        passed = (c.fuse_in.get("geo_context") or {}).get("background")
        bad = isinstance(passed, dict) and not _is_placeholder(passed) and not _same(passed.get("terrain"), t, ("slope_deg", "aspect_deg"))
        parts.append(_part("tool", text, UNVERIFIED if bad else VERIFIED, _ref(bg),
                           "融合使用的地形数值与 geo.background 输出不一致" if bad else ""))
    return _field(parts)


def _impact(c: _Ctx):
    parts, flags = [c.image_part("impact")], []
    nb = _last_ok(c.calls, "geo.nearby")
    if nb is None:
        reason = _missing_reason(c.calls, "geo.nearby")
        parts.append(_part("tool", None, UNAVAILABLE, note=("请求未提供经纬度；" if not c.has_coords else "") + reason))
    else:
        o = nb["output"]
        kinds: dict[str, int] = {}
        for f in o.get("features") or []:
            k = f"{f.get('type')}{':' + f['subtype'] if f.get('subtype') else ''}"
            try:
                feature_count = max(1, int(f.get("_count") or 1))
            except (TypeError, ValueError):
                feature_count = 1
            kinds[k] = kinds.get(k, 0) + feature_count
        listing = ", ".join(f"{k}×{v}" for k, v in kinds.items())
        text = f"Mapped facilities within {o.get('radius_m')} m: {o.get('count')}" + (f" ({listing})." if listing else ".")
        passed = (c.fuse_in.get("geo_context") or {}).get("nearby")
        bad = isinstance(passed, dict) and not _is_placeholder(passed) and not _same(passed, o, ("count",))
        parts.append(_part("tool", text, UNVERIFIED if bad else VERIFIED, _ref(nb),
                           "融合使用的设施数量与 geo.nearby 输出不一致" if bad else ""))
        said_none = bool(re.search(r"\bno (impact|visible|damage|affected)|not (affect|impact)", str(parts[0]["text"] or ""), re.I))
        if said_none and (o.get("count") or 0) > 0:
            flags.append(f"看图认为无影响，但 {o.get('radius_m')} m 内有 {o.get('count')} 处已登记设施，需人工确认")
    return _field(parts, flags)


def _reason(c: _Ctx):
    parts = [c.image_part("reason")]
    rec = c.fuse_out.get("classification_reconciliation") or {}
    cls = _last_ok(c.calls, "cls.run")
    if c.negative:
        pass
    elif cls is None:
        parts.append(_part("tool", None, UNAVAILABLE, note=_missing_reason(c.calls, "cls.run")))
    else:
        note = str(rec.get("report_note") or cls["output"].get("classification_note") or "").replace("Inference: ", "")
        parts.append(_part("tool", note or f"Classifier: {cls['output'].get('class_name')}.", VERIFIED, _ref(cls)))
    return _field(parts)


def _causation(c: _Ctx):
    parts = [c.image_part("causation")]
    bg = _last_ok(c.calls, "geo.background")
    if bg is None:
        parts.append(_part("tool", None, UNAVAILABLE, note=_missing_reason(c.calls, "geo.background")))
    else:
        t, g = bg["output"].get("terrain") or {}, bg["output"].get("geology") or {}
        parts.append(_part("tool", f"Measured predisposing factors: slope {t.get('slope_deg')}°, "
                                   f"lithology {g.get('lithology') or g.get('unit_name') or 'unknown'}.", VERIFIED, _ref(bg),
                           "成因为推断，工具只提供易发条件，不能证明触发因素"))
    return _field(parts)


_BUILDERS = {
    "presence": _presence, "type": _type, "position": _position,
    "morphology": _image_only("morphology"), "material": _image_only("material"),
    "movement": _image_only("movement"), "environment": _environment, "impact": _impact,
    "reason": _reason, "causation": _causation,
}


# --------------------------------------------------------------------------- #
# evidence chain + public API
# --------------------------------------------------------------------------- #
def _call_summary(c) -> str:
    out = c.get("output") if isinstance(c.get("output"), dict) else {}
    if out.get("error"):
        return "错误：" + _short(out["error"], 140)
    if _is_placeholder(out):
        return "记为不可用：" + _short(out.get("reason"), 140)
    t = c["tool"]
    if t == "llm.first_pass":
        return f"has_landslide={out.get('has_landslide')}，label={out.get('assessment_label')}，score={out.get('score')}"
    if t == "vlm.describe":
        f = out.get("fields") or {}
        return f"presence={f.get('presence')}，type={f.get('type')}，{len(f)} 个字段"
    if t == "seg.run":
        return f"面积 {float(out.get('area_ratio') or 0) * 100:.1f}%，{out.get('polygon_count')} 个多边形"
    if t == "seg.refine":
        return f"{len(out.get('regions') or [])} 个候选区域"
    if t == "region.locate":
        return f"位置={out.get('position')}"
    if t == "cls.run":
        return f"{out.get('class_name')}（{out.get('confidence')}），一致性={out.get('resolution', '—')}"
    if t == "geo.background":
        tr = out.get("terrain") or {}
        return f"坡度={tr.get('slope_deg')}，坡向={tr.get('aspect_deg')}"
    if t == "geo.nearby":
        return f"{out.get('count')} 处要素"
    if t == "seg.llm_review":
        sp = out.get("llm_second_pass")
        if isinstance(sp, dict):
            return f"复核={sp.get('decision')}"
        return "按设计跳过（面积较大）" if out.get("llm_second_pass_skipped_for_large_area") else "无复核结果"
    if t == "fuse.decision":
        return f"has_landslide={out.get('has_landslide')}，type={out.get('landslide_type')}"
    if t == "tiff.info":
        return f"{out.get('width')}×{out.get('height')}"
    return _short(", ".join(f"{k}={v}" for k, v in list(out.items())[:3]), 140)


def _boundary_review(calls) -> dict[str, Any]:
    """Whether the rule-defined boundary re-check was done (process fact, not a report field)."""
    for c in reversed(calls):
        if c["tool"] == "seg.llm_review" and _ok(c) and isinstance(c["output"].get("llm_second_pass"), dict):
            return {"status": VERIFIED, "text": "done", "source": _ref(c)}
    for c in reversed(calls):
        if c["tool"] == "seg.llm_review" and _ok(c) and c["output"].get("llm_second_pass_skipped_for_large_area"):
            return {"status": VERIFIED, "text": "skipped by design (large target)", "source": _ref(c)}
    seg, fp = _last_ok(calls, "seg.run"), _last_ok(calls, "llm.first_pass")
    if seg and fp:
        area = float(seg["output"].get("area_ratio") or 0.0)
        label = str(fp["output"].get("assessment_label") or "").lower()
        if area >= _review_area_threshold() and label not in ("uncertain", "error"):
            return {"status": VERIFIED, "text": "not required (large target, confident first pass)", "source": _ref(seg)}
    return {"status": UNAVAILABLE, "text": "required but not done", "source": "",
            "note": _missing_reason(calls, "seg.llm_review")}


def build_structured_report(trace, *, latitude=None, longitude=None) -> dict[str, Any]:
    calls = _calls(trace)
    c = _Ctx(calls, latitude is not None and longitude is not None)
    fields = {k: b(c) for k, b in _BUILDERS.items()}
    chain = [{"n": x["_n"], "tool": x["tool"], "state": x.get("execution_state") or x.get("status"),
              "summary": _call_summary(x)} for x in calls]
    counts = {s: sum(1 for f in fields.values() if f["status"] == s) for s in (VERIFIED, PARTIAL, UNAVAILABLE, UNVERIFIED)}
    parts = [p for f in fields.values() for p in f["parts"]]
    return {
        "report_version": "structured-2",
        "fields": fields,
        "boundary_review": _boundary_review(calls),
        "evidence_chain": chain,
        "counts": counts,
        "parts": {"total": len(parts), "verified": sum(p["status"] == VERIFIED for p in parts),
                  "unavailable": sum(p["status"] == UNAVAILABLE for p in parts),
                  "unverified": sum(p["status"] == UNVERIFIED for p in parts)},
        "flags": [f"{FIELD_TITLES[k]}: {x}" for k, f in fields.items() for x in f["flags"]],
        "request": {"latitude": latitude, "longitude": longitude},
    }


def _render_part(p: dict[str, Any]) -> str:
    who = "看图" if p["kind"] == "image" else "工具"
    if p["status"] == UNAVAILABLE:
        return f"（{who}不可用：{p.get('note') or '无'}）"
    tag = f"[{who} {p['source']}]" if p.get("source") else f"[{who}]"
    note = f"（{p['note']}）" if p.get("note") else ""
    warn = " ⚠未核实" if p["status"] == UNVERIFIED else ""
    return f"{p['text']} {tag}{note}{warn}"


def render_structured_report(report: dict[str, Any]) -> str:
    fields = report.get("fields") or {}
    lines = ["### Landslide Assessment Report", ""]
    for key, title in FIELD_TITLES.items():
        f = fields.get(key) or {"parts": [], "flags": [], "status": UNAVAILABLE}
        body = " ".join(_render_part(p) for p in f["parts"]) or "（无）"
        lines.append(f"**{title}:** {body}")
        for flag in f.get("flags") or []:
            lines.append(f"  - ⚠ {flag}")
        lines.append("")
    br = report.get("boundary_review") or {}
    lines += ["### 证据核对", "",
              "| 字段 | 状态 |", "|---|---|"]
    lines += [f"| {FIELD_TITLES[k]} | {STATUS_TEXT[f['status']]} |" for k, f in fields.items()]
    lines += ["", f"边界复核：{br.get('text', '—')} {('[' + br['source'] + ']') if br.get('source') else ''}"
              f"{('（' + br['note'] + '）') if br.get('note') else ''}"]
    lines += ["", "### 证据链（全部工具调用，按时间顺序）"]
    lines += [f"{c['n']}. {c['tool']} [{c['state']}] {c['summary']}" for c in report.get("evidence_chain") or []]
    return "\n".join(lines).strip()
