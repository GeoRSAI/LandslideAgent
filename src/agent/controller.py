"""Central policy controller for the landslide analysis agent.

This module is framework-neutral: the legacy JSON-RPC agent, FastAPI routes,
and LangGraph workflow use the same task rules here.  It contains analysis
policy and validation, not model inference implementation.
"""
from __future__ import annotations

import contextvars
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4


class ToolPreconditionError(ValueError):
    """Raised by :meth:`LandslidePolicy.prepare_tool_call` when a task rule
    forbids a tool call (missing upstream evidence, bad arguments, etc.).

    It subclasses ``ValueError`` so existing ``except (ValueError, ...)`` call
    sites keep working unchanged.
    """


def _is_existing_file(path: str) -> bool:
    try:
        return bool(path) and Path(path).is_file()
    except Exception:
        return False


def _looks_like_region_item(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    bbox = item.get("bbox")
    return isinstance(bbox, list) and len(bbox) == 4


def _looks_like_refinement_result(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    regions = value.get("regions")
    if not isinstance(regions, list):
        return False
    return all(_looks_like_region_item(d) for d in regions)


_ABSENT = object()

# Verification record of the call currently being prepared (None = not recorded).
_VERIFICATION: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "landslide_verification_record", default=None
)


# Evidence ledger of the call currently being prepared, used for diagnostics.
_LEDGER: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "landslide_evidence_ledger", default=None
)


def _note_checked(key: str) -> None:
    """Record that a supplied argument was verified against its reference."""
    record = _VERIFICATION.get()
    if record is not None:
        checked = record.setdefault("checked", [])
        if key not in checked:
            checked.append(key)


def _note_resolved(key: str, source: str) -> None:
    """Record that an omitted argument was resolved from a reference source."""
    record = _VERIFICATION.get()
    if record is not None:
        record.setdefault("resolved", {})[key] = source


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_number(value: Any) -> float | None:
    if _is_number(value):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _stated_half_unit(value: Any) -> float:
    """Half a unit in the last decimal place the agent actually wrote."""
    if isinstance(value, int) and not isinstance(value, bool):
        return 0.5
    text = value.strip() if isinstance(value, str) else repr(float(value))
    lowered = text.lower()
    if "e" in lowered or "inf" in lowered or "nan" in lowered:
        return 0.0
    decimals = len(text.split(".", 1)[1]) if "." in text else 0
    return 0.5 * 10 ** (-decimals)


def _evidence_conflicts(supplied: Any, recorded: Any, path: str) -> list[tuple[str, Any, Any]]:
    """Places where an agent-supplied value contradicts a recorded one.

    Agreement is judged only on what the agent wrote. Every key it supplies must
    exist in the record, every list element it supplies must match the element
    at the same position, strings must match exactly, and a number must be a
    valid rounding of the recorded number at the precision the agent stated.
    Leaving fields out is not a conflict: an abbreviated reference is still a
    faithful one.
    """
    if isinstance(supplied, dict):
        if not isinstance(recorded, dict):
            return [(path, supplied, recorded)]
        found: list[tuple[str, Any, Any]] = []
        for key, value in supplied.items():
            child = f"{path}.{key}"
            if key not in recorded:
                found.append((child, value, _ABSENT))
            else:
                found.extend(_evidence_conflicts(value, recorded[key], child))
        return found
    if isinstance(supplied, list):
        if not isinstance(recorded, list) or len(supplied) > len(recorded):
            return [(path, supplied, recorded)]
        found = []
        for index, value in enumerate(supplied):
            found.extend(_evidence_conflicts(value, recorded[index], f"{path}[{index}]"))
        return found
    if _is_number(recorded):
        number = _as_number(supplied)
        if number is None:
            return [(path, supplied, recorded)]
        if isinstance(recorded, int) and isinstance(supplied, int) and not isinstance(supplied, bool):
            return [] if supplied == recorded else [(path, supplied, recorded)]
        if abs(number - float(recorded)) <= _stated_half_unit(supplied) + 1e-12:
            return []
        return [(path, supplied, recorded)]
    if isinstance(recorded, str):
        if isinstance(supplied, str) and supplied.strip() == recorded.strip():
            return []
        return [(path, supplied, recorded)]
    return [] if supplied == recorded else [(path, supplied, recorded)]


def _short(value: Any, limit: int = 60) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        text = repr(value)
    if len(text) <= limit:
        return text
    head = (limit - 3) // 2
    return text[:head] + "..." + text[-(limit - 3 - head):]


def _describe_conflicts(conflicts: list[tuple[str, Any, Any]], limit: int = 3) -> str:
    parts = []
    for path, supplied, recorded in conflicts[:limit]:
        where = "absent from the record" if recorded is _ABSENT else f"recorded {_short(recorded)}"
        parts.append(f"`{path}` supplied {_short(supplied)}, {where}")
    if len(conflicts) > limit:
        parts.append(f"{len(conflicts) - limit} further conflict(s)")
    return "; ".join(parts)


def _explain_conflict(
    key: str,
    source: str,
    supplied: Any,
    recorded: Any,
    conflicts: list[tuple[str, Any, Any]],
) -> str:
    """State why a supplied evidence argument disagrees with its reference.

    The most useful diagnosis first: if the supplied object is in fact another
    recorded tool result, the argument was filled from the wrong source.
    Otherwise, fields the reference does not have are named together, and
    differing values are listed with both sides.
    """
    if isinstance(supplied, dict) and supplied:
        for owner, value in (_LEDGER.get() or {}).items():
            if owner in source or not isinstance(value, dict) or value.get("error"):
                continue
            if not _evidence_conflicts(supplied, value, key):
                return (
                    f"the supplied `{key}` matches the recorded {owner} result, but "
                    f"`{key}` refers to the {source} result"
                )

    parts: list[str] = []
    absent = [path for path, _supplied, ref in conflicts if ref is _ABSENT]
    if absent:
        shown = ", ".join(f"`{path}`" for path in absent[:6])
        if len(absent) > 6:
            shown += f" and {len(absent) - 6} more"
        verb = "is not a field" if len(absent) == 1 else "are not fields"
        text = f"{shown} {verb} of the recorded {source} result"
        top_level = all(path.count(".") == 1 and "[" not in path for path in absent)
        if top_level and isinstance(recorded, dict) and recorded:
            names = list(recorded.keys())
            listing = ", ".join(f"`{name}`" for name in names[:8])
            if len(names) > 8:
                listing += ", ..."
            text += f" (its fields are {listing})"
        parts.append(text)
    differing = [c for c in conflicts if c[2] is not _ABSENT]
    if differing:
        parts.append(_describe_conflicts(differing))
    return f"`{key}` disagrees with the recorded {source} result - " + "; ".join(parts)


def _same_path(a: str, b: str) -> bool:
    try:
        return Path(a).expanduser().resolve() == Path(b).expanduser().resolve()
    except Exception:
        return str(a).strip() == str(b).strip()


def _default_report_out_path(image_path: str | None = None) -> str:
    stem = Path(str(image_path or "")).stem.strip() if image_path else ""
    base = stem or "landslide_report"
    return str(Path("outputs") / "reports" / f"{base}_{uuid4().hex[:8]}.json")


# Auxiliary evidence the contract may record as unavailable instead of
# blocking the analysis forever: external geo services, the subtype
# classifier and the VLM boundary re-check. Core perception (tiff.info,
# llm.first_pass, seg.run, seg.refine) is never degradable.
DEGRADABLE_TOOLS: tuple[str, ...] = ("geo.background", "geo.nearby", "cls.run", "seg.llm_review", "vlm.describe")


def max_tool_failures() -> int:
    try:
        return max(1, int(os.getenv("AGENT_MAX_TOOL_FAILURES", "2") or "2"))
    except ValueError:
        return 2


def unavailable_placeholder(tool: str, reason: str) -> dict[str, Any]:
    """Ledger entry for evidence that could not be obtained.

    It carries no measured values: every field that would normally hold a
    measurement is empty, so nothing downstream can mistake it for data.
    """
    base: dict[str, Any] = {
        "evidence_unavailable": True,
        "tool": tool,
        "reason": reason,
        "source_status": "unavailable",
    }
    if tool == "geo.nearby":
        base.update({"count": None, "features": []})
    elif tool == "geo.background":
        base.update({"terrain": {}, "geology": {}})
    elif tool == "cls.run":
        base.update({"class_name": "", "confidence": None, "topk": []})
    elif tool == "seg.llm_review":
        base.update({"llm_second_pass": None})
    return base


def is_unavailable(result: Any) -> bool:
    return isinstance(result, dict) and bool(result.get("evidence_unavailable"))


def unavailable_evidence(outputs: dict[str, Any] | None) -> list[dict[str, str]]:
    return [
        {"tool": name, "reason": str(outputs[name].get("reason", ""))}
        for name in DEGRADABLE_TOOLS
        if is_unavailable((outputs or {}).get(name))
    ]


@dataclass
class ToolCallContext:
    """Per-request environment the policy needs to assemble a tool call.

    Framework-neutral: every entry point (free agent, FastAPI service, and the
    deterministic graph if it ever routes through the tool registry) fills this
    in from its own request state, so the precondition / dependency-assembly
    logic itself lives only in :meth:`LandslidePolicy.prepare_tool_call`.
    """

    image_path: str = ""
    latitude: float | None = None
    longitude: float | None = None
    nearby_radius: int = 300
    report_written: bool = False
    # Retained for interface compatibility. The contract layer never executes a
    # tool itself; it only admits, binds, or refuses the call it is given.
    run_tool: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None
    # Optional: enrich an ``image_info`` dict that has a path but no width/height.
    read_image_info: Callable[[str], dict[str, Any]] | None = None


@dataclass(frozen=True)
class LandslidePolicy:
    """Authoritative runtime rules for one landslide analysis request."""

    initial_cross_check: tuple[str, ...] = ("tiff.info", "llm.first_pass", "seg.run")
    tiny_area_review_threshold: float = 0.20
    small_area_ratio_threshold: float = 0.20
    require_classification: bool = True
    require_geo_context: bool = True
    require_report_write: bool = False
    # region.locate is descriptive-only, so it never entered the required order
    # and in practice was never called: neither next_required_tool nor the
    # workflow instructions named it. Gate it on a flag so the 3x3 frame
    # position is collected like any other evidence tool.
    require_region_locate: bool = True
    # Cross-module consistency gate (contribution 3: confidence-triggered
    # conditional re-perception, generalised beyond the tiny-area case).
    enforce_consistency_gate: bool = True
    # Segmentation area ratio that counts as a *material* landslide signal when
    # first-pass screening said "no" -> disagreement, forces seg.llm_review.
    screening_seg_disagreement_ratio: float = 0.05

    # --- fusion decision rule (landslide yes/no) --------------------------- #
    # The decision is the cross-validation of the two INDEPENDENT detection
    # modalities only: whole-image VLM screening + semantic segmentation.
    # Sub-type classification and geospatial context never vote — they enrich
    # the report. When the two modalities disagree, the VLM boundary re-check
    # (seg.llm_review) is the arbiter; without it the conservative outcome holds.
    require_modality_agreement: bool = True
    seg_positive_area_ratio: float = 0.01
    seg_positive_area_ratio_small: float = 0.005
    seg_positive_min_pixels: int = 512
    severity_high_area_ratio: float = 0.15
    severity_medium_area_ratio: float = 0.05

    @classmethod
    def from_environment(cls, thresholds_path: str = "configs/thresholds.json") -> "LandslidePolicy":
        thresholds: dict[str, Any] = {}
        try:
            thresholds = json.loads(Path(thresholds_path).read_text(encoding="utf-8"))
        except Exception:
            pass

        def number(name: str, default: float) -> float:
            raw = os.getenv(name, thresholds.get(name, default))
            try:
                return float(raw)
            except (TypeError, ValueError):
                return default

        raw_review = os.getenv("SEG_LLM_SECOND_PASS_MAX_AREA_RATIO")
        if raw_review is None:
            raw_review = thresholds.get("small_area_ratio_threshold", 0.20)
        try:
            review_threshold = float(raw_review)
        except (TypeError, ValueError):
            review_threshold = 0.20
        if review_threshold <= 0.0:
            review_threshold = 0.20

        gate_raw = os.getenv("AGENT_CONSISTENCY_GATE", thresholds.get("enforce_consistency_gate", "1"))
        enforce_gate = str(gate_raw).strip().lower() not in {"0", "false", "no", "off"}

        return cls(
            tiny_area_review_threshold=review_threshold,
            small_area_ratio_threshold=number("small_area_ratio_threshold", 0.20),
            require_report_write=os.getenv("AGENT_ENABLE_REPORT_WRITE", "1") in {"1", "true", "True"},
            require_region_locate=str(
                os.getenv("AGENT_ENABLE_REGION_LOCATE", thresholds.get("require_region_locate", "1"))
            ).strip().lower() not in {"0", "false", "no", "off"},
            enforce_consistency_gate=enforce_gate,
            screening_seg_disagreement_ratio=number("screening_seg_disagreement_ratio", 0.05),
            require_modality_agreement=str(
                os.getenv("FUSION_REQUIRE_AGREEMENT", thresholds.get("require_modality_agreement", "1"))
            ).strip().lower() not in {"0", "false", "no", "off"},
            seg_positive_area_ratio=number("seg_positive_area_ratio", 0.01),
            seg_positive_area_ratio_small=number("seg_positive_area_ratio_small", 0.005),
            seg_positive_min_pixels=int(number("seg_positive_min_pixels", 512)),
            severity_high_area_ratio=number("severity_high_area_ratio", 0.15),
            severity_medium_area_ratio=number("severity_medium_area_ratio", 0.05),
        )

    def requires_initial_cross_check(self, completed: set[str]) -> list[str]:
        return [name for name in self.initial_cross_check if name not in completed]

    def missing_fusion_prerequisites(self, completed: set[str]) -> list[str]:
        required = {"llm.first_pass", "seg.run", "cls.run", "geo.nearby", "geo.background"}
        return sorted(required - set(completed))

    def evidence_order(self, *, require_region_locate: bool | None = None) -> tuple[str, ...]:
        """Evidence tools required after the initial cross-check, in required order.

        Single source for the per-turn hint, the critic's auto-advance and the
        post-hoc check, so the three can never disagree about region.locate.
        """
        if require_region_locate is None:
            require_region_locate = self.require_region_locate
        order = ("seg.refine", "vlm.describe", "cls.run", "geo.background", "geo.nearby")
        # region.locate reads the refined candidate regions, so it runs immediately after seg.refine.
        if require_region_locate:
            return ("seg.refine", "region.locate", "vlm.describe", "cls.run", "geo.background", "geo.nearby")
        return order

    def report_write_allowed(self, completed: set[str], already_written: bool) -> bool:
        return self.require_report_write and "fuse.decision" in completed and not already_written

    @staticmethod
    def resolve_area_ratio(refinement: Any, segmentation: Any = None) -> float | None:
        for candidate in (refinement, segmentation):
            if not isinstance(candidate, dict):
                continue
            try:
                ratio = float(candidate.get("area_ratio"))
            except (TypeError, ValueError):
                continue
            if ratio >= 0.0:
                return ratio
        return None

    def review_required(self, refinement: Any, segmentation: Any = None) -> bool:
        ratio = self.resolve_area_ratio(refinement, segmentation)
        return ratio is not None and ratio < self.tiny_area_review_threshold

    @staticmethod
    def _stage1_label(stage1: Any) -> str:
        return str((stage1 or {}).get("assessment_label", "") or "").strip().lower()

    def should_run_second_pass(
        self,
        refinement: Any,
        segmentation: Any = None,
        stage1: Any = None,
        enabled: bool = False,
    ) -> bool:
        # Hard triggers - never gated by the UI toggle:
        #   * tiny-target segmentation
        #   * the screening model reporting its own uncertainty
        if self.review_required(refinement, segmentation):
            return True
        if self._stage1_label(stage1) in {"uncertain", "error"}:
            return True
        # The toggle only requests an extra (description-only, when the decision
        # is already settled) whole-image pass for narrative enrichment.
        return bool(enabled)

    def missing_fusion_requirements(self, args: dict[str, Any]) -> list[str]:
        missing: list[str] = []
        classification = args.get("classification")
        if self.require_classification and not is_unavailable(classification):
            if not isinstance(classification, dict):
                missing.append("classification")
            elif not str(classification.get("class_name", "") or "").strip():
                missing.append("classification.class_name")

        geo_context = args.get("geo_context")
        if not self.require_geo_context:
            return missing
        if not isinstance(geo_context, dict):
            missing.append("geo_context")
            return missing

        background = geo_context.get("background")
        nearby = geo_context.get("nearby")
        terrain = background.get("terrain") if isinstance(background, dict) else None
        geology = background.get("geology") if isinstance(background, dict) else None
        if not is_unavailable(background):
            if not isinstance(background, dict):
                missing.append("geo_context.background")
            if not isinstance(terrain, dict):
                missing.append("geo_context.background.terrain")
            else:
                for key in ("slope_deg", "aspect_deg"):
                    if key not in terrain:
                        missing.append(f"geo_context.background.terrain.{key}")
            if not isinstance(geology, dict):
                missing.append("geo_context.background.geology")
            elif not any(key in geology for key in ("lithology", "unit_name", "description", "age", "source")):
                missing.append("geo_context.background.geology.(lithology|unit_name|description|age|source)")
        if is_unavailable(nearby):
            pass
        elif not isinstance(nearby, dict):
            missing.append("geo_context.nearby")
        else:
            if "count" not in nearby:
                missing.append("geo_context.nearby.count")
            if "features" not in nearby:
                missing.append("geo_context.nearby.features")
            elif not isinstance(nearby.get("features"), list):
                missing.append("geo_context.nearby.features(list)")
        return missing

    # ------------------------------------------------------------------ #
    # cross-module consistency  (contribution 3, generalised)
    # ------------------------------------------------------------------ #
    @staticmethod
    def stage1_is_positive(stage1: Any) -> bool | None:
        """Tri-state read of the first-pass screening verdict (None = unknown)."""
        s = stage1 if isinstance(stage1, dict) else {}
        flag = s.get("has_landslide")
        if isinstance(flag, bool):
            return flag
        label = str(s.get("assessment_label", "") or "").strip().lower()
        if label == "likely":
            return True
        if label == "unlikely":
            return False
        return None

    @staticmethod
    def _geo_is_degraded(result: Any, *, is_background: bool) -> bool:
        if is_unavailable(result):
            return False  # explicitly recorded as unavailable; retrying is not required
        if not isinstance(result, dict):
            return True
        status = str(
            result.get("source_status", "") or result.get("status", "") or ""
        ).strip().lower()
        if status in {"error", "unavailable", "degraded", "failed"}:
            return True
        if is_background:
            terrain = result.get("terrain")
            if isinstance(terrain, dict) and "slope_deg" in terrain and terrain.get("slope_deg") is None:
                return True
        return False

    def consistency_needs_second_pass(self, outputs: dict[str, Any] | None) -> bool:
        """True when cross-module evidence disagrees in a way that a VLM
        boundary re-check (``seg.llm_review``) is the defined resolution for."""
        outputs = outputs or {}
        positive = self.stage1_is_positive(outputs.get("llm.first_pass"))
        ratio = self.resolve_area_ratio(
            outputs.get("seg.refine"), outputs.get("seg.run")
        )
        if positive is False and ratio is not None and ratio >= self.screening_seg_disagreement_ratio:
            return True
        fuse = outputs.get("fuse.decision")
        if isinstance(fuse, dict) and not fuse.get("error") and positive is not None:
            if bool(fuse.get("has_landslide")) != positive:
                return True
        return False

    def mandatory_second_pass_reason(self, outputs: dict[str, Any] | None) -> str | None:
        """The single place that answers 'is ``seg.llm_review`` a hard
        requirement right now, and why?'  Covers every trigger:
        tiny-target segmentation, screening-model uncertainty, and
        VLM/segmentation disagreement. Returns the reason string, or ``None``.
        Not gated by the UI second-pass toggle.
        """
        outputs = outputs or {}
        ratio = self.resolve_area_ratio(outputs.get("seg.refine"), outputs.get("seg.run"))
        if ratio is not None and ratio < self.tiny_area_review_threshold:
            return (
                f"segmentation area_ratio={ratio:.4f} is below "
                f"{self.tiny_area_review_threshold:.2f} (tiny-target re-check)"
            )
        label = self._stage1_label(outputs.get("llm.first_pass"))
        if label in {"uncertain", "error"}:
            return f"first-pass screening is '{label}' (screening-model uncertainty re-check)"
        if self.enforce_consistency_gate and self.consistency_needs_second_pass(outputs):
            return (
                "whole-image VLM screening and segmentation are in disagreement "
                "(cross-modality reconciliation re-check)"
            )
        return None

    def consistency_violations(
        self, outputs: dict[str, Any] | None, *, geo_expected: bool = False
    ) -> list[str]:
        outputs = outputs or {}

        def produced(name: str) -> bool:
            r = outputs.get(name)
            return isinstance(r, dict) and not r.get("error")

        out: list[str] = []

        # The screening/segmentation/fusion disagreement checks now live in
        # ``mandatory_second_pass_reason`` (a single re-perception trigger set,
        # surfaced by ``verify_analysis``). What remains here is degraded
        # geospatial evidence, which no re-perception can fix - it needs a retry.
        if geo_expected:
            if produced("geo.background") and self._geo_is_degraded(
                outputs.get("geo.background"), is_background=True
            ):
                out.append(
                    "geological background is a degraded placeholder though coordinates were "
                    "provided; retry geo.background with no arguments (request lat/lon are bound automatically)"
                )
            if produced("geo.nearby") and self._geo_is_degraded(
                outputs.get("geo.nearby"), is_background=False
            ):
                out.append(
                    "nearby-facility context failed though coordinates were provided; "
                    "retry geo.nearby with no arguments (request lat/lon and radius are bound automatically)"
                )

        return out

    # ------------------------------------------------------------------ #
    # fusion decision rule  (landslide yes / no)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _model_score(obj: Any) -> float | None:
        """A model-emitted score in [0, 1], or None when the model gave none."""
        try:
            s = float((obj if isinstance(obj, dict) else {}).get("score"))
        except (TypeError, ValueError):
            return None
        return s if 0.0 <= s <= 1.0 else None

    def seg_is_positive(self, segmentation: Any) -> bool:
        seg = segmentation if isinstance(segmentation, dict) else {}
        try:
            ratio = float(seg.get("area_ratio", 0.0) or 0.0)
        except (TypeError, ValueError):
            ratio = 0.0
        try:
            pixels = int(seg.get("landslide_pixels", 0) or 0)
        except (TypeError, ValueError):
            pixels = 0
        return ratio >= self.seg_positive_area_ratio or (
            ratio >= self.seg_positive_area_ratio_small and pixels >= self.seg_positive_min_pixels
        )

    def fuse_decision(
        self,
        *,
        stage1: Any,
        segmentation: Any = None,
        refinement: Any = None,
        llm_second_pass: Any = None,
    ) -> dict[str, Any]:
        """Cross-validate the two independent detection modalities into a
        landslide yes/no decision plus an areal-extent severity band.

        Inputs are ONLY the whole-image VLM screening (``stage1``) and the
        semantic segmentation result. ``refinement`` is used for extent/severity
        only; ``llm_second_pass`` acts as the disagreement arbiter through its
        three-valued verdict label alone (Support / NotSupport, i.e. decision
        ``positive`` / ``negative``). Sub-type classification and geospatial
        context are deliberately not arguments.

        ``confidence`` is **a real model score for the final verdict** and
        nothing else: the first-pass screening score when available, otherwise
        ``None``. The boundary re-check emits a verdict label, not a calibrated
        score, so its output never enters confidence. It is never synthesised,
        blended, or clamped - ``confidence_source`` names its origin.
        """
        vlm = self.stage1_is_positive(stage1)            # True / False / None
        vlm_pos = vlm is True
        seg_pos = self.seg_is_positive(segmentation)

        review = llm_second_pass if isinstance(llm_second_pass, dict) else {}
        purpose = str(
            review.get("review_purpose", review.get("purpose", "")) or ""
        ).strip().lower()
        r_decision = str(review.get("decision", "") or "").strip().lower()
        descriptive = purpose in {"description_only", "describe", "descriptive"}
        review_is_arbiter = bool(review) and not descriptive and r_decision in {"positive", "negative"}
        review_positive = review_is_arbiter and r_decision == "positive"
        review_negative = review_is_arbiter and r_decision == "negative"

        agree_positive = vlm_pos and seg_pos
        agree_negative = (vlm is False) and (not seg_pos)
        disagreement = not agree_positive and not agree_negative

        if not self.require_modality_agreement:
            # Aggressive primary rule: either visual screening or segmentation
            # is sufficient for a positive verdict.  The optional boundary
            # review remains audit evidence only; it must not create a third
            # positive path when both declared primary modalities are negative.
            has_landslide = vlm_pos or seg_pos
            basis = "either primary modality positive (aggressive OR rule)"
        elif agree_positive:
            has_landslide, basis = True, "whole-image VLM screening and segmentation agree (positive)"
        elif agree_negative:
            has_landslide, basis = False, "whole-image VLM screening and segmentation agree (negative)"
        elif review_is_arbiter:
            has_landslide = review_positive
            basis = "modalities disagreed; resolved by VLM boundary re-check (seg.llm_review)"
        else:
            has_landslide = False
            basis = "modalities disagreed and no VLM boundary re-check is available; conservative negative"

        # Confidence: strictly a real model score, never fabricated. The
        # boundary re-check returns a verdict label (its numeric ``score`` field
        # is a hardcoded label placeholder, not a model-emitted confidence), so
        # only the first-pass screening score is used here.
        stage1_score = self._model_score(stage1)
        if stage1_score is not None:
            confidence, confidence_source = stage1_score, "vlm_first_pass"
        else:
            confidence, confidence_source = None, "unavailable"

        try:
            seg_ratio = float((segmentation or {}).get("area_ratio", 0.0) or 0.0)
        except (TypeError, ValueError):
            seg_ratio = 0.0

        try:
            region_area_ratio = float((refinement or {}).get("area_ratio", 0.0) or 0.0)
        except (TypeError, ValueError):
            region_area_ratio = 0.0
        extent = max(seg_ratio, region_area_ratio)
        if not has_landslide:
            severity = "none"
        elif extent >= self.severity_high_area_ratio:
            severity = "high"
        elif extent >= self.severity_medium_area_ratio:
            severity = "medium"
        else:
            severity = "low"

        return {
            "has_landslide": has_landslide,
            "confidence": confidence,
            "confidence_source": confidence_source,
            "severity": severity,
            "decision_basis": basis,
            "modalities": {
                "vlm_verdict": "positive" if vlm is True else "negative" if vlm is False else "uncertain",
                "vlm_positive": vlm_pos,
                "segmentation_positive": seg_pos,
                "segmentation_area_ratio": round(seg_ratio, 6),
                "agreement": "positive" if agree_positive else "negative" if agree_negative else "disagreement",
                "second_pass_role": "arbiter" if review_is_arbiter else "descriptive" if descriptive else "none",
                "second_pass_positive": review_positive,
                "second_pass_negative": review_negative,
                "second_pass_descriptive": descriptive and bool(review),
            },
        }

    def verify_analysis(
        self,
        outputs: dict[str, Any] | None,
        *,
        require_report_write: bool | None = None,
        require_region_locate: bool | None = None,
        geo_expected: bool = False,
    ) -> list[str]:
        """Post-hoc rule check for one completed analysis.

        ``outputs`` maps tool name -> its result dict (as accumulated by the
        agent runtime). Returns a list of human-readable rule violations; an
        empty list means every mandatory task rule is satisfied.

        This is the authoritative "critic" gate: the agent is free to choose
        tool order and iteration, but it may not produce a final report while
        this returns any item.
        """
        outputs = outputs or {}
        if require_report_write is None:
            require_report_write = self.require_report_write
        if require_region_locate is None:
            require_region_locate = self.require_region_locate

        def produced(name: str) -> bool:
            result = outputs.get(name)
            return isinstance(result, dict) and not result.get("error")

        violations: list[str] = []

        for name in self.initial_cross_check:
            if not produced(name):
                violations.append(
                    f"initial cross-check incomplete: '{name}' has not produced a valid result"
                )

        if not produced("vlm.describe"):
            violations.append(
                "scene description missing: 'vlm.describe' has not produced a valid result"
            )

        if self.require_classification and not produced("cls.run"):
            violations.append(
                "landslide subtype reference missing: 'cls.run' has not produced a valid result"
            )

        if self.require_geo_context:
            if not produced("geo.background"):
                violations.append(
                    "geological background evidence missing: 'geo.background' "
                    "(terrain slope/aspect + geology) has not been collected"
                )
            if not produced("geo.nearby"):
                violations.append(
                    "nearby human-facility context missing: 'geo.nearby' has not been collected"
                )

        if require_region_locate and not produced("region.locate"):
            violations.append(
                "frame-relative position missing: 'region.locate' (3x3 grid position of "
                "the primary candidate) has not produced a valid result"
            )

        review_reason = self.mandatory_second_pass_reason(outputs)
        if review_reason and not produced("seg.llm_review"):
            violations.append(
                f"mandatory VLM boundary re-check missing: {review_reason}; "
                "call 'seg.llm_review' with no arguments before the final decision step"
            )

        if produced("fuse.decision") and self.fuse_decision_is_stale(outputs):
            violations.append(
                "stale fusion: fuse.decision was computed before seg.llm_review ran; "
                "call fuse.decision again so the boundary re-check is incorporated"
            )

        next_required = self.next_required_tool(
            outputs,
            require_report_write=require_report_write,
            require_region_locate=require_region_locate,
        )
        if next_required == "fuse.decision":
            violations.append(
                "final fusion missing: the final decision tool has not produced a valid result"
            )
        elif produced("fuse.decision"):
            fusion_view = {
                "classification": outputs.get("cls.run"),
                "geo_context": {
                    "background": outputs.get("geo.background"),
                    "nearby": outputs.get("geo.nearby"),
                },
            }
            for item in self.missing_fusion_requirements(fusion_view):
                violations.append(f"fusion evidence incomplete: {item}")

        if (
            require_report_write
            and produced("fuse.decision")
            and not self.fuse_decision_is_stale(outputs)
            and not produced("report.write")
        ):
            violations.append("final report not written: 'report.write' has not completed")

        if self.enforce_consistency_gate:
            violations.extend(
                self.consistency_violations(outputs, geo_expected=geo_expected)
            )

        return violations

    def fuse_decision_is_stale(self, outputs: dict[str, Any] | None) -> bool:
        """True when ``fuse.decision`` already ran but WITHOUT the boundary
        re-check, and a valid ``seg.llm_review`` verdict now exists.

        This closes the post-fusion gap: if the model called fuse.decision early
        and the consistency gate then forced ``seg.llm_review``, the stored
        fusion is stale and must be recomputed so the re-check is incorporated.
        """
        outputs = outputs or {}
        fuse = outputs.get("fuse.decision")
        review = outputs.get("seg.llm_review")
        if not isinstance(fuse, dict) or fuse.get("error"):
            return False
        if not isinstance(review, dict) or review.get("error"):
            return False
        lp = review.get("llm_second_pass")
        if not isinstance(lp, dict):
            return False
        decision = str(lp.get("decision", "") or "").strip().lower()
        if decision in ("", "descriptive"):
            return False
        role = str(
            (fuse.get("decision_support") or {}).get("second_pass_role", "") or ""
        ).strip().lower()
        return role in ("", "none")

    def tool_call_template(self, name: str) -> str:
        """One-line 'call it exactly like this' string for the graduated
        single-tool nudge the agent critic issues before it takes over."""
        templates = {
            "tiff.info": "call `tiff.info` with the image path.",
            "llm.first_pass": "call `llm.first_pass` with no arguments.",
            "vlm.describe": "call `vlm.describe` with no arguments.",
            "seg.run": "call `seg.run` with no arguments.",
            "seg.refine": "call `seg.refine` with no arguments.",
            "region.locate": "call `region.locate` with no arguments.",
            "seg.llm_review": "call `seg.llm_review` with no arguments (the segmentation-boundary overlay is assembled for you).",
            "cls.run": "call `cls.run` with no arguments.",
            "geo.background": "call `geo.background` with no arguments (coordinates are filled in).",
            "geo.nearby": "call `geo.nearby` with no arguments (coordinates and radius are filled in).",
            "fuse.decision": "call `fuse.decision` with no arguments; all prior evidence is assembled automatically.",
            "report.write": "call `report.write` with no arguments to persist the final report.",
        }
        return templates.get(name, "call `%s` with no arguments." % name)

    def fusion_argument_hint(self, missing: list[str] | None = None) -> str:
        missing_text = ", ".join(missing or []) or "unknown"
        return (
            "fuse.decision failed because its validated input is incomplete. "
            f"Missing or invalid fields: {missing_text}. "
            "Reuse successful outputs from this turn: stage1=llm.first_pass, "
            "segmentation=seg.run, refinement=seg.refine, classification=cls.run, "
            "geo_context={background:geo.background, nearby:geo.nearby}. "
            "Do not restart tiff.info -> llm.first_pass -> seg.run and do not hand-build "
            "classification or geo_context. Run only the specifically missing prerequisite "
            "tool(s), then retry fuse.decision with no arguments (or a valid llm_second_pass)."
        )

    def second_pass_required_instruction(self) -> str:
        return (
            "Hard requirement before fuse.decision for tiny-area cases: "
            f"if area_ratio is below {self.tiny_area_review_threshold:.2f}, call seg.llm_review first, "
            "then call fuse.decision again. Use arguments exactly as: refinement <- seg.refine; "
            "stage1 <- llm.first_pass; image_info <- tiff.info. "
            "Then set llm_second_pass <- seg.llm_review.llm_second_pass when calling fuse.decision."
        )

    def fusion_required_call_instruction(self) -> str:
        return (
            "How to call fuse.decision: call it with NO arguments after all required tools succeed. "
            "The controller assembles stage1 from llm.first_pass, segmentation from seg.run, "
            "refinement from seg.refine, classification from cls.run, and geo_context from "
            "geo.background + geo.nearby. Never hand-construct classification or geo_context. "
            "After a failed fuse.decision, preserve successful outputs and do not restart the "
            "initial tiff.info -> llm.first_pass -> seg.run chain; run only the missing prerequisite, "
            "then retry fuse.decision with no arguments."
        )

    def retry_instruction_from_error(self, error_text: str) -> str | None:
        lowered = str(error_text or "").strip().lower()
        if "fuse.decision" not in lowered:
            return None
        if "seg.llm_review" in lowered and ("mandatory" in lowered or "call seg.llm_review first" in lowered):
            return self.second_pass_required_instruction()
        if any(token in lowered for token in ("input is incomplete", "missing:", "requires", "classification", "geo_context")):
            return self.fusion_required_call_instruction()
        return None

    def workflow_instruction(self) -> str:
        report_instruction = (
            "Before finishing, call fuse.decision and then report.write to write the final report JSON to local disk."
            if self.require_report_write
            else "Before finishing, call fuse.decision to produce the final decision/report output."
        )
        return (
            "You are an autonomous but evidence-disciplined landslide remote-sensing agent. "
            "Use tools to inspect the image, choose useful optional analysis, and explain your reasoning; "
            "the system preserves your autonomy while enforcing only evidence dependencies. "
            "Treat the tool-result ledger as authoritative state: reuse successful results in later turns. "
            "Never claim a tool ran when its result is absent or errored. "
            "Hard gates: tiff.info before image analysis; seg.refine before region.locate; "
            "region.locate before fuse.decision; fuse.decision only after required evidence is complete. "
            "If fusion fails, diagnose the missing field, run only the missing prerequisite, and retry fusion. "
            "Do not restart the whole workflow or discard existing artifacts. "
            "You may select optional tools and the order of independent evidence collection. "
            "For image analysis, always complete this initial cross-check before final decision/report: "
            "tiff.info, llm.first_pass, seg.run. Intermediate tool usage is flexible, but all required evidence "
            "must be collected before fusion. "
            f"When landslide area ratio is below {self.tiny_area_review_threshold:.2f}, invoke seg.llm_review "
            "using the segmentation-mask boundary highlighted whole-image overlay for second-pass verification "
            "and narrative enrichment. Final conclusions and reports must include landslide subtype reference, "
            "terrain slope/aspect and geological background evidence, and nearby human-facility context. "
            + report_instruction
        )

    def autonomous_task_contract(self) -> str:
        """A non-prescriptive domain brief for the free Agent mode."""
        return (
            "You are an autonomous landslide remote-sensing agent. Your goal is to "
            "provide a traceable, image-grounded assessment rather than a merely "
            "plausible answer. You alone choose the scope of investigation, available "
            "tools, tool arguments, tool order, iteration, and when to stop. There is "
            "no prescribed workflow, tool quota, or call order. "
            "For a comprehensive assessment, the available evidence may include image "
            "metadata, whole-scene visual screening, semantic segmentation, candidate "
            "region refinement, boundary review, subtype classification, terrain and "
            "geology, nearby human facilities, and localized image tiles. Decide for "
            "yourself which of these are useful for the case. "
            "Treat tool outputs as the evidence record. Distinguish observed evidence "
            "from inference, do not claim a tool ran when it did not, and do not present "
            "unsupported context as fact. In the final report, mark direct image/tool facts "
            "with 'Observed:' and logical, causal, predictive, or cross-stage reasoning with "
            "'Inference:'. Use 'Not observed' or 'Not available' when evidence is missing; "
            "never turn a plausible explanation into an observed fact. "
            "When landslide subtypes from different sources disagree, reconcile them with the "
            "Cruden-Varnes movement-form parent: if the subtypes differ but their movement-form "
            "parent agrees, fall back to that parent; if the parent also differs, report an "
            "explicit classification conflict rather than forcing a single label. "
            "Explain agreement, conflict, absence, or "
            "uncertainty in the available evidence as you judge appropriate. "
            "When you give a final response, make the judgement, salient evidence, "
            "confidence or uncertainty, supported type/extent/location, useful spatial "
            "context, and material limitations clear when they are relevant. Keep the "
            "reasoning user-facing and concise; do not expose private chain-of-thought."
        )

    def next_required_tool(
        self,
        outputs: dict[str, Any] | None,
        *,
        require_report_write: bool | None = None,
        require_region_locate: bool | None = None,
    ) -> str | None:
        """Name of the next required tool that has not yet produced a valid result.

        Single source of truth for the *ordering* constraint: the per-turn hint,
        the deterministic pipeline, and the critic's auto-advance all derive the
        required order from here. Returns ``None`` once every mandatory step is
        satisfied.
        """
        outputs = outputs or {}
        if require_report_write is None:
            require_report_write = self.require_report_write

        def done(name: str) -> bool:
            result = outputs.get(name)
            return isinstance(result, dict) and not result.get("error")

        for name in self.initial_cross_check:
            if not done(name):
                return name
        review_pending = not done("seg.llm_review") and bool(
            self.mandatory_second_pass_reason(outputs)
        )
        for name in self.evidence_order(require_region_locate=require_region_locate):
            # The VLM boundary re-check operates on the same object as
            # seg.refine (the segmentation mask / boundary overlay). When it is
            # mandatory it runs right after refinement / region.locate and
            # before the enrichment tools (cls.run, geo.*).
            if review_pending and name not in ("seg.refine", "region.locate"):
                return "seg.llm_review"
            if not done(name):
                return name
        if review_pending:
            return "seg.llm_review"
        if not done("fuse.decision") or self.fuse_decision_is_stale(outputs):
            return "fuse.decision"
        if require_report_write and not done("report.write"):
            return "report.write"
        return None

    def next_step_hint(
        self,
        outputs: dict[str, Any] | None,
        *,
        require_report_write: bool | None = None,
        require_region_locate: bool | None = None,
    ) -> str:
        """One-line 'do this next' hint derived from what is already done.

        Injected every turn so a weaker model keeps following the required order
        without the order being hard-enforced.
        """
        outputs = outputs or {}
        if require_report_write is None:
            require_report_write = self.require_report_write

        def done(name: str) -> bool:
            result = outputs.get(name)
            return isinstance(result, dict) and not result.get("error")

        if not done("tiff.info"):
            return "Next required step: call tiff.info with the image path."

        cross_missing = [n for n in ("llm.first_pass", "seg.run") if not done(n)]
        if cross_missing:
            return (
                "Next required step: finish the initial cross-check - call "
                + " and ".join(cross_missing)
                + "."
            )

        evidence = self.evidence_order(require_region_locate=require_region_locate)
        refine_stage_done = all(
            done(n) for n in evidence if n in ("seg.refine", "region.locate")
        )
        if refine_stage_done and not done("seg.llm_review"):
            review_reason = self.mandatory_second_pass_reason(outputs)
            if review_reason:
                return (
                    f"Next required step: {review_reason} - call seg.llm_review "
                    "before cls.run / geo.*."
                )

        evidence_missing = [n for n in evidence if not done(n)]
        if evidence_missing:
            return "Next required step: collect evidence - call " + ", ".join(evidence_missing) + "."

        if not done("seg.llm_review"):
            review_reason = self.mandatory_second_pass_reason(outputs)
            if review_reason:
                return (
                    f"Next required step: {review_reason} - call seg.llm_review before fuse.decision."
                )

        if not done("fuse.decision"):
            return (
                "Next required step: all evidence is ready - call fuse.decision now. "
                "You can call it with no arguments; classification and geo_context are "
                "filled in from cls.run / geo.background / geo.nearby automatically."
            )

        if self.fuse_decision_is_stale(outputs):
            return (
                "Next required step: fuse.decision was computed before seg.llm_review ran - "
                "call fuse.decision again so the boundary re-check is incorporated."
            )

        if require_report_write and not done("report.write"):
            return "Next required step: call report.write to persist the final report."

        return ""

    # ------------------------------------------------------------------ #
    # tool-call preconditions + dependency assembly (single source)
    # ------------------------------------------------------------------ #
    def _resolve_image_info(
        self,
        args: dict[str, Any],
        outputs: dict[str, Any],
        ctx: "ToolCallContext",
    ) -> dict[str, Any]:
        info = args.get("image_info")
        if isinstance(info, dict):
            path = str(info.get("image_path", "") or "")
            has_size = ("width" in info) and ("height" in info)
            if path and not has_size and ctx.read_image_info is not None:
                try:
                    return {**ctx.read_image_info(path), **info}
                except Exception:
                    return info
            return info
        tiff = outputs.get("tiff.info")
        if isinstance(tiff, dict):
            return tiff
        path = str(args.get("image_path", "") or "") or ctx.image_path
        if path and ctx.read_image_info is not None:
            return ctx.read_image_info(path)
        raise ToolPreconditionError(
            "missing image_info/image_path. Provide image_path or call tiff.info first."
        )

    @staticmethod
    def _bind_evidence(
        tool: str,
        args: dict[str, Any],
        key: str,
        recorded: Any,
        source: str,
        *,
        ignore: tuple[str, ...] = (),
    ) -> None:
        """Verify ``args[key]`` against a recorded tool result, then bind it.

        Supplied and consistent: the recorded artifact is bound (the agent's
        value was a faithful, possibly abbreviated, reference to it). Supplied
        and contradictory, or supplied with nothing on record: the call is
        refused. Omitted: the recorded artifact is bound if there is one.
        """
        supplied = args.get(key)
        if recorded is None:
            if supplied is not None:
                raise ToolPreconditionError(
                    f"{tool} precondition unmet: `{key}` was supplied but no {source} "
                    "result is on record to support it. Evidence arguments must come "
                    "from recorded tool results."
                )
            args.pop(key, None)
            return
        if supplied is None:
            _note_resolved(key, f"evidence ledger: {source}")
        else:
            probe = supplied
            if ignore and isinstance(supplied, dict):
                probe = {k: v for k, v in supplied.items() if k not in ignore}
            conflicts = _evidence_conflicts(probe, recorded, key)
            if conflicts:
                raise ToolPreconditionError(
                    f"{tool} precondition unmet: "
                    f"{_explain_conflict(key, source, probe, recorded, conflicts)}. "
                    "Evidence arguments are verified against the evidence ledger, not "
                    "replaced; an omitted argument is bound to the recorded result."
                )
            _note_checked(key)
        args[key] = recorded

    @staticmethod
    def _bind_constant(tool: str, args: dict[str, Any], key: str, value: Any, label: str) -> None:
        """Verify ``args[key]`` against a request-level constant, then bind it."""
        supplied = args.get(key)
        if supplied is not None and _evidence_conflicts(supplied, value, key):
            raise ToolPreconditionError(
                f"{tool} precondition unmet: `{key}`={_short(supplied)} disagrees with the "
                f"{label} fixed for this request ({_short(value)}). Request-level "
                "parameters are verified against the request, not replaced; an omitted "
                "argument is bound to the request value."
            )
        if supplied is None:
            _note_resolved(key, f"request: {label}")
        else:
            _note_checked(key)
        args[key] = value

    @staticmethod
    def _check_image_reference(tool: str, args: dict[str, Any], scene_path: str) -> None:
        """Refuse a call whose image reference names a different image."""
        if not scene_path:
            return
        supplied = []
        top = str(args.get("image_path", "") or "").strip()
        if top:
            supplied.append(("image_path", top))
        info = args.get("image_info")
        if isinstance(info, dict):
            nested = str(info.get("image_path", "") or "").strip()
            if nested:
                supplied.append(("image_info.image_path", nested))
        for key, value in supplied:
            if not _same_path(value, scene_path):
                raise ToolPreconditionError(
                    f"{tool} precondition unmet: `{key}`={value!r} does not refer to the "
                    f"image under analysis ({scene_path!r})."
                )
            _note_checked(key)

    def _bind_image_info(
        self,
        tool: str,
        args: dict[str, Any],
        outputs: dict[str, Any],
        ctx: "ToolCallContext",
        scene_path: str,
        *,
        required: bool = True,
    ) -> None:
        """Check the call's image reference and bind the recorded raster metadata."""
        self._check_image_reference(tool, args, scene_path)
        tiff = outputs.get("tiff.info")
        if isinstance(tiff, dict) and not tiff.get("error"):
            # Paths were compared above with path semantics; compare the rest.
            self._bind_evidence(
                tool, args, "image_info", tiff, "tiff.info", ignore=("image_path",)
            )
        elif required:
            if not isinstance(args.get("image_info"), dict):
                _note_resolved("image_info", "request: image under analysis")
            args["image_info"] = self._resolve_image_info(args, outputs, ctx)

    def _prepare_tool_call(
        self,
        name: str,
        raw_args: dict[str, Any] | None,
        outputs: dict[str, Any],
        ctx: "ToolCallContext",
    ) -> dict[str, Any]:
        """Evaluate one tool invocation against its declarative execution contract.

        Domain rules are compiled into *preconditions* attached to individual
        tool nodes rather than into procedural instructions in the prompt or an
        external supervisory process. Arguments the agent supplies are *verified*,
        never silently replaced: an evidence argument must agree with the
        recorded result of the tool that produced it, and a request-level
        parameter (image, georeference, search radius) must agree with the
        request. Agreement admits the call and binds the recorded artifact;
        disagreement, or an evidence argument with no recorded source, raises
        :class:`ToolPreconditionError`, which the runtime surfaces to the
        cognitive engine as an ordinary observation. Arguments the agent omits
        are bound from the same sources.

        The layer never executes a tool the agent did not request and never
        reorders its plan: it can only admit or refuse the requested call. Tool
        selection, ordering and iteration therefore remain entirely with the
        cognitive engine, while the evidence-dependency graph of the final
        assessment stays non-bypassable.

        ``outputs`` maps tool name -> result dict for calls already completed in
        this request.
        """
        args = dict(raw_args or {})

        def recorded(tool_name: str) -> dict[str, Any] | None:
            value = outputs.get(tool_name)
            return value if isinstance(value, dict) and not value.get("error") else None

        tiff = recorded("tiff.info")
        # The image under analysis is fixed by the request (or, failing that, by
        # the recorded raster metadata); no call may silently switch images.
        scene_path = str(ctx.image_path or "").strip()
        if not scene_path and tiff is not None:
            scene_path = str(tiff.get("image_path", "") or "").strip()

        if name == "tiff.info":
            supplied = str(args.get("image_path", "") or "").strip()
            if supplied and scene_path and not _same_path(supplied, scene_path):
                raise ToolPreconditionError(
                    f"tiff.info precondition unmet: `image_path`={supplied!r} does not "
                    f"refer to the image under analysis ({scene_path!r})."
                )
            path = scene_path or supplied
            if not _is_existing_file(path):
                if path and Path(path).is_dir():
                    raise ToolPreconditionError(
                        f"tiff.info precondition unmet: {path!r} is a directory, not an "
                        "image file."
                    )
                raise ToolPreconditionError(
                    "tiff.info precondition unmet: no readable image file is on record "
                    "for this request."
                )
            if supplied:
                _note_checked("image_path")
            else:
                _note_resolved("image_path", "request: image under analysis")
            args["image_path"] = path

        elif name in ("geo.nearby", "geo.background"):
            if ctx.latitude is not None and ctx.longitude is not None:
                # The scene georeference is an operator-supplied request constant.
                # A supplied coordinate is checked against it rather than silently
                # replaced: a transcribed coordinate would relocate all geographic
                # evidence while every service still answers normally.
                self._bind_constant(name, args, "lat", float(ctx.latitude), "scene latitude")
                self._bind_constant(name, args, "lon", float(ctx.longitude), "scene longitude")
            if name == "geo.nearby":
                self._bind_constant(
                    name,
                    args,
                    "radius",
                    int(getattr(ctx, "nearby_radius", 300) or 300),
                    "user-confirmed search radius",
                )
            if "lat" not in args or "lon" not in args:
                raise ToolPreconditionError(
                    "geo tool precondition unmet: no georeference is on record for this "
                    "scene and none was supplied with the call."
                )
            try:
                _lat = float(args["lat"])
                _lon = float(args["lon"])
            except (TypeError, ValueError):
                raise ToolPreconditionError(
                    "geo tool precondition unmet: lat and lon must be numeric."
                )
            if not (-90.0 <= _lat <= 90.0) or not (-180.0 <= _lon <= 180.0):
                raise ToolPreconditionError(
                    f"geo tool precondition unmet: coordinates ({_lat}, {_lon}) fall "
                    "outside the valid geographic range."
                )
            args["lat"], args["lon"] = _lat, _lon

        elif name == "image.tile":
            # Tiling needs raster width/height, so it still requires resolvable
            # image metadata (tiff.info or a size-bearing image_info).
            self._bind_image_info(name, args, outputs, ctx, scene_path)

        elif name in ("llm.first_pass", "seg.run", "cls.run", "vlm.describe"):
            # Whole-scene screening (VLM), segmentation, and subtype classification
            # each read the scene image itself and consume only the image path, not
            # tiff's raster metadata, so tiff.info need not precede them. tiff.info
            # stays part of the required cross-check set (verify_analysis enforces
            # it) and next_step_hint still proposes it first; this only stops the
            # hard refusal when the model runs them before reading metadata.
            self._check_image_reference(name, args, scene_path)
            if tiff is not None:
                self._bind_evidence(
                    name, args, "image_info", tiff, "tiff.info", ignore=("image_path",)
                )
            else:
                info = args.get("image_info")
                info = dict(info) if isinstance(info, dict) else {}
                if not str(info.get("image_path", "") or "").strip() and scene_path:
                    info["image_path"] = scene_path
                    _note_resolved("image_info", "request: image under analysis")
                args["image_info"] = info

        elif name == "region.locate":
            refine = recorded("seg.refine")
            if not _looks_like_refinement_result(refine):
                raise ToolPreconditionError(
                    "region.locate precondition unmet: no segmentation-derived candidate "
                    "regions are on record. seg.run and seg.refine must have produced a "
                    "valid result before a candidate region can be located."
                )
            self._bind_evidence(name, args, "refinement", refine, "seg.refine")
            self._bind_image_info(name, args, outputs, ctx, scene_path, required=False)

        elif name == "seg.refine":
            segmentation = recorded("seg.run")
            if segmentation is None:
                raise ToolPreconditionError(
                    "seg.refine precondition unmet: no segmentation output is on record. "
                    "seg.run must have produced a valid result before its mask can be refined."
                )
            self._bind_evidence(name, args, "segmentation", segmentation, "seg.run")
            self._bind_image_info(name, args, outputs, ctx, scene_path)
            if "regions" in args and not all(
                _looks_like_region_item(d) for d in (args.get("regions") or [])
            ):
                args.pop("regions", None)

        elif name == "seg.llm_review":
            refine = recorded("seg.refine")
            if not _looks_like_refinement_result(refine):
                raise ToolPreconditionError(
                    "seg.llm_review precondition unmet: no segmentation-guided refinement "
                    "context is on record. seg.refine must have produced a valid result "
                    "before its boundary can be reviewed."
                )
            self._bind_evidence(name, args, "refinement", refine, "seg.refine")
            self._bind_evidence(name, args, "segmentation", recorded("seg.run"), "seg.run")
            self._bind_evidence(
                name, args, "stage1", recorded("llm.first_pass"), "llm.first_pass"
            )
            self._bind_image_info(name, args, outputs, ctx, scene_path)

        elif name == "fuse.decision":
            if self.require_region_locate and recorded("region.locate") is None:
                raise ToolPreconditionError(
                    "fuse.decision precondition unmet: region.locate has not produced a "
                    "valid result, so the primary candidate's 3x3 frame position is not "
                    "on record and cannot enter the fused assessment."
                )
            for prereq, hint_fields in (
                ("llm.first_pass", ["stage1", "classification", "geo_context"]),
                ("seg.run", ["refinement/segmentation", "classification", "geo_context"]),
                ("seg.refine", ["refinement"]),
                ("vlm.describe", ["scene description (vlm.describe)"]),
                ("cls.run", ["classification", "classification.class_name"]),
                ("geo.nearby", ["geo_context.nearby.count", "geo_context.nearby.features"]),
                (
                    "geo.background",
                    [
                        "geo_context.background.terrain.slope_deg",
                        "geo_context.background.terrain.aspect_deg",
                        "geo_context.background.geology",
                    ],
                ),
            ):
                if recorded(prereq) is None:
                    raise ToolPreconditionError(
                        f"fuse.decision prerequisite missing: {prereq}. "
                        + self.fusion_argument_hint(hint_fields)
                    )

            for geo_name, is_background in (
                ("geo.background", True), ("geo.nearby", False)
            ):
                if self._geo_is_degraded(
                    recorded(geo_name), is_background=is_background
                ):
                    raise ToolPreconditionError(
                        f"fuse.decision precondition unmet: {geo_name} returned degraded "
                        "evidence; retry that tool until it succeeds or is recorded "
                        "as unavailable."
                    )

            self._bind_evidence(name, args, "stage1", recorded("llm.first_pass"), "llm.first_pass")
            self._bind_evidence(name, args, "segmentation", recorded("seg.run"), "seg.run")
            self._bind_evidence(name, args, "refinement", recorded("seg.refine"), "seg.refine")

            review = recorded("seg.llm_review")
            review_reason = self.mandatory_second_pass_reason(outputs)
            if review_reason and review is None:
                raise ToolPreconditionError(
                    "fuse.decision precondition unmet: a vision-language boundary "
                    f"re-check is mandatory for this case ({review_reason}), and "
                    "seg.llm_review has not produced a valid result."
                )
            second_pass = review.get("llm_second_pass") if review is not None else None
            self._bind_evidence(
                name,
                args,
                "llm_second_pass",
                second_pass if isinstance(second_pass, dict) else None,
                "seg.llm_review",
            )

            self._bind_evidence(name, args, "classification", recorded("cls.run"), "cls.run")
            # geo_context is a composite of two recorded results.
            self._bind_evidence(
                name,
                args,
                "geo_context",
                {"background": recorded("geo.background"), "nearby": recorded("geo.nearby")},
                "geo.background / geo.nearby",
            )

            gaps = unavailable_evidence(outputs)
            if gaps:
                args["unavailable_evidence"] = gaps
            missing = self.missing_fusion_requirements(args)
            if missing:
                raise ToolPreconditionError(self.fusion_argument_hint(missing))
            if not _looks_like_refinement_result(args.get("refinement")):
                raise ToolPreconditionError(
                    "fuse.decision precondition unmet: the recorded seg.refine result "
                    "contains no candidate regions."
                )

        elif name == "report.write":
            if ctx.report_written or recorded("report.write") is not None:
                raise ToolPreconditionError(
                    "report.write precondition unmet: this analysis already has a persisted "
                    "report; it is written once."
                )
            fused = recorded("fuse.decision")
            if fused is None:
                raise ToolPreconditionError(
                    "report.write precondition unmet: no fused assessment is on record "
                    "to persist."
                )
            try:
                self._bind_evidence(name, args, "report", fused, "fuse.decision")
            except ToolPreconditionError:
                raise ToolPreconditionError(
                    "report.write precondition unmet: the report persisted for this "
                    "analysis is the recorded fuse.decision assessment, and the supplied "
                    "`report` differs from it. A separately composed report cannot be "
                    "written; an omitted `report` is bound to the recorded assessment."
                ) from None
            if not str(args.get("out_path", "") or "").strip():
                img = str(tiff.get("image_path", "") or "").strip() if tiff else ""
                args["out_path"] = _default_report_out_path(img or ctx.image_path)
                _note_resolved("out_path", "default report location")

        return args

    def prepare_tool_call(
        self,
        name: str,
        raw_args: dict[str, Any] | None,
        outputs: dict[str, Any],
        ctx: "ToolCallContext",
        *,
        record: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Verify one tool call; see :meth:`_prepare_tool_call`.

        When ``record`` is given it is filled with the verification outcome of
        an admitted call: ``checked`` lists supplied arguments that were
        verified against their reference, ``resolved`` maps omitted arguments to
        the source they were taken from. A refused call raises instead.
        """
        token = _VERIFICATION.set(record)
        ledger_token = _LEDGER.set(outputs)
        try:
            return self._prepare_tool_call(name, raw_args, outputs, ctx)
        finally:
            _LEDGER.reset(ledger_token)
            _VERIFICATION.reset(token)

    def normalize_coordinates(self, latitude: Any, longitude: Any) -> tuple[float, float] | None:
        if latitude is None or longitude is None:
            return None
        return float(latitude), float(longitude)


DEFAULT_POLICY = LandslidePolicy.from_environment()


def get_policy(thresholds_path: str = "configs/thresholds.json") -> LandslidePolicy:
    return LandslidePolicy.from_environment(thresholds_path)


def format_rule_violations(violations: list[str]) -> str:
    """Render unmet deliverable obligations as a structured observation.

    The wording deliberately mirrors a violated tool precondition: it states
    what the execution contract still requires, never which tool to call next
    or in what order. Planning stays with the cognitive engine.
    """
    bullet_list = "\n- ".join(violations)
    return (
        "[contract] The deliverable contract for this analysis is not yet satisfied. "
        "Tool selection, argument construction, ordering and iteration remain yours; "
        "the obligations below are structural preconditions of a final landslide "
        "assessment and cannot be waived:\n- "
        + bullet_list
        + "\n\nA text-only final answer cannot resolve missing tool evidence. "
        "Issue a real tool call for an unmet obligation, observe its result, and only then "
        "give your final answer. Choose the tool and order yourself."
    )
