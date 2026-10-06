r"""Contract-constrained free agent for landslide analysis.

The *free LLM agent* and the *domain rule layer* live together here, wired as a
LangGraph state machine::

    START -> agent -> (tool_calls?) --yes--> tools -> agent
                                      |
                                      +-- fuse.decision OK --> critic -> END
                          |
                          no
                          v
                       critic --contract satisfied--> final
                          |  \--contract unmet, budget left--> agent
                          \----contract unmet, budget spent---> fallback

Rule control is enforced *declaratively*, as preconditions on tool invocation,
rather than procedurally through the prompt or through supervisory
intervention:

* task brief             -> ``policy.autonomous_task_contract`` (identical to
                            the brief the unconstrained agent receives)
* per-tool preconditions -> ``policy.prepare_tool_call`` via
                            :meth:`RuleGuidedAgent._guarded_execute`; an unmet
                            precondition becomes an ordinary tool observation
* deliverable contract   -> ``policy.verify_analysis`` (critic node), reported
                            as unmet obligations, never as a next-step command
* last-resort guarantee  -> fallback event -> caller-selected failure handling

The agent chooses every tool, argument, order and iteration itself. The rule
layer can only admit or refuse a requested call; it never selects, reorders,
substitutes, or executes a tool on the model's behalf. This keeps the
evidence-dependency graph of the final assessment non-bypassable while leaving
planning autonomy intact, and makes the emitted trace an accurate record of the
model's own decisions.

The turn budget (``AGENT_MAX_TURNS``) and repeated identical contract checks
(``AGENT_MAX_IDENTICAL_CRITIC_CHECKS``) emit a ``fallback`` event and the caller decides (deterministic pipeline vs. error via
``AGENT_FALLBACK_TO_GRAPH``).

The module never imports the FastAPI layer.  The caller injects:

* ``model_fn(messages, tools) -> {"message": <assistant msg>, "raw": <str>}``

and consumes a stream of frontend-ready event dicts.  The terminal event is
either ``{"type": "final_core", "data": {...}}`` or
``{"type": "fallback", "reason": <str>}`` (the caller then runs its
deterministic pipeline and presents the result).
"""
from __future__ import annotations

import json
import hashlib
from dataclasses import replace
import operator
import os
import time
from pathlib import Path
from typing import Annotated, Any, Callable, Iterator, TypedDict
from uuid import uuid4

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from src.agent.controller import (
    LandslidePolicy,
    ToolCallContext,
    ToolPreconditionError,
    DEGRADABLE_TOOLS,
    format_rule_violations,
    get_policy,
    is_unavailable,
    max_tool_failures,
    unavailable_placeholder,
)
from src.agent.default_server import create_default_server
from src.pipelines.report_writer import free_report_request
from src.orchestration.evidence import record_output

ModelFn = Callable[[list[dict[str, Any]], list[dict[str, Any]]], dict[str, Any]]


# --------------------------------------------------------------------------- #
# small local helpers (ported so the module is self-contained)
# --------------------------------------------------------------------------- #
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


def _default_report_out_path(image_path: str | None = None) -> str:
    stem = Path(str(image_path or "")).stem.strip() if image_path else ""
    base = stem or "landslide_report"
    return str(Path("outputs") / "reports" / f"{base}_{uuid4().hex[:8]}.json")


def _tool_ok(result: Any) -> bool:
    return isinstance(result, dict) and not result.get("error")


# Arguments the contract layer binds from the evidence ledger or the request
# when they are omitted.
_LEDGER_BOUND_ARGS: frozenset[str] = frozenset(
    {
        "stage1",
        "segmentation",
        "refinement",
        "classification",
        "geo_context",
        "llm_second_pass",
        "image_info",
        "report",
        "out_path",
        "lat",
        "lon",
        "radius",
    }
)


def _retry_hint(name: str, args: dict[str, Any], error: str) -> str:
    """Concrete re-issue template for a call refused over a bound argument.

    A small model reads "an omitted argument is bound" and still repeats the
    same hand-copied evidence. The template only drops arguments the layer
    would bind anyway; the model must still re-issue the call itself.
    """
    if "omitted argument is bound" not in error:
        return ""
    kept = {k: v for k, v in (args or {}).items() if k not in _LEDGER_BOUND_ARGS}
    dropped = sorted(k for k in (args or {}) if k in _LEDGER_BOUND_ARGS)
    if not dropped:
        return ""
    call = json.dumps({"name": name, "arguments": kept}, ensure_ascii=False)
    return (
        f"Do not copy evidence into arguments. Omit {', '.join(dropped)}; the recorded "
        f"results are bound automatically. Re-issue the call as: {call}"
    )


# --------------------------------------------------------------------------- #
# graph state
# --------------------------------------------------------------------------- #
class _AgentState(TypedDict, total=False):
    messages: Annotated[list[dict[str, Any]], operator.add]
    events: Annotated[list[dict[str, Any]], operator.add]
    outputs: dict[str, Any]
    turns: int
    critic_retries: int
    critic_signature: str
    stalled_critic_checks: int
    verdict: str            # "continue" | "final" | "fallback"
    final_message: dict[str, Any]
    fallback_reason: str
    paused_reason: str       # non-empty => agent stopped to ask the caller something


# tools that are safe to serve from cache instead of re-running on a repeat call
_IDEMPOTENT_TOOLS: frozenset[str] = frozenset(
    {
        "tiff.info",
        "llm.first_pass",
        "seg.run",
        "seg.refine",
        "cls.run",
        "geo.background",
        "geo.nearby",
        "seg.llm_review",
        "region.locate",
    }
)


def _last_narration(messages: list[dict[str, Any]]) -> str:
    """Whitespace-normalised text of the most recent narration-only assistant turn.

    Narration-only = an assistant message carrying text but no tool calls. Tool
    turns in between do not reset it: a stalled model that repeats the same
    closing line either side of an auto-advanced tool is still repeating.
    """
    for message in reversed(messages or []):
        if message.get("role") != "assistant" or (message.get("tool_calls") or []):
            continue
        text = " ".join(str(message.get("content") or "").split())
        if text:
            return text
    return ""


def _truncate_for_model(text: str, limit: int) -> str:
    text = str(text)
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def _hydrate_outputs_from_messages(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Rebuild the successful tool ledger when a follow-up request resumes from chat history."""
    outputs: dict[str, Any] = {}
    for message in messages or []:
        if message.get("role") != "tool" or not message.get("name"):
            continue
        raw = message.get("content", "")
        if isinstance(raw, dict):
            value = raw
        else:
            text = str(raw or "").strip()
            try:
                value = json.loads(text)
            except Exception:
                # Cached tool messages may contain a short explanatory prefix.
                value = None
                for marker in ("{", "["):
                    pos = text.find(marker)
                    if pos < 0:
                        continue
                    try:
                        value = json.JSONDecoder().raw_decode(text[pos:])[0]
                        break
                    except Exception:
                        pass
        if isinstance(value, dict):
            verification = value.get("verification")
            if value.get("not_executed") or (isinstance(verification, dict) and verification.get("status") == "refused"):
                continue
            result = {k: v for k, v in value.items() if k != "verification"}
            record_output(outputs, str(message.get("name")), result)
    return {name: result for name, result in outputs.items() if _tool_ok(result)}


# --------------------------------------------------------------------------- #
# the agent
# --------------------------------------------------------------------------- #
FREE_REPORT_REQUEST = (
    "Write your final landslide assessment report now. Use exactly these ten fields, in this "
    "order, each as '**<Field>:** <text>': Landslide presence; Landslide type; Image relative "
    "position within the image frame; Morphological characteristics; Material composition and "
    "surface cover; Movement and deformation features; Surrounding environmental context; Impact "
    "on human infrastructure; Reason for landslide classification; Landslide causation inference. "
    "Begin the Landslide presence field with Yes or No, and the Landslide type field with the type "
    "name (or Undetermined / Not applicable)."
)


def _without_images(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy of the conversation with image parts removed: the report is written from
    the text record (tool results and the agent's own messages), not a second look."""
    out = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            c = [p for p in c if not (isinstance(p, dict) and p.get("type") in ("image", "image_url"))]
            m = dict(m, content=c or "")
        out.append(m)
    return out


class RuleGuidedAgent:
    def __init__(
        self,
        *,
        model_fn: ModelFn,
        thresholds_path: str = "configs/thresholds.json",
        enable_second_pass: bool = False,
        report_write_required: bool = False,
        image_path: str = "",
        latitude: float | None = None,
        longitude: float | None = None,
        nearby_radius: int | None = None,
        review_threshold: float | None = None,
        lazy_review_threshold: bool = False,
        max_turns: int | None = None,
        initial_turns: int = 0,
        initial_outputs: dict[str, Any] | None = None,
        initial_tool_inputs: dict[str, Any] | None = None,
        initial_tool_failures: dict[str, int] | None = None,
        critic_max_retries: int | None = None,
        react_mode: bool = True,
        enforce_contract: bool = True,
        loop_hygiene: bool = True,
    ) -> None:
        self.model_fn = model_fn
        self.thresholds_path = thresholds_path
        self.policy: LandslidePolicy = get_policy(thresholds_path)
        if review_threshold is not None:
            self.policy = replace(self.policy, tiny_area_review_threshold=float(review_threshold))
        self.image_path = str(image_path or "")
        self.latitude = latitude
        self.longitude = longitude
        # None => the OSM probe radius has not been chosen yet; the agent
        # pauses right before geo.background/geo.nearby and asks the caller.
        self.nearby_radius = None if nearby_radius is None else int(nearby_radius)
        self.lazy_review_threshold = bool(lazy_review_threshold)
        self.review_threshold = review_threshold
        self.report_write_required = bool(report_write_required)
        # Two orthogonal switches.
        #
        # ``enforce_contract`` is the experimental variable: it turns the domain
        # rule layer on or off (tool preconditions in ``prepare_tool_call`` plus
        # the deliverable contract checked by the critic). With it off the agent
        # is an unconstrained ReAct agent over the same tools and the same prompt.
        #
        # ``loop_hygiene`` is a resource control, not a domain rule: duplicate
        # suppression within a turn, idempotent-result reuse across turns, and
        # the post-fusion cut-off. It must be held constant across the two arms,
        # otherwise differences in tool-call counts, latency and cost are
        # attributable to caching rather than to rule enforcement.
        self.enforce_contract = bool(enforce_contract)
        self.loop_hygiene = bool(loop_hygiene)
        # The contract only applies to image analysis; plain chat is unconstrained.
        self.rule_mode = bool(self.image_path) and self.enforce_contract
        # Per-run count of backend failures of degradable tools (rule mode only).
        self._has_initial_outputs = initial_outputs is not None
        self._initial_outputs = dict(initial_outputs or {})
        self._initial_tool_inputs = dict(initial_tool_inputs or {})
        self._initial_tool_failures = dict(initial_tool_failures or {})
        self._tool_failures: dict[str, int] = {}
        self._cache_args: dict[str, str] = {}
        self.react_mode = bool(react_mode)

        server = create_default_server(
            thresholds_path,
            enable_seg_llm_second_pass=enable_second_pass,
            review_threshold=review_threshold,
        )
        self.registry = server.registry
        self.tools = self._openai_tools(report_write_required=self.report_write_required)

        # A complete, verified fusion/report is the success terminal. The turn
        # budget and the identical-critic-check limit bound unproductive runs.
        self.max_turns = int(
            max_turns if max_turns is not None else os.getenv("AGENT_MAX_TURNS", "30") or "30"
        )
        if self.max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        self.initial_turns = min(self.max_turns, max(0, int(initial_turns or 0)))
        try:
            self.max_identical_critic_checks = max(
                0, int(os.getenv("AGENT_MAX_IDENTICAL_CRITIC_CHECKS", "4") or "4")
            )
        except ValueError:
            self.max_identical_critic_checks = 4
        # Kept for backward compatibility / event labelling only.  It no longer
        # triggers an early fallback -- ``max_turns`` does.
        self.critic_max_retries = int(
            critic_max_retries
            if critic_max_retries is not None
            else os.getenv("AGENT_CRITIC_MAX_RETRIES", "0") or "0"
        )
        # Cap the size of a tool result fed back to the model (full result is
        # still kept in ``outputs`` for the rule logic).
        self.tool_result_char_limit = int(
            os.getenv("AGENT_TOOL_RESULT_CHARS", "6000") or "6000"
        )
        self.graph = self._build_graph()

    # ------------------------------------------------------------------ #
    def _tools_for_model(self, outputs: dict[str, Any]) -> list[dict[str, Any]]:
        """Expose the complete tool catalogue on every model turn.

        Fusion remains input-dependent: ``prepare_tool_call`` assembles prior
        evidence and returns a structured prerequisite error when it is missing.
        Keeping ``fuse.decision`` visible lets the model choose when to attempt
        fusion and observe that error through the normal tool loop.
        """
        return list(self.tools)

    def _enforce_tool_gate(self, message: dict[str, Any], outputs: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
        """Preserve model calls; enforce prerequisites at tool execution."""
        # Do not silently remove early fuse.decision calls.  _tools_node routes
        # every call through prepare_tool_call, which emits a structured error
        # (or auto-assembles arguments) back to the next model turn.
        return message

    def _next_required_tool(self, outputs: dict[str, Any]) -> str | None:
        """Delegate to the single rule module for the required-order decision."""
        return self.policy.next_required_tool(
            outputs, require_report_write=self.report_write_required
        )

    # ------------------------------------------------------------------ #
    # public
    # ------------------------------------------------------------------ #
    def stream(self, messages: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        """Run the agent and yield frontend event dicts.

        The final yielded event is either ``{"type": "final_core", ...}`` or
        ``{"type": "fallback", "reason": ...}``.
        """
        # No workflow is seeded: the task brief supplied by the caller is the
        # same non-prescriptive brief the unconstrained agent receives.
        seeded = list(messages)
        # A supplied trace is a complete replayed ledger, including removals.
        # Merging it into history would resurrect evidence invalidated in the
        # trace (failed attempts and newer upstream results may be truncated
        # or absent from model-visible messages).
        hydrated_outputs = (dict(self._initial_outputs) if self._has_initial_outputs
                            else _hydrate_outputs_from_messages(seeded))
        self._tool_failures = dict(self._initial_tool_failures)
        self._cache_args = {
            name: self._args_key(args)
            for name, args in self._initial_tool_inputs.items()
            if _tool_ok(hydrated_outputs.get(name))
        }
        initial_events: list[dict[str, Any]] = []
        if self.rule_mode and (self.latitude is None or self.longitude is None):
            # Without a georeference the geo tools cannot run at all; record them
            # as unavailable up front instead of letting the contract block fusion.
            for geo_tool in ("geo.background", "geo.nearby"):
                if not _tool_ok(hydrated_outputs.get(geo_tool)):
                    placeholder = unavailable_placeholder(geo_tool, "no coordinates were supplied with the request")
                    hydrated_outputs[geo_tool] = placeholder
                    initial_events.append({"type": "tool_result", "data": {
                        "tool": geo_tool, "status": "ok", "execution_state": "declared_unavailable",
                        "cached": False, "cost_ms": 0, "input": {}, "output": placeholder,
                    }})
        state: _AgentState = {
            "messages": seeded,
            "events": initial_events,
            "outputs": hydrated_outputs,
            "turns": self.initial_turns,
            "critic_retries": 0,
        }

        acc: dict[str, Any] = {
            "verdict": "",
            "final_message": {},
            "fallback_reason": "",
            "paused_reason": "",
            "critic_retries": 0,
            "turns": self.initial_turns,
            "outputs": dict(hydrated_outputs),
            "messages": list(seeded),
        }
        # Each model turn costs up to ~3 graph supersteps (agent -> tools/critic
        # -> agent).  Give the recursion limit enough headroom for the full
        # ``max_turns`` budget plus the terminal critic pass.
        recursion_limit = max(24, self.max_turns * 4 + 12)
        recursion_exhausted = False
        # Surface evidence the contract recorded as unavailable before any turn.
        for ev in initial_events:
            yield ev
        try:
            for chunk in self.graph.stream(
                state, stream_mode="updates", config={"recursion_limit": recursion_limit}
            ):
                for _node, delta in (chunk or {}).items():
                    delta = delta or {}
                    for ev in delta.get("events") or []:
                        yield ev
                    if "outputs" in delta:
                        acc["outputs"] = delta["outputs"]
                    if "turns" in delta:
                        acc["turns"] = int(delta.get("turns") or 0)
                    if delta.get("messages"):
                        acc["messages"].extend(delta["messages"])
                    for key in ("verdict", "final_message", "fallback_reason",
                                "critic_retries", "paused_reason"):
                        if delta.get(key):
                            acc[key] = delta[key]
        except GraphRecursionError:
            # The graph's own safety net fired before the turn budget did. Both
            # arms must end in a comparable, inspectable state, so this becomes
            # an ordinary fallback rather than an exception escaping to the
            # caller (and, for the sync endpoint, to the HTTP layer).
            recursion_exhausted = True

        if recursion_exhausted:
            yield {
                "type": "fallback",
                "turns_used": min(self.max_turns, int(acc.get("turns", self.initial_turns) or 0)),
                "reason": (
                    f"graph recursion limit ({recursion_limit}) reached "
                    "before the turn budget"
                ),
                "outputs": acc.get("outputs") or {},
            }
            return

        turns_used = min(self.max_turns, int(acc.get("turns", self.initial_turns) or 0))
        if acc.get("paused_reason") == "nearby_radius":
            yield {"type": "need_nearby_radius", "turns_used": turns_used}
            return
        if acc.get("paused_reason") == "review_threshold":
            yield {"type": "need_review_threshold", "turns_used": turns_used}
            return

        if acc["verdict"] == "fallback":
            # Carry the evidence ledger on the failure path too: a run that did
            # not satisfy the contract still has to be scorable against it.
            yield {
                "type": "fallback",
                "reason": acc["fallback_reason"] or "task rules not satisfied",
                "turns_used": turns_used,
                "outputs": acc.get("outputs") or {},
            }
            return

        final_message = acc["final_message"] or {"role": "assistant", "content": ""}
        yield {
            "type": "final_core",
            "data": {
                "message": final_message,
                "history": acc["messages"],
                "outputs": acc["outputs"],
                "turns_used": turns_used,
                "mode": "agent",
                "critic": {
                    "retries": acc.get("critic_retries", 0),
                    "degraded": False,
                    "unmet_rules": [],
                },
            },
        }

    # ------------------------------------------------------------------ #
    # graph wiring
    # ------------------------------------------------------------------ #
    def _build_graph(self):
        g = StateGraph(_AgentState)
        g.add_node("agent", self._agent_node)
        g.add_node("tools", self._tools_node)
        g.add_node("critic", self._critic_node)
        g.add_node("finalize", self._finalize_node)
        g.add_edge(START, "agent")
        g.add_conditional_edges(
            "agent",
            self._route_after_agent,
            {"tools": "tools", "critic": "critic", "finalize": "finalize", "pause": END, "fallback": END},
        )
        g.add_conditional_edges(
            "tools",
            self._route_after_tools,
            {"agent": "agent", "critic": "critic", "finalize": "finalize", "pause": END},
        )
        g.add_edge("finalize", END)
        g.add_conditional_edges(
            "critic",
            self._route_after_critic,
            {"agent": "agent", "final": END, "fallback": END},
        )
        return g.compile()

    def _route_after_agent(self, state: _AgentState) -> str:
        if state.get("verdict") == "fallback":
            return "fallback"
        if state.get("paused_reason"):
            return "pause"
        last = state["messages"][-1] if state.get("messages") else {}
        if last.get("tool_calls"):
            return "tools"
        if not self.enforce_contract:
            # Without a deliverable contract a narration-only turn is the
            # model's own terminal choice; there is nothing left to verify.
            return "finalize"
        return "critic"

    def _route_after_tools(self, state: _AgentState) -> str:
        if state.get("paused_reason"):
            return "pause"
        if not self.enforce_contract:
            # The observation is appended; the model decides what follows. Calling
            # fuse.decision is the model's own "done, write the report" signal.
            if _tool_ok((state.get("outputs") or {}).get("fuse.decision")) and os.getenv("FUSE_LENIENT", "0") in ("1", "true", "True"):
                return "finalize"
            return "agent"
        outputs = state.get("outputs") or {}
        if not _tool_ok(outputs.get("fuse.decision")):
            return "agent"
        if self.rule_mode and self.policy.fuse_decision_is_stale(outputs):
            # fuse.decision ran before seg.llm_review -> recompute before finishing.
            return "agent"
        # The critic persists the report and verifies every final obligation.
        return "critic"

    def _route_after_critic(self, state: _AgentState) -> str:
        return {"continue": "agent", "final": "final", "fallback": "fallback"}.get(
            state.get("verdict", "final"), "final"
        )

    # ------------------------------------------------------------------ #
    # nodes
    # ------------------------------------------------------------------ #
    def _agent_node(self, state: _AgentState) -> _AgentState:
        # The confirmed OSM radius is an argument the contract layer can supply,
        # not a queued action: after a radius pause the model re-issues geo.nearby
        # itself. Nothing is executed here on its behalf.
        model_messages = list(state["messages"])
        if self.react_mode:
            model_messages = [
                {
                    "role": "system",
                    "content": (
                        "[mode: react] Work as a ReAct agent: reason privately about the "
                        "user goal and available evidence, choose an action only when a tool "
                        "would help, observe each tool result, then reassess. You may call any "
                        "available tool with any arguments in any order, repeat a tool when useful, "
                        "or answer directly when you judge the work complete. Do not expose private "
                        "chain-of-thought; provide only concise, user-facing reasoning in your final answer."
                    ),
                }
            ] + model_messages

        turns = int(state.get("turns", 0)) + 1
        if turns > self.max_turns:
            # Stop asking the model; let the critic decide (clean -> final,
            # unmet -> fallback).
            return {
                "turns": self.max_turns,
                "events": [
                    {
                        "type": "model_raw",
                        "source": "agent",
                        "content": f"[agent] turn budget ({self.max_turns}) reached; verifying result.",
                    }
                ],
                "messages": [{"role": "assistant", "content": ""}],
            }

        try:
            result = self.model_fn(model_messages, self._tools_for_model(state.get("outputs") or {}))
            if not isinstance(result, dict) or not isinstance(result.get("message"), dict):
                raise ValueError("model response must contain an assistant message object")
            message = dict(result["message"])
            if message.get("role", "assistant") != "assistant":
                raise ValueError("model response role must be assistant")
            message.setdefault("role", "assistant")
            calls = message.get("tool_calls")
            if calls is not None:
                if not isinstance(calls, list):
                    raise ValueError("model tool_calls must be a list")
                ids = set()
                for call in calls:
                    if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                        raise ValueError("model tool call must contain a function object")
                    name, call_id = call["function"].get("name"), call.get("id")
                    if not isinstance(name, str) or not name.strip():
                        raise ValueError("model tool call name must be a nonempty string")
                    if not isinstance(call_id, str) or not call_id.strip() or call_id in ids:
                        raise ValueError("model tool call ids must be nonempty and unique")
                    ids.add(call_id)
        except Exception as exc:
            return {
                "turns": turns,
                "verdict": "fallback",
                "fallback_reason": f"agent model call failed: {exc}",
                "outputs": dict(state.get("outputs") or {}),
            }
        raw = str(result.get("raw", "") or "")
        events: list[dict[str, Any]] = []
        if raw.strip():
            events.append({"type": "model_raw", "source": "agent", "content": raw})
        content = str(message.get("content", "") or "")
        message = self._enforce_tool_gate(message, state.get("outputs") or {}, events)
        content = str(message.get("content", "") or "")
        tool_calls = message.get("tool_calls") or []
        # A stalled model re-emits its closing line every critic round-trip. The
        # first one is worth showing; the verbatim repeats are noise in the chat
        # transcript, so route them to the token stream instead.
        repeated_narration = (
            bool(content.strip())
            and not tool_calls
            and " ".join(content.split()) == _last_narration(state.get("messages") or [])
        )
        if repeated_narration:
            events.append(
                {
                    "type": "model_raw",
                    "source": "agent-repeat",
                    "content": content,
                }
            )
        else:
            events.append(
                {
                    "type": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                }
            )
        return {"messages": [message], "events": events, "turns": turns}

    @staticmethod
    def _args_key(args: dict[str, Any]) -> str:
        canonical = json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _defer_calls(self, calls, reason, messages, events) -> None:
        """Resolve unexecuted calls without interpreting or executing the plan."""
        for call in calls:
            fn = call.get("function") or {}
            name = fn.get("name", "")
            result = {"error": reason, "not_executed": True}
            messages.append({
                "role": "tool", "tool_call_id": call.get("id", ""),
                "name": name, "content": json.dumps(result, ensure_ascii=False),
            })
            events.append({"type": "tool_result", "data": {
                "tool": name, "status": "skipped", "execution_state": "deferred",
                "cached": False, "cost_ms": 0, "input": {}, "output": result,
            }})

    def _tools_node(self, state: _AgentState) -> _AgentState:
        last = state["messages"][-1]
        outputs = dict(state.get("outputs") or {})
        events: list[dict[str, Any]] = []
        new_messages: list[dict[str, Any]] = []
        seen_calls: dict[str, tuple[Any, str, str, Any]] = {}
        calls = last.get("tool_calls") or []

        for index, call in enumerate(calls):
            fn = call.get("function", {}) or {}
            name = fn.get("name", "")
            raw_args = fn.get("arguments", "{}")
            args = {}
            parse_error = ""
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                if not isinstance(args, dict):
                    raise ValueError("arguments must be a JSON object")
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                parse_error = f"invalid tool arguments: {exc}"
                args = {}

            if (not parse_error and name == "geo.nearby"
                    and self.nearby_radius is None
                    and self.latitude is not None and self.longitude is not None):
                self._defer_calls(calls[index:],
                                 "Run paused for the user-confirmed nearby radius; re-issue the call after resume.",
                                 new_messages, events)
                return {"messages": new_messages, "events": events, "outputs": outputs,
                        "paused_reason": "nearby_radius"}

            # An earlier refusal/result is reusable only while its evidence
            # context is unchanged. Later calls in the same batch may have
            # satisfied a missing prerequisite or invalidated a derived result.
            context = {key: value for key, value in outputs.items()
                       if key != name or name == "report.write"}
            call_key = name + "|" + self._args_key({"arguments": args, "context": context})
            used_args = args
            verification = None
            note = ""
            cached_hit = False
            cost_ms = 0
            if parse_error:
                result, status, execution_state = {"error": parse_error}, "error", "refused"
            elif self.loop_hygiene and call_key in seen_calls:
                result, status, prior_state, verification = seen_calls[call_key]
                execution_state = "reused" if status == "ok" else prior_state
                cached_hit = True
                note = f"{name} was already called with identical arguments in this turn; result unchanged."
            else:
                t0 = time.perf_counter()
                try:
                    cached = outputs.get(name)
                    degraded_geo = (self.rule_mode and name in ("geo.background", "geo.nearby")
                                    and _tool_ok(cached)
                                    and self.policy._geo_is_degraded(cached, is_background=(name == "geo.background")))
                    can_reuse = self.loop_hygiene and name in _IDEMPOTENT_TOOLS and _tool_ok(cached) and not degraded_geo
                    if can_reuse:
                        # Cached observations are still tool invocations: validate
                        # every model-supplied reference before admitting them.
                        used_args, verification = self._prepare_args(name, args, outputs)
                        previous_key = self._cache_args.get(name)
                        if previous_key is None and self.enforce_contract:
                            default_args, _ = self._prepare_args(name, {}, outputs)
                            previous_key = self._args_key(default_args)
                        can_reuse = previous_key == self._args_key(used_args)
                    if can_reuse:
                        result, status, execution_state = cached, "ok", "reused"
                        cached_hit = True
                        note = (f"{name} is recorded as unavailable ({cached.get('reason', '')}); do not retry it."
                                if is_unavailable(cached) else f"{name} already completed successfully; result unchanged.")
                    else:
                        result, used_args, verification = self._guarded_execute(name, args, outputs)
                        status = "ok" if _tool_ok(result) else "error"
                        execution_state = "completed" if status == "ok" else "failed"
                except ToolPreconditionError as exc:
                    result, used_args, status = {"error": str(exc)}, args, "error"
                    hint = _retry_hint(name, args, str(exc))
                    if hint:
                        result["retry"] = hint
                    execution_state = "refused"
                    verification = {"status": "refused"}
                except Exception as exc:
                    result, used_args, status = {"error": str(exc)}, args, "error"
                    execution_state = "failed"
                cost_ms = int((time.perf_counter() - t0) * 1000)
                degraded_geo = (self.rule_mode and name in ("geo.background", "geo.nearby")
                                and status == "ok" and not cached_hit
                                and self.policy._geo_is_degraded(result, is_background=(name == "geo.background")))
                if degraded_geo:
                    status, execution_state = "degraded", "degraded"
                if self.rule_mode and name in DEGRADABLE_TOOLS and execution_state in ("failed", "degraded"):
                    self._tool_failures[name] = self._tool_failures.get(name, 0) + 1
                    if self._tool_failures[name] >= max_tool_failures():
                        issue = (result.get("error") or result.get("source_status") or result.get("warnings")
                                 or "degraded result") if isinstance(result, dict) else result
                        result = unavailable_placeholder(name, f"failed or degraded {self._tool_failures[name]} times ({str(issue)[:160]})")
                        self._record_output(outputs, name, result)
                        self._cache_args[name] = self._args_key(used_args)
                        status, execution_state = "degraded", "declared_unavailable"
                        note = f"{name} was retried and is now recorded as unavailable; continue without inventing this evidence."
                elif execution_state == "completed" and name in DEGRADABLE_TOOLS:
                    self._tool_failures.pop(name, None)

            if not parse_error:
                seen_calls[call_key] = result, status, execution_state, verification
            trace = {"tool": name, "status": status, "execution_state": execution_state,
                     "cached": cached_hit, "cost_ms": cost_ms, "input": used_args,
                     "output": result if isinstance(result, dict) else {"value": result}}
            if verification is not None:
                trace["verification"] = verification
            events.append({"type": "tool_call", "name": name, "arguments": used_args})
            events.append({"type": "tool_result", "data": trace})
            observed = result
            if verification is not None:
                observed = {"verification": verification}
                if isinstance(result, dict):
                    observed.update({k: v for k, v in result.items() if k != "verification"})
                else:
                    observed["result"] = result
            tool_text = json.dumps(observed, ensure_ascii=False, default=str)
            if note:
                tool_text = note + "\n" + tool_text
            new_messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "name": name,
                                 "content": _truncate_for_model(tool_text, self.tool_result_char_limit)})
            if name == "seg.refine" and status == "ok" and self.lazy_review_threshold and self.review_threshold is None:
                self._defer_calls(calls[index + 1:],
                                 "Run paused for the review threshold; re-issue the call after resume.",
                                 new_messages, events)
                return {"messages": new_messages, "events": events, "outputs": outputs,
                        "paused_reason": "review_threshold"}
            if self.loop_hygiene and name == "fuse.decision" and status == "ok":
                self._defer_calls(calls[index + 1:],
                                 "Fusion completed; this remaining call was not executed.", new_messages, events)
                break

        return {"messages": new_messages, "events": events, "outputs": outputs}

    def _finalize_node(self, state: _AgentState) -> _AgentState:
        outputs = dict(state.get("outputs") or {})
        events: list[dict[str, Any]] = []
        messages: list[dict[str, Any]] = []
        if not self.enforce_contract:
            # No deliverable contract: the model's own closing turn is the
            # result, and nothing is written on its behalf.
            text = ""
            for message in reversed(state.get("messages") or []):
                if message.get("role") == "assistant" and not (message.get("tool_calls") or []):
                    text = str(message.get("content") or "").strip()
                    if text:
                        break
            if os.getenv("FUSE_LENIENT", "0") in ("1", "true", "True") and self.image_path:
                # The agent writes its final report itself, with the image and all
                # of its own observations in context. Only the deliverable format is
                # stated; what to conclude is entirely the agent's.
                try:
                    result = self.model_fn(
                        _without_images(list(state.get("messages") or []))
                        + [{"role": "user", "content": free_report_request()}],
                        [],
                    )
                    own = str(((result or {}).get("message") or {}).get("content") or "").strip()
                    if own:
                        text = own
                except Exception as exc:  # the run's result stands without the report turn
                    events.append({"type": "model_raw", "source": "agent", "content": f"[agent] report turn failed: {exc}"})
            text = text or "Agent stopped without a final written conclusion."
            return {
                "outputs": outputs,
                "messages": [{"role": "assistant", "content": text}],
                "events": events,
                "verdict": "final",
                "final_message": {"role": "assistant", "content": text},
            }
        events.extend(self._persist_report(outputs))
        fuse = outputs.get("fuse.decision")
        text = str(fuse.get("final_description") or fuse.get("summary") or "Analysis complete.").strip() if isinstance(fuse, dict) else "Analysis complete."
        messages.append({"role":"assistant","content":text})
        return {"outputs":outputs,"messages":messages,"events":events,"verdict":"final","final_message":{"role":"assistant","content":text}}

    def _critic_node(self, state: _AgentState) -> _AgentState:
        """Terminal contract check.

        Runs only when the model has chosen to stop. Apart from persisting an
        already-fused report (see ``_persist_report``) it never selects, orders,
        repairs or executes a tool: it either accepts the run or returns the
        outstanding obligations to the cognitive engine as a structured
        observation, in the same shape as a violated tool precondition.
        """
        outputs = dict(state.get("outputs") or {})
        persisted = self._persist_report(outputs)
        if (
            self.rule_mode
            and self.report_write_required
            and _tool_ok(outputs.get("fuse.decision"))
            and not self.policy.fuse_decision_is_stale(outputs)
            and not _tool_ok(outputs.get("report.write"))
        ):
            report_result = outputs.get("report.write") or {}
            detail = str(report_result.get("error") or "report persistence failed")
            return {
                "verdict": "fallback",
                "fallback_reason": f"report.write failed: {detail}",
                "outputs": outputs,
                "events": persisted,
            }
        violations = (
            self.policy.verify_analysis(
                outputs,
                require_report_write=self.report_write_required,
                geo_expected=(self.latitude is not None and self.longitude is not None),
            )
            if self.rule_mode
            else []
        )

        if not violations:
            fuse = outputs.get("fuse.decision")
            text = ""
            if isinstance(fuse, dict):
                text = str(fuse.get("final_description", "") or "").strip()
            if not text:
                last_assistant = ""
                for msg in reversed(state.get("messages") or []):
                    if msg.get("role") == "assistant" and not (msg.get("tool_calls") or []):
                        last_assistant = str(msg.get("content", "") or "").strip()
                        break
                text = last_assistant or "Analysis complete."
            return {
                "verdict": "final",
                "final_message": {"role": "assistant", "content": text},
                "outputs": outputs,
                "events": persisted,
            }

        # The deliverable contract is unmet. Return actionable obligations to
        # the model; repeated identical checks stop early to avoid empty loops.
        turns = int(state.get("turns", 0))
        retries = int(state.get("critic_retries", 0))
        signature = hashlib.sha256(json.dumps(
            {"violations": violations, "outputs": outputs},
            sort_keys=True, ensure_ascii=False, default=str,
        ).encode("utf-8")).hexdigest()
        same_checks = (
            int(state.get("stalled_critic_checks", 0)) + 1
            if signature == state.get("critic_signature")
            else 1
        )

        if (
            self.max_identical_critic_checks
            and same_checks >= self.max_identical_critic_checks
        ):
            reason = (
                f"no progress after {same_checks} identical contract checks; "
                + "; ".join(violations)
            )
            return {
                "verdict": "fallback",
                "fallback_reason": reason,
                "outputs": outputs,
                "events": persisted + [{
                    "type": "model_raw", "source": "critic",
                    "content": f"[contract] {reason}",
                }],
                "critic_signature": signature,
                "stalled_critic_checks": same_checks,
            }

        if turns >= self.max_turns:
            return {
                "verdict": "fallback",
                "fallback_reason": "; ".join(violations),
                "outputs": outputs,
                "events": persisted,
            }

        return {
            "outputs": outputs,
            "verdict": "continue",
            "critic_retries": retries + 1,
            "critic_signature": signature,
            "stalled_critic_checks": same_checks,
            "messages": [
                {"role": "system", "content": format_rule_violations(violations)}
            ],
            "events": persisted + [
                {
                    "type": "model_raw",
                    "source": "critic",
                    "content": (
                        f"[contract check {retries + 1}; turn {turns}/{self.max_turns}] "
                        "unsatisfied obligations:\n- " + "\n- ".join(violations)
                    ),
                }
            ],
        }

    def _persist_report(self, outputs: dict[str, Any]) -> list[dict[str, Any]]:
        """Write the recorded fusion to disk once it is final.

        ``report.write`` only serialises the recorded ``fuse.decision`` result to
        a default path, and the model does not reliably issue it after fusion
        (it narrates the report instead until the turn budget runs out). The
        runtime therefore performs this single step itself, through the same
        contract-checked path, and marks it in the trace.
        """
        if not (self.report_write_required and self.rule_mode):
            return []
        if _tool_ok(outputs.get("report.write")) or not _tool_ok(outputs.get("fuse.decision")):
            return []
        if self.policy.fuse_decision_is_stale(outputs):
            return []
        name = "report.write"
        t0 = time.perf_counter()
        try:
            result, used_args, verification = self._guarded_execute(name, {}, outputs)
            status = "ok" if _tool_ok(result) else "error"
        except Exception as exc:
            result, used_args, verification, status = {"error": str(exc)}, {}, None, "error"
        # Retain the path internally for the API, while keeping this out of the
        # model-visible tool trace and tool-call metric.
        outputs["report.write"] = result
        trace = {
            "name": "report.persist",
            "status": status,
            "execution_state": "system_persisted" if status == "ok" else "failed",
            "cost_ms": int((time.perf_counter() - t0) * 1000),
            "report_path": (result or {}).get("report_path", "") if isinstance(result, dict) else "",
            "invoked_by": "system",
        }
        return [{"type": "system_persist", "data": trace}]

    # ------------------------------------------------------------------ #
    # rule control: declarative tool preconditions
    # ------------------------------------------------------------------ #
    def _guarded_execute(
        self, name: str, raw_args: dict[str, Any], outputs: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
        args, verification = self._prepare_args(name, raw_args, outputs)
        # Refusals happen above and preserve valid evidence. Once the backend
        # is attempted, exceptions and invalid responses are failed observations
        # and must invalidate the prior result just like returned error objects.
        try:
            result = self.registry.call_tool(name, args)
            if not isinstance(result, dict):
                raise TypeError(f"{name} returned an invalid result; expected an object")
        except Exception as exc:
            result = {"error": str(exc) or type(exc).__name__}
        invalidated = self._record_output(outputs, name, result)
        if invalidated and verification is not None:
            verification["invalidated_evidence"] = invalidated
        if _tool_ok(result):
            self._cache_args[name] = self._args_key(args)
        return result, args, verification

    def _record_output(self, outputs, name, result):
        invalidated = record_output(outputs, name, result)
        for consumer in invalidated:
            self._cache_args.pop(consumer, None)
        return invalidated

    def _prepare_args(self, name, raw_args, outputs):
        if not self.enforce_contract:
            return dict(raw_args), None
        ctx = ToolCallContext(
            image_path=self.image_path, latitude=self.latitude, longitude=self.longitude,
            nearby_radius=self.nearby_radius, report_written=_tool_ok(outputs.get("report.write")),
            run_tool=self.registry.call_tool, read_image_info=None,
        )
        record: dict[str, Any] = {}
        args = self.policy.prepare_tool_call(name, raw_args, outputs, ctx, record=record)
        return args, {"status": "admitted", "checked": record.get("checked", []),
                      "resolved": record.get("resolved", {})}

    # ------------------------------------------------------------------ #
    def _openai_tools(self, *, report_write_required: bool) -> list[dict[str, Any]]:
        # Report persistence is system-owned, so it is never offered to the model.
        specs = [
            spec for spec in self.registry.list_tools()
            if str(spec.get("name", "")) != "report.write"
        ]
        # Do not present the registry's canonical workflow order as an implicit
        # plan: the catalogue the model sees must not encode an ordering the
        # rule layer no longer imposes.
        specs = sorted(specs, key=lambda spec: str(spec.get("name", "")))
        tools = [
            {
                "type": "function",
                "function": {
                    "name": s["name"],
                    "description": s["description"],
                    "parameters": s["input_schema"],
                },
            }
            for s in specs
        ]
        neutral_descriptions = {
            "cls.run": "Classify visible landslide subtype and return class labels and scores.",
            "geo.background": "Retrieve terrain, geological, and address context for supplied coordinates.",
            "geo.nearby": "Retrieve nearby mapped human facilities for supplied coordinates and radius.",
            "image.tile": "Create localized image tiles for optional detailed inspection.",
            "llm.first_pass": "Assess whole-scene landslide evidence with the vision-language model.",
            "vlm.describe": (
                "Describe the landslide in the image (morphology, material, movement, environment, "
                "impact, classification reason, causation) with the fine-tuned vision-language model."
            ),
            "region.locate": (
                "Describe the frame position and bounds of the candidate region. "
                "`refinement` and `image_info` are bound from recorded results when omitted."
            ),
            "seg.llm_review": (
                "Review a segmentation-boundary overlay with the vision-language model. "
                "`stage1`, `refinement` and `image_info` are bound from recorded results "
                "when omitted; call it with no arguments."
            ),
            "seg.refine": (
                "Derive candidate landslide regions from image and segmentation evidence. "
                "`segmentation` and `image_info` are bound from recorded results when omitted."
            ),
            "seg.run": "Segment possible landslide area in a supplied image and return mask and overlay artifacts.",
            "tiff.info": "Read raster metadata and available geospatial fields from an image.",
        }
        for tool in tools:
            function = tool.get("function") or {}
            name = function.get("name")
            if name in neutral_descriptions:
                function["description"] = neutral_descriptions[name]
            elif name == "fuse.decision":
                function["description"] = (
                    "Fuse recorded evidence into decision and report fields. You may call this "
                    "tool at any time. Every evidence argument is bound from recorded tool "
                    "results when omitted, so call it with no arguments; a supplied argument "
                    "must match its recorded result exactly. Missing prerequisites return an "
                    "ordinary error for you to interpret."
                )
        return tools
