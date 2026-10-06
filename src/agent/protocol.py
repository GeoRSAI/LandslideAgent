from __future__ import annotations

import os

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


class ToolRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {}

    def register(self, spec: ToolSpec, handler: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        self._specs[spec.name] = spec
        self._handlers[spec.name] = handler

    # Canonical workflow order for presenting tools to the model. Keeping the
    # tool list in required-execution order (rather than registration order)
    # stops a weaker model from calling steps out of sequence.
    _WORKFLOW_ORDER = (
        "tiff.info", "llm.first_pass", "seg.run", "seg.refine",
        "region.locate", "seg.llm_review", "cls.run",
        "geo.background", "geo.nearby", "fuse.decision",
        "report.write", "image.tile",
    )

    def list_tools(self) -> list[dict[str, Any]]:
        order = {name: i for i, name in enumerate(self._WORKFLOW_ORDER)}
        specs = sorted(
            self._specs.values(),
            key=lambda s: (order.get(s.name, len(order)), s.name),
        )
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "input_schema": spec.input_schema,
            }
            for spec in specs
        ]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in self._handlers:
            raise KeyError(f"unknown tool: {name}")
        # Robustness testing only: AGENT_FAULT_INJECT="geo.nearby,seg.llm_review"
        # makes the listed tools fail as if their backend were down. Unset in
        # normal runs; applies identically to every agent mode.
        injected = {t.strip() for t in os.getenv("AGENT_FAULT_INJECT", "").split(",") if t.strip()}
        if name in injected:
            raise RuntimeError(f"{name} is unavailable: backend service did not respond")
        return self._handlers[name](arguments)


class JsonRpcAgentServer:
    """
    Minimal JSON-RPC 2.0 tool server:
    - method: tools/list
    - method: tools/call
    - method: chat
    """

    def __init__(
        self,
        registry: ToolRegistry,
        chat_handler: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.registry = registry
        self.chat_handler = chat_handler

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        req_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {}) or {}

        try:
            if method == "tools/list":
                result = {"tools": self.registry.list_tools()}
            elif method == "tools/call":
                tool_name = params["name"]
                arguments = params.get("arguments", {}) or {}
                tool_result = self.registry.call_tool(tool_name, arguments)
                result = {"content": tool_result}
            elif method == "chat":
                if not self.chat_handler:
                    return self._error(req_id, -32601, "method not found: chat")
                result = self.chat_handler(params)
            else:
                return self._error(req_id, -32601, f"method not found: {method}")

            return {"jsonrpc": "2.0", "id": req_id, "result": result}
        except KeyError as exc:
            message = exc.args[0] if exc.args else str(exc)
            return self._error(req_id, -32602, str(message))
        except Exception as exc:  # pragma: no cover
            return self._error(req_id, -32000, str(exc))

    @staticmethod
    def _error(req_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}
