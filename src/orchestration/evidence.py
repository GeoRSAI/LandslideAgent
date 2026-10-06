"""Evidence freshness for autonomous tool iterations; never schedules tools."""
from typing import Any

# Direct consumers of a recorded result. Invalidating their old observations
# lets the model decide whether/how to obtain fresh evidence itself.
DEPENDENTS = {
    "tiff.info": ("seg.refine", "region.locate", "seg.llm_review", "fuse.decision"),
    "llm.first_pass": ("seg.llm_review", "fuse.decision"),
    "seg.run": ("seg.refine", "seg.llm_review", "fuse.decision"),
    "seg.refine": ("region.locate", "seg.llm_review", "fuse.decision"),
    "region.locate": ("fuse.decision",),
    "vlm.describe": ("fuse.decision",),
    "cls.run": ("fuse.decision",),
    "geo.background": ("fuse.decision",),
    "geo.nearby": ("fuse.decision",),
    "seg.llm_review": ("fuse.decision",),
    "fuse.decision": ("report.write",),
}


def record_output(outputs: dict[str, Any], name: str, result: dict[str, Any]) -> list[str]:
    invalidated: list[str] = []
    if name in outputs and outputs[name] != result:
        pending = list(DEPENDENTS.get(name, ()))
        visited: set[str] = set()
        while pending:
            consumer = pending.pop()
            if consumer in visited:
                continue
            visited.add(consumer)
            pending.extend(DEPENDENTS.get(consumer, ()))
            if consumer in outputs:
                outputs.pop(consumer)
                invalidated.append(consumer)
    outputs[name] = result
    return sorted(invalidated)
