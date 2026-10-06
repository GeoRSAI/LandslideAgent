"""Verify the two experimental arms differ in exactly one variable.

Builds both arms with identical inputs, captures everything the cognitive
engine actually receives on its first turn, and diffs it.
"""
import difflib, hashlib, json, os, sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
os.chdir(str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from src.agent.controller import get_policy
from src.orchestration.agent_runner import AgentRunner

CAPTURED = {}


def make_arm(name, **flags):
    def model_fn(messages, tools):
        CAPTURED.setdefault(name, {"messages": messages, "tools": tools})
        return {"message": {"role": "assistant", "content": "stop"}, "raw": ""}

    agent = AgentRunner(
        model_fn=model_fn,
        image_path="data/sample.tif",
        latitude=29.6,
        longitude=103.0,
        nearby_radius=300,
        max_turns=40,
        **flags,
    )
    # Same brief the service injects for both arms.
    brief = get_policy().autonomous_task_contract()
    list(agent.stream([
        {"role": "system", "content": "For any tool call that needs an image path, "
                                      "use this exact path: data/sample.tif. " + brief},
        {"role": "user", "content": "Analyse this scene."},
    ]))
    return agent


a = make_arm("contract", enforce_contract=True, loop_hygiene=True)
b = make_arm("free", enforce_contract=False, loop_hygiene=True)


def render(entry):
    lines = []
    for m in entry["messages"]:
        lines.append("[%s] %s" % (m.get("role"), " ".join(str(m.get("content") or "").split())))
    lines.append("--- tool catalogue ---")
    for tool in entry["tools"]:
        fn = tool.get("function") or {}
        lines.append("%s :: %s" % (fn.get("name"), fn.get("description")))
        lines.append("   schema: %s" % json.dumps(fn.get("parameters"), sort_keys=True))
    return lines


left, right = render(CAPTURED["contract"]), render(CAPTURED["free"])

print("=" * 70)
print("MODEL-FACING CONTEXT ON TURN 1")
print("  contract arm : %d messages, %d tools, md5 %s"
      % (len(CAPTURED["contract"]["messages"]), len(CAPTURED["contract"]["tools"]),
         hashlib.md5("\n".join(left).encode()).hexdigest()))
print("  free arm     : %d messages, %d tools, md5 %s"
      % (len(CAPTURED["free"]["messages"]), len(CAPTURED["free"]["tools"]),
         hashlib.md5("\n".join(right).encode()).hexdigest()))

diff = list(difflib.unified_diff(left, right, "contract", "free", lineterm="", n=0))
if diff:
    print("\n!! DIFFERS:")
    print("\n".join(diff[:40]))
else:
    print("\n  IDENTICAL - the model sees exactly the same context in both arms.")

print("=" * 70)
print("RUNTIME FLAGS")
for label, agent in (("contract", a), ("free", b)):
    print("  %-9s enforce_contract=%-5s loop_hygiene=%-5s rule_mode=%-5s "
          "react_mode=%-5s max_turns=%s"
          % (label, agent.enforce_contract, agent.loop_hygiene, agent.rule_mode,
             agent.react_mode, agent.max_turns))
print("=" * 70)
