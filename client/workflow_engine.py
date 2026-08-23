"""
User-defined workflows: YAML in, deterministic execution out.

Users author workflows in ./workflows/*.yaml — no Python. A workflow is a
list of steps; each step is one of three kinds:

    - read: workspace://calendar/today          # read an MCP resource
      as: today

    - tool: list_tasks                          # call an MCP tool (through the gate!)
      args: {overdue_only: true}
      as: overdue

    - agent: "Pick the most urgent item in {overdue} and explain why"
      spec: task-assistant                      # an AgentSpec (see agent_specs.py)
      as: recommendation

`as:` names the step's result; later steps reference results with
`{name}` or `{name.field}` / `{name.0}` placeholders. A string that is
EXACTLY one placeholder keeps its original type ("{task.id}" -> int), so
results can be piped into typed tool arguments.

Design point, deliberately: the engine has no if/else, no loops, no
expressions. Deterministic control flow belongs to the workflow author;
open-ended judgement belongs inside an `agent:` step. If you feel the urge to
add conditionals here, that step probably wants to be an agent goal instead.

Every `tool:` call and every tool the `agent:` steps request goes through the
SAME ApprovalPolicy — user-authored workflows inherit the human-in-the-loop
gate and the audit trail for free.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml
from mcp import Client

from .agent_specs import load_specs
from .discovery import tool_result_text
from .gate import ApprovalPolicy
from .providers import load_provider

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = PROJECT_ROOT / "workflows"

_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*)\}")


class WorkflowError(SystemExit):
    """Authoring/runtime error in a user workflow — reported, not a stack trace."""


# --------------------------------------------------------------------------- #
# Templating: {name}, {name.field}, {name.0}
# --------------------------------------------------------------------------- #
def _lookup(path: str, ctx: dict[str, Any]) -> Any:
    head, *rest = path.split(".")
    if head not in ctx:
        raise WorkflowError(f"workflow references {{{path}}} but no earlier step is named {head!r} "
                            f"(known: {sorted(ctx)})")
    value = ctx[head]
    for part in rest:
        if isinstance(value, list) and part.isdigit():
            value = value[int(part)]
        elif isinstance(value, dict) and part in value:
            value = value[part]
        else:
            raise WorkflowError(f"cannot resolve {{{path}}}: {part!r} not found in {type(value).__name__}")
    return value


def _render(value: Any, ctx: dict[str, Any]) -> Any:
    """Substitute placeholders. Whole-string placeholders keep their type."""
    if isinstance(value, str):
        whole = _PLACEHOLDER.fullmatch(value.strip())
        if whole:
            return _lookup(whole.group(1), ctx)
        return _PLACEHOLDER.sub(lambda m: _to_text(_lookup(m.group(1), ctx)), value)
    if isinstance(value, dict):
        return {k: _render(v, ctx) for k, v in value.items()}
    if isinstance(value, list):
        return [_render(v, ctx) for v in value]
    return value


def _to_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
# Discovery of user workflow files
# --------------------------------------------------------------------------- #
def discover_workflows() -> dict[str, Path]:
    """{name: path} for every ./workflows/*.yaml (file stem = workflow name)."""
    return {p.stem: p for p in sorted(WORKFLOWS_DIR.glob("*.yaml"))}


def load_workflow(path: Path) -> dict[str, Any]:
    wf = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    steps = wf.get("steps")
    if not isinstance(steps, list) or not steps:
        raise WorkflowError(f"{path.name}: a workflow needs a non-empty `steps:` list")
    for i, step in enumerate(steps, 1):
        kinds = [k for k in ("read", "tool", "agent") if k in step]
        if len(kinds) != 1:
            raise WorkflowError(f"{path.name} step {i}: each step needs exactly one of read:/tool:/agent:")
        unknown = set(step) - {"read", "tool", "agent", "args", "spec", "as"}
        if unknown:
            raise WorkflowError(f"{path.name} step {i}: unknown field(s) {sorted(unknown)}")
    return wf


# --------------------------------------------------------------------------- #
# The runner
# --------------------------------------------------------------------------- #
async def run_yaml_workflow(client: Client, policy: ApprovalPolicy, path: Path, *,
                            verbose: bool = True) -> str:
    wf = load_workflow(path)
    specs = load_specs()
    ctx: dict[str, Any] = {}

    for i, step in enumerate(wf["steps"], 1):
        name = step.get("as", f"step{i}")

        if "read" in step:                                     # ---- resource read
            uri = _render(step["read"], ctx)
            if verbose:
                print(f"[step {i}] read {uri}", file=sys.stderr)
            res = await client.read_resource(uri)
            text = res.contents[0].text
            try:
                value = json.loads(text)                       # JSON resources become data
            except (json.JSONDecodeError, TypeError):
                value = text                                   # markdown etc. stay text

        elif "tool" in step:                                   # ---- gated tool call
            args = _render(step.get("args", {}), ctx)
            if verbose:
                print(f"[step {i}] tool {step['tool']}({json.dumps(args, ensure_ascii=False)})", file=sys.stderr)
            result = await policy.call_tool(client, step["tool"], args)
            if result.is_error:
                # A denial or tool error stops the workflow honestly — no silent skips.
                raise WorkflowError(f"{path.name} stopped at step {i} ({step['tool']}): "
                                    f"{tool_result_text(result)}")
            sc = result.structured_content
            value = sc.get("result", sc) if isinstance(sc, dict) else tool_result_text(result)

        else:                                                  # ---- bounded agent step
            goal = _render(step["agent"], ctx)
            spec = specs.get(step.get("spec", "generalist"))
            if spec is None:
                raise WorkflowError(f"{path.name} step {i}: unknown agent spec {step.get('spec')!r} "
                                    f"(known: {sorted(specs)})")
            run_agent = load_provider(spec.provider)           # checks SDK + credentials
            if verbose:
                print(f"[step {i}] agent[{spec.name}] {goal[:100]}", file=sys.stderr)
            value = await run_agent(client, policy, goal, spec=spec, verbose=verbose)

        ctx[name] = value

    output = wf.get("output")
    return _to_text(_render(output, ctx)) if output else _to_text(ctx[name])
