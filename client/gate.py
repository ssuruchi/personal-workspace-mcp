"""
Human-in-the-loop: the client-side *intercept layer*.

The model proposes; the client disposes. Before any tool call reaches the
server, `ApprovalPolicy.authorize()` looks at the tool's self-declared
ToolAnnotations and decides:

    read_only_hint = True          -> run automatically
    anything that writes           -> ask the human (y/N) unless --yes
    destructive_hint / open_world  -> ask the human, and say why it is risky

Two independent gates exist, and they compose:

  1. This one — lives in the CLIENT. Based on annotations the server published.
     It is where "require explicit human confirmation before the server executes
     the tool" actually happens.
  2. Elicitation — initiated by the SERVER mid-call (delete_note does this).
     The SDK delivers it to `handle_elicitation()` below; we ask the human and
     return accept / decline.

Every decision is appended to an in-memory audit trail so a workflow or agent
run can print "what happened and who approved it" at the end.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Any

from mcp import Client
from mcp.types import CallToolResult, ElicitResult, ErrorData, Tool


@dataclass
class AuditEntry:
    tool: str
    arguments: dict[str, Any]
    risk: str
    decision: str       # "auto" | "approved" | "denied"
    outcome: str = ""   # short result / error text


@dataclass
class ApprovalPolicy:
    """Decides which tool calls may proceed and asks the human when needed."""

    auto_approve: bool = False            # --yes : never prompt (CI / demos)
    interactive: bool = True              # False : deny anything that needs a prompt
    audit: list[AuditEntry] = field(default_factory=list)
    _tools: dict[str, Tool] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Tool-level gate
    # ------------------------------------------------------------------ #
    def register_tools(self, tools: list[Tool]) -> None:
        """Remember the server's tool descriptions so we can read annotations."""
        self._tools = {t.name: t for t in tools}

    @staticmethod
    def classify(tool: Tool | None) -> str:
        """Turn ToolAnnotations into a one-word risk class."""
        if tool is None or tool.annotations is None:
            return "unknown"            # no annotations => treat as a write
        a = tool.annotations
        if a.read_only_hint:
            return "read-only"
        if a.open_world_hint:
            return "external"           # leaves the workspace (email, payments…)
        if a.destructive_hint:
            return "destructive"
        return "write"

    def authorize(self, name: str, arguments: dict[str, Any]) -> bool:
        """Return True if the call may proceed. May block on human input."""
        risk = self.classify(self._tools.get(name))
        if risk == "read-only":
            self.audit.append(AuditEntry(name, arguments, risk, "auto"))
            return True
        if self.auto_approve:
            self.audit.append(AuditEntry(name, arguments, risk, "approved (auto)"))
            return True
        if not self.interactive:
            self.audit.append(AuditEntry(name, arguments, risk, "denied (non-interactive)"))
            return False
        print(f"\n  ⚠  The model wants to call `{name}` [{risk.upper()}]", file=sys.stderr)
        print("     arguments: " + json.dumps(arguments, ensure_ascii=False), file=sys.stderr)
        answer = input("     Approve? [y/N] ").strip().lower()
        ok = answer in ("y", "yes")
        self.audit.append(AuditEntry(name, arguments, risk, "approved" if ok else "denied"))
        return ok

    async def call_tool(self, client: Client, name: str, arguments: dict[str, Any] | None = None) -> CallToolResult:
        """The ONLY path through which a tool call reaches the server."""
        arguments = arguments or {}
        if not self.authorize(name, arguments):
            # Tell the model the truth: the human said no. The model can adapt.
            return CallToolResult(
                content=[{"type": "text", "text": f"Tool call `{name}` was DENIED by the user. Do not retry it; ask the user what to do instead."}],
                is_error=True,
            )
        result = await client.call_tool(name, arguments)
        summary = " ".join(_first_text(result).split())[:110]
        self.audit[-1].outcome = ("ERROR: " if result.is_error else "") + summary
        return result

    # ------------------------------------------------------------------ #
    # Server-initiated elicitation (MCP `elicitation/create`)
    # ------------------------------------------------------------------ #
    async def handle_elicitation(self, context, params) -> ElicitResult | ErrorData:
        """The server paused a tool call to ask the human something."""
        schema = getattr(params, "requested_schema", None)  # None for URL-mode elicitation
        props = (schema or {}).get("properties", {}) if isinstance(schema, dict) else {}
        print(f"\n  ❓ Server asks: {params.message}", file=sys.stderr)
        if self.auto_approve:
            # Fill every boolean with True, everything else left to defaults.
            content = {k: True for k, v in props.items() if v.get("type") == "boolean"}
            print(f"     (auto-approve) -> {content}", file=sys.stderr)
            return ElicitResult(action="accept", content=content)
        if not self.interactive:
            return ElicitResult(action="decline")
        content: dict[str, Any] = {}
        for key, spec in props.items():
            desc = spec.get("description", "")
            raw = input(f"     {key} ({spec.get('type','string')}) {desc} : ").strip()
            if spec.get("type") == "boolean":
                content[key] = raw.lower() in ("y", "yes", "true", "1")
            elif spec.get("type") in ("integer", "number"):
                content[key] = float(raw) if spec.get("type") == "number" else int(raw or 0)
            else:
                content[key] = raw
        if not content:
            raw = input("     accept? [y/N] ").strip().lower()
            return ElicitResult(action="accept" if raw in ("y", "yes") else "decline")
        return ElicitResult(action="accept", content=content)

    # ------------------------------------------------------------------ #
    def print_audit(self) -> None:
        if not self.audit:
            print("\n(no tool calls were made)")
            return
        print("\n── Audit trail ──────────────────────────────────────────────")
        for i, e in enumerate(self.audit, 1):
            print(f"{i:>2}. {e.tool:<14} [{e.risk:<11}] {e.decision:<24} {json.dumps(e.arguments, ensure_ascii=False)[:60]}")
            if e.outcome:
                print(f"    ↳ {e.outcome}")


def _first_text(result: CallToolResult) -> str:
    for block in result.content:
        if getattr(block, "type", None) == "text":
            return block.text
    return ""
