"""
AgentSpec — an "agent" as DATA, not code.

The loop in agent*.py is fully generic; what makes an agent "the notes
librarian" vs "the task assistant" is only:

    1. a system prompt        (persona + rules)
    2. a tool surface         (subset of what the server offers)
    3. model / provider / turn budget

So a specialised agent is just a small record. Built-in specs live below;
users add their own by dropping YAML files into ./agents/ — no Python:

    # agents/notes-librarian.yaml
    description: Organises notes; never deletes unprompted.
    system_prompt: |
      You are the notes librarian. You only work with notes. ...
    allowed_tools: [search_notes, create_note, delete_note]
    provider: anthropic          # optional (default anthropic)
    model: claude-opus-5         # optional (provider default otherwise)
    max_turns: 8                 # optional

Run one with:  python -m client.main agent --spec notes-librarian "..."
or reference it from a workflow step:  - agent: "...", spec: notes-librarian
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml
from mcp.types import Tool

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENTS_DIR = PROJECT_ROOT / "agents"


@dataclass
class AgentSpec:
    name: str
    description: str = ""
    system_prompt: str = ""
    allowed_tools: list[str] = field(default_factory=list)   # [] = all tools
    provider: str = "anthropic"
    model: str | None = None
    max_turns: int = 12


GENERALIST = AgentSpec(
    name="generalist",
    description="Default agent: every tool, no extra persona.",
)


def load_specs() -> dict[str, AgentSpec]:
    """Built-ins + every ./agents/*.yaml (file stem = spec name)."""
    specs = {GENERALIST.name: GENERALIST}
    for path in sorted(AGENTS_DIR.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        allowed = {"description", "system_prompt", "allowed_tools", "provider", "model", "max_turns"}
        unknown = set(raw) - allowed
        if unknown:
            raise SystemExit(f"{path.name}: unknown field(s) {sorted(unknown)}; allowed: {sorted(allowed)}")
        specs[path.stem] = AgentSpec(name=path.stem, **raw)
    return specs


def filter_tools_for_spec(tools: list[Tool], spec: AgentSpec | None) -> list[Tool]:
    """The tool surface the MODEL sees. (The gate still registers the full list.)"""
    if spec is None or not spec.allowed_tools:
        return tools
    known = {t.name for t in tools}
    missing = set(spec.allowed_tools) - known
    if missing:
        raise SystemExit(f"agent spec {spec.name!r} allows tools the server doesn't have: {sorted(missing)}")
    return [t for t in tools if t.name in spec.allowed_tools]


def compose_system_prompt(base: str, spec: AgentSpec | None) -> str:
    """Server instructions + resource map first, then the spec's persona."""
    if spec is None or not spec.system_prompt:
        return base
    return f"{base}\n\n## Your role: {spec.name}\n{spec.system_prompt.strip()}"
