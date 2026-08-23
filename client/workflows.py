"""
WORKFLOWS — code decides the steps, MCP supplies the data/actions.

Contrast with agent.py: here there is *no* model choosing which tool to call.
The control flow is ordinary Python. MCP is used purely as the uniform adapter
to the workspace — read a resource, call a tool — so the same workflow code
keeps working if the server is rewritten, moved to HTTP, or swapped for
another MCP server that exposes the same primitives.

  daily_brief      resources only                       (no LLM, no prompts)
  capture_todos    resources + write tool + HITL gate   (no LLM)
  weekly_review    MCP prompt + resources -> ONE Claude call (LLM, but no tool loop)
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from mcp import Client

from .gate import ApprovalPolicy

MODEL = "claude-opus-5"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
async def read_json_resource(client: Client, uri: str) -> Any:
    res = await client.read_resource(uri)
    return json.loads(res.contents[0].text)


async def read_text_resource(client: Client, uri: str) -> str:
    res = await client.read_resource(uri)
    return res.contents[0].text


# --------------------------------------------------------------------------- #
# 1. daily_brief — pure resources
# --------------------------------------------------------------------------- #
async def daily_brief(client: Client, policy: ApprovalPolicy) -> str:
    today = await read_json_resource(client, "workspace://calendar/today")
    tasks = await read_json_resource(client, "workspace://tasks")
    notes = await read_json_resource(client, "workspace://notes")

    date = today["date"]
    overdue = [t for t in tasks if not t["done"] and t["due"] < date]
    due_today = [t for t in tasks if not t["done"] and t["due"] == date]
    lines = [f"Daily brief for {date}", "=" * 28, "", "Schedule:"]
    lines += [f"  {e['start'][11:]}–{e['end'][11:]}  {e['title']}" for e in today["events"]] or ["  (nothing scheduled)"]
    lines += ["", "Overdue:"] + ([f"  ! #{t['id']} {t['title']} (due {t['due']}, {t['priority']})" for t in overdue] or ["  none"])
    lines += ["", "Due today:"] + ([f"  • #{t['id']} {t['title']} ({t['priority']})" for t in due_today] or ["  none"])
    lines += ["", f"Notes in workspace: {len(notes)} — " + ", ".join(n["slug"] for n in notes)]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 2. capture_todos — read notes, create tasks (write tool => gate applies)
# --------------------------------------------------------------------------- #
TODO_RE = re.compile(r"^\s*(?:[-*]\s*)?TODO:\s*(.+)$", re.IGNORECASE)


async def capture_todos(client: Client, policy: ApprovalPolicy, *, due_in_days: int = 7) -> str:
    notes = await read_json_resource(client, "workspace://notes")
    tasks = await read_json_resource(client, "workspace://tasks")
    existing = {t["title"].lower() for t in tasks}
    today = (await read_json_resource(client, "workspace://calendar/today"))["date"]

    import datetime as dt
    due = (dt.date.fromisoformat(today) + dt.timedelta(days=due_in_days)).isoformat()

    created, skipped, denied = [], [], []
    for n in notes:
        text = await read_text_resource(client, n["uri"])        # resource template in action
        for line in text.splitlines():
            m = TODO_RE.match(line)
            if not m:
                continue
            title = m.group(1).strip()
            if title.lower() in existing:
                skipped.append(title)
                continue
            # add_task is a WRITE tool -> ApprovalPolicy will ask (unless --yes)
            result = await policy.call_tool(client, "add_task", {"title": title, "due": due, "priority": "medium"})
            if result.is_error:
                denied.append(title)
            else:
                created.append(title)
                existing.add(title.lower())
    out = [f"capture_todos: {len(created)} created, {len(skipped)} already existed, {len(denied)} denied/failed"]
    out += [f"  + {t}" for t in created] + [f"  = {t}" for t in skipped] + [f"  x {t}" for t in denied]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# 3. weekly_review — MCP PROMPT + RESOURCES, one Claude call, no tool loop
# --------------------------------------------------------------------------- #
async def weekly_review(client: Client, policy: ApprovalPolicy, *, focus: str = "Project Atlas") -> str:
    import anthropic  # imported lazily: the other workflows don't need an API key

    # (a) Ask the server for its prompt template. The server owns the wording;
    #     the client just fills in arguments. Think "slash command".
    prompt = await client.get_prompt("weekly_review", {"focus": focus})

    # (b) Ground the prompt with resources. We read them over MCP and attach
    #     them as plain text blocks (documents) in the *first* user message.
    context_blocks: list[dict[str, Any]] = []
    for uri in ("workspace://notes", "workspace://tasks", "workspace://calendar/upcoming"):
        context_blocks.append({"type": "text", "text": f"<resource uri=\"{uri}\">\n{await read_text_resource(client, uri)}\n</resource>"})
    notes_index = await read_json_resource(client, "workspace://notes")
    for n in notes_index:
        context_blocks.append({"type": "text", "text": f"<resource uri=\"{n['uri']}\">\n{await read_text_resource(client, n['uri'])}\n</resource>"})

    # (c) Convert MCP prompt messages -> Anthropic messages.
    messages: list[dict[str, Any]] = []
    for m in prompt.messages:
        text = m.content.text if getattr(m.content, "type", None) == "text" else str(m.content)
        messages.append({"role": m.role, "content": [{"type": "text", "text": text}]})
    messages[0]["content"] = context_blocks + messages[0]["content"]   # context first, question last

    # (d) ONE model call. No tools, no loop: this is a workflow step, not an agent.
    anthropic_client = anthropic.Anthropic()
    response = anthropic_client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=client.instructions or "",
        thinking={"type": "adaptive"},
        output_config={"effort": "medium"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",          # re-run on a fallback model if a safety classifier declines
        messages=messages,
    )
    if response.stop_reason == "refusal":
        return "Model declined the request."
    return "".join(b.text for b in response.content if b.type == "text")


WORKFLOWS = {
    "daily-brief": daily_brief,
    "capture-todos": capture_todos,
    "weekly-review": weekly_review,
}
