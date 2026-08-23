"""
Personal Workspace Assistant — an MCP *server*.

This file is the "adapter" side of MCP. It wraps a tiny personal workspace
(markdown notes, a task list, a calendar, an outbox) and exposes it through the
three MCP primitives ONLY:

    RESOURCES  -> read-only context the model can load   (workspace://...)
    TOOLS      -> actions the model may ask to execute    (add_task, delete_note, ...)
    PROMPTS    -> reusable, parameterised prompt templates (daily_briefing, weekly_review)

Everything the server can do is *self-described*: the client asks
`tools/list`, `resources/list`, `resources/templates/list`, `prompts/list` and
receives JSON Schemas + metadata it can hand straight to an LLM.

Run it:
    python -m server.workspace_server            # stdio transport (default)
    python -m server.workspace_server --http     # streamable-http on 127.0.0.1:8000/mcp

The server never decides *whether* an action is allowed — that is the client's
job (human-in-the-loop). The server only *declares* what each tool does via
ToolAnnotations (read_only_hint / destructive_hint / ...) and, for the one
truly irreversible tool, asks the client to confirm via *elicitation*.
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from mcp.server.mcpserver import (
    AcceptedElicitation,
    CancelledElicitation,
    Context,
    DeclinedElicitation,
    Elicit,
    ElicitationResult,
    MCPServer,
    Resolve,
)
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Storage — a deliberately boring "database": JSON files + markdown in ./data
# --------------------------------------------------------------------------- #

DATA_DIR = Path(os.environ.get("WORKSPACE_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
NOTES_DIR = DATA_DIR / "notes"
TASKS_FILE = DATA_DIR / "tasks.json"
CALENDAR_FILE = DATA_DIR / "calendar.json"
OUTBOX_FILE = DATA_DIR / "outbox.json"


def _read_json(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def _write_json(path: Path, data: list[dict]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _slug_ok(slug: str) -> bool:
    # Tools are the *only* way the model can touch the filesystem, and even then
    # only inside NOTES_DIR with a whitelisted slug. This is "least privilege".
    return re.fullmatch(r"[a-z0-9][a-z0-9\-]{0,63}", slug) is not None


def _today() -> date:
    # Overridable so the demo/tests are deterministic.
    override = os.environ.get("WORKSPACE_TODAY")
    return date.fromisoformat(override) if override else date.today()


# --------------------------------------------------------------------------- #
# The server object. `instructions` is sent to the client at handshake time and
# is meant to be put in the LLM's system prompt — part of being self-describing.
# --------------------------------------------------------------------------- #

mcp = MCPServer(
    name="personal-workspace",
    version="0.1.0",
    instructions=(
        "You are connected to the user's personal workspace: markdown notes, a task list, "
        "a calendar and an e-mail outbox. Read resources before acting. Prefer read-only tools; "
        "only call write/destructive tools when the user clearly asked for that outcome. "
        "Dates are ISO-8601 (YYYY-MM-DD)."
    ),
)

# =========================================================================== #
# 1. RESOURCES — "here is context you may read"                                #
#    Static URIs and URI *templates*. No side effects. Think GET, not POST.    #
# =========================================================================== #


@mcp.resource(
    "workspace://notes",
    name="notes-index",
    description="Index of all notes in the workspace (slug, title, size).",
    mime_type="application/json",
)
def notes_index() -> str:
    items = []
    for p in sorted(NOTES_DIR.glob("*.md")):
        first = p.read_text(encoding="utf-8").splitlines()[0] if p.stat().st_size else ""
        items.append({"slug": p.stem, "title": first.lstrip("# ").strip(), "bytes": p.stat().st_size,
                      "uri": f"workspace://notes/{p.stem}"})
    return json.dumps(items, indent=2)


@mcp.resource(
    "workspace://notes/{slug}",
    name="note",
    description="Full markdown content of a single note, addressed by slug.",
    mime_type="text/markdown",
)
def note_content(slug: str) -> str:
    # This is a *resource template*: the client fills in {slug}.
    if not _slug_ok(slug):
        raise ValueError(f"invalid slug: {slug!r}")
    path = NOTES_DIR / f"{slug}.md"
    if not path.exists():
        raise FileNotFoundError(f"no note with slug {slug!r}")
    return path.read_text(encoding="utf-8")


@mcp.resource(
    "workspace://tasks",
    name="tasks",
    description="All tasks as JSON (id, title, due, priority, done).",
    mime_type="application/json",
)
def tasks_resource() -> str:
    return json.dumps(_read_json(TASKS_FILE), indent=2)


@mcp.resource(
    "workspace://calendar/today",
    name="calendar-today",
    description="Calendar events for today (server clock, or WORKSPACE_TODAY override).",
    mime_type="application/json",
)
def calendar_today() -> str:
    today = _today().isoformat()
    events = [e for e in _read_json(CALENDAR_FILE) if e["start"].startswith(today)]
    return json.dumps({"date": today, "events": events}, indent=2)


@mcp.resource(
    "workspace://calendar/upcoming",
    name="calendar-upcoming",
    description="All calendar events from today onward, soonest first.",
    mime_type="application/json",
)
def calendar_upcoming() -> str:
    today = _today().isoformat()
    events = sorted((e for e in _read_json(CALENDAR_FILE) if e["start"][:10] >= today), key=lambda e: e["start"])
    return json.dumps(events, indent=2)


# =========================================================================== #
# 2. TOOLS — "here are actions you may *request*"                              #
#    Each tool publishes a JSON Schema (derived from the type hints) and        #
#    ToolAnnotations that tell the client how risky it is.                     #
# =========================================================================== #

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)
WRITE_SAFE = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=False)
WRITE_IDEMPOTENT = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=True, open_world_hint=False)
DESTRUCTIVE = ToolAnnotations(read_only_hint=False, destructive_hint=True, idempotent_hint=True, open_world_hint=False)
EXTERNAL = ToolAnnotations(read_only_hint=False, destructive_hint=True, idempotent_hint=False, open_world_hint=True)


@mcp.tool(annotations=READ_ONLY, description="Case-insensitive full-text search across all notes. Returns matching lines with their note slug.")
def search_notes(query: str, max_results: int = 20) -> list[dict[str, Any]]:
    q = query.lower()
    hits: list[dict] = []
    for p in sorted(NOTES_DIR.glob("*.md")):
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
            if q in line.lower():
                hits.append({"slug": p.stem, "line": n, "text": line.strip()})
                if len(hits) >= max_results:
                    return hits
    return hits


@mcp.tool(annotations=READ_ONLY, description="List tasks, optionally filtered. `status` is 'open', 'done' or 'all'. `overdue_only` keeps open tasks whose due date is before today.")
def list_tasks(status: Literal["open", "done", "all"] = "open", overdue_only: bool = False) -> list[dict[str, Any]]:
    tasks = _read_json(TASKS_FILE)
    if status != "all":
        tasks = [t for t in tasks if t["done"] == (status == "done")]
    if overdue_only:
        today = _today().isoformat()
        tasks = [t for t in tasks if not t["done"] and t["due"] < today]
    return tasks


@mcp.tool(annotations=WRITE_SAFE, description="Create a new task. `due` must be YYYY-MM-DD.")
def add_task(
    title: str,
    due: str,
    priority: Literal["low", "medium", "high"] = "medium",
) -> dict[str, Any]:
    date.fromisoformat(due)  # validate — raises ValueError which MCP returns as a tool error
    tasks = _read_json(TASKS_FILE)
    task = {"id": (max((t["id"] for t in tasks), default=0) + 1), "title": title, "due": due,
            "priority": priority, "done": False}
    tasks.append(task)
    _write_json(TASKS_FILE, tasks)
    return task


@mcp.tool(annotations=WRITE_IDEMPOTENT, description="Mark a task as done by id.")
def complete_task(task_id: int) -> dict[str, Any]:
    tasks = _read_json(TASKS_FILE)
    for t in tasks:
        if t["id"] == task_id:
            t["done"] = True
            _write_json(TASKS_FILE, tasks)
            return t
    raise ValueError(f"no task with id {task_id}")


@mcp.tool(annotations=WRITE_SAFE, description="Create a new markdown note. Fails if the slug already exists.")
def create_note(slug: str, content: str) -> dict[str, Any]:
    if not _slug_ok(slug):
        raise ValueError("slug must be lowercase letters, digits and dashes")
    path = NOTES_DIR / f"{slug}.md"
    if path.exists():
        raise FileExistsError(f"note {slug!r} already exists")
    path.write_text(content, encoding="utf-8")
    return {"slug": slug, "uri": f"workspace://notes/{slug}", "bytes": len(content.encode())}


# ---- Human-in-the-loop, spec-native flavour: ELICITATION ------------------- #
# The client *already* gates destructive tools (see client/gate.py). On top of
# that, the server itself can ask the human a question mid-call. The MCP way to
# do that is `elicitation/create`: server -> client -> human -> client -> server.
# In the mcp 2.0 SDK you express it as a *resolver* parameter.


class DeleteConfirmation(BaseModel):
    confirm: bool = Field(description="Set to true to permanently delete the note.")


def _ask_delete_confirmation(slug: str) -> Elicit[DeleteConfirmation]:
    # The resolver receives the tool argument `slug` by name and returns an
    # Elicit request; the framework does the round-trip to the client.
    return Elicit(f"Permanently delete note '{slug}'? This cannot be undone.", DeleteConfirmation)


@mcp.tool(annotations=DESTRUCTIVE, description="Permanently delete a note by slug. The user will be asked to confirm.")
def delete_note(
    slug: str,
    confirmation: Annotated[ElicitationResult[DeleteConfirmation], Resolve(_ask_delete_confirmation)],
) -> str:
    # `confirmation` is NOT in the tool's inputSchema — the LLM cannot supply it.
    # It is filled by the elicitation round-trip with the human.
    match confirmation:
        case AcceptedElicitation(data=DeleteConfirmation(confirm=True)):
            pass
        case AcceptedElicitation():
            return f"Deletion of '{slug}' not confirmed — nothing deleted."
        case DeclinedElicitation():
            return f"User declined to delete '{slug}' — nothing deleted."
        case CancelledElicitation():
            return f"User cancelled — nothing deleted."
    if not _slug_ok(slug):
        raise ValueError("invalid slug")
    path = NOTES_DIR / f"{slug}.md"
    if not path.exists():
        raise FileNotFoundError(f"no note with slug {slug!r}")
    path.unlink()
    return f"Deleted note '{slug}'."


@mcp.tool(annotations=EXTERNAL, description="Send an e-mail on the user's behalf. (Demo: appends to an outbox file instead of really sending.)")
async def send_email(to: str, subject: str, body: str, ctx: Context) -> dict[str, Any]:
    # `ctx: Context` is injected by the framework (not part of the schema). It lets
    # the tool talk back to the client over the *same* stateful connection —
    # e.g. progress notifications streamed while the call is still running.
    # (This is something plain request/response REST cannot do.)
    await ctx.report_progress(0.5, 1.0, "writing to outbox")
    outbox = _read_json(OUTBOX_FILE)
    msg = {"id": len(outbox) + 1, "to": to, "subject": subject, "body": body,
           "queued_at": datetime.now().isoformat(timespec="seconds")}
    outbox.append(msg)
    _write_json(OUTBOX_FILE, outbox)
    await ctx.report_progress(1.0, 1.0, "done")
    return {"status": "queued (demo outbox)", "message": msg}


# =========================================================================== #
# 3. PROMPTS — "here are good ways to ask me things"                           #
#    Templates the *user/client* picks (slash-command style). They return       #
#    messages; they do not call the model themselves.                          #
# =========================================================================== #


@mcp.prompt(description="Produce a short daily briefing from today's calendar, open tasks and recent notes.")
def daily_briefing(tone: Literal["brief", "detailed"] = "brief") -> str:
    return (
        "You are my workspace assistant. Using the resources workspace://calendar/today, "
        "workspace://tasks and workspace://notes (read them with the tools/resources available), "
        f"write a {tone} daily briefing: (1) today's schedule, (2) overdue and due-today tasks, "
        "(3) one suggested focus for the day. Do not invent data."
    )


@mcp.prompt(description="Guide a weekly review of a focus area, grounded in the workspace notes and tasks.")
def weekly_review(focus: str) -> list[dict[str, Any]]:
    # A prompt may return several messages (a mini conversation scaffold).
    return [
        {"role": "user", "content": f"Let's do a weekly review focused on: {focus}."},
        {"role": "assistant", "content": "Sure. I'll look at related notes, open tasks and upcoming events first."},
        {"role": "user", "content": "Summarise progress, list blockers, and propose at most three concrete next actions. Cite note slugs / task ids you used."},
    ]


# --------------------------------------------------------------------------- #
# Entry point — pick a transport. stdio = local pipes (no network at all);      #
# streamable-http = remote-capable, put TLS/auth in front of it in production.  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    if "--http" in sys.argv:
        port = int(os.environ.get("WORKSPACE_PORT", "8000"))
        print(f"[workspace-server] streamable-http on http://127.0.0.1:{port}/mcp", file=sys.stderr)
        mcp.run(transport="streamable-http", port=port)
    else:
        # IMPORTANT: with stdio, stdout IS the protocol channel. Never print() to
        # stdout in a stdio server — use stderr.
        mcp.run(transport="stdio")
