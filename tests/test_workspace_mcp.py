"""
Tests run the *real* server in-process (mcp.Client accepts an MCPServer object
and speaks the protocol in memory — no subprocess, no sockets). Each test gets
a fresh copy of ./data so writes never leak.

    .venv/Scripts/python -m pytest -q
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
from pathlib import Path

import pytest
from mcp import Client
from mcp.types import ElicitResult

from client.discovery import describe_server, mcp_tools_to_anthropic
from client.gate import ApprovalPolicy
from client.workflows import capture_todos, daily_brief

PROJECT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def server(tmp_path, monkeypatch):
    """Fresh data dir + a freshly imported server module bound to it.

    Seeded from tests/fixtures/data (a frozen snapshot), NOT the live ./data —
    playing with the CLI must never break the tests."""
    data = tmp_path / "data"
    shutil.copytree(Path(__file__).parent / "fixtures" / "data", data)
    monkeypatch.setenv("WORKSPACE_DATA_DIR", str(data))
    monkeypatch.setenv("WORKSPACE_TODAY", "2026-08-23")
    import server.workspace_server as ws
    ws = importlib.reload(ws)            # re-evaluate DATA_DIR with the env var
    ws.DATA = data                        # handy for assertions
    return ws


# --------------------------------------------------------------------------- #
# Self-description
# --------------------------------------------------------------------------- #
@pytest.mark.anyio
async def test_server_is_self_describing(server):
    async with Client(server.mcp) as c:
        desc = await describe_server(c)
    names = {t["name"] for t in desc["tools"]}
    assert names == {"search_notes", "list_tasks", "add_task", "complete_task", "create_note", "delete_note", "send_email"}
    # every tool ships a JSON Schema the LLM can be given verbatim
    for t in desc["tools"]:
        assert t["inputSchema"]["type"] == "object"
    # elicitation-resolved params never leak into the schema the model sees
    delete = next(t for t in desc["tools"] if t["name"] == "delete_note")
    assert list(delete["inputSchema"]["properties"]) == ["slug"]
    assert {r["uri"] for r in desc["resources"]} >= {"workspace://notes", "workspace://tasks", "workspace://calendar/today"}
    assert desc["resource_templates"][0]["uriTemplate"] == "workspace://notes/{slug}"
    assert {p["name"] for p in desc["prompts"]} == {"daily_briefing", "weekly_review"}
    assert "prompts" in desc["capabilities"] and "tools" in desc["capabilities"]


@pytest.mark.anyio
async def test_mcp_to_anthropic_tool_conversion(server):
    async with Client(server.mcp) as c:
        tools = (await c.list_tools()).tools
    converted = mcp_tools_to_anthropic(tools)
    assert {"name", "description", "input_schema"} <= set(converted[0])
    assert converted[0]["input_schema"] == tools[0].input_schema   # pure passthrough


# --------------------------------------------------------------------------- #
# Resources
# --------------------------------------------------------------------------- #
@pytest.mark.anyio
async def test_resources_and_template(server):
    async with Client(server.mcp) as c:
        idx = json.loads((await c.read_resource("workspace://notes")).contents[0].text)
        assert {n["slug"] for n in idx} == {"ideas", "project-atlas", "reading-list"}
        note = (await c.read_resource("workspace://notes/ideas")).contents[0]
        assert note.mime_type == "text/markdown" and note.text.startswith("# Random ideas")
        today = json.loads((await c.read_resource("workspace://calendar/today")).contents[0].text)
        assert today["date"] == "2026-08-23" and {e["id"] for e in today["events"]} == {"evt-2", "evt-3"}


@pytest.mark.anyio
async def test_template_rejects_bad_slug(server):
    async with Client(server.mcp) as c:
        with pytest.raises(Exception):
            await c.read_resource("workspace://notes/..%2Fsecrets")


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
@pytest.mark.anyio
async def test_read_tools(server):
    async with Client(server.mcp) as c:
        hits = (await c.call_tool("search_notes", {"query": "todo"})).structured_content["result"]
        assert len(hits) == 3
        overdue = (await c.call_tool("list_tasks", {"overdue_only": True})).structured_content["result"]
        assert [t["id"] for t in overdue] == [1]


@pytest.mark.anyio
async def test_write_tools_and_errors(server):
    async with Client(server.mcp) as c:
        r = await c.call_tool("add_task", {"title": "x", "due": "2026-09-01", "priority": "high"})
        assert not r.is_error and r.structured_content["id"] == 5
        r = await c.call_tool("complete_task", {"task_id": 5})
        assert r.structured_content["done"] is True
        bad = await c.call_tool("add_task", {"title": "x", "due": "not-a-date"})
        assert bad.is_error                      # server-side validation -> isError, not a crash
        bad = await c.call_tool("complete_task", {"task_id": 999})
        assert bad.is_error and "no task" in bad.content[0].text
    assert json.loads((server.DATA / "tasks.json").read_text())[4]["done"] is True


@pytest.mark.anyio
async def test_tool_annotations_drive_the_gate(server):
    async with Client(server.mcp) as c:
        tools = (await c.list_tools()).tools
    risk = {t.name: ApprovalPolicy.classify(t) for t in tools}
    assert risk["search_notes"] == "read-only"
    assert risk["add_task"] == "write"
    assert risk["delete_note"] == "destructive"
    assert risk["send_email"] == "external"


# --------------------------------------------------------------------------- #
# Human-in-the-loop: client gate + server elicitation
# --------------------------------------------------------------------------- #
@pytest.mark.anyio
async def test_gate_denies_writes_when_not_interactive(server):
    policy = ApprovalPolicy(auto_approve=False, interactive=False)
    async with Client(server.mcp, elicitation_callback=policy.handle_elicitation) as c:
        policy.register_tools((await c.list_tools()).tools)
        ok = await policy.call_tool(c, "search_notes", {"query": "atlas"})      # read-only: auto
        assert not ok.is_error
        denied = await policy.call_tool(c, "add_task", {"title": "nope", "due": "2026-09-01"})
        assert denied.is_error and "DENIED" in denied.content[0].text
    assert [e.decision for e in policy.audit] == ["auto", "denied (non-interactive)"]
    assert len(json.loads((server.DATA / "tasks.json").read_text())) == 4   # nothing written


@pytest.mark.anyio
async def test_elicitation_accept_deletes(server):
    asked: list[str] = []

    async def accept(ctx, params):
        asked.append(params.message)
        return ElicitResult(action="accept", content={"confirm": True})

    async with Client(server.mcp, elicitation_callback=accept) as c:
        r = await c.call_tool("delete_note", {"slug": "ideas"})
    assert not r.is_error and r.content[0].text == "Deleted note 'ideas'."
    assert asked and "ideas" in asked[0]
    assert not (server.DATA / "notes" / "ideas.md").exists()


@pytest.mark.anyio
async def test_elicitation_decline_keeps_note(server):
    async def decline(ctx, params):
        return ElicitResult(action="decline")

    async with Client(server.mcp, elicitation_callback=decline) as c:
        r = await c.call_tool("delete_note", {"slug": "ideas"})
    assert not r.is_error and "declined" in r.content[0].text
    assert (server.DATA / "notes" / "ideas.md").exists()


@pytest.mark.anyio
async def test_elicitation_accept_but_unconfirmed(server):
    async def accept_false(ctx, params):
        return ElicitResult(action="accept", content={"confirm": False})

    async with Client(server.mcp, elicitation_callback=accept_false) as c:
        r = await c.call_tool("delete_note", {"slug": "ideas"})
    assert "not confirmed" in r.content[0].text
    assert (server.DATA / "notes" / "ideas.md").exists()


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
@pytest.mark.anyio
async def test_prompts(server):
    async with Client(server.mcp) as c:
        p = await c.get_prompt("weekly_review", {"focus": "Atlas"})
        assert [m.role for m in p.messages] == ["user", "assistant", "user"]
        assert "Atlas" in p.messages[0].content.text
        p2 = await c.get_prompt("daily_briefing", {"tone": "detailed"})
        assert "detailed daily briefing" in p2.messages[0].content.text


# --------------------------------------------------------------------------- #
# Workflows (no LLM)
# --------------------------------------------------------------------------- #
@pytest.mark.anyio
async def test_daily_brief_workflow(server):
    policy = ApprovalPolicy(interactive=False)
    async with Client(server.mcp) as c:
        out = await daily_brief(c, policy)
    assert "Daily brief for 2026-08-23" in out
    assert "#1 Draft Atlas migration plan" in out and "1:1 with Sam" in out


@pytest.mark.anyio
async def test_capture_todos_workflow_is_idempotent(server):
    policy = ApprovalPolicy(auto_approve=True)
    async with Client(server.mcp) as c:
        policy.register_tools((await c.list_tools()).tools)
        first = await capture_todos(c, policy)
        second = await capture_todos(c, policy)
    assert first.startswith("capture_todos: 3 created")
    assert second.startswith("capture_todos: 0 created, 3 already existed")
    assert len(json.loads((server.DATA / "tasks.json").read_text())) == 7


# --------------------------------------------------------------------------- #
# Provider-agnostic adapter: the same MCP schema feeds every LLM API
# --------------------------------------------------------------------------- #
@pytest.mark.anyio
async def test_tool_conversion_is_provider_neutral(server):
    from client.discovery import mcp_tools_to_gemini, mcp_tools_to_openai

    async with Client(server.mcp) as c:
        tools = (await c.list_tools()).tools

    anthropic_t = mcp_tools_to_anthropic(tools)
    openai_t = mcp_tools_to_openai(tools)
    gemini_t = mcp_tools_to_gemini(tools)

    # envelopes differ …
    assert set(anthropic_t[0]) == {"name", "description", "input_schema"}
    assert openai_t[0]["type"] == "function" and set(openai_t[0]["function"]) == {"name", "description", "parameters"}
    assert list(gemini_t[0]) == ["function_declarations"] and len(gemini_t[0]["function_declarations"]) == len(tools)

    # … but the JSON Schema inside is the SAME object, untouched
    assert (anthropic_t[0]["input_schema"]
            == openai_t[0]["function"]["parameters"]
            == gemini_t[0]["function_declarations"][0]["parameters_json_schema"]
            == tools[0].input_schema)

    # and the Gemini dicts validate against the google-genai SDK types
    genai_types = pytest.importorskip("google.genai.types")
    cfg = genai_types.GenerateContentConfig(system_instruction="s", tools=gemini_t)
    assert cfg.tools[0].function_declarations[0].name == tools[0].name


# --------------------------------------------------------------------------- #
# User-defined YAML workflows (client/workflow_engine.py) + AgentSpecs
# --------------------------------------------------------------------------- #
def _write_wf(tmp_path, text):
    p = tmp_path / "wf.yaml"
    p.write_text(text, encoding="utf-8")
    return p


@pytest.mark.anyio
async def test_yaml_workflow_read_tool_and_templating(server, tmp_path):
    from client.workflow_engine import run_yaml_workflow

    wf = _write_wf(tmp_path, """
steps:
  - read: workspace://calendar/today
    as: today
  - tool: list_tasks
    args: {status: open, overdue_only: true}
    as: overdue
  - tool: add_task
    args:
      title: "Follow up: {overdue.0.title}"
      due: "{today.date}"
      priority: "{overdue.0.priority}"     # whole-string placeholder keeps type
    as: created
output: "created #{created.id} due {created.due}"
""")
    policy = ApprovalPolicy(auto_approve=True)
    async with Client(server.mcp) as c:
        policy.register_tools((await c.list_tools()).tools)
        out = await run_yaml_workflow(c, policy, wf, verbose=False)
    assert out == "created #5 due 2026-08-23"
    tasks = json.loads((server.DATA / "tasks.json").read_text())
    assert tasks[-1]["title"] == "Follow up: Draft Atlas migration plan"
    assert tasks[-1]["priority"] == "high"


@pytest.mark.anyio
async def test_yaml_workflow_denied_write_stops_run(server, tmp_path):
    from client.workflow_engine import WorkflowError, run_yaml_workflow

    wf = _write_wf(tmp_path, """
steps:
  - tool: add_task
    args: {title: nope, due: "2026-09-01"}
""")
    policy = ApprovalPolicy(auto_approve=False, interactive=False)
    async with Client(server.mcp) as c:
        policy.register_tools((await c.list_tools()).tools)
        with pytest.raises(WorkflowError, match="DENIED"):
            await run_yaml_workflow(c, policy, wf, verbose=False)
    assert len(json.loads((server.DATA / "tasks.json").read_text())) == 4   # nothing written


@pytest.mark.anyio
async def test_yaml_workflow_agent_step_uses_spec_and_context(server, tmp_path, monkeypatch):
    """The agent: step resolves a spec and receives templated context — the LLM
    itself is stubbed out so the test needs no API key."""
    import client.workflow_engine as eng

    seen = {}

    async def fake_run_agent(client, policy, goal, *, spec=None, verbose=True, model=None):
        seen["goal"], seen["spec"] = goal, spec
        return "do task 1 first"

    monkeypatch.setattr(eng, "load_provider", lambda name: fake_run_agent)
    wf = _write_wf(tmp_path, """
steps:
  - tool: list_tasks
    args: {overdue_only: true}
    as: overdue
  - agent: "Prioritise: {overdue}"
    spec: task-assistant
    as: plan
output: "{plan}"
""")
    policy = ApprovalPolicy(auto_approve=True)
    async with Client(server.mcp) as c:
        policy.register_tools((await c.list_tools()).tools)
        out = await run_or(eng, c, policy, wf)
    assert out == "do task 1 first"
    assert "Draft Atlas migration plan" in seen["goal"]      # context was templated in
    assert seen["spec"].name == "task-assistant"             # loaded from agents/*.yaml
    assert seen["spec"].allowed_tools == ["list_tasks", "add_task", "complete_task"]


async def run_or(eng, c, policy, wf):
    return await eng.run_yaml_workflow(c, policy, wf, verbose=False)


@pytest.mark.anyio
async def test_yaml_workflow_bad_reference_fails_clearly(server, tmp_path):
    from client.workflow_engine import WorkflowError, run_yaml_workflow

    wf = _write_wf(tmp_path, """
steps:
  - tool: list_tasks
    as: tasks
output: "{tazks}"
""")
    policy = ApprovalPolicy(auto_approve=True)
    async with Client(server.mcp) as c:
        policy.register_tools((await c.list_tools()).tools)
        with pytest.raises(WorkflowError, match="tazks"):
            await run_yaml_workflow(c, policy, wf, verbose=False)


@pytest.mark.anyio
async def test_agent_spec_filters_tool_surface(server):
    from client.agent_specs import filter_tools_for_spec, load_specs

    specs = load_specs()
    assert "generalist" in specs and "task-assistant" in specs and "notes-librarian" in specs
    async with Client(server.mcp) as c:
        tools = (await c.list_tools()).tools
    filtered = filter_tools_for_spec(tools, specs["notes-librarian"])
    assert {t.name for t in filtered} == {"search_notes", "create_note", "delete_note"}
    assert len(filter_tools_for_spec(tools, specs["generalist"])) == len(tools)   # [] = all
    with pytest.raises(SystemExit, match="doesn't have"):
        from client.agent_specs import AgentSpec
        filter_tools_for_spec(tools, AgentSpec(name="bad", allowed_tools=["launch_rockets"]))
