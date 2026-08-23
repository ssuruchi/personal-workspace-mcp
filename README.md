# Personal Workspace Assistant — learning MCP by building both sides

A small, complete **Model Context Protocol** setup you can read top to bottom:

* an **MCP server** that wraps a "personal workspace" (markdown notes, tasks, calendar, e-mail outbox) — [`server/workspace_server.py`](server/workspace_server.py)
* an **MCP client** with three *workflows* and one *agent* (selectable brain: Claude, OpenAI, Gemini, or a local model) — [`client/`](client/)
* tests that run the real protocol in memory — [`tests/`](tests/)
* a line-by-line trace of one real agent run — [`docs/walkthrough.md`](docs/walkthrough.md)
* what "harness", "agent", "config" and "workflow" actually mean — [`docs/concepts.md`](docs/concepts.md)

Everything maps back to the ideas in the MCP write-up you read. The table at the bottom is the cheat-sheet.

```
            ┌──────────────────────────────┐   JSON-RPC 2.0 over       ┌───────────────────────────────┐
            │  client/                     │   stdio pipes  –or–       │  server/workspace_server.py   │
            │  ├ connection.py  (transport)│   streamable HTTP         │  MCPServer("personal-workspace")
            │  ├ discovery.py   (list_*)   │◄────────────────────────►│   RESOURCES workspace://…      │
 you ─────► │  ├ gate.py        (HITL)     │   ONE stateful session    │   TOOLS     add_task, …        │
            │  ├ workflows.py   (code)     │                           │   PROMPTS   daily_briefing, …  │
            │  └ agent*.py      (LLM loop) │                           │        │                       │
            └──────────┬───────────────────┘                           └────────┼───────────────────────┘
                       │ provider API (tools = the SAME MCP JSON Schemas)       ▼
                       ▼                                                   ./data/  (notes/*.md, tasks.json, …)
        Claude ─or─ OpenAI/local ─or─ Gemini      (agent.py)  (agent_openai.py)  (agent_gemini.py)
```

---

## 1. Setup

```powershell
cd personal-workspace-mcp
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env        # only needed for the Claude-backed commands
```

Everything below is run from this folder with `.venv\Scripts\python`.
The demo data is dated around **2026‑08‑23**; `.env.example` pins `WORKSPACE_TODAY` to that so "today"/"overdue" make sense.

## 2. Run it

### See the server describe itself (no LLM)

```powershell
.venv\Scripts\python -m client.main inspect          # human-readable
.venv\Scripts\python -m client.main inspect --json   # the raw JSON Schemas
```

Output (abridged):

```
MCP protocol  : 2026-07-28
Server        : personal-workspace v0.1.0
Capabilities  : prompts, resources, tools

TOOLS (7)  — what the model may *request*
  • search_notes(query: string, max_results?: integer)  [read-only]
  • add_task(title: string, due: string, priority?: string)  [write]
  • delete_note(slug: string)  [destructive]
  • send_email(to: string, subject: string, body: string)  [external]
RESOURCES (4)  — context the model may *read*
  • workspace://notes                  application/json
  • workspace://notes/{slug}           text/markdown      (template)
PROMPTS (2)  — reusable templates the *user/client* picks
  • daily_briefing(tone?)   • weekly_review(focus)
```

### Workflows (code decides the steps)

```powershell
.venv\Scripts\python -m client.main workflow daily-brief      # resources only, no LLM
.venv\Scripts\python -m client.main workflow capture-todos    # scans notes for "TODO:" → add_task (you approve each)
.venv\Scripts\python -m client.main workflow weekly-review    # MCP prompt + resources → ONE Claude call
```

`capture-todos` will stop and ask `Approve? [y/N]` for every `add_task` — that is the human-in-the-loop gate.
Use `--yes` to auto-approve (demos/CI) or `--no-input` to see the deny path.

### Agent (an LLM decides the steps)

```powershell
.venv\Scripts\python -m client.main agent "What is overdue? Add a task to unblock it and tell me what you did."
.venv\Scripts\python -m client.main agent "Delete the note about random ideas"     # watch both gates fire
```

The agent's "brain" is configurable — the MCP side does not change at all:

```powershell
.venv\Scripts\python -m client.main agent --provider openai "..."        # needs OPENAI_API_KEY
.venv\Scripts\python -m client.main agent --provider gemini "..."        # needs GEMINI_API_KEY
.venv\Scripts\python -m client.main agent --provider openai --model qwen3 "..."   # local, see below
```

| provider | file | SDK | key (in `.env`) | default model (override) |
|---|---|---|---|---|
| `anthropic` (default) | [`client/agent.py`](client/agent.py) | `anthropic` | `ANTHROPIC_API_KEY` | `claude-opus-5` (`ANTHROPIC_MODEL`) |
| `openai` | [`client/agent_openai.py`](client/agent_openai.py) | `openai` | `OPENAI_API_KEY` | `gpt-5.1` (`OPENAI_MODEL`) |
| `gemini` | [`client/agent_gemini.py`](client/agent_gemini.py) | `google-genai` | `GEMINI_API_KEY` | `gemini-2.5-pro` (`GEMINI_MODEL`) |

**Local models** (Ollama, LM Studio, vLLM, llama.cpp server) speak the OpenAI wire format, so they use `--provider openai` with a redirected base URL — in `.env`:

```
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=qwen3
OPENAI_API_KEY=ollama        # any non-empty value; local servers ignore it
```

Read the three `agent*.py` files side by side: steps 1/3/4/5 (discover, intercept, execute, feed back) are the same shared code — only the **think** step and its wire format differ. The tool schemas are converted by the three tiny functions at the bottom of [`client/discovery.py`](client/discovery.py), and the test `test_tool_conversion_is_provider_neutral` asserts all three carry the *identical* JSON Schema from `tools/list`. That is the "swap the model without touching the tool code" promise of MCP, demonstrated.

You'll see `[tool_use] …` / `[tool_result] …` lines as Claude discovers and calls tools, approval prompts for anything that writes, and an **audit trail** at the end.

### User-defined workflows and agents (no Python required)

Drop a YAML file into [`workflows/`](workflows/) and it becomes runnable — `workflow --list` shows everything:

```yaml
# workflows/morning-routine.yaml — fully deterministic, no LLM
steps:
  - read: workspace://calendar/today
    as: today
  - tool: list_tasks
    args: {status: open, overdue_only: true}
    as: overdue
  - tool: send_email                    # [external] -> the gate asks for approval
    args: {to: you@example.com, subject: "Brief {today.date}", body: "Overdue: {overdue}"}
output: "Brief for {today.date} sent."
```

Three step kinds: `read:` (MCP resource), `tool:` (MCP tool, **through the approval gate**), and `agent:` (a bounded LLM step — the workflow decides *what/when*, the agent decides *how*). Results are named with `as:` and referenced with `{name}` / `{name.field}` / `{name.0}`; a whole-string placeholder keeps its type so `"{task.id}"` stays an integer. There is deliberately no if/else in the engine — deterministic control flow belongs to the author, judgement belongs in an `agent:` step. See [`workflows/plan-my-day.yaml`](workflows/plan-my-day.yaml) for the hybrid pattern.

Agents are data too: a YAML file in [`agents/`](agents/) = persona + tool subset + model ("an agent is the generic loop plus a spec"):

```yaml
# agents/task-assistant.yaml
system_prompt: |
  You are the task assistant. You manage the user's task list and nothing else.
allowed_tools: [list_tasks, add_task, complete_task]   # the model never sees the rest
max_turns: 8
```

```powershell
.venv\Scripts\python -m client.main agent --spec task-assistant "what should I do first today?"
.venv\Scripts\python -m client.main workflow plan-my-day        # workflow invoking that spec
```

The engine ([`client/workflow_engine.py`](client/workflow_engine.py)) and spec loader ([`client/agent_specs.py`](client/agent_specs.py)) funnel every action through the same `ApprovalPolicy` — user-authored workflows inherit human-in-the-loop gating and the audit trail for free.

### Two transports, same code

```powershell
# Terminal 1 — run the server as a standalone HTTP service
.venv\Scripts\python -m server.workspace_server --http        # http://127.0.0.1:8000/mcp

# Terminal 2 — point the same client at it
.venv\Scripts\python -m client.main --transport http inspect
.venv\Scripts\python -m client.main --transport http workflow daily-brief
```

Default (`--transport stdio`) spawns the server as a child process and talks over its stdin/stdout — no port, no network.

### Tests

```powershell
.venv\Scripts\python -m pytest -q     # 20 tests, in-memory client⇄server (seeded from tests/fixtures, so playing with the CLI can't break them)
```

---

## 3. What to read, in order

| # | File | What it teaches |
|---|------|-----------------|
| 1 | [`server/workspace_server.py`](server/workspace_server.py) | The three primitives, ToolAnnotations, elicitation, transports. Read the comments top to bottom. |
| 2 | [`client/discovery.py`](client/discovery.py) | Runtime discovery (`list_tools`, `list_resources`, `list_resource_templates`, `list_prompts`) and the 5‑line MCP→Claude tool conversion. |
| 3 | [`client/gate.py`](client/gate.py) | The client-side intercept layer (annotations → auto / ask / deny) + handling server elicitation. |
| 4 | [`client/connection.py`](client/connection.py) | One stateful session; stdio vs streamable HTTP. |
| 5 | [`client/workflows.py`](client/workflows.py) | MCP used *from code*: resources, a write tool through the gate, and prompt+resources→single LLM call. |
| 6 | [`client/agent.py`](client/agent.py) | The hand-written agent loop: discover → think → intercept → execute → feed back. |
| 6b | [`client/agent_openai.py`](client/agent_openai.py), [`client/agent_gemini.py`](client/agent_gemini.py) | The same loop on other providers/local models — diff them against agent.py to see exactly what an LLM swap touches (only the think step). |
| 7 | [`client/agent_specs.py`](client/agent_specs.py) | An "agent" as data: persona + tool subset + model. Users add agents via `agents/*.yaml`. |
| 8 | [`client/workflow_engine.py`](client/workflow_engine.py) | User-defined YAML workflows: `read`/`tool`/`agent` steps, templating, gate-aware. |
| 9 | [`tests/test_workspace_mcp.py`](tests/test_workspace_mcp.py) | Every claim above, asserted. |

---

## 4. Mapping the reading to the code

### "Three structural primitives — Resources, Tools, Prompts"

| Primitive | In the server | In the client |
|-----------|---------------|---------------|
| **Resources** — read-only context | `@mcp.resource("workspace://tasks")`, template `@mcp.resource("workspace://notes/{slug}")` | `client.read_resource(uri)` in `workflows.py`; listed in the agent's system prompt |
| **Tools** — actions | `@mcp.tool(annotations=…)` on `add_task`, `delete_note`, … JSON Schema is generated from the type hints | `mcp_tools_to_anthropic()` → Claude `tools=[…]`; executed via `policy.call_tool()` |
| **Prompts** — reusable templates | `@mcp.prompt()` `daily_briefing(tone)`, `weekly_review(focus)` (returns 1 or N messages) | `client.get_prompt("weekly_review", {"focus": …})` in `weekly_review()` workflow |

The server has *no other way* to expose anything. A REST endpoint would have been a URL + verb; here it is forced into one of three shapes with a schema attached.

### "Self-describing requirement"

Run `inspect --json`. Every tool carries `inputSchema` (and `outputSchema` when the return type is structured), every resource a `mimeType`, every prompt its argument list, and the handshake carries `instructions` + `capabilities`. [`discovery.py`](client/discovery.py) is generic — it would print the same report for *any* MCP server. The agent never hard-codes a tool name: it is handed `tools/list` at runtime and Claude picks.

### "Stateful transport"

[`connection.py`](client/connection.py): `async with Client(...)` opens one session; handshake once; then every `list_*`, `read_resource`, `call_tool`, `get_prompt` rides that same connection. Two things in the demo are only possible because the connection is stateful and bidirectional:

* `send_email` calls `ctx.report_progress(...)` — the **server streams progress back mid-call**.
* `delete_note` triggers **elicitation** — the server pauses the call, asks the *client* a question, and resumes when the human answers.

With stdio the "connection" is a pair of OS pipes to a child process (`StdioServerParameters(command=sys.executable, args=["-m","server.workspace_server"])`). With `--http` it is Streamable HTTP to a URL; in production that is where TLS and a bearer/OAuth/mTLS handshake go. The application code above the transport is identical.

### "Human-in-the-loop constraint"

Two layers, and they compose:

1. **Client intercept** — [`gate.py`](client/gate.py). The server *declares* risk with `ToolAnnotations(read_only_hint / destructive_hint / open_world_hint)`; the client *decides*: read-only runs, everything else prompts `Approve? [y/N]`, and a denial is returned to the model as an `is_error` tool result ("DENIED by the user … do not retry"). Every decision lands in an audit trail.
2. **Server elicitation** — `delete_note` has a parameter `Annotated[ElicitationResult[DeleteConfirmation], Resolve(_ask_delete_confirmation)]`. That parameter is **not** in the schema the model sees (test `test_server_is_self_describing` asserts this); it is filled by an `elicitation/create` round-trip to the human. Accept / decline / cancel are all handled.

### "Least privilege"

The model cannot run arbitrary code or read arbitrary files. It can only call the seven functions advertised, each validated by Pydantic, each confined to `./data` with a slug whitelist (`_slug_ok`). The SDK additionally rejects path traversal in resource templates (`test_template_rejects_bad_slug`).

### "You will USE MCP as a client and CREATE an MCP server"

* **Create**: `server/` wraps *your* data (notes/tasks/calendar) — the proprietary part.
* **Use**: `client/` could be pointed at any other MCP server (GitHub, filesystem, memory…) with only the `StdioServerParameters`/URL changed — the discovery, gate and agent code stay the same.

### Workflow vs agent

| | `workflows.py` | `agent.py` |
|---|---|---|
| Who decides the next step | your Python code | Claude, from the tool list |
| LLM calls | 0 (`daily-brief`, `capture-todos`) or exactly 1 (`weekly-review`) | a loop, until `stop_reason == "end_turn"` |
| MCP usage | `read_resource`, `call_tool`, `get_prompt` directly | `list_tools` → Claude `tools`; `call_tool` for each `tool_use` |
| Human gate | same `ApprovalPolicy` | same `ApprovalPolicy` |

Both go through the same `ApprovalPolicy.call_tool()` — that single choke point is the intercept layer.

---

## 5. Notes on versions

* `mcp` **2.0** (Python SDK). Names changed from the 1.x tutorials you may find online: `FastMCP` → `MCPServer`, fields are snake_case (`tool.input_schema`, `annotations.read_only_hint`), and elicitation is expressed as a *resolver* parameter (`Annotated[T, Resolve(fn)]` where `fn` returns `Elicit(...)`) instead of `ctx.elicit()`. Negotiated protocol version is `2026-07-28`.
* `anthropic` **1.0**. The agent uses the plain Messages API with `claude-opus-5`, adaptive thinking, and `fallbacks="default"` (beta `server-side-fallback-2026-07-01`) so a safety-classifier decline is retried server-side instead of surfacing as a refusal; `stop_reason == "refusal"` is still checked. The SDK also ships a beta `tool_runner` and `anthropic.lib.tools.mcp.async_mcp_tool` helper that can drive the same loop in fewer lines; the hand-written loop is kept here so the intercept step is visible.
* Credentials: `ANTHROPIC_API_KEY` in `.env`, or an `ant auth login` profile — the SDK picks either up automatically.
