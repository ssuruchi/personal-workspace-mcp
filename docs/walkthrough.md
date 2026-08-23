# Walkthrough: anatomy of one agent run

This document traces a single real run, line by line, and maps every line of
output to the code and to the MCP protocol messages underneath. The command:

```bash
python -m client.main agent --provider openai "what's overdue?"
```

And the output it produced:

```
[agent] provider=openai
[tool_call] list_tasks({"status":"open","overdue_only":true})
[tool_result] {"result": [{"id": 1, "title": "Draft Atlas migration plan", "due": "2026-08-22", "priority": "high", "done": false}]}

[assistant · turn 2] You have 1 overdue task:

- [ ] Draft Atlas migration plan — due 2026-08-22 (priority: high)

── Answer ───────────────────────────────────────────────────
You have 1 overdue task:

- [ ] Draft Atlas migration plan — due 2026-08-22 (priority: high)

── Audit trail ──────────────────────────────────────────────
 1. list_tasks     [read-only  ] auto                     {"status": "open", "overdue_only": true}
    ↳ { "id": 1, "title": "Draft Atlas migration plan", "due": "2026-08-22", "priority": "high", "done": false }
```

One command, two LLM API calls, one MCP tool call, zero hard-coded knowledge
about tasks in the agent. Here is the timeline.

---

## Phase 0 — before any AI is involved

1. [`client/main.py`](../client/main.py) parses the arguments. `load_provider("openai")`
   lazily imports [`client/agent_openai.py`](../client/agent_openai.py) and checks that
   `OPENAI_API_KEY` is set → prints `[agent] provider=openai`. A missing key fails
   *here*, fast, before anything is spawned.
2. [`client/connection.py`](../client/connection.py) **spawns the MCP server as a child
   process** (`python -m server.workspace_server`) and connects to its stdin/stdout
   pipes. No port, no network — that is the **stdio transport**. The connection
   inherits OS process isolation; there is nothing to sniff remotely.
3. **Handshake** (once per session): client and server negotiate the protocol
   version (`2026-07-28`), and the server sends its `capabilities`
   (tools / resources / prompts) plus its `instructions` string. The connection
   now stays open — **one stateful session** carries everything that follows.
   This is MCP's "stateful transport" constraint in action.

## Phase 1 — discovery (the self-describing requirement)

The agent sends `tools/list` over the session. The server answers with all
seven tools — name, description, **JSON Schema** (generated from the Python
type hints), and `ToolAnnotations`. Two things happen with that answer:

* `policy.register_tools(tools)` — the human-in-the-loop **gate**
  ([`client/gate.py`](../client/gate.py)) memorises each tool's risk annotations
  for later.
* `mcp_tools_to_openai(tools)` — the schemas are re-enveloped into OpenAI's
  `{type: "function", function: {name, parameters}}` format
  ([`client/discovery.py`](../client/discovery.py)). The schema JSON itself is
  passed through **untouched** — the converter is a field rename.

The server's handshake `instructions` plus its resource list become the system
prompt. Nothing about "tasks" or "overdue" is hard-coded in the agent — the
model is about to learn what is possible entirely from this discovery step.

## Phase 2 — think, turn 1 (LLM call #1)

The client POSTs to the provider: system prompt + the goal (`what's overdue?`)
+ the seven tool definitions. The model reads the schemas, notices that
`list_tasks` has an `overdue_only: boolean` parameter (which exists only
because the server's type hints said so), and responds **not with text** but
with a tool request:

```json
"tool_calls": [{"function": {"name": "list_tasks",
                             "arguments": "{\"status\":\"open\",\"overdue_only\":true}"}}]
```

That is the output line `[tool_call] list_tasks({...})`.

**The model didn't *do* anything — it *requested* something.** The model
proposes; the client disposes.

## Phase 3 — intercept (the human-in-the-loop gate)

Before the call touches the server, `ApprovalPolicy.authorize()` runs
([`client/gate.py`](../client/gate.py)). It looks up `list_tasks`, finds the
annotation the *server* published — `read_only_hint=True` — classifies the call
`read-only`, and lets it through **without asking the human**. Hence the audit
line:

```
1. list_tasks  [read-only ] auto ...
```

Had the model asked for `add_task` (a write), the run would have stopped at
`Approve? [y/N]`. Had it asked for `delete_note`, that *plus* the server's own
elicitation question ("Permanently delete …?"). Neither fired here because the
model only needed to read.

## Phase 4 — execute over MCP

Only now does the client send `tools/call` down the still-open stdio session.
The server:

1. validates the arguments against the tool's schema (Pydantic),
2. runs the actual Python function `list_tasks(status="open", overdue_only=True)`,
3. filters `data/tasks.json` against today's date,
4. returns a `CallToolResult` with `structuredContent`.

`tool_result_text()` flattens that into the JSON on the `[tool_result]` line.
Note what the model could **not** have done instead: read an arbitrary file,
run a shell command, touch anything outside the seven advertised functions.
That is the least-privilege constraint — the tool surface *is* the permission
surface.

## Phase 5 — feed back + think, turn 2 (LLM call #2)

The result goes back to the provider as a `role: "tool"` message appended to
the conversation. The model now has real data, needs nothing further, and
answers in prose instead of requesting more tools — the
`[assistant · turn 2]` line. No `tool_calls` in the response → the loop exits
and prints the final **Answer**.

Then the audit trail prints (every tool call, its risk class, who approved it,
what came back), the `async with` blocks unwind, the stdio session closes, and
the server child process exits. Nothing is left running.

---

## The scorecard

| | count | |
|---|---|---|
| LLM API calls | **2** | think → tool request; result → final answer |
| MCP tool calls | **1** | `tools/call list_tasks` |
| MCP discovery calls | 3–4 | `tools/list`, `resources/list`, templates — handshake + loop start |
| Human approvals | **0** | only a read-only tool was needed |
| Lines of agent code that know about "tasks" | **0** | all knowledge came from discovery |

## Experiments to run next

* Run the **same question** with `--provider anthropic` or `--provider gemini`:
  same `[tool_call]` → `[tool_result]` → answer shape, identical audit trail —
  everything except Phase 2's wire format is shared code. That is MCP's
  "swap the model without touching the tool code" promise, live.
* Force a write: `agent "add a task to draft the migration plan tomorrow"` —
  Phase 3 will stop and ask you.
* Force both gates: `agent "delete the note about random ideas"` — client
  approval *and* server elicitation.
* Watch the wire: run the server with `--http` in one terminal and the client
  with `--transport http` in the other; same phases, different transport.
