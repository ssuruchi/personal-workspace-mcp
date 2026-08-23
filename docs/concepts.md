# Concepts: harness, agent, config, workflow

Four words that get used loosely everywhere. This page pins down what each one
means *in this repository* — every claim links to the code that implements it.

---

## 1. Harness

An LLM API does exactly one thing: you send it messages, it returns one
message. It cannot loop, cannot execute a tool, cannot ask a human, cannot
remember anything past the request. The **harness** is everything wrapped
around the model that turns raw text-prediction into a working system:

```
            ┌─────────────────── THE HARNESS ────────────────────┐
            │  the loop            (keep calling until done)      │
            │  tool execution      (actually run what it asked)   │
            │  context management  (history, truncation, memory)  │
            │  permissions/HITL    (gate, audit)                  │
user ──────►│  error handling      (retries, denied calls)        │──────► model API
            │  termination         (max turns, stop conditions)   │◄────── (just text in,
            │  I/O                 (CLI, transcripts, progress)   │         text out)
            └─────────────────────────────────────────────────────┘
```

This repo **is** a harness, written by hand:

| Harness responsibility | Where it lives here |
|---|---|
| The loop | [`client/agent.py`](../client/agent.py) (and its two siblings) |
| Tool execution | [`client/gate.py`](../client/gate.py) `ApprovalPolicy.call_tool` → MCP `tools/call` |
| Context management | the `messages` list each loop appends to |
| Permissions / human-in-the-loop | [`client/gate.py`](../client/gate.py) + server elicitation |
| Error handling | denied calls fed back as `is_error` results; `MCP error:` catch |
| Termination | `max_turns`, "no tool calls → done" |
| I/O | [`client/main.py`](../client/main.py), the `[tool_use]` trace lines, the audit trail |

The model contributes **none** of that. That is why swapping the brain
(Claude ⇄ GPT ⇄ Gemini ⇄ local) was a one-file change per provider: the harness
is provider-independent; only the single API call inside it changes.

The word matters because it is the real axis on which products differ:

* **Anthropic tool runner / OpenAI Agents SDK** — minimal harness: the loop is
  driven for you, you supply tools, you host everything.
* **Claude Code / Claude Agent SDK** — a large harness: loop + built-in
  file/bash tools + permission prompts + subagents + context management.
* **LangChain / LangGraph, AutoGen, CrewAI** — harness construction kits.
* **Anthropic Managed Agents** — harness *plus* hosting: the provider runs the
  loop and a sandbox for you.

None of these adds "intelligence". A framework agent is the same loop wearing a
coat. If you understand the ~90 lines of `agent.py`, you understand what every
one of these products is doing on your behalf.

---

## 2. Agent

"Agent" is not a library or a product — it is a **control-flow pattern**:

> An agent = an LLM + a set of tools + a loop, where **the model (not your
> code) decides what to do next**, and the loop continues until the model
> decides it is done.

Check [`client/agent.py`](../client/agent.py) against that definition:

```
while turns < MAX_TURNS:          # the loop
    response = llm(goal, tools)   # the model decides
    if no tool requests: return   # the model decides when it's done
    execute the requested tools   # act
    feed results back             # observe
```

Contrast with [`client/workflows.py`](../client/workflows.py), where *Python*
decides every step and the LLM (if used at all) is called exactly once as a
text transformer. Same ingredients, opposite **locus of control**. That is the
entire agent-vs-workflow distinction.

**Raw Python or a framework?** You do not need LangChain to "create an agent" —
`agent.py` is a complete, real agent in ~90 lines of raw Python. Frameworks add
harness plumbing (tracing, persistence, multi-agent graphs, provider swapping),
not agent-ness. A sane progression: write them raw until a framework solves a
problem you *actually have* — usually observability, durable state across
restarts, or multi-agent orchestration.

---

## 3. Config — an "agent" as data, not code

Watch what actually differs when the agent handles "what's overdue?" versus
"file my meeting notes": **nothing in the code**. The loop is fully generic.
Task-specificity lives entirely in three inputs:

1. **the system prompt** — persona and rules
2. **the tool surface** — which tools the model is allowed to see
3. **the goal** — the user's message

So a "specialised agent" is just a record holding 1 and 2 (plus model/provider
and a turn budget). That record here is
[`AgentSpec`](../client/agent_specs.py):

```yaml
# agents/task-assistant.yaml  — a new agent, zero Python
system_prompt: |
  You are the task assistant. You manage the user's task list and nothing else.
allowed_tools: [list_tasks, add_task, complete_task]
max_turns: 8
```

```bash
python -m client.main agent --spec task-assistant "what should I do first today?"
```

One detail worth copying into any system you build: the **gate registers the
full tool list** (so risk classification always works), while the **model sees
only the spec's subset**. Restriction of the model is a prompt-side concern;
safety enforcement stays global.

**When to specialise** — not reflexively by task type, but on these signals:

* the tool count grows past what a model picks from reliably (dozens+),
* two tasks need conflicting instructions ("be exhaustive" vs "be terse"),
* different risk profiles (a read-only reporter you can auto-approve, next to
  a writer that always gates),
* different cost points (cheap model + narrow tools for routine work).

Start with one generalist. Split when you observe it failing. This
"agent = generic loop + spec file" design is not a toy convention — it is how
real products define agents (e.g. Claude Code subagents are markdown files:
frontmatter with a tool list, body as the system prompt).

---

## 4. Workflow — and letting users write their own

A workflow is the opposite pattern: **your code (or a config file) decides the
steps**; any LLM involvement is a bounded, single-purpose call. This repo has
both forms:

* **code workflows** — [`client/workflows.py`](../client/workflows.py)
  (`daily-brief`, `capture-todos`, `weekly-review`)
* **user-defined workflows** — YAML in [`workflows/`](../workflows/), executed
  by [`client/workflow_engine.py`](../client/workflow_engine.py)

User-defined workflows sit on a capability ladder:

| Level | What | Here |
|---|---|---|
| 1 | **Declarative steps**: `read:` resources, `tool:` calls, `{name.field}` templating. Deterministic, auditable, no code. | [`workflows/morning-routine.yaml`](../workflows/morning-routine.yaml) |
| 2 | **Hybrid**: add an `agent:` step — the workflow decides *what and when*, a scoped agent decides *how*. | [`workflows/plan-my-day.yaml`](../workflows/plan-my-day.yaml) |
| 3 | **Server-owned templates**: MCP *prompts* are workflows the server author publishes for any client to discover. | `daily_briefing`, `weekly_review` in [`server/workspace_server.py`](../server/workspace_server.py) |
| 4 | **Model-planned**: an LLM *generates* a Level-1 file from a goal; a human reviews it; the deterministic engine executes it. | (exercise for the reader) |

Two design decisions in the engine worth understanding:

* **No if/else, no loops, no expressions — on purpose.** Deterministic control
  flow belongs to the workflow author; open-ended judgement belongs inside an
  `agent:` step. If a step seems to need a conditional, it probably wants to be
  an agent goal instead. Keeping the engine dumb keeps it predictable,
  auditable, and safe to hand to users.
* **Everything funnels through one choke point.** Both `tool:` steps and every
  tool an `agent:` step requests go through the same
  `ApprovalPolicy.call_tool()`. User-authored workflows therefore inherit
  human-in-the-loop gating and the audit trail *for free* — there is no path
  around the gate to forget about.

Skip heavyweight engines (Temporal, Airflow, n8n) until you need durability
across crashes or scheduling at scale; the concepts transfer directly from the
60-line runner here.

---

## The one-sentence versions

* **Harness** — everything around the model that makes it able to *do* things;
  the model only ever returns one message.
* **Agent** — a loop in which the model chooses the next action.
* **Config (AgentSpec)** — an agent's identity as data: prompt + tool subset +
  model; one loop, many agents.
* **Workflow** — a sequence chosen by code/config, where the model (if present)
  is a bounded step, not the driver.

And the composition rule that ties the whole repo together: **the gate does not
care which of these is calling.** Agent, workflow, or user-authored YAML — every
action crosses the same intercept layer, which is exactly the
human-in-the-loop architecture MCP's design anticipates.
