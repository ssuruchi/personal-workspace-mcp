"""
AGENT (OpenAI-compatible) — same loop as agent.py, different "brain".

Read this side by side with agent.py. The MCP half — discover, intercept,
execute, audit — is byte-for-byte the same imports (discovery.py, gate.py).
Only the THINK step changes:

    Anthropic                          OpenAI Chat Completions
    ---------                          -----------------------
    tools=[{name, input_schema}]       tools=[{type:"function", function:{name, parameters}}]
    response.content tool_use blocks   message.tool_calls (arguments arrive as a JSON *string*)
    role:"user" tool_result blocks     role:"tool" messages, one per call
    stop_reason == "tool_use"          finish_reason == "tool_calls"

Because this uses the OpenAI *wire format*, the same file also drives local
models: anything served by Ollama, LM Studio, vLLM, llama.cpp, Groq, etc.
exposes this API. Point OPENAI_BASE_URL at it, e.g.

    OPENAI_BASE_URL=http://localhost:11434/v1  OPENAI_MODEL=qwen3  OPENAI_API_KEY=ollama
"""

from __future__ import annotations

import json
import os
import sys

import openai
from mcp import Client

from .agent_specs import AgentSpec, compose_system_prompt, filter_tools_for_spec
from .discovery import build_agent_system_prompt, mcp_tools_to_openai, tool_result_text
from .gate import ApprovalPolicy

DEFAULT_MODEL = "gpt-5.1"
MAX_TURNS = 12


async def run_agent(client: Client, policy: ApprovalPolicy, goal: str, *,
                    model: str | None = None, verbose: bool = True,
                    spec: AgentSpec | None = None) -> str:
    model = model or (spec.model if spec else None) or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
    max_turns = spec.max_turns if spec else MAX_TURNS

    # 1. DISCOVER — identical to agent.py.
    tools = (await client.list_tools()).tools
    policy.register_tools(tools)                  # gate sees all; model sees the spec's subset
    oa_tools = mcp_tools_to_openai(filter_tools_for_spec(tools, spec))
    system = compose_system_prompt(await build_agent_system_prompt(client), spec)

    # base_url from OPENAI_BASE_URL (env) makes this work with local servers.
    oa = openai.OpenAI()
    messages: list[dict] = [
        {"role": "system", "content": system},    # OpenAI: system prompt is a message
        {"role": "user", "content": goal},
    ]

    for turn in range(1, max_turns + 1):
        # 2. THINK
        response = oa.chat.completions.create(model=model, messages=messages, tools=oa_tools)
        msg = response.choices[0].message

        if verbose and msg.content:
            print(f"\n[assistant · turn {turn}] {msg.content}", file=sys.stderr)

        if not msg.tool_calls:
            return msg.content or ""              # final answer

        # Echo the assistant turn (with its tool_calls) back into history.
        messages.append({"role": "assistant", "content": msg.content,
                         "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})

        # 3 + 4. INTERCEPT + EXECUTE — same gate, same MCP session.
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")   # note: JSON string, not dict
            if verbose:
                print(f"[tool_call] {tc.function.name}({tc.function.arguments})", file=sys.stderr)
            try:
                mcp_result = await policy.call_tool(client, tc.function.name, args)   # <- the gate
                content = tool_result_text(mcp_result)
                if mcp_result.is_error:
                    content = f"ERROR: {content}"              # no is_error flag in this API
            except Exception as exc:
                content = f"MCP error: {exc}"
            if verbose:
                print(f"[tool_result] {content[:200].replace(chr(10), ' ')}", file=sys.stderr)
            # 5. FEED BACK — one role:"tool" message per call, keyed by tool_call_id.
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})

    return "Stopped: reached the maximum number of agent turns."
