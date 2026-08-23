"""
AGENT (Anthropic / Claude) — the model decides which MCP tools to call,
the client executes them.

This is the piece of the reading that says "MCP is built for AI agents to
autonomously discover and navigate tools at runtime". The loop is written out
by hand on purpose, so every hop is visible:

   1. discover   tools/list over MCP  ->  Claude `tools=[...]` (JSON Schema passthrough)
   2. think      Claude returns tool_use blocks (or a final answer)
   3. intercept  ApprovalPolicy decides: auto / ask human / deny
   4. execute    tools/call over the SAME stateful MCP session
   5. feed back  tool_result blocks -> Claude, repeat

Nothing in this file knows what the tools *are*. Swap the server and the agent
adapts at runtime.

Provider comparison: agent_openai.py and agent_gemini.py implement the SAME
loop against other LLM APIs. Steps 1, 3, 4 are identical (shared MCP-side code
in discovery.py / gate.py); only step 2's wire format differs. Select with
`--provider` on the CLI. (The Anthropic SDK also has a beta "tool runner" that
can drive this loop for you; the manual loop keeps the intercept explicit.)
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import anthropic
from mcp import Client

from .discovery import build_agent_system_prompt, mcp_tools_to_anthropic, tool_result_text
from .gate import ApprovalPolicy

DEFAULT_MODEL = "claude-opus-5"
MAX_TURNS = 12


async def run_agent(client: Client, policy: ApprovalPolicy, goal: str, *,
                    model: str | None = None, verbose: bool = True) -> str:
    model = model or os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)

    # 1. DISCOVER — ask the server what it can do, right now.
    tools = (await client.list_tools()).tools
    policy.register_tools(tools)                  # gate needs the annotations
    claude_tools = mcp_tools_to_anthropic(tools)  # schema passthrough
    system = await build_agent_system_prompt(client)

    anthropic_client = anthropic.Anthropic()
    messages: list[dict[str, Any]] = [{"role": "user", "content": goal}]

    for turn in range(1, MAX_TURNS + 1):
        # 2. THINK — one model call; Claude may answer or ask for tools.
        response = anthropic_client.beta.messages.create(
            model=model,
            max_tokens=16000,
            system=system,
            tools=claude_tools,
            thinking={"type": "adaptive"},
            output_config={"effort": "medium"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",      # safety-classifier decline -> retried on a fallback model server-side
            messages=messages,
        )

        if response.stop_reason == "refusal":
            return "Model declined the request."
        if response.stop_reason == "max_tokens":
            return "Response was cut off (max_tokens)."
        if response.stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": response.content})
            continue

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        texts = [b.text for b in response.content if b.type == "text"]
        if verbose and texts:
            print(f"\n[assistant · turn {turn}] " + " ".join(texts), file=sys.stderr)

        if response.stop_reason != "tool_use" or not tool_uses:
            return "\n".join(texts)             # final answer

        # Echo the assistant turn back *unchanged* (thinking + tool_use blocks included).
        messages.append({"role": "assistant", "content": response.content})

        # 3 + 4. INTERCEPT + EXECUTE — all tool_results go back in ONE user message.
        results: list[dict[str, Any]] = []
        for tu in tool_uses:
            args = dict(tu.input) if isinstance(tu.input, dict) else json.loads(tu.input)
            if verbose:
                print(f"[tool_use] {tu.name}({json.dumps(args, ensure_ascii=False)})", file=sys.stderr)
            try:
                mcp_result = await policy.call_tool(client, tu.name, args)   # <- the gate
                content = tool_result_text(mcp_result)
                is_error = bool(mcp_result.is_error)
            except Exception as exc:  # transport/protocol failure: tell the model, don't crash
                content, is_error = f"MCP error: {exc}", True
            if verbose:
                print(f"[tool_result] {'ERROR ' if is_error else ''}{content[:200].replace(chr(10), ' ')}", file=sys.stderr)
            results.append({"type": "tool_result", "tool_use_id": tu.id, "content": content, "is_error": is_error})

        # 5. FEED BACK
        messages.append({"role": "user", "content": results})

    return "Stopped: reached the maximum number of agent turns."
