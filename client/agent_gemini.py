"""
AGENT (Google Gemini) — same loop as agent.py, third wire format.

The MCP half is again identical (discovery.py, gate.py). The THINK step:

    Anthropic                          Gemini (google-genai SDK)
    ---------                          -------------------------
    tools=[{name, input_schema}]       config.tools=[{function_declarations:[{name, parameters_json_schema}]}]
    response.content tool_use blocks   response.function_calls (args already a dict)
    role:"user" tool_result blocks     Part.from_function_response(...) in a "user" turn
    system= parameter                  config.system_instruction

Gemini's `parameters_json_schema` accepts the MCP inputSchema verbatim —
the clearest illustration that MCP's self-description is just JSON Schema.

Auth: GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment / .env.
The SDK's own automatic-function-calling is NOT used: it would execute tools
itself and bypass our human-in-the-loop gate. We keep the loop manual.
"""

from __future__ import annotations

import json
import os
import sys

from google import genai
from google.genai import types as gt
from mcp import Client

from .agent_specs import AgentSpec, compose_system_prompt, filter_tools_for_spec
from .discovery import build_agent_system_prompt, mcp_tools_to_gemini, tool_result_text
from .gate import ApprovalPolicy

DEFAULT_MODEL = "gemini-2.5-pro"
MAX_TURNS = 12


async def run_agent(client: Client, policy: ApprovalPolicy, goal: str, *,
                    model: str | None = None, verbose: bool = True,
                    spec: AgentSpec | None = None) -> str:
    model = model or (spec.model if spec else None) or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
    max_turns = spec.max_turns if spec else MAX_TURNS

    # 1. DISCOVER — identical to agent.py.
    tools = (await client.list_tools()).tools
    policy.register_tools(tools)                  # gate sees all; model sees the spec's subset
    config = gt.GenerateContentConfig(
        system_instruction=compose_system_prompt(await build_agent_system_prompt(client), spec),
        tools=mcp_tools_to_gemini(filter_tools_for_spec(tools, spec)),
    )

    g = genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY from the environment
    contents: list[gt.Content] = [gt.Content(role="user", parts=[gt.Part.from_text(text=goal)])]

    for turn in range(1, max_turns + 1):
        # 2. THINK
        response = g.models.generate_content(model=model, contents=contents, config=config)
        candidate = response.candidates[0]
        calls = response.function_calls or []

        if verbose and response.text:
            print(f"\n[assistant · turn {turn}] {response.text}", file=sys.stderr)

        if not calls:
            return response.text or ""             # final answer

        # Echo the model turn (function_call parts included) back into history.
        contents.append(candidate.content)

        # 3 + 4. INTERCEPT + EXECUTE — same gate, same MCP session.
        result_parts: list[gt.Part] = []
        for fc in calls:
            args = dict(fc.args or {})             # already a dict here
            if verbose:
                print(f"[function_call] {fc.name}({json.dumps(args, ensure_ascii=False)})", file=sys.stderr)
            try:
                mcp_result = await policy.call_tool(client, fc.name, args)   # <- the gate
                payload = {"result": tool_result_text(mcp_result)}
                if mcp_result.is_error:
                    payload = {"error": tool_result_text(mcp_result)}
            except Exception as exc:
                payload = {"error": f"MCP error: {exc}"}
            if verbose:
                print(f"[tool_result] {str(payload)[:200]}", file=sys.stderr)
            result_parts.append(gt.Part.from_function_response(name=fc.name, response=payload))

        # 5. FEED BACK — all function responses in one user turn.
        contents.append(gt.Content(role="user", parts=result_parts))

    return "Stopped: reached the maximum number of agent turns."
