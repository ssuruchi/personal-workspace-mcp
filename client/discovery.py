"""
Discovery — how a client learns what a server can do, at runtime.

Nothing here is hard-coded to *this* server. Point it at any MCP server and it
prints the same report, because every MCP server must answer:

    tools/list               -> name, description, inputSchema (JSON Schema), annotations
    resources/list           -> static URIs + mime types
    resources/templates/list -> parameterised URIs like workspace://notes/{slug}
    prompts/list             -> prompt names + their arguments

This is the "self-describing requirement" made concrete, and it is exactly what
the agent (agent.py) feeds to Claude as its tool definitions.
"""

from __future__ import annotations

import json
from typing import Any

from mcp import Client
from mcp.types import CallToolResult, Tool

from .gate import ApprovalPolicy


async def describe_server(client: Client) -> dict[str, Any]:
    """Collect the server's full self-description into one plain dict."""
    tools = (await client.list_tools()).tools
    resources = (await client.list_resources()).resources
    templates = (await client.list_resource_templates()).resource_templates
    prompts = (await client.list_prompts()).prompts
    info = client.server_info
    return {
        "protocol_version": client.protocol_version,
        "server": {"name": info.name if info else None, "version": info.version if info else None},
        "instructions": client.instructions,
        "capabilities": client.server_capabilities.model_dump(exclude_none=True),
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "risk": ApprovalPolicy.classify(t),
                "inputSchema": t.input_schema,
                "outputSchema": t.output_schema,
            }
            for t in tools
        ],
        "resources": [{"uri": r.uri, "name": r.name, "mimeType": r.mime_type, "description": r.description} for r in resources],
        "resource_templates": [{"uriTemplate": r.uri_template, "name": r.name, "mimeType": r.mime_type} for r in templates],
        "prompts": [{"name": p.name, "description": p.description,
                     "arguments": [{"name": a.name, "required": a.required} for a in (p.arguments or [])]} for p in prompts],
    }


def print_report(desc: dict[str, Any], *, verbose: bool = False) -> None:
    print(f"MCP protocol  : {desc['protocol_version']}")
    print(f"Server        : {desc['server']['name']} v{desc['server']['version']}")
    print(f"Capabilities  : {', '.join(desc['capabilities'].keys())}")
    print(f"Instructions  : {(desc['instructions'] or '').strip()[:110]}…")

    print(f"\nTOOLS ({len(desc['tools'])})  — what the model may *request*")
    for t in desc["tools"]:
        req = t["inputSchema"].get("required", [])
        params = ", ".join(f"{k}{'' if k in req else '?'}: {v.get('type', 'any')}" for k, v in t["inputSchema"].get("properties", {}).items())
        print(f"  • {t['name']}({params})  [{t['risk']}]")
        print(f"      {t['description']}")
        if verbose:
            print("      inputSchema: " + json.dumps(t["inputSchema"]))

    print(f"\nRESOURCES ({len(desc['resources'])})  — context the model may *read*")
    for r in desc["resources"]:
        print(f"  • {r['uri']:<34} {r['mimeType'] or '':<18} {r['description'] or ''}")
    for r in desc["resource_templates"]:
        print(f"  • {r['uriTemplate']:<34} {r['mimeType'] or '':<18} (template)")

    print(f"\nPROMPTS ({len(desc['prompts'])})  — reusable templates the *user/client* picks")
    for p in desc["prompts"]:
        args = ", ".join(a["name"] + ("" if a["required"] else "?") for a in p["arguments"])
        print(f"  • {p['name']}({args})  — {p['description']}")


# --------------------------------------------------------------------------- #
# MCP -> provider tool-schema converters.
#
# This is the whole "adapter" layer, side by side. MCP already speaks JSON
# Schema and so does every provider's function-calling API, so each converter
# is a field rename — no per-tool glue code for ANY provider. This is why you
# do NOT need a different MCP for each LLM: the protocol is provider-neutral,
# only these few lines differ.
# --------------------------------------------------------------------------- #


def mcp_tools_to_anthropic(tools: list[Tool]) -> list[dict[str, Any]]:
    """Claude Messages API: tools=[{name, description, input_schema}]."""
    return [
        {
            "name": t.name,
            "description": t.description or "",
            "input_schema": t.input_schema,
        }
        for t in tools
    ]


def mcp_tools_to_openai(tools: list[Tool]) -> list[dict[str, Any]]:
    """OpenAI Chat Completions (also Ollama / LM Studio / vLLM / Groq — anything
    OpenAI-compatible): tools=[{type: "function", function: {...}}]."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.input_schema,
            },
        }
        for t in tools
    ]


def mcp_tools_to_gemini(tools: list[Tool]) -> list[dict[str, Any]]:
    """Google Gemini: one Tool holding function_declarations. The google-genai
    SDK accepts plain dicts; `parameters_json_schema` takes JSON Schema verbatim."""
    return [
        {
            "function_declarations": [
                {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters_json_schema": t.input_schema,
                }
                for t in tools
            ]
        }
    ]


# --------------------------------------------------------------------------- #
# Provider-neutral helpers shared by all three agents.
# --------------------------------------------------------------------------- #


async def build_agent_system_prompt(client: Client) -> str:
    """System prompt = the server's handshake `instructions` + its resource map."""
    resources = (await client.list_resources()).resources
    templates = (await client.list_resource_templates()).resource_templates
    lines = [f"- {r.uri}: {r.description or ''}" for r in resources]
    lines += [f"- {t.uri_template}: {t.description or ''} (template)" for t in templates]
    return (
        (client.instructions or "") + "\n\n"
        "You are operating through the Model Context Protocol. Tools are the only way to act. "
        "The user may deny a tool call; if a result says DENIED, do not retry — explain and stop or ask. "
        "Resources in this workspace (read via tools where available):\n" + "\n".join(lines)
    )


def tool_result_text(result: CallToolResult) -> str:
    """Flatten an MCP CallToolResult into plain text any model can read."""
    if result.structured_content is not None:
        return json.dumps(result.structured_content, ensure_ascii=False)
    parts: list[str] = []
    for block in result.content:
        t = getattr(block, "type", None)
        if t == "text":
            parts.append(block.text)
        elif t == "resource":
            parts.append(getattr(block.resource, "text", "") or f"<binary resource {block.resource.uri}>")
        else:
            parts.append(f"<{t} content>")
    return "\n".join(parts) or "(empty result)"
