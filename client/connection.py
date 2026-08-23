"""
Connecting to the workspace MCP server — the *client* side of the protocol.

Two transports are supported, chosen at runtime:

  stdio  (default) The client *spawns the server as a child process* and talks
                   JSON-RPC over its stdin/stdout pipes. Nothing touches the
                   network; the connection inherits OS process isolation.
  http             The server is already running somewhere
                   (`python -m server.workspace_server --http`) and the client
                   connects to its URL (Streamable HTTP). In production you put
                   TLS + auth (bearer / OAuth / mTLS) in front of this.

Either way, `mcp.Client` gives us ONE persistent, stateful session:
the handshake happens once (capabilities + server info + instructions are
exchanged), and every later tools/resources/prompts call rides the same
connection. That is the "stateful transport" constraint from the reading.
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from mcp import Client
from mcp.client.stdio import StdioServerParameters, stdio_client

from .gate import ApprovalPolicy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HTTP_URL = "http://127.0.0.1:8000/mcp"


@asynccontextmanager
async def connect(
    policy: ApprovalPolicy,
    *,
    transport: str = "stdio",
    url: str = DEFAULT_HTTP_URL,
) -> AsyncIterator[Client]:
    """Open a stateful MCP session to the workspace server.

    `policy` is the human-in-the-loop layer. It is wired in *here* so that the
    server's elicitation requests (server -> client questions) are routed to it,
    and so every call_tool can be intercepted (see gate.py).
    """
    if transport == "stdio":
        # Same interpreter, server started as `python -m server.workspace_server`.
        # Env is passed explicitly: stdio servers inherit only what we give them.
        env = {k: v for k, v in os.environ.items() if k.startswith("WORKSPACE_")}
        env.setdefault("PYTHONUTF8", "1")
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "server.workspace_server"],
            cwd=str(PROJECT_ROOT),
            env=env,
        )
        # stdio_client yields (read_stream, write_stream); Client accepts any such
        # transport object directly.
        target = stdio_client(server_params)
    elif transport == "http":
        target = url  # a URL string => streamable_http_client under the hood
    else:
        raise ValueError(f"unknown transport {transport!r}")

    async with Client(
        target,
        elicitation_callback=policy.handle_elicitation,   # server asks the human
        client_info=None,
    ) as client:
        yield client
