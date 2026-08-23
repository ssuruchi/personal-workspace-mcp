"""
LLM provider registry — which "brain" drives the agent loop.

Moved out of main.py so both the CLI and the workflow engine (agent: steps)
can resolve providers without circular imports. Imports are lazy: you only
need the SDK — and the API key — of the provider you actually pick.

Every provider module exposes the same callable:

    run_agent(client, policy, goal, *, model=None, verbose=True, spec=None)

The MCP side (discovery, gate, execution) is shared; only the think step
differs. See the agent*.py headers.
"""

from __future__ import annotations

import importlib
import os

PROVIDERS = {
    "anthropic": {"module": ".agent", "keys": ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"],
                  "hint": "set ANTHROPIC_API_KEY (or run `ant auth login`)"},
    "openai": {"module": ".agent_openai", "keys": ["OPENAI_API_KEY"],
               "hint": "set OPENAI_API_KEY (any non-empty value for a local OPENAI_BASE_URL server)"},
    "gemini": {"module": ".agent_gemini", "keys": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
               "hint": "set GEMINI_API_KEY (or GOOGLE_API_KEY)"},
}


def has_credentials(provider: str) -> bool:
    if any(os.environ.get(k) for k in PROVIDERS[provider]["keys"]):
        return True
    if provider == "anthropic":
        # `ant auth login` profiles live here and are picked up by the SDK automatically.
        cfg = os.path.expanduser("~/.config/anthropic")
        return os.path.isdir(cfg) and any(os.scandir(cfg))
    return False


def load_provider(name: str):
    """Import the provider module and return its run_agent. Fails fast + clean."""
    if name not in PROVIDERS:
        raise SystemExit(f"unknown provider {name!r}; choose from {sorted(PROVIDERS)}")
    spec = PROVIDERS[name]
    try:
        module = importlib.import_module(spec["module"], package=__package__)
    except ImportError as exc:
        raise SystemExit(f"provider {name!r} needs its SDK installed: {exc}")
    if not has_credentials(name):
        raise SystemExit(f"provider {name!r}: {spec['hint']}")
    return module.run_agent
