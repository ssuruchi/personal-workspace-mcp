"""
CLI entry point for the MCP client.

    python -m client.main inspect                         # what can the server do?
    python -m client.main workflow daily-brief            # deterministic, no LLM
    python -m client.main workflow capture-todos          # writes -> you will be asked
    python -m client.main workflow weekly-review          # MCP prompt + resources -> Claude
    python -m client.main workflow morning-routine        # USER-DEFINED: workflows/*.yaml
    python -m client.main workflow --list                 # everything runnable
    python -m client.main agent "what's overdue? add a task to fix it"
    python -m client.main agent --spec task-assistant "..."   # agents/*.yaml persona + tool subset
    python -m client.main agent --provider openai "..."   # or gemini; default anthropic

Flags:
    --transport stdio|http   (default stdio: spawns the server as a subprocess)
    --url URL                (http only; default http://127.0.0.1:8000/mcp)
    --yes                    auto-approve every write/destructive tool (no prompts)
    --no-input               never prompt; deny anything that would need approval
    --json                   (inspect) dump the raw self-description as JSON

The agent's LLM is configurable (same MCP code, different "brain"):
    --provider anthropic|openai|gemini   or env AGENT_PROVIDER (default anthropic)
    --model NAME                         or env ANTHROPIC_MODEL / OPENAI_MODEL / GEMINI_MODEL
Local models: --provider openai with OPENAI_BASE_URL pointing at Ollama/LM Studio/vLLM.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from dotenv import load_dotenv

from .agent_specs import load_specs
from .connection import DEFAULT_HTTP_URL, connect
from .discovery import describe_server, print_report
from .gate import ApprovalPolicy
from .providers import PROVIDERS, has_credentials, load_provider
from .workflow_engine import discover_workflows, run_yaml_workflow
from .workflows import WORKFLOWS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m client.main", description="Personal Workspace MCP client")
    p.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    p.add_argument("--url", default=DEFAULT_HTTP_URL)
    p.add_argument("--yes", action="store_true", help="auto-approve all tool calls")
    p.add_argument("--no-input", action="store_true", help="never prompt; deny gated calls")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("inspect", help="print the server's self-description")
    s.add_argument("--json", action="store_true")
    s.add_argument("-v", "--verbose", action="store_true")

    yaml_workflows = discover_workflows()
    w = sub.add_parser("workflow", help="run a built-in or user-defined (workflows/*.yaml) workflow")
    w.add_argument("name", nargs="?", choices=sorted(WORKFLOWS) + sorted(yaml_workflows))
    w.add_argument("--focus", default="Project Atlas", help="(weekly-review) focus area")
    w.add_argument("--list", action="store_true", help="list all runnable workflows")

    a = sub.add_parser("agent", help="let an LLM pursue a goal using the server's tools")
    a.add_argument("goal")
    a.add_argument("--provider", choices=sorted(PROVIDERS), default=None,
                   help="LLM brain (default: the spec's provider, or AGENT_PROVIDER, or anthropic)")
    a.add_argument("--spec", default=None, choices=sorted(load_specs()),
                   help="agent persona/tool-subset from agents/*.yaml (see agent_specs.py)")
    a.add_argument("--model", default=None, help="override the provider's default model")
    a.add_argument("-q", "--quiet", action="store_true")
    return p


async def amain(args: argparse.Namespace) -> int:
    policy = ApprovalPolicy(auto_approve=args.yes, interactive=not args.no_input and sys.stdin.isatty())

    # Resolve spec + LLM provider (SDK import + credential check) BEFORE opening
    # the MCP session, so a missing key fails fast with a clean message.
    run_agent, agent_spec = None, None
    if args.cmd == "agent":
        agent_spec = load_specs()[args.spec] if args.spec else None
        provider = (args.provider
                    or (agent_spec.provider if agent_spec else None)
                    or os.environ.get("AGENT_PROVIDER", "anthropic"))
        args.provider = provider
        run_agent = load_provider(provider)

    async with connect(policy, transport=args.transport, url=args.url) as client:
        if args.cmd == "inspect":
            desc = await describe_server(client)
            if args.json:
                print(json.dumps(desc, indent=2))
            else:
                print_report(desc, verbose=args.verbose)
            return 0

        # Both workflows and the agent need the tool annotations for the gate.
        policy.register_tools((await client.list_tools()).tools)

        if args.cmd == "workflow":
            yaml_workflows = discover_workflows()
            if args.list or not args.name:
                print("built-in (client/workflows.py):")
                for n in sorted(WORKFLOWS):
                    print(f"  {n}")
                print("user-defined (workflows/*.yaml):")
                for n, path in sorted(yaml_workflows.items()):
                    print(f"  {n}  ({path.name})")
                return 0
            if args.name in WORKFLOWS:
                fn = WORKFLOWS[args.name]
                kwargs = {"focus": args.focus} if args.name == "weekly-review" else {}
                if args.name == "weekly-review" and not has_credentials("anthropic"):
                    print("weekly-review calls Claude: set ANTHROPIC_API_KEY (or run `ant auth login`).", file=sys.stderr)
                    return 2
                print(await fn(client, policy, **kwargs))
            else:
                from .workflow_engine import WorkflowError
                try:
                    print(await run_yaml_workflow(client, policy, yaml_workflows[args.name]))
                except WorkflowError as exc:
                    print(f"\nworkflow stopped: {exc}", file=sys.stderr)
                    policy.print_audit()
                    return 1
            policy.print_audit()
            return 0

        if args.cmd == "agent":
            print(f"[agent] provider={args.provider}"
                  + (f" spec={agent_spec.name}" if agent_spec else ""), file=sys.stderr)
            answer = await run_agent(client, policy, args.goal, model=args.model,
                                     verbose=not args.quiet, spec=agent_spec)
            print("\n── Answer ───────────────────────────────────────────────────")
            print(answer)
            policy.print_audit()
            return 0
    return 1


def main() -> None:
    load_dotenv()  # optional .env with ANTHROPIC_API_KEY / WORKSPACE_TODAY
    args = build_parser().parse_args()
    raise SystemExit(asyncio.run(amain(args)))


if __name__ == "__main__":
    main()
