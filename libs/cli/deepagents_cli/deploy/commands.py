"""CLI commands for `deepagents init`, `deploy`, `agents`, and `mcp-servers`.

Wired into the root argparse subparsers by `setup_deploy_parsers` (called from
`deepagents_cli.main`). Each top-level command has an `execute_*_command`
entrypoint that the main module dispatches.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

_BETA_WARNING = (
    "\033[33mWarning: `deepagents deploy` is in beta. "
    "APIs, configuration format, and behavior may change between releases.\033[0m\n"
)


def setup_deploy_parsers(
    subparsers: Any,  # noqa: ANN401
    *,
    make_help_action: Callable[[Callable[[], None]], type[argparse.Action]],
) -> None:
    """Register the top-level subparsers for the migrated deploy CLI."""
    _add_init_parser(subparsers, make_help_action)
    _add_deploy_parser(subparsers, make_help_action)
    _add_agents_parser(subparsers, make_help_action)
    _add_mcp_servers_parser(subparsers, make_help_action)


# --- init -------------------------------------------------------------------


def _add_init_parser(subparsers: Any, make_help_action) -> None:  # noqa: ANN001
    p = subparsers.add_parser(
        "init",
        help="(beta) Scaffold a new managed-agent project",
        add_help=False,
    )
    p.add_argument("name", nargs="?", default=None)
    p.add_argument(
        "-h", "--help",
        action=make_help_action(lambda: p.print_help()),
        help="show this help message and exit",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing files")


def execute_init_command(args: argparse.Namespace) -> None:
    print(_BETA_WARNING)
    name = args.name
    if name is None:
        try:
            name = input("Project name: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            raise SystemExit(1) from None
        if not name:
            print("Error: project name is required.")
            raise SystemExit(1)
    _scaffold(name=name, force=args.force)


def _scaffold(*, name: str, force: bool) -> None:
    project_dir = Path.cwd() / name
    if project_dir.exists() and not force:
        print(f"Error: {name}/ already exists. Use --force to overwrite.")
        raise SystemExit(1)
    project_dir.mkdir(parents=True, exist_ok=True)

    (project_dir / "agent.json").write_text(_STARTER_AGENT_JSON.format(name=name))
    (project_dir / "AGENTS.md").write_text(_STARTER_AGENTS_MD)
    (project_dir / ".gitignore").write_text(_STARTER_GITIGNORE)
    (project_dir / ".env").write_text(_STARTER_ENV)
    (project_dir / "skills").mkdir(exist_ok=True)

    print(f"Created {name}/ with: agent.json, AGENTS.md, .gitignore, .env, skills/")
    print("\nNext steps:")
    print(f"  cd {name}")
    print("  # edit AGENTS.md, optionally add tools.json / skills/ / subagents/")
    print("  deepagents deploy")


_STARTER_AGENT_JSON = """\
{{
  "name": "{name}",
  "description": "A managed deep agent.",
  "runtime": {{
    "model": {{"model_id": "anthropic:claude-sonnet-4-6"}},
    "backend_type": "thread_scoped_sandbox"
  }}
}}
"""

_STARTER_AGENTS_MD = """\
# Agent Instructions

You are a helpful AI agent.

## Guidelines

- Follow the user's instructions carefully.
- Ask for clarification when the request is ambiguous.
"""

_STARTER_GITIGNORE = """\
.env
.deepagents/
"""

_STARTER_ENV = """\
# Required: LangSmith API key for /v1/deepagents/* (private preview)
LANGSMITH_API_KEY=

# Optional: override the API endpoint (defaults to https://api.smith.langchain.com)
# LANGSMITH_ENDPOINT=
"""


# --- deploy / agents / mcp-servers (stubs filled by later tasks) ------------


def _add_deploy_parser(subparsers: Any, make_help_action) -> None:  # noqa: ANN001
    p = subparsers.add_parser(
        "deploy",
        help="(beta) Upsert the project as a managed deep agent",
        add_help=False,
    )
    p.add_argument("-h", "--help",
                   action=make_help_action(lambda: p.print_help()),
                   help="show this help message and exit")
    p.add_argument("--dir", type=str, default=None,
                   help="Project directory (default: cwd)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print payload without sending")
    p.add_argument("--detach", action="store_true",
                   help="Exit immediately after upsert without polling health")
    p.add_argument("--reset", action="store_true",
                   help="Discard local state and create a fresh agent")


def execute_deploy_command(args: argparse.Namespace) -> None:
    from deepagents_cli.config import _load_dotenv  # existing helper
    from deepagents_cli.deploy.api_client import ApiClient, ApiError
    from deepagents_cli.deploy.mcp_resolver import (
        UnresolvedServersError,
        resolve_referenced_servers,
    )
    from deepagents_cli.deploy.payload import build_payload
    from deepagents_cli.deploy.project import Project, ProjectError
    from deepagents_cli.deploy.state import State

    print(_BETA_WARNING)
    root = Path(args.dir).resolve() if args.dir else Path.cwd().resolve()
    _load_dotenv(start_path=root)

    try:
        project = Project.load(root)
    except ProjectError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from None

    state = State.load(root, reset=args.reset)
    payload = build_payload(project, mode="patch" if state.agent_id else "create")

    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return

    client = ApiClient.from_env()
    state.endpoint = client.endpoint

    try:
        state.mcp_servers = resolve_referenced_servers(
            client, payload, cache=state.mcp_servers
        )
    except UnresolvedServersError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from None

    try:
        agent = _upsert_agent(client, state.agent_id, payload)
    except ApiError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from None

    state.save(agent_id=agent["id"], revision=agent.get("revision"))
    _print_deploy_result(agent, client.endpoint, detach=args.detach, client=client)


def _upsert_agent(
    client,  # type: ignore[no-untyped-def]
    agent_id: str | None,
    payload: dict[str, Any],
) -> dict[str, Any]:
    from deepagents_cli.deploy.api_client import ApiError

    if agent_id:
        try:
            return client.patch_agent(agent_id, payload)
        except ApiError as exc:
            if exc.status == 404:
                print(
                    f"Note: agent {agent_id} no longer exists — creating a new one."
                )
            else:
                raise
    return client.create_agent(payload)


def _print_deploy_result(
    agent: dict[str, Any],
    endpoint: str,
    *,
    detach: bool,
    client,  # type: ignore[no-untyped-def]
) -> None:
    name = agent.get("name", "?")
    agent_id = agent.get("id", "?")
    revision = agent.get("revision", "")[:8]
    smith_endpoint = endpoint.replace("api.smith.langchain.com", "smith.langchain.com")
    print(f"\nDeployed: {name}")
    print(f"  agent_id: {agent_id}")
    print(f"  revision: {revision}")
    print(f"  {smith_endpoint}/o/-/agents/{agent_id}")
    if detach:
        return
    try:
        health = client._request("GET", f"/v1/deepagents/agents/{agent_id}/health")
        print(f"  health:   {health}")
    except Exception as exc:  # noqa: BLE001
        print(f"  health check skipped: {exc}")


def _add_agents_parser(subparsers: Any, make_help_action) -> None:  # noqa: ANN001
    p = subparsers.add_parser("agents", help="Manage agents", add_help=False)
    p.add_argument("-h", "--help",
                   action=make_help_action(lambda: p.print_help()),
                   help="show this help message and exit")
    sub = p.add_subparsers(dest="agents_cmd", required=True)
    sub.add_parser("list")
    g = sub.add_parser("get"); g.add_argument("agent_id"); g.add_argument("--include-files", action="store_true")
    d = sub.add_parser("delete"); d.add_argument("agent_id"); d.add_argument("--yes", action="store_true")


def execute_agents_command(args: argparse.Namespace) -> None:
    from deepagents_cli.deploy.api_client import ApiClient, ApiError

    client = ApiClient.from_env()
    try:
        if args.agents_cmd == "list":
            for agent in client.iter_agents(page_size=50):
                print(f"{agent.get('id')}\t{agent.get('name', '')}\t{agent.get('updated_at', '')}")
        elif args.agents_cmd == "get":
            agent = client.get_agent(args.agent_id, include_files=args.include_files)
            print(json.dumps(agent, indent=2))
        elif args.agents_cmd == "delete":
            if not args.yes:
                try:
                    answer = input(f"Delete agent {args.agent_id}? [y/N]: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print()
                    print("Aborted.")
                    return
                if answer not in {"y", "yes"}:
                    print("Aborted.")
                    return
            client.delete_agent(args.agent_id)
            print(f"Deleted {args.agent_id}")
    except ApiError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from None


def _add_mcp_servers_parser(subparsers: Any, make_help_action) -> None:  # noqa: ANN001
    p = subparsers.add_parser("mcp-servers", help="Manage MCP servers", add_help=False)
    p.add_argument("-h", "--help",
                   action=make_help_action(lambda: p.print_help()),
                   help="show this help message and exit")
    sub = p.add_subparsers(dest="mcp_cmd", required=True)
    sub.add_parser("list")
    a = sub.add_parser("add")
    a.add_argument("--url", required=True)
    a.add_argument("--name", default=None)
    a.add_argument("--header", action="append", default=[], metavar="KEY=VALUE")
    a.add_argument("--auth-type", default="headers", choices=["headers"])
    g = sub.add_parser("get"); g.add_argument("mcp_server_id")
    d = sub.add_parser("delete"); d.add_argument("mcp_server_id"); d.add_argument("--yes", action="store_true")


def execute_mcp_servers_command(args: argparse.Namespace) -> None:
    from urllib.parse import urlparse

    from deepagents_cli.deploy.api_client import ApiClient, ApiError

    client = ApiClient.from_env()
    try:
        if args.mcp_cmd == "list":
            for srv in client.list_mcp_servers():
                print(f"{srv.get('id')}\t{srv.get('name', '')}\t{srv.get('url', '')}")
        elif args.mcp_cmd == "add":
            headers = _parse_header_args(args.header)
            name = args.name or urlparse(args.url).hostname or args.url
            srv = client.create_mcp_server(
                name=name,
                url=args.url,
                headers=headers,
                auth_type=args.auth_type,
            )
            print(f"Created mcp_server {srv.get('id')}: {srv.get('name')} → {srv.get('url')}")
        elif args.mcp_cmd == "get":
            print(json.dumps(client.get_mcp_server(args.mcp_server_id), indent=2))
        elif args.mcp_cmd == "delete":
            if not args.yes:
                try:
                    answer = input(f"Delete MCP server {args.mcp_server_id}? [y/N]: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print()
                    print("Aborted.")
                    return
                if answer not in {"y", "yes"}:
                    print("Aborted.")
                    return
            client.delete_mcp_server(args.mcp_server_id)
            print(f"Deleted {args.mcp_server_id}")
    except ApiError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from None


def _parse_header_args(raw: list[str]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for entry in raw:
        if "=" not in entry:
            print(f"Error: --header must be KEY=VALUE, got {entry!r}")
            raise SystemExit(1)
        key, _, value = entry.partition("=")
        out.append({"key": key.strip(), "value": value})
    return out
