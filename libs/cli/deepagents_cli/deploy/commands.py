"""CLI commands for `deepagents init`, `deploy`, `agents`, and `mcp-servers`.

Wired into the root argparse subparsers by `setup_deploy_parsers` (called from
`deepagents_cli.main`). Each top-level command has an `execute_*_command`
entrypoint that the main module dispatches.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from deepagents_cli.deploy.api_client import ApiClient
    from deepagents_cli.deploy.state import State

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


def _add_init_parser(
    subparsers: Any,  # noqa: ANN401
    make_help_action: Callable[[Callable[[], None]], type[argparse.Action]],
) -> None:
    p = subparsers.add_parser(
        "init",
        help="(beta) Scaffold a new managed-agent project",
        add_help=False,
    )
    p.add_argument("name", nargs="?", default=None)
    p.add_argument(
        "-h",
        "--help",
        action=make_help_action(lambda: p.print_help()),
        help="show this help message and exit",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing files")


def execute_init_command(args: argparse.Namespace) -> None:
    """Run the `deepagents init` command."""
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
    _print_post_init_welcome(name)


def _scaffold(*, name: str, force: bool) -> None:
    project_dir = Path.cwd() / name
    if project_dir.exists() and not force:
        print(f"Error: {name}/ already exists. Use --force to overwrite.")
        raise SystemExit(1)
    project_dir.mkdir(parents=True, exist_ok=True)

    (project_dir / "agent.json").write_text(_STARTER_AGENT_JSON.format(name=name))
    (project_dir / "AGENTS.md").write_text(_STARTER_AGENTS_MD)
    (project_dir / ".gitignore").write_text(_STARTER_GITIGNORE)
    (project_dir / "tools.json").write_text(_STARTER_TOOLS_JSON)

    skill_dir = project_dir / "skills" / _STARTER_SKILL_NAME
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(_STARTER_SKILL_MD)

    subagent_dir = project_dir / "subagents" / _STARTER_SUBAGENT_NAME
    subagent_dir.mkdir(parents=True, exist_ok=True)
    (subagent_dir / "agent.json").write_text(_STARTER_SUBAGENT_AGENT_JSON)
    (subagent_dir / "AGENTS.md").write_text(_STARTER_SUBAGENT_AGENTS_MD)

    print(
        f"Created {name}/ with: agent.json, AGENTS.md, .gitignore, tools.json, "
        f"skills/{_STARTER_SKILL_NAME}/, subagents/{_STARTER_SUBAGENT_NAME}/"
    )


def _print_post_init_welcome(name: str) -> None:
    """Print a formatted, scannable walkthrough after scaffolding."""
    print("\nNext steps")
    print("──────────")
    print("  1. Edit your agent's files")
    print("       AGENTS.md    system prompt / instructions")
    print("       agent.json   name, model, backend")
    print("       tools.json   tools the agent can call (starts empty)")
    print("       skills/      reusable instructions (optional; example included)")
    print("       subagents/   delegated agents (optional; example included)")
    print()
    print("  2. Connect tools (optional)")
    print("       Set LANGSMITH_API_KEY, then register or reuse an MCP server:")
    print("         deepagents mcp-servers list      # servers already connected")
    print("         deepagents mcp-servers add ...   # register a new server")
    print("       Auth options for `add`:")
    print("         --header X-Api-Key=$LANGSMITH_API_KEY   # API key")
    print("         --auth-type oauth --connect             # OAuth (browser)")
    print("       List a server's tools (prints a tools.json snippet):")
    print("         deepagents mcp-servers tools <id|name|url>")
    print()
    print("  3. Deploy")
    print(f"       cd {name}")
    print("       deepagents deploy")


_STARTER_AGENT_JSON = """\
{{
  "name": "{name}",
  "description": "A managed deep agent.",
  "model": "openai:gpt-5.5",
  "backend": {{
    "type": "state"
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
"""

# Empty by default so the first `deepagents deploy` succeeds out of the box.
# Tools reference an MCP server that must already be registered in the
# workspace (see `deepagents mcp-servers add`); add entries to `tools` once a
# server exists, e.g.:
#   {"name": "read_url_content", "mcp_server_url": "https://tools.langchain.com",
#    "mcp_server_name": "Fleet", "display_name": "read_url_content"}
_STARTER_TOOLS_JSON = """\
{
  "tools": [],
  "interrupt_config": {}
}
"""

# Example subagent. The main agent can delegate to it via the Task tool.
_STARTER_SUBAGENT_NAME = "researcher"

_STARTER_SUBAGENT_AGENT_JSON = """\
{
  "description": "Researches a topic and returns a concise summary.",
  "model": "openai:gpt-5.5"
}
"""

_STARTER_SUBAGENT_AGENTS_MD = """\
# Researcher

You are a focused research subagent.

## Guidelines

- Gather the requested information and return a concise, well-sourced summary.
- State assumptions and call out anything you could not verify.
"""

# Example skill. Skills are progressive-disclosure instructions the agent loads
# on demand; the SKILL.md frontmatter `name` and `description` are required.
_STARTER_SKILL_NAME = "example-skill"

_STARTER_SKILL_MD = """\
---
name: example-skill
description: Worked example of the skill format; replace with your own trigger.
---

# Example skill

Skills hold detailed, reusable instructions the agent pulls in only when the
description above matches the task.

## Steps

1. Describe the first step the agent should take.
2. Add any constraints, formats, or examples it should follow.
3. Delete or replace this skill once you have your own.
"""

# --- deploy / agents / mcp-servers (stubs filled by later tasks) ------------


def _add_deploy_parser(
    subparsers: Any,  # noqa: ANN401
    make_help_action: Callable[[Callable[[], None]], type[argparse.Action]],
) -> None:
    p = subparsers.add_parser(
        "deploy",
        help="(beta) Upsert the project as a managed deep agent",
        add_help=False,
    )
    p.add_argument(
        "-h",
        "--help",
        action=make_help_action(lambda: p.print_help()),
        help="show this help message and exit",
    )
    p.add_argument(
        "--dir", type=str, default=None, help="Project directory (default: cwd)"
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Print payload without sending"
    )
    p.add_argument(
        "--detach",
        action="store_true",
        help="Exit immediately after upsert without polling health",
    )
    p.add_argument(
        "--reset",
        action="store_true",
        help="Discard local state and create a fresh agent",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="Confirm deploy target changes without prompting",
    )


def execute_deploy_command(args: argparse.Namespace) -> None:
    """Run the `deepagents deploy` command."""
    from deepagents_cli.config import _load_dotenv  # existing helper
    from deepagents_cli.deploy.api_client import ApiClient, ApiError
    from deepagents_cli.deploy.mcp_resolver import (
        UninvokableServersError,
        UnresolvedServersError,
        resolve_referenced_servers,
    )
    from deepagents_cli.deploy.payload import (
        build_directory_files,
        build_metadata_payload,
        build_payload,
    )
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

    create_payload = build_payload(project, mode="create")
    metadata_payload = build_metadata_payload(project)
    directory_files = build_directory_files(project)
    agent_payload = metadata_payload if project.agent_id else create_payload

    if args.dry_run:
        directory_entries = {
            path: {"type": "file", "content": content}
            for path, content in directory_files.items()
        }
        print(
            json.dumps(
                {
                    "agent_payload": agent_payload,
                    "directory_files": directory_entries,
                },
                indent=2,
            )
        )
        return

    client = ApiClient.from_env()
    if args.reset and project.agent_id:
        print(
            "Error: --reset cannot create a fresh agent while agent.json declares "
            f"agent_id {project.agent_id!r}. Remove agent_id or deploy without --reset."
        )
        raise SystemExit(1)
    try:
        state = State.load(root, endpoint=client.endpoint, reset=args.reset)
    except ValueError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from None

    target_agent_id = project.agent_id or state.agent_id
    if project.agent_id:
        try:
            _confirm_agent_json_target(
                client,
                project.agent_id,
                state,
                assume_yes=args.yes,
            )
        except ApiError as exc:
            print(f"Error: {exc}")
            raise SystemExit(1) from None

    try:
        state.mcp_servers = resolve_referenced_servers(
            client, create_payload, cache=state.mcp_servers
        )
    except (UnresolvedServersError, UninvokableServersError) as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from None

    try:
        agent, revision = _deploy_agent(
            client,
            target_agent_id,
            create_payload=create_payload,
            metadata_payload=metadata_payload,
            directory_files=directory_files,
            allow_create_on_missing=project.agent_id is None,
        )
    except ApiError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from None

    state.save(agent_id=agent["id"], revision=revision)
    _print_deploy_result(
        agent,
        client.endpoint,
        detach=args.detach,
        client=client,
        revision=revision,
    )


def _confirm_agent_json_target(
    client: ApiClient,
    agent_id: str,
    state: State,
    *,
    assume_yes: bool,
) -> None:
    """Confirm first use of an `agent_id` declared by project configuration."""
    if state.agent_id == agent_id:
        return

    agent = client.get_agent(agent_id, include_files=False)
    name = agent.get("name") if isinstance(agent.get("name"), str) else "<unnamed>"
    if assume_yes:
        print(f"Using agent_id from agent.json: {agent_id} ({name})")
        return

    try:
        prompt = (
            f"Deploy to agent {name} ({agent_id}) from agent.json? "
            "This will update that remote agent. [y/N]: "
        )
        answer = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        print("Aborted.")
        raise SystemExit(1) from None
    if answer not in {"y", "yes"}:
        print("Aborted.")
        raise SystemExit(1)


def _deploy_agent(
    client: ApiClient,
    agent_id: str | None,
    *,
    create_payload: dict[str, Any],
    metadata_payload: dict[str, Any],
    directory_files: dict[str, str],
    allow_create_on_missing: bool = True,
) -> tuple[dict[str, Any], str | None]:
    """Create the agent or patch metadata and sync managed files."""
    from deepagents_cli.deploy.api_client import ApiError

    if agent_id:
        try:
            agent = client.patch_agent(agent_id, metadata_payload)
        except ApiError as exc:
            if exc.status == 404 and allow_create_on_missing:  # noqa: PLR2004
                print(f"Note: agent {agent_id} no longer exists — creating a new one.")
            else:
                raise
        else:
            agent.setdefault("id", agent_id)
            revision = _sync_agent_directory(client, agent_id, directory_files)
            return agent, revision or agent.get("revision")
    agent = client.create_agent(create_payload)
    return agent, agent.get("revision")


def _sync_agent_directory(
    client: ApiClient,
    agent_id: str,
    directory_files: dict[str, str],
) -> str | None:
    """Commit managed project files so the remote directory matches local."""
    from deepagents_cli.deploy.api_client import ApiError
    from deepagents_cli.deploy.payload import build_directory_delta

    directory = _get_agent_directory(client, agent_id)
    remote_files = _extract_directory_files(directory)
    parent_commit = _extract_commit_hash(directory)
    delta = build_directory_delta(remote_files, directory_files)
    if not delta:
        return parent_commit
    try:
        commit = client.commit_agent_directory(
            agent_id,
            files=delta,
            parent_commit=parent_commit,
        )
    except ApiError as exc:
        if exc.status not in {409, 412}:
            raise
        directory = client.get_agent_directory(agent_id)
        delta = build_directory_delta(
            _extract_directory_files(directory), directory_files
        )
        if not delta:
            return _extract_commit_hash(directory)
        commit = client.commit_agent_directory(
            agent_id,
            files=delta,
            parent_commit=_extract_commit_hash(directory),
        )
    return _extract_commit_hash(commit) or parent_commit


def _get_agent_directory(client: ApiClient, agent_id: str) -> dict[str, Any]:
    from deepagents_cli.deploy.api_client import ApiError

    try:
        return client.get_agent_directory(agent_id)
    except ApiError as exc:
        if exc.status == 404:  # noqa: PLR2004
            return {}
        raise


def _extract_directory_files(directory: dict[str, Any]) -> dict[str, Any]:
    files = directory.get("files")
    if isinstance(files, dict):
        return files
    nested = directory.get("directory")
    if isinstance(nested, dict) and isinstance(nested.get("files"), dict):
        return nested["files"]
    return {}


def _extract_commit_hash(payload: dict[str, Any]) -> str | None:
    for key in ("commit_hash", "revision", "hash"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    for key in ("commit", "directory"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            value = _extract_commit_hash(nested)
            if value:
                return value
    return None


def _print_deploy_result(
    agent: dict[str, Any],
    endpoint: str,
    *,
    detach: bool,
    client: ApiClient,
    revision: str | None = None,
) -> None:
    """Print a deploy summary and optionally poll agent health."""
    name = agent.get("name", "?")
    agent_id = agent.get("id", "?")
    revision_display = (revision or agent.get("revision") or "")[:8]
    smith_endpoint = endpoint.replace("api.smith.langchain.com", "smith.langchain.com")
    print(f"\nDeployed: {name}")
    print(f"  agent_id: {agent_id}")
    print(f"  revision: {revision_display}")
    print(f"  {smith_endpoint}/o/-/agents/{agent_id}")
    if detach:
        return
    try:
        health = client._request("GET", f"/v1/deepagents/agents/{agent_id}/health")
        print(f"  health:   {health}")
    except Exception as exc:
        print(f"  health check skipped: {exc}")


def _add_agents_parser(
    subparsers: Any,  # noqa: ANN401
    make_help_action: Callable[[Callable[[], None]], type[argparse.Action]],
) -> None:
    p = subparsers.add_parser("agents", help="Manage agents", add_help=False)
    p.add_argument(
        "-h",
        "--help",
        action=make_help_action(lambda: p.print_help()),
        help="show this help message and exit",
    )
    sub = p.add_subparsers(dest="agents_cmd", required=True)
    sub.add_parser("list")
    g = sub.add_parser("get")
    g.add_argument("agent_id")
    g.add_argument("--include-files", action="store_true")
    d = sub.add_parser("delete")
    d.add_argument("agent_id")
    d.add_argument("--yes", action="store_true")


def execute_agents_command(args: argparse.Namespace) -> None:
    """Run the `deepagents agents` sub-command."""
    from deepagents_cli.config import _load_dotenv
    from deepagents_cli.deploy.api_client import ApiClient, ApiError

    _load_dotenv(start_path=Path.cwd())
    client = ApiClient.from_env()
    try:
        _execute_agents_command(args, client)
    except ApiError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from None


def _execute_agents_command(args: argparse.Namespace, client: ApiClient) -> None:
    if args.agents_cmd == "list":
        for agent in client.iter_agents(page_size=50):
            updated = agent.get("updated_at", "")
            print(f"{agent.get('id')}\t{agent.get('name', '')}\t{updated}")
    elif args.agents_cmd == "get":
        agent = client.get_agent(args.agent_id, include_files=args.include_files)
        print(json.dumps(agent, indent=2))
    elif args.agents_cmd == "delete":
        if not args.yes:
            try:
                prompt = f"Delete agent {args.agent_id}? [y/N]: "
                answer = input(prompt).strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                print("Aborted.")
                return
            if answer not in {"y", "yes"}:
                print("Aborted.")
                return
        client.delete_agent(args.agent_id)
        print(f"Deleted {args.agent_id}")


def _add_mcp_servers_parser(
    subparsers: Any,  # noqa: ANN401
    make_help_action: Callable[[Callable[[], None]], type[argparse.Action]],
) -> None:
    p = subparsers.add_parser("mcp-servers", help="Manage MCP servers", add_help=False)
    p.add_argument(
        "-h",
        "--help",
        action=make_help_action(lambda: p.print_help()),
        help="show this help message and exit",
    )
    sub = p.add_subparsers(dest="mcp_cmd", required=True)
    sub.add_parser("list")
    a = sub.add_parser("add")
    a.add_argument("--url", required=True)
    a.add_argument("--name", default=None)
    a.add_argument("--header", action="append", default=[], metavar="KEY=VALUE")
    a.add_argument("--auth-type", default="headers", choices=["headers", "oauth"])
    a.add_argument(
        "--no-tools",
        action="store_true",
        help="Skip listing the server's tools after registering",
    )
    a.add_argument(
        "--connect",
        action="store_true",
        help="Start OAuth connection after creating an OAuth MCP server",
    )
    _add_oauth_connect_options(a)
    id_help = "MCP server id, exact name, or URL"
    g = sub.add_parser("get")
    g.add_argument("mcp_server_id", metavar="ID|NAME|URL", help=id_help)
    u = sub.add_parser("update")
    u.add_argument("mcp_server_id", metavar="ID|NAME|URL", help=id_help)
    u.add_argument("--url", default=None)
    u.add_argument("--header", action="append", default=None, metavar="KEY=VALUE")
    u.add_argument("--clear-headers", action="store_true")
    u.add_argument("--auth-type", default=None, choices=["headers"])
    d = sub.add_parser("delete")
    d.add_argument("mcp_server_id", metavar="ID|NAME|URL", help=id_help)
    d.add_argument("--yes", action="store_true")
    c = sub.add_parser("connect")
    c.add_argument("mcp_server_id", metavar="ID|NAME|URL", help=id_help)
    _add_oauth_connect_options(c)
    tl = sub.add_parser("tools")
    tl.add_argument("mcp_server_id", metavar="ID|NAME|URL", help=id_help)


def _add_oauth_connect_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--scope",
        action="append",
        default=[],
        metavar="SCOPE",
        help="OAuth scope to request; repeat for multiple scopes",
    )
    parser.add_argument(
        "--force-new",
        action="store_true",
        help="Create a fresh OAuth session instead of reusing an existing token",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Seconds to wait for OAuth completion; use 0 to skip polling",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Print the verification URL without opening a browser",
    )


_MCP_UUID_RE = re.compile(
    r"\A[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\Z"
)


def _resolve_mcp_server_id(client: ApiClient, identifier: str) -> str:
    """Resolve an MCP server *identifier* to its id.

    The platform addresses MCP servers by id, so a non-UUID *identifier* is
    matched against the workspace server list by exact name first, then by
    normalized URL (lowercased, trailing slash stripped — same rules deploy
    uses). A UUID is returned as-is without a lookup.

    Args:
        client: Authenticated deploy API client.
        identifier: An MCP server id (UUID), exact name, or URL.

    Returns:
        The resolved MCP server id.

    Raises:
        SystemExit: If nothing matches or the identifier is ambiguous.
    """
    from deepagents_cli.deploy.mcp_resolver import _normalize_url

    candidate = identifier.strip()
    if _MCP_UUID_RE.match(candidate):
        return candidate

    servers = client.list_mcp_servers()
    matches = [s for s in servers if s.get("name") == identifier]
    if not matches:
        target = _normalize_url(identifier)
        matches = [
            s
            for s in servers
            if isinstance(s.get("url"), str) and _normalize_url(s["url"]) == target
        ]

    ids = sorted({s["id"] for s in matches if isinstance(s.get("id"), str)})
    if not ids:
        print(
            f"Error: no MCP server matches {identifier!r}. "
            "Run `deepagents mcp-servers list` to see ids, names, and URLs."
        )
        raise SystemExit(1)
    if len(ids) > 1:
        listed = ", ".join(ids)
        print(
            f"Error: {identifier!r} matches multiple MCP servers ({listed}). "
            "Re-run with the id."
        )
        raise SystemExit(1)
    return ids[0]


def execute_mcp_servers_command(args: argparse.Namespace) -> None:
    """Run the `deepagents mcp-servers` sub-command."""
    from deepagents_cli.config import _load_dotenv
    from deepagents_cli.deploy.api_client import ApiClient, ApiError

    _load_dotenv(start_path=Path.cwd())
    client = ApiClient.from_env()
    try:
        _execute_mcp_servers_command(args, client)
    except ApiError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from None


def _execute_mcp_servers_command(args: argparse.Namespace, client: ApiClient) -> None:
    if args.mcp_cmd == "list":
        for srv in client.list_mcp_servers():
            print(f"{srv.get('id')}\t{srv.get('name', '')}\t{srv.get('url', '')}")
    elif args.mcp_cmd == "add":
        _execute_mcp_server_add(args, client)
    elif args.mcp_cmd == "get":
        server_id = _resolve_mcp_server_id(client, args.mcp_server_id)
        server = client.get_mcp_server(server_id)
        print(json.dumps(_redact_mcp_server(server), indent=2))
    elif args.mcp_cmd == "update":
        _execute_mcp_server_update(args, client)
    elif args.mcp_cmd == "delete":
        _execute_mcp_server_delete(args, client)
    elif args.mcp_cmd == "connect":
        _connect_mcp_server_oauth(
            client,
            _resolve_mcp_server_id(client, args.mcp_server_id),
            scopes=args.scope,
            force_new=args.force_new,
            timeout_seconds=args.timeout,
            no_browser=args.no_browser,
        )
    elif args.mcp_cmd == "tools":
        _execute_mcp_server_tools(args, client)


def _execute_mcp_server_add(args: argparse.Namespace, client: ApiClient) -> None:
    from urllib.parse import urlparse

    if args.auth_type == "oauth" and args.header:
        print("Error: --header cannot be used with --auth-type oauth.")
        raise SystemExit(1)
    if args.auth_type != "oauth" and getattr(args, "connect", False):
        print("Error: --connect requires --auth-type oauth.")
        raise SystemExit(1)
    headers = _parse_header_args(args.header)
    name = args.name or urlparse(args.url).hostname or args.url
    srv = client.create_mcp_server(
        name=name,
        url=args.url,
        headers=headers,
        auth_type=args.auth_type,
        oauth_mode="per_user_dynamic_client" if args.auth_type == "oauth" else None,
    )
    srv_id = srv.get("id")
    srv_name = srv.get("name")
    srv_url = srv.get("url")
    print(f"Created mcp_server {srv_id}: {srv_name} → {srv_url}")
    connect_attempted = False
    if getattr(args, "connect", False):
        if not isinstance(srv_id, str) or not srv_id:
            print("Error: created OAuth MCP server did not include an id.")
            raise SystemExit(1)
        _connect_mcp_server_oauth(
            client,
            srv_id,
            scopes=args.scope,
            force_new=args.force_new,
            timeout_seconds=args.timeout,
            no_browser=args.no_browser,
        )
        connect_attempted = True
    _show_tools_after_add(
        client,
        srv,
        auth_type=args.auth_type,
        connect_attempted=connect_attempted,
        suppressed=getattr(args, "no_tools", False),
    )


def _execute_mcp_server_update(args: argparse.Namespace, client: ApiClient) -> None:
    headers = _parse_update_headers(args.header, args.clear_headers)
    if args.url is None and headers is None and args.auth_type is None:
        print("Error: provide at least one of --url, --header, or --clear-headers.")
        raise SystemExit(1)
    srv = client.update_mcp_server(
        _resolve_mcp_server_id(client, args.mcp_server_id),
        url=args.url,
        headers=headers,
        auth_type=args.auth_type,
    )
    srv_id = srv.get("id")
    srv_name = srv.get("name")
    srv_url = srv.get("url")
    print(f"Updated mcp_server {srv_id}: {srv_name} → {srv_url}")


def _execute_mcp_server_delete(args: argparse.Namespace, client: ApiClient) -> None:
    server_id = _resolve_mcp_server_id(client, args.mcp_server_id)
    label = (
        server_id
        if args.mcp_server_id == server_id
        else f"{args.mcp_server_id} ({server_id})"
    )
    if not args.yes:
        try:
            prompt = f"Delete MCP server {label}? [y/N]: "
            answer = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            print("Aborted.")
            return
        if answer not in {"y", "yes"}:
            print("Aborted.")
            return
    client.delete_mcp_server(server_id)
    print(f"Deleted {server_id}")


def _execute_mcp_server_tools(args: argparse.Namespace, client: ApiClient) -> None:
    server = client.get_mcp_server(_resolve_mcp_server_id(client, args.mcp_server_id))
    url = server.get("url")
    if not isinstance(url, str) or not url:
        print("Error: that MCP server record has no URL.")
        raise SystemExit(1)
    tools = client.list_mcp_server_tools(
        url, oauth_provider_id=server.get("oauth_provider_id")
    )
    _print_mcp_tools(server, tools)


def _connect_mcp_server_oauth(
    client: ApiClient,
    mcp_server_id: str,
    *,
    scopes: list[str],
    force_new: bool,
    timeout_seconds: int,
    no_browser: bool,
) -> None:
    if timeout_seconds < 0:
        print("Error: --timeout must be greater than or equal to 0.")
        raise SystemExit(1)

    provider = client.register_mcp_oauth_provider(mcp_server_id)
    provider_id = provider.get("oauth_provider_id")
    if not isinstance(provider_id, str) or not provider_id:
        print("Error: OAuth provider registration did not return oauth_provider_id.")
        raise SystemExit(1)

    strategy = "CREATE" if force_new else "REUSE"
    session = client.create_auth_session(
        provider_id=provider_id,
        scopes=scopes,
        strategy=strategy,
    )
    _handle_auth_session(
        client,
        session,
        timeout_seconds=timeout_seconds,
        no_browser=no_browser,
    )


def _handle_auth_session(
    client: ApiClient,
    session: dict[str, Any],
    *,
    timeout_seconds: int,
    no_browser: bool,
) -> None:
    status = _auth_session_status(session)
    if status == "COMPLETED":
        print("MCP OAuth connection is ready.")
        return
    if status != "PENDING":
        _raise_auth_session_status(status)

    verification_url = session.get("verification_url")
    if not isinstance(verification_url, str) or not verification_url:
        print("Error: OAuth session is pending but no verification_url was returned.")
        raise SystemExit(1)

    print("Open this URL to authorize the MCP server:")
    print(f"  {verification_url}")
    if not no_browser:
        _open_browser(verification_url)

    session_id = session.get("id")
    if not isinstance(session_id, str) or not session_id:
        print("Error: OAuth session is pending but no session id was returned.")
        raise SystemExit(1)
    if timeout_seconds == 0:
        print("Authorization started. Re-run `deepagents mcp-servers connect` later.")
        return

    deadline = time.monotonic() + timeout_seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print("Timed out waiting for OAuth completion.")
            raise SystemExit(1)
        wait_seconds = max(1, min(5, int(remaining)))
        session = client.get_auth_session(session_id, wait_seconds=wait_seconds)
        status = _auth_session_status(session)
        if status == "COMPLETED":
            print("MCP OAuth connection is ready.")
            return
        if status != "PENDING":
            _raise_auth_session_status(status)


def _auth_session_status(session: dict[str, Any]) -> str:
    status = session.get("status")
    return status.upper() if isinstance(status, str) else ""


def _raise_auth_session_status(status: str) -> None:
    if status in {"CONNECTION_REQUIRED", "TOKEN_EXPIRED"}:
        print(
            "Error: OAuth connection is required or expired. "
            "Run `deepagents mcp-servers connect` again."
        )
    else:
        print(f"Error: OAuth session ended with status {status or '<missing>'}.")
    raise SystemExit(1)


def _open_browser(url: str) -> None:
    import webbrowser

    try:
        opened = webbrowser.open(url)
    except Exception as exc:
        print(f"Could not open browser automatically: {exc}")
        return
    if not opened:
        print("Could not open browser automatically.")


def _parse_header_args(raw: list[str]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for entry in raw:
        if "=" not in entry:
            print(f"Error: --header must be KEY=VALUE, got {entry!r}")
            raise SystemExit(1)
        key, _, value = entry.partition("=")
        out.append({"key": key.strip(), "value": value})
    return out


def _parse_update_headers(
    raw: list[str] | None,
    clear_headers: bool,
) -> list[dict[str, str]] | None:
    if raw and clear_headers:
        print("Error: use either --header or --clear-headers, not both.")
        raise SystemExit(1)
    if clear_headers:
        return []
    if raw is None:
        return None
    return _parse_header_args(raw)


def _show_tools_after_add(
    client: ApiClient,
    srv: dict[str, Any],
    *,
    auth_type: str,
    connect_attempted: bool,
    suppressed: bool,
) -> None:
    """Best-effort: list a freshly registered server's tools after `add`.

    Closes the discovery loop so the user sees what the server exposes (and a
    paste-ready tools.json snippet) without a separate `tools` call. It never
    raises: a missing tools endpoint, OAuth server that isn't connected yet, or
    any API error degrades to a hint pointing at `mcp-servers tools`.

    Args:
        client: Authenticated deploy API client.
        srv: The MCP server record returned by `create_mcp_server`.
        auth_type: The server's auth type (`headers` or `oauth`).
        connect_attempted: Whether an OAuth connect flow was just run.
        suppressed: When `True` (the `--no-tools` flag), skip listing entirely.
    """
    if suppressed:
        return
    srv_id = srv.get("id")
    if not isinstance(srv_id, str) or not srv_id:
        return
    if auth_type == "oauth" and not connect_attempted:
        print(f"  After connecting, run: deepagents mcp-servers tools {srv_id}")
        return

    try:
        # OAuth servers need `oauth_provider_id`, which is only populated after
        # connect, so refresh the record before listing.
        record = client.get_mcp_server(srv_id) if auth_type == "oauth" else srv
        url = record.get("url")
        if not isinstance(url, str) or not url:
            return
        tools = client.list_mcp_server_tools(
            url, oauth_provider_id=record.get("oauth_provider_id")
        )
    except Exception:  # tool listing is best-effort; it must never fail `add`
        print(f"  Run `deepagents mcp-servers tools {srv_id}` to list its tools.")
        return
    print()
    _print_mcp_tools(record, tools)


def _print_mcp_tools(server: dict[str, Any], tools: list[dict[str, Any]]) -> None:
    """Print an MCP server's tools and a paste-ready tools.json snippet."""
    url = str(server.get("url") or "")
    name = str(server.get("name") or "")
    label = name or url
    if not tools:
        print(f"No tools found for {label}.")
        return

    print(f"Tools for {label} ({url}):")
    width = max(len(str(t.get("name") or "")) for t in tools)
    for tool in tools:
        tname = str(tool.get("name") or "")
        description = str(tool.get("description") or "").strip()
        summary = description.splitlines()[0] if description else ""
        print(f"  {tname.ljust(width)}  {summary}")

    entries: list[dict[str, str]] = []
    for tool in tools:
        tname = str(tool.get("name") or "")
        if not tname:
            continue
        entry = {"name": tname, "mcp_server_url": url, "display_name": tname}
        if name:
            entry["mcp_server_name"] = name
        entries.append(entry)
    snippet = {"tools": entries, "interrupt_config": {}}
    print("\nAdd to tools.json:")
    print(json.dumps(snippet, indent=2))


def _redact_mcp_server(server: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(server)
    headers = redacted.get("headers")
    if isinstance(headers, list):
        redacted["headers"] = [
            _redact_mcp_header(header) if isinstance(header, dict) else header
            for header in headers
        ]
    return redacted


def _redact_mcp_header(header: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(header)
    if "value" in redacted:
        redacted["value"] = "***"
    return redacted
