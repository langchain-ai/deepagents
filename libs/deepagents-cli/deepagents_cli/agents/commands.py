import argparse
import re
from pathlib import Path
from typing import Any

from deepagents_cli.config import COLORS, Settings, console

from .client import (
    AgentConflictError,
    AgentFilesystemClient,
    AgentFilesystemError,
    AgentNotFoundError,
    AuthenticationError,
)


def _validate_agent_name(name: str) -> tuple[bool, str]:
    if not name or not name.strip():
        return False, "Agent name cannot be empty"

    if ".." in name:
        return False, "Agent name cannot contain '..'"

    if name.startswith(("/", "\\")):
        return False, "Agent name cannot be an absolute path"

    if "/" in name or "\\" in name:
        return False, "Agent name cannot contain path separators"

    # Only allow alphanumeric, hyphens, underscores
    if not re.match(r"^[a-zA-Z0-9_-]+$", name):
        return False, "Agent name can only contain letters, numbers, hyphens, and underscores"

    return True, ""


def _load_local_files(local_dir: Path) -> list[dict[str, Any]]:
    files = []
    for path in local_dir.rglob("*"):
        if path.is_file():
            # Store relative path (without leading slash for cleaner storage)
            relative_path = str(path.relative_to(local_dir))
            try:
                content = path.read_text()
                files.append({
                    "path": relative_path,
                    "content": content,
                    "size": len(content.encode("utf-8")),
                })
            except UnicodeDecodeError:
                # Skip binary files with a warning
                console.print(
                    f"[yellow]Warning:[/yellow] Skipping binary file: {relative_path}",
                    style=COLORS["dim"],
                )
    return files


def _save_agent_files(agent_name: str, files: list, settings: Settings) -> Path:
    agent_dir = settings.get_agent_dir(agent_name)

    # Clear existing directory if it exists
    if agent_dir.exists():
        import shutil
        shutil.rmtree(agent_dir)

    # Create agent directory
    agent_dir.mkdir(parents=True, exist_ok=True)

    # Write each file
    for file in files:
        # Handle both leading slash and no leading slash in path
        file_path = file.path.lstrip("/")
        full_path = agent_dir / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(file.content)

    return agent_dir


def _push(args: argparse.Namespace) -> None:
    settings = Settings.from_environment()

    # Validate agent name
    is_valid, error_msg = _validate_agent_name(args.name)
    if not is_valid:
        console.print(f"[bold red]Error:[/bold red] {error_msg}")
        return

    # Check required environment variables
    if not settings.has_agent_fs_url:
        console.print("[bold red]Error:[/bold red] AGENT_FS_URL not set.")
        console.print(
            "\n[dim]Set the agent filesystem URL:[/dim]",
            style=COLORS["dim"],
        )
        console.print("  export AGENT_FS_URL=https://your-agent-fs-server.com", style=COLORS["dim"])
        return

    if not settings.has_agent_fs_api_key:
        console.print("[bold red]Error:[/bold red] AGENT_FS_API_KEY not set.")
        console.print(
            "\n[dim]Authentication is required for push. Set your API key:[/dim]",
            style=COLORS["dim"],
        )
        console.print("  export AGENT_FS_API_KEY=your_api_key", style=COLORS["dim"])
        return

    # Get agent directory from ~/.deepagents/<name>/
    agent_dir = settings.get_agent_dir(args.name)
    if not agent_dir.exists():
        console.print(f"[bold red]Error:[/bold red] Agent directory not found: {agent_dir}")
        console.print(
            f"\n[dim]Create the directory first, or use 'deepagents pull {args.name}' "
            "to fetch an existing agent.[/dim]",
            style=COLORS["dim"],
        )
        return

    if not agent_dir.is_dir():
        console.print(f"[bold red]Error:[/bold red] Not a directory: {agent_dir}")
        return

    # Determine if public or private
    is_public = args.public if hasattr(args, "public") and args.public else False

    # Load files from agent directory
    console.print(f"Loading files from {agent_dir}...", style=COLORS["dim"])
    files = _load_local_files(agent_dir)

    if not files:
        console.print("[bold red]Error:[/bold red] No files found in directory.")
        return

    console.print(f"Found {len(files)} file(s) to push.", style=COLORS["dim"])

    # Create client and push
    client = AgentFilesystemClient(
        base_url=settings.agent_fs_url,
        api_key=settings.agent_fs_api_key,
    )

    try:
        response = client.push(args.name, files, is_public=is_public)
        visibility = "public" if is_public else "private"
        console.print(
            f"\n[bold green]Success![/bold green] "
            f"Agent '{response.name}' pushed as {visibility}.",
            style=COLORS["primary"],
        )
        console.print(f"  Version: {response.version}", style=COLORS["dim"])
        console.print(f"  Files: {response.files_count}", style=COLORS["dim"])
    except AuthenticationError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
    except AgentConflictError:
        console.print(
            f"[bold red]Error:[/bold red] A public agent named '{args.name}' already exists "
            "and is owned by another user."
        )
    except AgentFilesystemError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")


def _pull(args: argparse.Namespace) -> None:
    settings = Settings.from_environment()

    # Validate agent name
    is_valid, error_msg = _validate_agent_name(args.name)
    if not is_valid:
        console.print(f"[bold red]Error:[/bold red] {error_msg}")
        return

    # Check required environment variables
    if not settings.has_agent_fs_url:
        console.print("[bold red]Error:[/bold red] AGENT_FS_URL not set.")
        console.print(
            "\n[dim]Set the agent filesystem URL:[/dim]",
            style=COLORS["dim"],
        )
        console.print("  export AGENT_FS_URL=https://your-agent-fs-server.com", style=COLORS["dim"])
        return

    # Determine if pulling public only
    is_public = args.public if hasattr(args, "public") and args.public else None
    version = args.version if hasattr(args, "version") else None

    # Warn if no API key and not explicitly pulling public
    if not settings.has_agent_fs_api_key and not is_public:
        console.print(
            "[yellow]Note:[/yellow] No AGENT_FS_API_KEY set. Only public agents available.",
            style=COLORS["dim"],
        )

    # Create client and pull
    client = AgentFilesystemClient(
        base_url=settings.agent_fs_url,
        api_key=settings.agent_fs_api_key,
    )

    try:
        console.print(f"Pulling agent '{args.name}'...", style=COLORS["dim"])
        response = client.pull(args.name, version=version, is_public=is_public)

        # Save files locally
        agent_dir = _save_agent_files(args.name, response.files, settings)

        console.print(
            f"\n[bold green]Success![/bold green] Agent '{response.name}' pulled.",
            style=COLORS["primary"],
        )
        console.print(f"  Version: {response.version}", style=COLORS["dim"])
        console.print(f"  Files: {len(response.files)}", style=COLORS["dim"])
        console.print(f"  Location: {agent_dir}", style=COLORS["dim"])
        console.print(
            f"\n[dim]Use it with: deepagents --agent {args.name}[/dim]",
            style=COLORS["dim"],
        )
    except AgentNotFoundError:
        console.print(f"[bold red]Error:[/bold red] Agent '{args.name}' not found.")
        if not settings.has_agent_fs_api_key:
            console.print(
                "\n[dim]If this is a private agent, set AGENT_FS_API_KEY to access it.[/dim]",
                style=COLORS["dim"],
            )
    except AuthenticationError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
    except AgentFilesystemError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")


def _display_agent_group(agents: list, title: str, color: str) -> None:
    if not agents:
        return
    console.print(f"[bold {color}]{title}:[/bold {color}]", style=COLORS["primary"])
    for agent in agents:
        console.print(f"  - [bold]{agent.name}[/bold]", style=COLORS["primary"])
        console.print(f"    Version: {agent.latest_version}", style=COLORS["dim"])
    console.print()


def _get_empty_message(*, show_public: bool, show_private: bool) -> str:
    if show_public:
        return "[yellow]No public agents found.[/yellow]"
    if show_private:
        return "[yellow]No private agents found.[/yellow]"
    return "[yellow]No agents found.[/yellow]"


def _list_agents(args: argparse.Namespace) -> None:
    settings = Settings.from_environment()

    if not settings.has_agent_fs_url:
        console.print("[bold red]Error:[/bold red] AGENT_FS_URL not set.")
        console.print("\n[dim]Set the agent filesystem URL:[/dim]", style=COLORS["dim"])
        console.print("  export AGENT_FS_URL=https://your-agent-fs-server.com", style=COLORS["dim"])
        return

    show_public = getattr(args, "public", False)
    show_private = getattr(args, "private", False)

    if show_private and not settings.has_agent_fs_api_key:
        console.print(
            "[bold red]Error:[/bold red] AGENT_FS_API_KEY required to list private agents."
        )
        console.print("\n[dim]Set your API key:[/dim]", style=COLORS["dim"])
        console.print("  export AGENT_FS_API_KEY=your_api_key", style=COLORS["dim"])
        return

    client = AgentFilesystemClient(
        base_url=settings.agent_fs_url,
        api_key=settings.agent_fs_api_key,
    )

    # Determine is_public filter for API
    is_public_filter: bool | None = None
    if show_public and not show_private:
        is_public_filter = True
    elif show_private and not show_public:
        is_public_filter = False

    try:
        agents = client.list_agents(is_public=is_public_filter)

        if not agents:
            console.print(_get_empty_message(show_public=show_public, show_private=show_private))
            return

        console.print("\n[bold]Available Agents:[/bold]\n", style=COLORS["primary"])

        public_agents = [p for p in agents if p.is_public]
        private_agents = [p for p in agents if not p.is_public]

        _display_agent_group(public_agents, "Public Agents", "cyan")
        _display_agent_group(private_agents, "Private Agents", "green")

    except (AuthenticationError, AgentFilesystemError) as e:
        console.print(f"[bold red]Error:[/bold red] {e}")


def setup_push_parser(subparsers: Any) -> argparse.ArgumentParser:
    push_parser = subparsers.add_parser(
        "push",
        help="Push an agent from ~/.deepagents/<name>/ to the remote filesystem",
        description="Push an agent from ~/.deepagents/<name>/ to the remote filesystem",
    )
    push_parser.add_argument("name", help="Agent name (must match folder name in ~/.deepagents/)")
    push_parser.add_argument(
        "--public",
        action="store_true",
        help="Make the agent public (default: private)",
    )
    push_parser.add_argument(
        "--private",
        action="store_true",
        help="Make the agent private (this is the default)",
    )
    return push_parser


def setup_pull_parser(subparsers: Any) -> argparse.ArgumentParser:
    pull_parser = subparsers.add_parser(
        "pull",
        help="Pull an agent from the remote filesystem",
        description="Pull an agent from the agent filesystem to local storage",
    )
    pull_parser.add_argument("name", help="Agent name to pull")
    pull_parser.add_argument(
        "--public",
        action="store_true",
        help="Pull public agent only (no authentication required)",
    )
    pull_parser.add_argument(
        "--version",
        type=int,
        help="Specific version to pull (default: latest)",
    )
    return pull_parser


def setup_agents_parser(subparsers: Any) -> argparse.ArgumentParser:
    agents_parser = subparsers.add_parser(
        "agents",
        help="List available agents on the remote filesystem",
        description="List agents from the agent filesystem",
    )
    agents_parser.add_argument(
        "--public",
        action="store_true",
        help="Show only public agents",
    )
    agents_parser.add_argument(
        "--private",
        action="store_true",
        help="Show only private agents (requires AGENT_FS_API_KEY)",
    )
    return agents_parser


def execute_push_command(args: argparse.Namespace) -> None:
    _push(args)


def execute_pull_command(args: argparse.Namespace) -> None:
    _pull(args)


def execute_agents_command(args: argparse.Namespace) -> None:
    _list_agents(args)


__all__ = [
    "execute_agents_command",
    "execute_pull_command",
    "execute_push_command",
    "setup_agents_parser",
    "setup_pull_parser",
    "setup_push_parser",
]
