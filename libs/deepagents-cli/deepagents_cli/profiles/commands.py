"""CLI commands for profile management.

These commands are registered with the CLI via main.py:
- deepagents push <name> <local_dir> [--public|--private]
- deepagents pull <name> [--public] [--version N]
- deepagents profiles [--public|--private]
"""

import argparse
import re
from pathlib import Path
from typing import Any

from deepagents_cli.config import COLORS, Settings, console

from .client import (
    AgentFilesystemClient,
    AgentFilesystemError,
    AuthenticationError,
    ProfileConflictError,
    ProfileNotFoundError,
)


def _validate_profile_name(name: str) -> tuple[bool, str]:
    """Validate profile name to prevent path traversal attacks.

    Args:
        name: The profile name to validate

    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is empty.
    """
    if not name or not name.strip():
        return False, "Profile name cannot be empty"

    if ".." in name:
        return False, "Profile name cannot contain '..'"

    if name.startswith(("/", "\\")):
        return False, "Profile name cannot be an absolute path"

    if "/" in name or "\\" in name:
        return False, "Profile name cannot contain path separators"

    # Only allow alphanumeric, hyphens, underscores
    if not re.match(r"^[a-zA-Z0-9_-]+$", name):
        return False, "Profile name can only contain letters, numbers, hyphens, and underscores"

    return True, ""


def _load_local_files(local_dir: Path) -> list[dict[str, Any]]:
    """Load all files from a local directory for pushing.

    Args:
        local_dir: Path to the local directory

    Returns:
        List of file dicts with path (relative), content, and size
    """
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


def _save_profile_files(profile_name: str, files: list, settings: Settings) -> Path:
    """Save pulled profile files to local storage.

    Args:
        profile_name: Name of the profile
        files: List of ProfileFile objects
        settings: Settings instance

    Returns:
        Path to the profile directory
    """
    profile_dir = settings.get_agent_dir(profile_name)

    # Clear existing directory if it exists
    if profile_dir.exists():
        import shutil
        shutil.rmtree(profile_dir)

    # Create profile directory
    profile_dir.mkdir(parents=True, exist_ok=True)

    # Write each file
    for file in files:
        # Handle both leading slash and no leading slash in path
        file_path = file.path.lstrip("/")
        full_path = profile_dir / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(file.content)

    return profile_dir


def _push(args: argparse.Namespace) -> None:
    """Handle the push command.

    Push a local directory to the remote filesystem as a profile.
    """
    settings = Settings.from_environment()

    # Validate profile name
    is_valid, error_msg = _validate_profile_name(args.name)
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

    # Validate local directory
    local_dir = Path(args.local_dir).resolve()
    if not local_dir.exists():
        console.print(f"[bold red]Error:[/bold red] Directory not found: {local_dir}")
        return

    if not local_dir.is_dir():
        console.print(f"[bold red]Error:[/bold red] Not a directory: {local_dir}")
        return

    # Determine if public or private
    is_public = args.public if hasattr(args, "public") and args.public else False

    # Load files from local directory
    console.print(f"Loading files from {local_dir}...", style=COLORS["dim"])
    files = _load_local_files(local_dir)

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
            f"Profile '{response.name}' pushed as {visibility}.",
            style=COLORS["primary"],
        )
        console.print(f"  Version: {response.version}", style=COLORS["dim"])
        console.print(f"  Files: {response.files_count}", style=COLORS["dim"])
    except AuthenticationError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
    except ProfileConflictError:
        console.print(
            f"[bold red]Error:[/bold red] A public profile named '{args.name}' already exists "
            "and is owned by another user."
        )
    except AgentFilesystemError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")


def _pull(args: argparse.Namespace) -> None:
    """Handle the pull command.

    Pull a profile from the remote filesystem to local storage.
    """
    settings = Settings.from_environment()

    # Validate profile name
    is_valid, error_msg = _validate_profile_name(args.name)
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
            "[yellow]Note:[/yellow] No AGENT_FS_API_KEY set. Only public profiles available.",
            style=COLORS["dim"],
        )

    # Create client and pull
    client = AgentFilesystemClient(
        base_url=settings.agent_fs_url,
        api_key=settings.agent_fs_api_key,
    )

    try:
        console.print(f"Pulling profile '{args.name}'...", style=COLORS["dim"])
        response = client.pull(args.name, version=version, is_public=is_public)

        # Save files locally
        profile_dir = _save_profile_files(args.name, response.files, settings)

        console.print(
            f"\n[bold green]Success![/bold green] Profile '{response.name}' pulled.",
            style=COLORS["primary"],
        )
        console.print(f"  Version: {response.version}", style=COLORS["dim"])
        console.print(f"  Files: {len(response.files)}", style=COLORS["dim"])
        console.print(f"  Location: {profile_dir}", style=COLORS["dim"])
        console.print(
            f"\n[dim]Use it with: deepagents --agent {args.name}[/dim]",
            style=COLORS["dim"],
        )
    except ProfileNotFoundError:
        console.print(f"[bold red]Error:[/bold red] Profile '{args.name}' not found.")
        if not settings.has_agent_fs_api_key:
            console.print(
                "\n[dim]If this is a private profile, set AGENT_FS_API_KEY to access it.[/dim]",
                style=COLORS["dim"],
            )
    except AuthenticationError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
    except AgentFilesystemError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")


def _display_profile_group(profiles: list, title: str, color: str) -> None:
    """Display a group of profiles with a title."""
    if not profiles:
        return
    console.print(f"[bold {color}]{title}:[/bold {color}]", style=COLORS["primary"])
    for profile in profiles:
        console.print(f"  - [bold]{profile.name}[/bold]", style=COLORS["primary"])
        console.print(f"    Version: {profile.latest_version}", style=COLORS["dim"])
    console.print()


def _get_empty_message(*, show_public: bool, show_private: bool) -> str:
    """Get the appropriate 'no profiles found' message."""
    if show_public:
        return "[yellow]No public profiles found.[/yellow]"
    if show_private:
        return "[yellow]No private profiles found.[/yellow]"
    return "[yellow]No profiles found.[/yellow]"


def _list_profiles(args: argparse.Namespace) -> None:
    """Handle the profiles (list) command."""
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
            "[bold red]Error:[/bold red] AGENT_FS_API_KEY required to list private profiles."
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
        profiles = client.list_profiles(is_public=is_public_filter)

        if not profiles:
            console.print(_get_empty_message(show_public=show_public, show_private=show_private))
            return

        console.print("\n[bold]Available Profiles:[/bold]\n", style=COLORS["primary"])

        public_profiles = [p for p in profiles if p.is_public]
        private_profiles = [p for p in profiles if not p.is_public]

        _display_profile_group(public_profiles, "Public Profiles", "cyan")
        _display_profile_group(private_profiles, "Private Profiles", "green")

    except (AuthenticationError, AgentFilesystemError) as e:
        console.print(f"[bold red]Error:[/bold red] {e}")


def setup_push_parser(subparsers: Any) -> argparse.ArgumentParser:
    """Setup the push subcommand parser."""
    push_parser = subparsers.add_parser(
        "push",
        help="Push a local directory as a profile to the remote filesystem",
        description="Push a local directory to the agent filesystem as a profile",
    )
    push_parser.add_argument("name", help="Profile name")
    push_parser.add_argument("local_dir", help="Local directory to push")
    push_parser.add_argument(
        "--public",
        action="store_true",
        help="Make the profile public (default: private)",
    )
    push_parser.add_argument(
        "--private",
        action="store_true",
        help="Make the profile private (this is the default)",
    )
    return push_parser


def setup_pull_parser(subparsers: Any) -> argparse.ArgumentParser:
    """Setup the pull subcommand parser."""
    pull_parser = subparsers.add_parser(
        "pull",
        help="Pull a profile from the remote filesystem",
        description="Pull a profile from the agent filesystem to local storage",
    )
    pull_parser.add_argument("name", help="Profile name to pull")
    pull_parser.add_argument(
        "--public",
        action="store_true",
        help="Pull public profile only (no authentication required)",
    )
    pull_parser.add_argument(
        "--version",
        type=int,
        help="Specific version to pull (default: latest)",
    )
    return pull_parser


def setup_profiles_parser(subparsers: Any) -> argparse.ArgumentParser:
    """Setup the profiles subcommand parser."""
    profiles_parser = subparsers.add_parser(
        "profiles",
        help="List available profiles on the remote filesystem",
        description="List profiles from the agent filesystem",
    )
    profiles_parser.add_argument(
        "--public",
        action="store_true",
        help="Show only public profiles",
    )
    profiles_parser.add_argument(
        "--private",
        action="store_true",
        help="Show only private profiles (requires AGENT_FS_API_KEY)",
    )
    return profiles_parser


def execute_push_command(args: argparse.Namespace) -> None:
    """Execute the push command."""
    _push(args)


def execute_pull_command(args: argparse.Namespace) -> None:
    """Execute the pull command."""
    _pull(args)


def execute_profiles_command(args: argparse.Namespace) -> None:
    """Execute the profiles command."""
    _list_profiles(args)


__all__ = [
    "execute_profiles_command",
    "execute_pull_command",
    "execute_push_command",
    "setup_profiles_parser",
    "setup_pull_parser",
    "setup_push_parser",
]
