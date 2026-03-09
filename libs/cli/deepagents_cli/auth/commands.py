"""Auth subcommand for managing provider authentication."""

from __future__ import annotations

import argparse
import functools
import sys
from typing import TYPE_CHECKING, Any

from rich.console import Console

if TYPE_CHECKING:
    from collections.abc import Callable

console = Console()


def _show_auth_help() -> None:
    console.print("[bold]deepagents auth[/bold] - Manage provider authentication\n")
    console.print("Subcommands:")
    console.print("  login   Log in to a provider")
    console.print("  status  Show authentication status")
    console.print("  logout  Log out from a provider")
    console.print("\nUse [bold]deepagents auth <command> -h[/bold] for more info.")


def _show_auth_login_help() -> None:
    console.print("[bold]deepagents auth login[/bold] - Log in to a provider\n")
    console.print("Options:")
    console.print("  --provider  Provider to authenticate with (default: codex)")
    console.print("  --headless  Print URL instead of opening browser")


def _show_auth_status_help() -> None:
    console.print("[bold]deepagents auth status[/bold] - Show authentication status\n")
    console.print("Options:")
    console.print("  --provider  Provider to check (default: codex)")


def _show_auth_logout_help() -> None:
    console.print("[bold]deepagents auth logout[/bold] - Log out from a provider\n")
    console.print("Options:")
    console.print("  --provider  Provider to log out from (default: codex)")


def build_help_parent(
    help_fn: Callable[[], None],
    *,
    make_help_action: Callable[[Callable[[], None]], type[argparse.Action]],
) -> list[argparse.ArgumentParser]:
    """Build a parent parser that wires -h/--help to a custom help function.

    Returns:
        List containing a single parent parser.
    """
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "-h",
        "--help",
        action=make_help_action(help_fn),
        default=argparse.SUPPRESS,
        help="Show this help message and exit",
    )
    return [parent]


def setup_auth_parser(
    subparsers: Any,  # noqa: ANN401
    *,
    make_help_action: Callable[[Callable[[], None]], type[argparse.Action]],
) -> argparse.ArgumentParser:
    """Set up the auth subcommand parser with login/status/logout subcommands.

    Args:
        subparsers: The parent subparsers object to add the auth parser to.
        make_help_action: Factory that accepts a zero-argument help
            callable and returns an argparse Action class wired to it.

    Returns:
        The auth subparser.
    """
    help_parent = functools.partial(
        build_help_parent, make_help_action=make_help_action
    )

    auth_parser = subparsers.add_parser(
        "auth",
        help="Manage provider authentication",
        description="Manage provider authentication - login, check status, and logout.",
        add_help=False,
        parents=help_parent(_show_auth_help),
    )
    auth_subparsers = auth_parser.add_subparsers(
        dest="auth_command", help="Auth command"
    )

    # auth login
    login_parser = auth_subparsers.add_parser(
        "login",
        help="Log in to a provider",
        add_help=False,
        parents=help_parent(_show_auth_login_help),
    )
    login_parser.add_argument(
        "--provider",
        default="codex",
        help="Provider to authenticate with (default: codex)",
    )
    login_parser.add_argument(
        "--headless",
        action="store_true",
        help="Print URL instead of opening browser",
    )

    # auth status
    status_parser = auth_subparsers.add_parser(
        "status",
        help="Show authentication status",
        add_help=False,
        parents=help_parent(_show_auth_status_help),
    )
    status_parser.add_argument(
        "--provider",
        default="codex",
        help="Provider to check (default: codex)",
    )

    # auth logout
    logout_parser = auth_subparsers.add_parser(
        "logout",
        help="Log out from a provider",
        add_help=False,
        parents=help_parent(_show_auth_logout_help),
    )
    logout_parser.add_argument(
        "--provider",
        default="codex",
        help="Provider to log out from (default: codex)",
    )

    return auth_parser


def _handle_login(provider: str, *, headless: bool = False) -> None:
    """Handle the auth login command."""
    if provider != "codex":
        console.print(
            f"[bold red]Error:[/bold red] Unknown provider '{provider}'. "
            "Supported providers: codex"
        )
        sys.exit(1)

    try:
        from deepagents_codex import login
        from deepagents_codex.errors import CodexAuthError
    except ImportError:
        console.print(
            "[bold red]Error:[/bold red] Codex provider requires deepagents-codex.\n"
            "Install: pip install 'deepagents-cli[codex]'"
        )
        sys.exit(1)

    try:
        creds = login(headless=headless)
        email_info = f" ({creds.user_email})" if creds.user_email else ""
        console.print(f"[green]Successfully logged in to Codex{email_info}[/green]")
    except (CodexAuthError, OSError) as e:
        console.print(f"[bold red]Error:[/bold red] Login failed: {e}")
        sys.exit(1)


def _handle_status(provider: str) -> None:
    """Handle the auth status command."""
    if provider != "codex":
        console.print(
            f"[bold red]Error:[/bold red] Unknown provider '{provider}'. "
            "Supported providers: codex"
        )
        sys.exit(1)

    try:
        from deepagents_codex import get_auth_status
        from deepagents_codex.status import CodexAuthStatus
    except ImportError:
        console.print(
            "[bold yellow]Warning:[/bold yellow] deepagents-codex not installed.\n"
            "Install: pip install 'deepagents-cli[codex]'"
        )
        sys.exit(1)

    info = get_auth_status()
    if info.status == CodexAuthStatus.AUTHENTICATED:
        email = f"  Email: {info.user_email}" if info.user_email else ""
        console.print(f"[green]Codex: Authenticated[/green]{email}")
        if info.expires_at:
            import datetime

            exp = datetime.datetime.fromtimestamp(info.expires_at, tz=datetime.UTC)
            console.print(f"  Token expires: {exp.isoformat()}")
    elif info.status == CodexAuthStatus.EXPIRED:
        console.print("[yellow]Codex: Session expired[/yellow]")
        console.print("  Run: deepagents auth login --provider codex")
    elif info.status == CodexAuthStatus.CORRUPT:
        console.print("[red]Codex: Corrupt credentials[/red]")
        console.print(
            "  Run: deepagents auth logout --provider codex && "
            "deepagents auth login --provider codex"
        )
    else:
        console.print("[dim]Codex: Not authenticated[/dim]")
        console.print("  Run: deepagents auth login --provider codex")


def _handle_logout(provider: str) -> None:
    """Handle the auth logout command."""
    if provider != "codex":
        console.print(
            f"[bold red]Error:[/bold red] Unknown provider '{provider}'. "
            "Supported providers: codex"
        )
        sys.exit(1)

    try:
        from deepagents_codex import logout
    except ImportError:
        console.print(
            "[bold red]Error:[/bold red] Codex provider requires deepagents-codex.\n"
            "Install: pip install 'deepagents-cli[codex]'"
        )
        sys.exit(1)

    if logout():
        console.print("[green]Successfully logged out from Codex[/green]")
    else:
        console.print("[dim]No Codex credentials found[/dim]")


def execute_auth_command(args: argparse.Namespace) -> None:
    """Execute auth subcommands based on parsed arguments.

    Args:
        args: Parsed command line arguments with auth_command attribute.
    """
    if args.auth_command == "login":
        _handle_login(args.provider, headless=getattr(args, "headless", False))
    elif args.auth_command == "status":
        _handle_status(args.provider)
    elif args.auth_command == "logout":
        _handle_logout(args.provider)
    else:
        _show_auth_help()


__all__ = [
    "execute_auth_command",
    "setup_auth_parser",
]
