"""Entry point for the `deepagents` CLI.

As of `deepagents-cli==0.1.0` the interactive Textual REPL has moved to the
[`deepagents-code`](https://pypi.org/project/deepagents-code/) package. This
CLI now exposes only the deployment-oriented commands: `init`, `dev`, and
`deploy`. Bare invocations print a deprecation notice and exit non-zero.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from typing import TYPE_CHECKING, Any

from deepagents_cli._version import __version__

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


_REPL_REDIRECT_MESSAGE = (
    "The interactive `deepagents` REPL has moved to the `deepagents-code` "
    "package.\n"
    "Install it with:\n\n"
    "  pip install deepagents-code\n\n"
    "Then run `deepagents-code` to start an interactive session.\n\n"
    "The `deepagents` CLI now only provides the `init`, `dev`, and `deploy` "
    "subcommands."
)


def _make_help_action(
    help_fn: Callable[[], None],
) -> type[argparse.Action]:
    """Create an argparse Action that calls `help_fn` and exits.

    argparse requires a *class* (not a callable) for custom actions; this
    factory uses a closure so each subcommand can wire `-h` to its own
    `print_help()`.

    Args:
        help_fn: Callable that prints help text to stdout.

    Returns:
        An argparse Action class wired to the given help function.
    """

    class _ShowHelp(argparse.Action):
        def __init__(
            self,
            option_strings: list[str],
            dest: str = argparse.SUPPRESS,
            default: str = argparse.SUPPRESS,
            **kwargs: Any,
        ) -> None:
            super().__init__(
                option_strings=option_strings,
                dest=dest,
                default=default,
                nargs=0,
                **kwargs,
            )

        def __call__(
            self,
            parser: argparse.ArgumentParser,
            namespace: argparse.Namespace,  # noqa: ARG002
            values: str | Sequence[Any] | None,  # noqa: ARG002
            option_string: str | None = None,  # noqa: ARG002
        ) -> None:
            with contextlib.suppress(BrokenPipeError):
                help_fn()
            parser.exit()

    return _ShowHelp


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argparse parser for the deploy/dev/init CLI.

    Returns:
        Configured `ArgumentParser` with `init`, `dev`, and `deploy`
        subparsers registered.
    """
    from deepagents_cli.deploy import setup_deploy_parsers

    parser = argparse.ArgumentParser(
        prog="deepagents",
        description=(
            "Deep Agents - deployment tooling.\n\n"
            "For interactive chat, install `deepagents-code` instead."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    subparsers.add_parser(
        "help",
        help="Show help information",
        add_help=False,
        parents=help_parent(_lazy_help("show_help")),
    )

    agents_parser = subparsers.add_parser(
        "agents",
        help="Manage agents",
        add_help=False,
        parents=help_parent(_lazy_help("show_agents_help")),
    )
    add_json_output_arg(agents_parser)
    agents_sub = agents_parser.add_subparsers(dest="agents_command")

    agents_list = agents_sub.add_parser(
        "list",
        aliases=["ls"],
        help="List all agents",
        add_help=False,
        parents=help_parent(_lazy_help("show_list_help")),
    )
    add_json_output_arg(agents_list)

    agents_reset = agents_sub.add_parser(
        "reset",
        help="Reset an agent's prompt to default",
        add_help=False,
        parents=help_parent(_lazy_help("show_reset_help")),
    )
    add_json_output_arg(agents_reset)
    agents_reset.add_argument("--agent", required=True, help="Name of agent to reset")
    agents_reset.add_argument(
        "--target", dest="source_agent", help="Copy prompt from another agent"
    )
    agents_reset.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without making changes",
    )

    setup_skills_parser(
        subparsers,
        make_help_action=_make_help_action,
        add_output_args=add_json_output_arg,
    )

    threads_parser = subparsers.add_parser(
        "threads",
        help="Manage conversation threads",
        add_help=False,
        parents=help_parent(_lazy_help("show_threads_help")),
    )
    add_json_output_arg(threads_parser)
    threads_sub = threads_parser.add_subparsers(dest="threads_command")

    threads_list = threads_sub.add_parser(
        "list",
        aliases=["ls"],
        help="List threads",
        add_help=False,
        parents=help_parent(_lazy_help("show_threads_list_help")),
    )
    add_json_output_arg(threads_list)
    threads_list.add_argument(
        "--agent", default=None, help="Filter by agent name (default: show all)"
    )
    threads_list.add_argument(
        "-n",
        "--limit",
        type=int,
        default=None,
        help="Max number of threads to display (default: 20)",
    )
    threads_list.add_argument(
        "--sort",
        choices=["created", "updated"],
        default=None,
        help="Sort threads by timestamp (default: from config, or updated)",
    )
    threads_list.add_argument(
        "--branch",
        default=None,
        help="Filter by git branch name",
    )
    threads_list.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Show all columns (branch, created, prompt)",
    )
    threads_list.add_argument(
        "-r",
        "--relative",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show timestamps as relative time (default: from config, or absolute)",
    )
    threads_delete = threads_sub.add_parser(
        "delete",
        help="Delete a thread",
        add_help=False,
        parents=help_parent(_lazy_help("show_threads_delete_help")),
    )
    add_json_output_arg(threads_delete)
    threads_delete.add_argument("thread_id", help="Thread ID to delete")
    threads_delete.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without making changes",
    )

    update_parser = subparsers.add_parser(
        "update",
        help="Check for and install CLI updates",
        add_help=False,
        parents=help_parent(_lazy_help("show_update_help")),
    )
    add_json_output_arg(update_parser)

    # Default interactive mode — argument order here determines the
    # usage line printed by argparse; keep in sync with ui.show_help().
    parser.add_argument(
        "-r",
        "--resume",
        dest="resume_thread",
        nargs="?",
        const="__MOST_RECENT__",
        default=None,
        metavar="ID",
        help="Resume thread: -r for most recent, -r <ID> for specific thread",
    )

    parser.add_argument(
        "-a",
        "--agent",
        default=_DEFAULT_AGENT_NAME,
        metavar="NAME",
        help="Agent to use (e.g., coder, researcher).",
    )

    parser.add_argument(
        "-M",
        "--model",
        metavar="MODEL",
        help="Model to use (e.g., claude-sonnet-4-6, gpt-5.2). "
        "Provider is auto-detected from model name.",
    )

    parser.add_argument(
        "--model-params",
        metavar="JSON",
        help="Extra kwargs to pass to the model as a JSON string "
        '(e.g., \'{"temperature": 0.7, "max_tokens": 4096}\'). '
        "These take priority, overriding config file values.",
    )

    parser.add_argument(
        "--profile-override",
        metavar="JSON",
        help="Override model profile fields as a JSON string "
        "(e.g., '{\"max_input_tokens\": 4096}'). "
        "Merged on top of config file profile overrides.",
    )

    parser.add_argument(
        "--default-model",
        metavar="MODEL",
        nargs="?",
        const="__SHOW__",
        default=None,
        help="Set the default model for future launches "
        "(e.g., anthropic:claude-opus-4-6). "
        "Use --default-model with no argument to show the current default. "
        "Use --clear-default-model to remove it.",
    )

    parser.add_argument(
        "--clear-default-model",
        action="store_true",
        help="Clear the default model, falling back to recent model "
        "or environment auto-detection.",
    )

    parser.add_argument(
        "-m",
        "--message",
        dest="initial_prompt",
        metavar="TEXT",
        help="Initial prompt to auto-submit when session starts",
    )

    parser.add_argument(
        "-n",
        "--non-interactive",
        dest="non_interactive_message",
        metavar="TEXT",
        help="Run a single task non-interactively and exit "
        "(shell disabled unless --shell-allow-list is set)",
    )

    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Clean output for piping — only the agent's response "
        "goes to stdout. Requires -n or piped stdin.",
    )

    parser.add_argument(
        "--no-stream",
        dest="no_stream",
        action="store_true",
        help="Buffer the full response and write it to stdout at once "
        "instead of streaming token-by-token. Requires -n or piped stdin.",
    )

    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read input from stdin explicitly (instead of auto-detection)",
    )

    add_json_output_arg(parser, default="text")

    parser.add_argument(
        "-y",
        "--auto-approve",
        action="store_true",
        help=(
            "Auto-approve all tool calls without prompting "
            "(disables human-in-the-loop). Affected tools: shell "
            "execution, file writes/edits, web search, and URL fetch. "
            "Use with caution — the agent can execute arbitrary commands."
        ),
    )

    parser.add_argument(
        "--sandbox",
        choices=["none", "agentcore", "modal", "daytona", "runloop", "langsmith", "tensorlake"],
        default="none",
        metavar="TYPE",
        help=(
            "Remote sandbox for code execution "
            "(default: none - local only; langsmith is included, "
            "agentcore/modal/daytona/runloop/tensorlake require downloading extras)"
        ),
    )

    parser.add_argument(
        "--sandbox-id",
        metavar="ID",
        help="Existing sandbox ID to reuse (skips creation and cleanup)",
    )

    parser.add_argument(
        "--sandbox-setup",
        metavar="PATH",
        help="Path to setup script to run in sandbox after creation",
    )
    parser.add_argument(
        "-S",
        "--shell-allow-list",
        metavar="LIST",
        help="Comma-separated list of shell commands to auto-approve, "
        "'recommended' for safe defaults, or 'all' to allow any command. "
        "Applies to both -n and interactive modes.",
    )
    parser.add_argument(
        "--mcp-config",
        help="Path to MCP servers JSON configuration file (Claude Desktop format). "
        "Merged on top of auto-discovered configs (highest precedence).",
    )
    parser.add_argument(
        "--no-mcp",
        action="store_true",
        help="Disable all MCP tool loading (skip auto-discovery and explicit config)",
    )
    parser.add_argument(
        "--trust-project-mcp",
        action="store_true",
        help="Trust project-level MCP configs with stdio servers "
        "(skip interactive approval prompt)",
    )

    try:
        from importlib.metadata import (
            PackageNotFoundError,
            version as _pkg_version,
        )

        sdk_version = _pkg_version("deepagents")
    except PackageNotFoundError:
        logger.debug("deepagents SDK package not found in environment")
        sdk_version = "unknown"
    except Exception:
        logger.warning("Unexpected error looking up SDK version", exc_info=True)
        sdk_version = "unknown"
    parser.add_argument(
        "--update",
        action="store_true",
        help="Check for and install updates, then exit",
    )
    parser.add_argument(
        "--acp",
        action="store_true",
        help="Run as an ACP server over stdio instead of launching the Textual UI",
    )

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"deepagents-cli {__version__}\ndeepagents (SDK) {sdk_version}",
    )
    parser.add_argument(
        "-h",
        "--help",
        action=_make_help_action(parser.print_help),
        help="show this help message and exit",
    )

    setup_deploy_parsers(subparsers, make_help_action=_make_help_action)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments and return the resulting namespace.

    Args:
        argv: Optional argument list (defaults to `sys.argv[1:]`).

    Returns:
        Parsed argparse `Namespace`.
    """
    parser = _build_parser()
    return parser.parse_args(argv)


def cli_main() -> None:
    """Entry point for the `deepagents` and `deepagents-cli` console scripts.

    Raises:
        SystemExit: On `--help`/`--version`, after a subcommand finishes,
            on `KeyboardInterrupt`, when invoked without a subcommand (the
            user is redirected to `deepagents-code`), or when a subcommand
            handler raises `SystemExit` directly (e.g. config validation
            failures in `execute_deploy_command`).
    """
    if sys.platform == "darwin":
        # gRPC (pulled in transitively by LangSmith deps) crashes on macOS
        # when the process forks after gRPC has been initialized. Disable
        # fork support to avoid the abort; the current deploy/dev paths
        # don't fork, so disabling fork support is a safe default —
        # reconsider this env var if a future subcommand spawns workers.
        os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "0")

    args = parse_args()

    try:
        if args.command == "init":
            from deepagents_cli.deploy import execute_init_command

            execute_init_command(args)
        elif args.command == "dev":
            from deepagents_cli.deploy import execute_dev_command

            execute_dev_command(args)
        elif args.command == "deploy":
            from deepagents_cli.deploy import execute_deploy_command

            execute_deploy_command(args)
        else:
            sys.stderr.write(_REPL_REDIRECT_MESSAGE + "\n")
            raise SystemExit(1)
    except KeyboardInterrupt:
        sys.stderr.write("\nInterrupted.\n")
        raise SystemExit(130) from None
