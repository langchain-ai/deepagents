"""Main entry point and CLI loop for deepagents."""

# ruff: noqa: E402
# Imports placed after warning filters to suppress deprecation warnings

# Suppress deprecation warnings from langchain_core (e.g., Pydantic V1 on Python 3.14+)
import warnings

warnings.filterwarnings("ignore", module="langchain_core._api.deprecation")

import argparse
import asyncio
import contextlib
import functools
import importlib.util
import json
import logging
import os
import shutil
import sys
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from deepagents_cli.app import AppResult

# Suppress Pydantic v1 compatibility warnings from langchain on Python 3.14+
warnings.filterwarnings("ignore", message=".*Pydantic V1.*", category=UserWarning)

from deepagents_cli._version import __version__

logger = logging.getLogger(__name__)

# Duplicated from agent.DEFAULT_AGENT_NAME to avoid importing the heavy agent
# module at startup. Keep in sync with agent.py. Tested.
_DEFAULT_AGENT_NAME = "agent"


def check_cli_dependencies() -> None:
    """Check if CLI optional dependencies are installed."""
    missing = []

    if importlib.util.find_spec("requests") is None:
        missing.append("requests")

    if importlib.util.find_spec("dotenv") is None:
        missing.append("python-dotenv")

    if importlib.util.find_spec("tavily") is None:
        missing.append("tavily-python")

    if importlib.util.find_spec("textual") is None:
        missing.append("textual")

    if missing:
        print("\nMissing required CLI dependencies!")  # noqa: T201  # CLI output for missing dependencies
        print("\nThe following packages are required to use the deepagents CLI:")  # noqa: T201  # CLI output for missing dependencies
        for pkg in missing:
            print(f"  - {pkg}")  # noqa: T201  # CLI output for missing dependencies
        print("\nPlease install them with:")  # noqa: T201  # CLI output for missing dependencies
        print("  pip install deepagents[cli]")  # noqa: T201  # CLI output for missing dependencies
        print("\nOr install all dependencies:")  # noqa: T201  # CLI output for missing dependencies
        print("  pip install 'deepagents[cli]'")  # noqa: T201  # CLI output for missing dependencies
        sys.exit(1)


_RIPGREP_URL = "https://github.com/BurntSushi/ripgrep#installation"

_RIPGREP_SUPPRESS_HINT = (
    "To suppress, add to ~/.deepagents/config.toml:\n"
    "\\[warnings]\n"
    'suppress = \\["ripgrep"]'
)


def check_optional_tools(*, config_path: Path | None = None) -> list[str]:
    """Check for recommended external tools and return missing tool names.

    Skips tools that the user has suppressed via
    `[warnings].suppress` in `config.toml`.

    Args:
        config_path: Path to config file.

            Defaults to `~/.deepagents/config.toml`.

    Returns:
        List of missing tool names (e.g. `["ripgrep"]`).
    """
    from deepagents_cli.model_config import is_warning_suppressed

    missing: list[str] = []
    if shutil.which("rg") is None and not is_warning_suppressed("ripgrep", config_path):
        missing.append("ripgrep")
    return missing


def format_tool_warning_tui(tool: str) -> str:
    """Format a missing-tool warning for the TUI toast.

    Args:
        tool: Name of the missing tool.

    Returns:
        Plain-text warning suitable for `App.notify`.
    """
    if tool == "ripgrep":
        return (
            "ripgrep is not installed; the grep tool will use a slower fallback.\n"
            f"\nInstall: {_RIPGREP_URL}\n\n"
            f"{_RIPGREP_SUPPRESS_HINT}"
        )
    return f"{tool} is not installed."


def format_tool_warning_cli(tool: str) -> str:
    """Format a missing-tool warning for non-interactive Rich console output.

    Args:
        tool: Name of the missing tool.

    Returns:
        Rich-markup string suitable for `console.print`.
    """
    if tool == "ripgrep":
        return (
            "ripgrep is not installed; the grep tool will use a slower fallback.\n"
            f"Install: [link={_RIPGREP_URL}]{_RIPGREP_URL}[/link]\n\n"
            f"{_RIPGREP_SUPPRESS_HINT}\n"
        )
    return f"{tool} is not installed."


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    from deepagents_cli.skills import setup_skills_parser
    from deepagents_cli.ui import (
        build_help_parent,
        show_help,
        show_list_help,
        show_reset_help,
        show_threads_delete_help,
        show_threads_help,
        show_threads_list_help,
    )

    # Factory that builds an argparse Action whose __call__ invokes the
    # supplied *help_fn* instead of argparse's default help text.  Each
    # subcommand can pass its own Rich-formatted help screen so that
    # `deepagents <subcommand> -h` shows context-specific help.
    def _make_help_action(
        help_fn: Callable[[], None],
    ) -> type[argparse.Action]:
        """Create an argparse Action that displays *help_fn* and exits.

        argparse requires a *class* (not a callable) for custom actions.
        This factory uses a closure: the returned `_ShowHelp` class captures
        *help_fn* from the enclosing scope so that each subcommand can wire `-h`
        to its own Rich help screen.

        Args:
            help_fn: Callable that prints help text to the console.

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
                namespace: argparse.Namespace,  # noqa: ARG002  # Required by argparse Action interface
                values: str | Sequence[Any] | None,  # noqa: ARG002  # Required by argparse Action interface
                option_string: str | None = None,  # noqa: ARG002  # Required by argparse Action interface
            ) -> None:
                with contextlib.suppress(BrokenPipeError):
                    help_fn()
                parser.exit()

        return _ShowHelp

    help_parent = functools.partial(
        build_help_parent, make_help_action=_make_help_action
    )

    parser = argparse.ArgumentParser(
        description=("Deep Agents - AI Coding Assistant"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    subparsers.add_parser(
        "help",
        help="Show help information",
        add_help=False,
        parents=help_parent(show_help),
    )

    subparsers.add_parser(
        "list",
        help="List all available agents",
        add_help=False,
        parents=help_parent(show_list_help),
    )

    reset_parser = subparsers.add_parser(
        "reset",
        help="Reset an agent",
        add_help=False,
        parents=help_parent(show_reset_help),
    )
    reset_parser.add_argument("--agent", required=True, help="Name of agent to reset")
    reset_parser.add_argument(
        "--target", dest="source_agent", help="Copy prompt from another agent"
    )

    setup_skills_parser(subparsers, make_help_action=_make_help_action)

    threads_parser = subparsers.add_parser(
        "threads",
        help="Manage conversation threads",
        add_help=False,
        parents=help_parent(show_threads_help),
    )
    threads_sub = threads_parser.add_subparsers(dest="threads_command")

    threads_list = threads_sub.add_parser(
        "list",
        aliases=["ls"],
        help="List threads",
        add_help=False,
        parents=help_parent(show_threads_list_help),
    )
    threads_list.add_argument(
        "--agent", default=None, help="Filter by agent name (default: show all)"
    )
    threads_list.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of threads to display (default: 20)",
    )
    threads_delete = threads_sub.add_parser(
        "delete",
        help="Delete a thread",
        add_help=False,
        parents=help_parent(show_threads_delete_help),
    )
    threads_delete.add_argument("thread_id", help="Thread ID to delete")

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
        help="Model to use (e.g., claude-sonnet-4-5-20250929, gpt-5.2). "
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
        choices=["none", "modal", "daytona", "runloop", "langsmith"],
        default="none",
        metavar="TYPE",
        help="Remote sandbox for code execution (default: none - local only)",
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
        "--shell-allow-list",
        metavar="LIST",
        help="Comma-separated list of shell commands to auto-approve, "
        "or 'recommended' for safe defaults. "
        "Applies to both -n and interactive modes.",
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
        "-v",
        "--version",
        action="version",
        version=f"deepagents-cli {__version__}\ndeepagents (SDK) {sdk_version}",
    )
    parser.add_argument(
        "-h",
        "--help",
        action=_make_help_action(show_help),
    )

    return parser.parse_args()


async def run_textual_cli_async(
    assistant_id: str,
    *,
    auto_approve: bool = False,
    sandbox_type: str = "none",  # str (not None) to match argparse choices
    sandbox_id: str | None = None,
    sandbox_setup: str | None = None,
    model_name: str | None = None,
    model_params: dict[str, Any] | None = None,
    thread_id: str | None = None,
    is_resumed: bool = False,
    initial_prompt: str | None = None,
) -> "AppResult":
    """Run the Textual CLI interface (async version).

    Args:
        assistant_id: Agent identifier for memory storage
        auto_approve: Whether to auto-approve tool usage
        sandbox_type: Type of sandbox
            ("none", "modal", "runloop", "daytona", "langsmith")
        sandbox_id: Optional existing sandbox ID to reuse
        sandbox_setup: Optional path to setup script to run in the sandbox
            after creation.
        model_name: Optional model name to use
        model_params: Extra kwargs from `--model-params` to pass to the model.

            These override config file values.
        thread_id: Thread ID to use (new or resumed)
        is_resumed: Whether this is a resumed session
        initial_prompt: Optional prompt to auto-submit when session starts

    Returns:
        An `AppResult` with the return code and final thread ID.
    """
    from rich.text import Text

    from deepagents_cli.agent import create_cli_agent
    from deepagents_cli.app import run_textual_app
    from deepagents_cli.config import console, create_model, settings
    from deepagents_cli.model_config import ModelConfigError
    from deepagents_cli.sessions import get_checkpointer
    from deepagents_cli.tools import fetch_url, http_request, web_search

    try:
        result = create_model(model_name, extra_kwargs=model_params)
    except ModelConfigError as e:
        from deepagents_cli.app import AppResult

        console.print(f"[bold red]Error:[/bold red] {e}")
        return AppResult(return_code=1, thread_id=None)

    model = result.model
    result.apply_to_settings()

    # Show thread info
    if is_resumed:
        msg = Text("Resuming thread: ", style="dim")
        msg.append(str(thread_id), style="dim")
        console.print(msg)
    else:
        msg = Text("Starting with thread: ", style="dim")
        msg.append(str(thread_id), style="dim")
        console.print(msg)

    # Use async context manager for checkpointer
    async with get_checkpointer() as checkpointer:
        # Create agent with conditional tools
        tools: list[Callable[..., Any] | dict[str, Any]] = [http_request, fetch_url]
        if settings.has_tavily:
            tools.append(web_search)

        # Handle sandbox mode
        sandbox_backend = None
        sandbox_cm = None

        if sandbox_type != "none":
            # Deferred: sandbox_factory imports provider-specific SDKs,
            # only needed when a sandbox is actually requested.
            from deepagents_cli.integrations.sandbox_factory import (
                create_sandbox,
            )

            try:
                # Create sandbox context manager but keep it open
                sandbox_cm = create_sandbox(
                    sandbox_type,
                    sandbox_id=sandbox_id,
                    setup_script_path=sandbox_setup,
                )
                sandbox_backend = sandbox_cm.__enter__()  # noqa: PLC2801  # Context manager used without `with` for long-lived sandbox lifecycle
            except (ImportError, ValueError, RuntimeError, NotImplementedError) as e:
                console.print()
                console.print("[red]Sandbox creation failed[/red]")
                console.print(Text(str(e), style="dim"))
                sys.exit(1)

        try:
            agent, composite_backend = create_cli_agent(
                model=model,
                assistant_id=assistant_id,
                tools=tools,
                sandbox=sandbox_backend,
                sandbox_type=sandbox_type if sandbox_type != "none" else None,
                auto_approve=auto_approve,
                checkpointer=checkpointer,
            )
        except Exception as e:  # noqa: BLE001  # CLI needs robust error handling to show friendly error messages
            error_text = Text("Failed to create agent: ", style="red")
            error_text.append(str(e))
            console.print(error_text)
            sys.exit(1)

        # Run Textual app - errors propagate to caller
        from deepagents_cli.app import AppResult

        result = AppResult(return_code=1, thread_id=None)
        try:
            result = await run_textual_app(
                agent=agent,
                assistant_id=assistant_id,
                backend=composite_backend,
                auto_approve=auto_approve,
                cwd=Path.cwd(),
                thread_id=thread_id,
                initial_prompt=initial_prompt,
                checkpointer=checkpointer,
                tools=tools,
                sandbox=sandbox_backend,
                sandbox_type=sandbox_type if sandbox_type != "none" else None,
            )
        finally:
            # Clean up sandbox after app exits (success or error)
            if sandbox_cm is not None:
                try:
                    sandbox_cm.__exit__(None, None, None)
                except Exception:
                    logger.warning("Sandbox cleanup failed", exc_info=True)
        return result


def apply_stdin_pipe(args: argparse.Namespace) -> None:
    r"""Read piped stdin and merge it into the parsed CLI arguments.

    When stdin is not a TTY (i.e. input is piped), reads all available text
    and applies it to the argument namespace. If stdin is a TTY or the piped
    input is empty/whitespace-only, the function returns without modifying
    `args`. Leading and trailing whitespace is stripped from piped input.

    - If `non_interactive_message` is already set (`-n`), prepends the
        piped text to it (the CLI still runs non-interactively):

        ```bash
        cat context.txt | deepagents -n "summarize this"
        # non_interactive_message = "{contents of context.txt}\n\nsummarize this"
        ```

    - If `initial_prompt` is already set (`-m`, but not `-n`), prepends
        the piped text to it (the CLI still runs interactively):

        ```bash
        cat error.log | deepagents -m "explain this"
        # initial_prompt = "{contents of error.log}\n\nexplain this"
        ```

    - Otherwise, sets `non_interactive_message` to the piped text, causing
        the CLI to run non-interactively with it as the prompt:

        ```bash
        echo "fix the typo in README.md" | deepagents
        # non_interactive_message = "fix the typo in README.md"
        ```

    Args:
        args: The parsed argument namespace (mutated in place).
    """
    from deepagents_cli.config import console

    if sys.stdin is None:
        return

    try:
        is_tty = sys.stdin.isatty()
    except (ValueError, OSError):
        return

    if is_tty:
        return

    max_stdin_bytes = 10 * 1024 * 1024  # 10 MiB

    try:
        stdin_text = sys.stdin.read(max_stdin_bytes + 1)
    except UnicodeDecodeError:
        msg = "Could not read piped input — ensure the input is valid text"
        console.print(f"[bold red]Error:[/bold red] {msg}")
        sys.exit(1)
    except (OSError, ValueError) as exc:
        msg = f"Failed to read piped input: {exc}"
        console.print(f"[bold red]Error:[/bold red] {msg}")
        sys.exit(1)

    if len(stdin_text) > max_stdin_bytes:
        msg = (
            f"Piped input exceeds {max_stdin_bytes // (1024 * 1024)} MiB limit. "
            "Consider writing the content to a file and referencing it instead."
        )
        console.print(f"[bold red]Error:[/bold red] {msg}")
        sys.exit(1)

    stdin_text = stdin_text.strip()

    if not stdin_text:
        return

    if args.non_interactive_message:
        args.non_interactive_message = f"{stdin_text}\n\n{args.non_interactive_message}"
    elif args.initial_prompt:
        args.initial_prompt = f"{stdin_text}\n\n{args.initial_prompt}"
    else:
        args.non_interactive_message = stdin_text

    # Restore stdin from the real terminal so the interactive Textual app
    # (used by the -m path) can read keyboard/mouse input normally.
    # Textual's driver reads from file descriptor 0 directly (not sys.stdin),
    # so we must replace the underlying fd with /dev/tty using os.dup2.
    try:
        tty_fd = os.open("/dev/tty", os.O_RDONLY)
    except OSError:
        # No controlling terminal (CI, Docker, headless). Non-interactive
        # path still works; interactive -m path will fail later with a
        # clear "not a terminal" error from Textual.
        return

    try:
        os.dup2(tty_fd, 0)
        os.close(tty_fd)
        sys.stdin = open(0, encoding="utf-8", closefd=False)  # noqa: SIM115  # fd 0 requires open() for TTY restoration
    except OSError:
        console.print(
            "[yellow]Warning:[/yellow] TTY restoration failed. "
            "Interactive mode (-m) may not work correctly."
        )
        logger.warning(
            "TTY restoration failed after opening /dev/tty",
            exc_info=True,
        )
        try:
            os.close(tty_fd)
        except OSError:
            logger.warning(
                "Failed to close TTY fd %d during cleanup",
                tty_fd,
                exc_info=True,
            )


def cli_main() -> None:
    """Entry point for console script."""
    # Fix for gRPC fork issue on macOS
    # https://github.com/grpc/grpc/issues/37642
    if sys.platform == "darwin":
        os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"

    # Note: LANGSMITH_PROJECT is already overridden in config.py
    # (before LangChain imports). This ensures agent traces use
    # DEEPAGENTS_LANGSMITH_PROJECT while shell commands use the
    # user's original LANGSMITH_PROJECT (via LocalShellBackend env).

    # Fast path: print version without loading heavy dependencies
    if len(sys.argv) == 2 and sys.argv[1] in {"-v", "--version"}:  # noqa: PLR2004  # argv length check for fast-path
        try:
            from importlib.metadata import (
                PackageNotFoundError,
                version as _pkg_version,
            )

            sdk_version = _pkg_version("deepagents")
        except PackageNotFoundError:
            sdk_version = "unknown"
        except Exception:  # Best-effort SDK version lookup
            logger.debug("Unexpected error looking up SDK version", exc_info=True)
            sdk_version = "unknown"
        print(f"deepagents-cli {__version__}\ndeepagents (SDK) {sdk_version}")  # noqa: T201  # CLI version output
        sys.exit(0)

    # Check dependencies first
    check_cli_dependencies()

    from deepagents_cli.config import console, settings

    try:
        args = parse_args()

        # Apply shell-allow-list from command line if provided (overrides env var)
        if args.shell_allow_list:
            from deepagents_cli.config import parse_shell_allow_list

            settings.shell_allow_list = parse_shell_allow_list(args.shell_allow_list)

        model_params: dict[str, Any] | None = None
        raw_kwargs = getattr(args, "model_params", None)
        if raw_kwargs:
            try:
                model_params = json.loads(raw_kwargs)
            except json.JSONDecodeError as e:
                console.print(
                    f"[bold red]Error:[/bold red] --model-params is not valid JSON: {e}"
                )
                sys.exit(1)
            if not isinstance(model_params, dict):
                console.print(
                    "[bold red]Error:[/bold red] --model-params must be a JSON object"
                )
                sys.exit(1)

        apply_stdin_pipe(args)

        if (args.quiet or args.no_stream) and not args.non_interactive_message:
            # Print to stderr (not the module-level stdout console) and exit
            # with code 2 to match the POSIX convention for usage errors, as
            # argparse's parser.error() would.
            from rich.console import Console as _Console

            flags = []
            if args.quiet:
                flags.append("--quiet")
            if args.no_stream:
                flags.append("--no-stream")
            flag = " and ".join(flags)
            _Console(stderr=True).print(
                f"[bold red]Error:[/bold red] {flag} requires "
                "--non-interactive (-n) or piped stdin"
            )
            sys.exit(2)

        # Handle --default-model / --clear-default-model (headless, no session)
        if args.clear_default_model:
            from deepagents_cli.model_config import clear_default_model

            if clear_default_model():
                console.print("Default model cleared.")
            else:
                console.print(
                    "[bold red]Error:[/bold red] Could not clear default model. "
                    "Check permissions for ~/.deepagents/"
                )
                sys.exit(1)
            sys.exit(0)

        if args.default_model is not None:
            from deepagents_cli.model_config import (
                ModelConfig,
                save_default_model,
            )

            if args.default_model == "__SHOW__":
                config = ModelConfig.load()
                if config.default_model:
                    console.print(f"Default model: {config.default_model}")
                else:
                    console.print("No default model set.")
                sys.exit(0)

            model_spec = args.default_model
            # Auto-detect provider for bare model names
            from deepagents_cli.config import detect_provider
            from deepagents_cli.model_config import ModelSpec

            parsed = ModelSpec.try_parse(model_spec)
            if not parsed:
                provider = detect_provider(model_spec)
                if provider:
                    model_spec = f"{provider}:{model_spec}"

            if save_default_model(model_spec):
                console.print(f"Default model set to {model_spec}")
            else:
                console.print(
                    "[bold red]Error:[/bold red] Could not save default model. "
                    "Check permissions for ~/.deepagents/"
                )
                sys.exit(1)
            sys.exit(0)

        if args.command == "help":
            from deepagents_cli.ui import show_help

            show_help()
        elif args.command == "list":
            from deepagents_cli.agent import list_agents

            list_agents()
        elif args.command == "reset":
            from deepagents_cli.agent import reset_agent

            reset_agent(args.agent, args.source_agent)
        elif args.command == "skills":
            from deepagents_cli.skills import execute_skills_command

            execute_skills_command(args)
        elif args.command == "threads":
            from deepagents_cli.sessions import (
                delete_thread_command,
                list_threads_command,
            )
            from deepagents_cli.ui import show_threads_help

            # "ls" is an argparse alias for "list" — argparse stores the
            # alias as-is in the namespace, so we must match both values.
            if args.threads_command in {"list", "ls"}:
                asyncio.run(
                    list_threads_command(
                        agent_name=getattr(args, "agent", None),
                        limit=getattr(args, "limit", None),
                    )
                )
            elif args.threads_command == "delete":
                asyncio.run(delete_thread_command(args.thread_id))
            else:
                # No subcommand provided, show threads help screen
                show_threads_help()
        elif args.non_interactive_message:
            # Check for optional tools before running agent (stderr so
            # --quiet piped output stays clean)
            try:
                from rich.console import Console as _Console
            except ImportError:
                logger.warning(
                    "Could not import rich.console; skipping tool warnings",
                    exc_info=True,
                )
            else:
                try:
                    warn_console = _Console(stderr=True)
                    for tool in check_optional_tools():
                        warn_console.print(
                            f"[yellow]Warning:[/yellow] {format_tool_warning_cli(tool)}"
                        )
                except Exception:
                    logger.debug("Failed to check for optional tools", exc_info=True)
            # Non-interactive mode - execute single task and exit
            from deepagents_cli.non_interactive import run_non_interactive

            exit_code = asyncio.run(
                run_non_interactive(
                    message=args.non_interactive_message,
                    assistant_id=args.agent,
                    model_name=getattr(args, "model", None),
                    model_params=model_params,
                    sandbox_type=args.sandbox,
                    sandbox_id=args.sandbox_id,
                    sandbox_setup=getattr(args, "sandbox_setup", None),
                    quiet=args.quiet,
                    stream=not args.no_stream,
                )
            )
            sys.exit(exit_code)
        else:
            # Interactive mode - handle thread resume
            from rich.style import Style
            from rich.text import Text

            from deepagents_cli.config import (
                build_langsmith_thread_url,
            )
            from deepagents_cli.sessions import (
                find_similar_threads,
                generate_thread_id,
                get_most_recent,
                get_thread_agent,
                thread_exists,
            )

            thread_id = None
            is_resumed = False

            if args.resume_thread == "__MOST_RECENT__":
                # -r (no ID): Get most recent thread
                # If --agent specified, filter by that agent; otherwise get
                # most recent overall
                agent_filter = args.agent if args.agent != _DEFAULT_AGENT_NAME else None
                thread_id = asyncio.run(get_most_recent(agent_filter))
                if thread_id:
                    is_resumed = True
                    agent_name = asyncio.run(get_thread_agent(thread_id))
                    if agent_name:
                        args.agent = agent_name
                else:
                    if agent_filter:
                        msg = Text("No previous thread for '", style="yellow")
                        msg.append(args.agent)
                        msg.append("', starting new.", style="yellow")
                    else:
                        msg = Text("No previous threads, starting new.", style="yellow")
                    console.print(msg)

            elif args.resume_thread:
                # -r <ID>: Resume specific thread
                if asyncio.run(thread_exists(args.resume_thread)):
                    thread_id = args.resume_thread
                    is_resumed = True
                    if args.agent == _DEFAULT_AGENT_NAME:
                        agent_name = asyncio.run(get_thread_agent(thread_id))
                        if agent_name:
                            args.agent = agent_name
                else:
                    error_msg = Text("Thread '", style="red")
                    error_msg.append(args.resume_thread)
                    error_msg.append("' not found.", style="red")
                    console.print(error_msg)

                    # Check for similar thread IDs
                    similar = asyncio.run(find_similar_threads(args.resume_thread))
                    if similar:
                        console.print()
                        console.print("[yellow]Did you mean?[/yellow]")
                        for tid in similar:
                            hint = Text("  deepagents -r ", style="cyan")
                            hint.append(str(tid), style="cyan")
                            console.print(hint)
                        console.print()

                    console.print(
                        "[dim]Use 'deepagents threads list' to see "
                        "available threads.[/dim]"
                    )
                    console.print(
                        "[dim]Use 'deepagents -r' to resume the most "
                        "recent thread.[/dim]"
                    )
                    sys.exit(1)

            # Generate new thread ID if not resuming
            if thread_id is None:
                thread_id = generate_thread_id()

            # Run Textual CLI
            return_code = 0
            try:
                result = asyncio.run(
                    run_textual_cli_async(
                        assistant_id=args.agent,
                        auto_approve=args.auto_approve,
                        sandbox_type=args.sandbox,
                        sandbox_id=args.sandbox_id,
                        sandbox_setup=getattr(args, "sandbox_setup", None),
                        model_name=getattr(args, "model", None),
                        model_params=model_params,
                        thread_id=thread_id,
                        is_resumed=is_resumed,
                        initial_prompt=getattr(args, "initial_prompt", None),
                    )
                )
                return_code = result.return_code
                # The user may have switched threads via /threads during the
                # session; use the final thread ID for teardown messages.
                thread_id = result.thread_id or thread_id
            except Exception as e:  # noqa: BLE001  # Top-level error handler for the application
                error_msg = Text("\nApplication error: ", style="red")
                error_msg.append(str(e))
                console.print(error_msg)
                console.print(Text(traceback.format_exc(), style="dim"))
                sys.exit(1)

            # Show LangSmith thread link for threads with checkpointed
            # content (same table that backs the `/threads` listing).
            try:
                thread_url = build_langsmith_thread_url(thread_id)
                if thread_url and asyncio.run(thread_exists(thread_id)):
                    console.print()
                    ls_hint = Text("View this thread in LangSmith: ", style="dim")
                    ls_hint.append(
                        thread_url,
                        style=Style(dim=True, link=thread_url),
                    )
                    console.print(ls_hint)
            except Exception:
                logger.debug(
                    "Could not display LangSmith thread URL on teardown",
                    exc_info=True,
                )

            # Show resume hint on exit for threads with checkpointed content.
            if thread_id and return_code == 0 and asyncio.run(thread_exists(thread_id)):
                console.print()
                console.print("[dim]Resume this thread with:[/dim]")
                hint = Text("deepagents -r ", style="cyan")
                hint.append(str(thread_id), style="cyan")
                console.print(hint)
    except KeyboardInterrupt:
        # Clean exit on Ctrl+C - suppress ugly traceback
        console.print("\n\n[yellow]Interrupted[/yellow]")
        sys.exit(0)


if __name__ == "__main__":
    cli_main()
