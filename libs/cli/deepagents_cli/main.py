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
import os
import sys
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

# Suppress Pydantic v1 compatibility warnings from langchain on Python 3.14+
warnings.filterwarnings("ignore", message=".*Pydantic V1.*", category=UserWarning)

from rich.text import Text

from deepagents_cli._version import __version__

# Now safe to import agent (which imports LangChain modules)
from deepagents_cli.agent import (
    DEFAULT_AGENT_NAME,
    create_cli_agent,
    list_agents,
    reset_agent,
)

# CRITICAL: Import config FIRST to set LANGSMITH_PROJECT before LangChain loads
from deepagents_cli.config import (
    console,
    create_model,
    settings,
)
from deepagents_cli.integrations.sandbox_factory import create_sandbox
from deepagents_cli.model_config import ModelConfigError
from deepagents_cli.sessions import (
    delete_thread_command,
    find_similar_threads,
    generate_thread_id,
    get_checkpointer,
    get_most_recent,
    get_thread_agent,
    list_threads_command,
    thread_exists,
)
from deepagents_cli.skills import execute_skills_command, setup_skills_parser
from deepagents_cli.tools import fetch_url, http_request, web_search
from deepagents_cli.ui import (
    build_help_parent,
    show_help,
    show_list_help,
    show_reset_help,
    show_threads_delete_help,
    show_threads_help,
    show_threads_list_help,
)


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
        print("\n❌ Missing required CLI dependencies!")
        print("\nThe following packages are required to use the deepagents CLI:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nPlease install them with:")
        print("  pip install deepagents[cli]")
        print("\nOr install all dependencies:")
        print("  pip install 'deepagents[cli]'")
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """

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
                namespace: argparse.Namespace,  # noqa: ARG002
                values: str | Sequence[Any] | None,  # noqa: ARG002
                option_string: str | None = None,  # noqa: ARG002
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
        default=20,
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
        default=DEFAULT_AGENT_NAME,
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
        "goes to stdout. Requires -n.",
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

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"deepagents-cli {__version__}",
    )
    parser.add_argument(
        "-h",
        "--help",
        action=_make_help_action(show_help),
    )

    args = parser.parse_args()

    if args.quiet and not args.non_interactive_message:
        parser.error("--quiet requires --non-interactive (-n)")

    return args


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
) -> int:
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
        The app's return code (0 for success, non-zero for error).
    """
    from deepagents_cli.app import run_textual_app

    try:
        result = create_model(model_name, extra_kwargs=model_params)
    except ModelConfigError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        return 1

    model = result.model
    result.apply_to_settings()

    # Show thread info
    if is_resumed:
        msg = Text("Resuming thread: ", style="green")
        msg.append(str(thread_id))
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
            try:
                # Create sandbox context manager but keep it open
                sandbox_cm = create_sandbox(
                    sandbox_type,
                    sandbox_id=sandbox_id,
                    setup_script_path=sandbox_setup,
                )
                sandbox_backend = sandbox_cm.__enter__()  # noqa: PLC2801
            except (ImportError, ValueError, RuntimeError, NotImplementedError) as e:
                console.print()
                console.print("[red]❌ Sandbox creation failed[/red]")
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
        except Exception as e:
            error_text = Text("❌ Failed to create agent: ", style="red")
            error_text.append(str(e))
            console.print(error_text)
            sys.exit(1)

        # Run Textual app - errors propagate to caller
        return_code = 0
        try:
            return_code = await run_textual_app(
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
                with contextlib.suppress(Exception):
                    sandbox_cm.__exit__(None, None, None)
        return return_code


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

    # Check dependencies first
    check_cli_dependencies()

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
            show_help()
        elif args.command == "list":
            list_agents()
        elif args.command == "reset":
            reset_agent(args.agent, args.source_agent)
        elif args.command == "skills":
            execute_skills_command(args)
        elif args.command == "threads":
            # "ls" is an argparse alias for "list" — argparse stores the
            # alias as-is in the namespace, so we must match both values.
            if args.threads_command in {"list", "ls"}:
                asyncio.run(
                    list_threads_command(
                        agent_name=getattr(args, "agent", None),
                        limit=getattr(args, "limit", 20),
                    )
                )
            elif args.threads_command == "delete":
                asyncio.run(delete_thread_command(args.thread_id))
            else:
                # No subcommand provided, show threads help screen
                show_threads_help()
        elif args.non_interactive_message:
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
                )
            )
            sys.exit(exit_code)
        else:
            # Interactive mode - handle thread resume
            thread_id = None
            is_resumed = False

            if args.resume_thread == "__MOST_RECENT__":
                # -r (no ID): Get most recent thread
                # If --agent specified, filter by that agent; otherwise get
                # most recent overall
                agent_filter = args.agent if args.agent != DEFAULT_AGENT_NAME else None
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
                    if args.agent == DEFAULT_AGENT_NAME:
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
                return_code = asyncio.run(
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
            except Exception as e:
                error_msg = Text("\nApplication error: ", style="red")
                error_msg.append(str(e))
                console.print(error_msg)
                console.print(Text(traceback.format_exc(), style="dim"))
                sys.exit(1)

            # Show resume hint on exit (only for new threads with successful exit)
            if thread_id and not is_resumed and return_code == 0:
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
