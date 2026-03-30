"""Main entry point and CLI loop for deepagents."""

# ruff: noqa: E402
# Imports placed after warning filters to suppress deprecation warnings

# Suppress deprecation warnings from langchain_core (e.g., Pydantic V1 on Python 3.14+)
import warnings

warnings.filterwarnings("ignore", module="langchain_core._api.deprecation")

import argparse
import asyncio
import contextlib
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
    from deepagents_cli.mcp_tools import MCPServerInfo

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


def _ripgrep_install_hint() -> str:
    """Return a platform-specific install command for ripgrep.

    Falls back to the GitHub URL when the platform isn't recognized.
    """
    plat = sys.platform
    if plat == "darwin":
        if shutil.which("brew"):
            return "brew install ripgrep"
        if shutil.which("port"):
            return "sudo port install ripgrep"
    elif plat == "linux":
        if shutil.which("apt-get"):
            return "sudo apt-get install ripgrep"
        if shutil.which("dnf"):
            return "sudo dnf install ripgrep"
        if shutil.which("pacman"):
            return "sudo pacman -S ripgrep"
        if shutil.which("zypper"):
            return "sudo zypper install ripgrep"
        if shutil.which("apk"):
            return "sudo apk add ripgrep"
        if shutil.which("nix-env"):
            return "nix-env -iA nixpkgs.ripgrep"
    elif plat == "win32":
        if shutil.which("choco"):
            return "choco install ripgrep"
        if shutil.which("scoop"):
            return "scoop install ripgrep"
        if shutil.which("winget"):
            return "winget install BurntSushi.ripgrep"
    # Cross-platform fallbacks
    if shutil.which("cargo"):
        return "cargo install ripgrep"
    if shutil.which("conda"):
        return "conda install -c conda-forge ripgrep"
    return _RIPGREP_URL


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
        hint = _ripgrep_install_hint()
        return (
            "ripgrep is not installed; the grep tool will use a slower fallback.\n"
            f"\nInstall: {hint}\n\n"
            f"{_RIPGREP_SUPPRESS_HINT}"
        )
    return f"{tool} is not installed."


def format_tool_warning_cli(tool: str) -> str:
    """Format a missing-tool warning for non-interactive console output.

    Args:
        tool: Name of the missing tool.

    Returns:
        Warning string suitable for `console.print`.
    """
    if tool == "ripgrep":
        hint = _ripgrep_install_hint()
        if hint.startswith("http"):
            hint = f"[link={hint}]{hint}[/link]"
        return (
            "ripgrep is not installed; the grep tool will use a slower fallback.\n"
            f"Install: {hint}\n\n"
            f"{_RIPGREP_SUPPRESS_HINT}\n"
        )
    return f"{tool} is not installed."


async def _preload_session_mcp_server_info(
    *,
    mcp_config_path: str | None,
    no_mcp: bool,
    trust_project_mcp: bool | None,
) -> list["MCPServerInfo"] | None:
    """Load MCP metadata for the interactive TUI in server mode.

    In server mode the actual MCP tools are created inside the LangGraph server
    process, but the local Textual app still needs MCP metadata for the welcome
    banner and `/mcp` viewer. This preloads the metadata in the CLI process and
    immediately cleans up any temporary MCP sessions it opened.

    Args:
        mcp_config_path: Optional explicit MCP config path.
        no_mcp: Whether MCP loading is disabled.
        trust_project_mcp: Project-level MCP trust decision.

    Returns:
        MCP server metadata for the TUI, or `None` when MCP is disabled.
    """
    if no_mcp:
        return None

    from deepagents_cli.mcp_tools import resolve_and_load_mcp_tools
    from deepagents_cli.project_utils import ProjectContext

    session_manager = None
    try:
        try:
            project_context = ProjectContext.from_user_cwd(Path.cwd())
        except OSError:
            logger.warning("Could not determine working directory for MCP preload")
            project_context = None
        _tools, session_manager, server_info = await resolve_and_load_mcp_tools(
            explicit_config_path=mcp_config_path,
            no_mcp=no_mcp,
            trust_project_mcp=trust_project_mcp,
            project_context=project_context,
        )
        return server_info
    finally:
        if session_manager is not None:
            try:
                await session_manager.cleanup()
            except Exception:
                logger.warning(
                    "MCP metadata preload cleanup failed",
                    exc_info=True,
                )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    from deepagents_cli.output import add_json_output_arg
    from deepagents_cli.skills import setup_skills_parser

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

    # Lazy wrapper: defers `ui` import until the help action fires (i.e.,
    # only when the user passes `-h`). This avoids pulling in Rich and config at
    # parse time for the common non-help path.
    def _lazy_help(fn_name: str) -> Callable[[], None]:
        def _show() -> None:
            from deepagents_cli import ui

            getattr(ui, fn_name)()

        return _show

    def help_parent(help_fn: Callable[[], None]) -> list[argparse.ArgumentParser]:
        parent = argparse.ArgumentParser(add_help=False)
        parent.add_argument("-h", "--help", action=_make_help_action(help_fn))
        return [parent]

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
        choices=["none", "agentcore", "modal", "daytona", "runloop", "langsmith"],
        default="none",
        metavar="TYPE",
        help=(
            "Remote sandbox for code execution "
            "(default: none - local only; langsmith is included, "
            "agentcore/modal/daytona/runloop require downloading extras)"
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
        action=_make_help_action(_lazy_help("show_help")),
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
    profile_override: dict[str, Any] | None = None,
    thread_id: str | None = None,
    resume_thread: str | None = None,
    initial_prompt: str | None = None,
    mcp_config_path: str | None = None,
    no_mcp: bool = False,
    trust_project_mcp: bool | None = None,
) -> "AppResult":
    """Run the Textual CLI interface (async version).

    Starts a LangGraph server in a subprocess and connects the TUI to it via the
    `langgraph-sdk` client.

    Args:
        assistant_id: Agent identifier for memory storage
        auto_approve: Whether to auto-approve tool usage
        sandbox_type: Type of sandbox
            ("none", "agentcore", "modal", "runloop", "daytona", "langsmith")
        sandbox_id: Optional existing sandbox ID to reuse.
        sandbox_setup: Optional path to setup script to run in the sandbox
            after creation.
        model_name: Optional model name to use
        model_params: Extra kwargs from `--model-params` to pass to the model.

            These override config file values.
        profile_override: Extra profile fields from `--profile-override`.

            Merged on top of config file profile overrides.
        thread_id: Thread ID for the session.

            `None` when `resume_thread` is provided (the TUI resolves the final
            ID asynchronously).
        resume_thread: Raw resume intent from `-r` flag.

            `'__MOST_RECENT__'` for bare `-r`, a thread ID string for `-r <id>`,
            or `None` for new sessions.

            Resolved asynchronously inside the TUI.
        initial_prompt: Optional prompt to auto-submit when session starts
        mcp_config_path: Optional path to MCP servers JSON configuration file.

            Merged on top of auto-discovered configs (highest precedence).
        no_mcp: Disable all MCP tool loading.
        trust_project_mcp: Controls project-level stdio server trust.

            `True` to allow, `False` to deny, `None` to check trust store.

    Returns:
        An `AppResult` with the return code and final thread ID.
    """
    from rich.text import Text

    from deepagents_cli.app import AppResult, run_textual_app
    from deepagents_cli.config import (
        _get_default_model_spec,
        detect_provider,
        settings,
    )
    from deepagents_cli.model_config import ModelConfigError, ModelSpec

    # Resolve display-name cheaply (<1ms, no langchain) so the status
    # bar can show the model on first paint. The expensive create_model()
    # (~560ms) is deferred to a background worker.

    try:
        resolved_spec = model_name or _get_default_model_spec()
    except ModelConfigError as e:
        from rich.markup import escape

        from deepagents_cli.config import console

        console.print(f"[bold red]Error:[/bold red] {escape(str(e))}", highlight=False)
        return AppResult(return_code=1, thread_id=None)

    parsed = ModelSpec.try_parse(resolved_spec)
    if parsed:
        settings.model_provider = parsed.provider
        settings.model_name = parsed.model
    else:
        settings.model_name = resolved_spec
        settings.model_provider = detect_provider(resolved_spec) or ""

    model_kwargs: dict[str, Any] = {
        "model_spec": model_name,
        "extra_kwargs": model_params,
        "profile_overrides": profile_override,
    }

    # Build kwargs for deferred server startup (runs inside the TUI).
    # Never pass auto_approve to the server — the interactive server must
    # always configure full HITL interrupts so that Shift+Tab can toggle
    # approval mode mid-session. The -y flag is handled client-side via
    # session_state.auto_approve in textual_adapter.py.
    server_kwargs: dict[str, Any] = {
        "assistant_id": assistant_id,
        "model_name": model_name,
        "model_params": model_params,
        "sandbox_type": sandbox_type,
        "sandbox_id": sandbox_id,
        "sandbox_setup": sandbox_setup,
        "enable_ask_user": True,
        "mcp_config_path": mcp_config_path,
        "no_mcp": no_mcp,
        "trust_project_mcp": trust_project_mcp,
        "interactive": True,
    }

    mcp_preload_kwargs: dict[str, Any] | None = None
    if not no_mcp:
        mcp_preload_kwargs = {
            "mcp_config_path": mcp_config_path,
            "no_mcp": no_mcp,
            "trust_project_mcp": trust_project_mcp,
        }

    try:
        result = await run_textual_app(
            assistant_id=assistant_id,
            backend=None,
            auto_approve=auto_approve,
            cwd=Path.cwd(),
            thread_id=thread_id,
            resume_thread=resume_thread,
            initial_prompt=initial_prompt,
            profile_override=profile_override,
            server_kwargs=server_kwargs,
            mcp_preload_kwargs=mcp_preload_kwargs,
            model_kwargs=model_kwargs,
        )
    except Exception as e:
        logger.debug("App error", exc_info=True)
        from deepagents_cli.config import console

        error_text = Text("Application error: ", style="red")
        error_text.append(str(e))
        console.print(error_text)
        if logger.isEnabledFor(logging.DEBUG):
            console.print(Text(traceback.format_exc(), style="dim"))
        return AppResult(return_code=1, thread_id=None)

    return result


async def _run_acp_cli_async(
    assistant_id: str,
    *,
    run_acp_agent: Callable[[Any], Any],
    agent_server_cls: type[Any],
    model_name: str | None = None,
    model_params: dict[str, Any] | None = None,
    profile_override: dict[str, Any] | None = None,
    mcp_config_path: str | None = None,
    no_mcp: bool = False,
    trust_project_mcp: bool | None = None,
) -> int:
    """Run ACP server mode and return a process exit code.

    Args:
        assistant_id: Agent identifier to initialize.
        run_acp_agent: ACP server runner function.
        agent_server_cls: ACP server class constructor.
        model_name: Optional model name to use.
        model_params: Extra kwargs from `--model-params` to pass to the model.
        profile_override: Extra profile fields from `--profile-override`.
        mcp_config_path: Optional path to MCP servers JSON configuration file.
        no_mcp: Disable all MCP tool loading.
        trust_project_mcp: Controls project-level stdio server trust.

    Returns:
        Exit code for ACP mode.
    """
    from deepagents_cli.agent import create_cli_agent, load_async_subagents
    from deepagents_cli.config import create_model, settings
    from deepagents_cli.model_config import ModelConfigError, save_recent_model
    from deepagents_cli.tools import fetch_url, http_request, web_search

    try:
        model_result = create_model(
            model_name,
            extra_kwargs=model_params,
            profile_overrides=profile_override,
        )
    except ModelConfigError as exc:
        sys.stderr.write(f"Error: {exc}\n")
        sys.stderr.flush()
        return 1
    model_result.apply_to_settings()

    # Persist the resolved model so [models].recent is always populated.
    save_recent_model(f"{model_result.provider}:{model_result.model_name}")

    tools: list[Any] = [http_request, fetch_url]
    if settings.has_tavily:
        tools.append(web_search)

    mcp_session_manager = None
    mcp_server_info = None
    try:
        from deepagents_cli.mcp_tools import resolve_and_load_mcp_tools

        (
            mcp_tools,
            mcp_session_manager,
            mcp_server_info,
        ) = await resolve_and_load_mcp_tools(
            explicit_config_path=mcp_config_path,
            no_mcp=no_mcp,
            trust_project_mcp=trust_project_mcp,
        )
        tools.extend(mcp_tools)
    except FileNotFoundError as exc:
        msg = f"Error: MCP config file not found: {exc}\n"
        sys.stderr.write(msg)
        sys.stderr.flush()
        return 1
    except RuntimeError as exc:
        msg = f"Error: Failed to load MCP tools: {exc}\n"
        sys.stderr.write(msg)
        sys.stderr.flush()
        return 1

    async_subagents = load_async_subagents() or None

    try:
        from langgraph.checkpoint.memory import InMemorySaver

        agent_graph, _backend = create_cli_agent(
            model=model_result.model,
            assistant_id=assistant_id,
            tools=tools,
            mcp_server_info=mcp_server_info,
            checkpointer=InMemorySaver(),
            async_subagents=async_subagents,
        )
    except Exception as exc:
        sys.stderr.write(f"Error: failed to create agent: {exc}\n")
        sys.stderr.flush()
        logger.debug("ACP agent creation failed", exc_info=True)
        return 1

    server = agent_server_cls(agent_graph)  # Pregel is a CompiledStateGraph at runtime
    exit_code = 0
    try:
        await run_acp_agent(server)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        sys.stderr.write(f"Error: ACP server failed: {exc}\n")
        sys.stderr.flush()
        logger.exception("ACP server crashed")
        exit_code = 1
    finally:
        if mcp_session_manager is not None:
            try:
                await mcp_session_manager.cleanup()
            except Exception:
                logger.warning("MCP session cleanup failed", exc_info=True)
    return exit_code


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

    explicit_stdin = args.stdin

    if sys.stdin is None:
        if explicit_stdin:
            console.print(
                "[bold red]Error:[/bold red] --stdin was passed but stdin "
                "is not available."
            )
            sys.exit(1)
        return

    try:
        is_tty = sys.stdin.isatty()
    except (ValueError, OSError):
        if explicit_stdin:
            console.print(
                "[bold red]Error:[/bold red] --stdin was passed but stdin "
                "state could not be determined."
            )
            sys.exit(1)
        return

    if is_tty:
        if explicit_stdin:
            console.print(
                "[bold red]Error:[/bold red] --stdin was passed but stdin "
                "is a terminal. Pipe input or use -n instead.\n"
                "  cat prompt.txt | deepagents --stdin -q"
            )
            sys.exit(1)
        return

    max_stdin_bytes = 10 * 1024 * 1024  # 10 MiB

    try:
        stdin_text = sys.stdin.read(max_stdin_bytes + 1)
    except UnicodeDecodeError:
        msg = "Could not read piped input — ensure the input is valid text"
        console.print(f"[bold red]Error:[/bold red] {msg}")
        sys.exit(1)
    except (OSError, ValueError) as exc:
        from rich.markup import escape

        console.print(
            f"[bold red]Error:[/bold red] Failed to read piped input: "
            f"{escape(str(exc))}"
        )
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


def _print_session_stats(stats: Any, console: Any) -> None:  # noqa: ANN401
    """Print a session-level usage stats table to the console on TUI exit.

    Args:
        stats: The cumulative session stats from the Textual app.
        console: Rich console for output.
    """
    from deepagents_cli.textual_adapter import SessionStats, print_usage_table

    if not isinstance(stats, SessionStats):
        return
    print_usage_table(stats, stats.wall_time_seconds, console)


def _check_mcp_project_trust(*, trust_flag: bool = False) -> bool | None:
    """Check whether project-level MCP stdio servers should be trusted.

    When the project has no stdio servers in project-level configs, returns
    `None` (no gate needed). When `--trust-project-mcp` was passed, returns
    `True`. Otherwise checks the persistent trust store; if untrusted, shows
    an interactive approval prompt.

    Args:
        trust_flag: Whether `--trust-project-mcp` was passed.

    Returns:
        `True` to allow project stdio servers, `False` to deny, or `None`
            when no project stdio servers exist.
    """
    from deepagents_cli.mcp_tools import (
        classify_discovered_configs,
        discover_mcp_configs,
        extract_stdio_server_commands,
        load_mcp_config_lenient,
    )
    from deepagents_cli.project_utils import ProjectContext

    try:
        project_context = ProjectContext.from_user_cwd(Path.cwd())
        config_paths = discover_mcp_configs(project_context=project_context)
    except (OSError, RuntimeError):
        return None

    _, project_configs = classify_discovered_configs(config_paths)
    if not project_configs:
        return None

    # Collect all stdio servers across project configs
    all_stdio: list[tuple[str, str, list[str]]] = []
    for path in project_configs:
        cfg = load_mcp_config_lenient(path)
        if cfg is not None:
            all_stdio.extend(extract_stdio_server_commands(cfg))

    if not all_stdio:
        return None

    if trust_flag:
        return True

    # Check trust store
    from deepagents_cli.mcp_trust import (
        compute_config_fingerprint,
        is_project_mcp_trusted,
        trust_project_mcp,
    )

    project_root = str(
        (project_context.project_root or project_context.user_cwd).resolve()
    )
    fingerprint = compute_config_fingerprint(project_configs)

    if is_project_mcp_trusted(project_root, fingerprint):
        return True

    # Interactive prompt
    from rich.console import Console as _Console

    prompt_console = _Console(stderr=True)
    prompt_console.print()
    prompt_console.print(
        "[bold yellow]Project MCP servers require approval:[/bold yellow]"
    )
    for name, cmd, args in all_stdio:
        args_str = " ".join(args) if args else ""
        prompt_console.print(f'  [bold]"{name}"[/bold]:  {cmd} {args_str}')
    prompt_console.print()

    try:
        answer = input("Allow? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = ""

    if answer == "y":
        trust_project_mcp(project_root, fingerprint)
        return True
    return False


def cli_main() -> None:
    """Entry point for console script."""
    # Fix for gRPC fork issue on macOS
    # https://github.com/grpc/grpc/issues/37642
    if sys.platform == "darwin":
        os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"

    # Note: LANGSMITH_PROJECT override is handled lazily by config.py's
    # _ensure_bootstrap() (triggered on first access of `settings`).
    # This ensures agent traces use DEEPAGENTS_CLI_LANGSMITH_PROJECT while
    # shell commands use the user's original LANGSMITH_PROJECT.

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

    # ACP mode does not require Textual, so skip UI dependency checks when
    # the flag is present in raw argv.
    if "--acp" not in sys.argv[1:]:
        check_cli_dependencies()

    try:
        args = parse_args()

        # Import console/settings AFTER arg parsing so --help (which exits
        # inside parse_args) never pays the settings bootstrap cost.
        from deepagents_cli.config import console, settings

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

        profile_override: dict[str, Any] | None = None
        raw_profile = getattr(args, "profile_override", None)
        if raw_profile:
            try:
                profile_override = json.loads(raw_profile)
            except json.JSONDecodeError as e:
                console.print(
                    "[bold red]Error:[/bold red] "
                    f"--profile-override is not valid JSON: {e}"
                )
                sys.exit(1)
            if not isinstance(profile_override, dict):
                console.print(
                    "[bold red]Error:[/bold red] "
                    "--profile-override must be a JSON object"
                )
                sys.exit(1)

        if getattr(args, "acp", False):
            try:
                from acp import run_agent as run_acp_agent
                from deepagents_acp.server import AgentServerACP
            except ImportError as exc:
                msg = (
                    f"ACP dependencies not available: {exc}\n"
                    "Install with: pip install deepagents-acp\n"
                )
                sys.stderr.write(msg)
                sys.stderr.flush()
                sys.exit(1)

            if getattr(args, "no_mcp", False) and getattr(args, "mcp_config", None):
                msg = (
                    "Error: --no-mcp and --mcp-config are mutually exclusive."
                    " Use one or the other.\n"
                    "  deepagents --mcp-config path/to/config.json\n"
                    "  deepagents --no-mcp\n"
                )
                sys.stderr.write(msg)
                sys.stderr.flush()
                sys.exit(2)

            exit_code = asyncio.run(
                _run_acp_cli_async(
                    assistant_id=args.agent,
                    run_acp_agent=run_acp_agent,
                    agent_server_cls=AgentServerACP,
                    model_name=getattr(args, "model", None),
                    model_params=model_params,
                    profile_override=profile_override,
                    mcp_config_path=getattr(args, "mcp_config", None),
                    no_mcp=getattr(args, "no_mcp", False),
                    trust_project_mcp=getattr(args, "trust_project_mcp", False),
                )
            )
            sys.exit(exit_code)

        # Apply shell-allow-list from command line if provided (overrides env var)
        if args.shell_allow_list:
            from deepagents_cli.config import parse_shell_allow_list

            settings.shell_allow_list = parse_shell_allow_list(args.shell_allow_list)

        apply_stdin_pipe(args)

        if getattr(args, "no_mcp", False) and getattr(args, "mcp_config", None):
            from rich.console import Console as _Console

            _Console(stderr=True).print(
                "[bold red]Error:[/bold red] --no-mcp and --mcp-config "
                "are mutually exclusive. Use one or the other.\n"
                "  deepagents --mcp-config path/to/config.json\n"
                "  deepagents --no-mcp"
            )
            sys.exit(2)

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
                "--non-interactive (-n) or piped stdin\n"
                "  deepagents -n 'summarize README.md' --quiet"
            )
            sys.exit(2)

        # Handle --update flag or `update` subcommand (headless, no session)
        if args.update or args.command == "update":
            try:
                from rich.markup import escape

                from deepagents_cli._version import __version__ as cli_version
                from deepagents_cli.update_check import (
                    is_update_available,
                    perform_upgrade,
                    upgrade_command,
                )

                console.print("Checking for updates...", style="dim")
                available, latest = is_update_available(bypass_cache=True)
                if latest is None:
                    console.print(
                        "[bold yellow]Warning:[/bold yellow] Could not "
                        "reach PyPI. Check your network and try again."
                    )
                    sys.exit(1)
                if not available:
                    console.print(f"Already on the latest version (v{cli_version}).")
                    sys.exit(0)

                console.print(
                    f"Update available: v{latest} "
                    f"(current: v{cli_version}). Upgrading..."
                )
                success, output = asyncio.run(perform_upgrade())
                if success:
                    console.print(f"[green]Updated to v{latest}.[/green]")
                else:
                    cmd = upgrade_command()
                    detail = f": {escape(output[:200])}" if output else ""
                    console.print(
                        f"[bold red]Auto-update failed{detail}[/bold red]\n"
                        f"Run manually: [cyan]{cmd}[/cyan]"
                    )
                    sys.exit(1)
                sys.exit(0)
            except Exception:
                logger.warning("--update failed", exc_info=True)
                console.print(
                    "[bold red]Error:[/bold red] Update failed.\n"
                    "Run manually: [cyan]uv tool upgrade "
                    "deepagents-cli[/cyan]"
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

        output_format = getattr(args, "output_format", "text")

        if args.command == "help":
            from deepagents_cli.ui import show_help

            show_help()
        elif args.command == "agents":
            from deepagents_cli.agent import list_agents, reset_agent
            from deepagents_cli.ui import show_agents_help

            # "ls" is an argparse alias for "list"
            if args.agents_command in {"list", "ls"}:
                list_agents(output_format=output_format)
            elif args.agents_command == "reset":
                reset_agent(
                    args.agent,
                    args.source_agent,
                    dry_run=args.dry_run,
                    output_format=output_format,
                )
            else:
                show_agents_help()
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
                        sort_by=getattr(args, "sort", None),
                        branch=getattr(args, "branch", None),
                        verbose=getattr(args, "verbose", False),
                        relative=getattr(args, "relative", None),
                        output_format=output_format,
                    )
                )
            elif args.threads_command == "delete":
                asyncio.run(
                    delete_thread_command(
                        args.thread_id,
                        dry_run=args.dry_run,
                        output_format=output_format,
                    )
                )
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
            # Validate sandbox provider deps before spawning server subprocess
            if args.sandbox and args.sandbox not in {"none", "langsmith"}:
                from deepagents_cli.integrations.sandbox_factory import (
                    verify_sandbox_deps,
                )

                try:
                    verify_sandbox_deps(args.sandbox)
                except ImportError as exc:
                    from rich.markup import escape

                    console.print(f"[bold red]Error:[/bold red] {escape(str(exc))}")
                    sys.exit(1)

            # Non-interactive mode - execute single task and exit
            from deepagents_cli.non_interactive import run_non_interactive

            exit_code = asyncio.run(
                run_non_interactive(
                    message=args.non_interactive_message,
                    assistant_id=args.agent,
                    model_name=getattr(args, "model", None),
                    model_params=model_params,
                    profile_override=profile_override,
                    sandbox_type=args.sandbox,
                    sandbox_id=args.sandbox_id,
                    sandbox_setup=getattr(args, "sandbox_setup", None),
                    quiet=args.quiet,
                    stream=not args.no_stream,
                    mcp_config_path=getattr(args, "mcp_config", None),
                    no_mcp=getattr(args, "no_mcp", False),
                    trust_project_mcp=getattr(args, "trust_project_mcp", False),
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
                generate_thread_id,
                thread_exists,
            )

            # Instead of resolving thread_id here with synchronous asyncio.run()
            # DB calls, pass the raw resume request to the TUI and let it
            # resolve asynchronously during startup.
            resume_thread = args.resume_thread  # "__MOST_RECENT__", "<id>", or None
            thread_id = None if resume_thread else generate_thread_id()

            # Validate sandbox provider deps before spawning server subprocess
            if args.sandbox and args.sandbox not in {"none", "langsmith"}:
                from deepagents_cli.integrations.sandbox_factory import (
                    verify_sandbox_deps,
                )

                try:
                    verify_sandbox_deps(args.sandbox)
                except ImportError as exc:
                    from rich.markup import escape

                    console.print(f"[bold red]Error:[/bold red] {escape(str(exc))}")
                    sys.exit(1)

            # Check project MCP trust before launching TUI
            mcp_trust_decision = _check_mcp_project_trust(
                trust_flag=getattr(args, "trust_project_mcp", False),
            )

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
                        profile_override=profile_override,
                        thread_id=thread_id,
                        resume_thread=resume_thread,
                        initial_prompt=getattr(args, "initial_prompt", None),
                        mcp_config_path=getattr(args, "mcp_config", None),
                        no_mcp=getattr(args, "no_mcp", False),
                        trust_project_mcp=mcp_trust_decision,
                    )
                )
                return_code = result.return_code
                # The user may have switched threads via /threads during the
                # session; use the final thread ID for teardown messages.
                thread_id = result.thread_id or thread_id
                _print_session_stats(result.session_stats, console)
            except Exception as e:  # noqa: BLE001  # Top-level error handler for the application
                error_msg = Text("\nApplication error: ", style="red")
                error_msg.append(str(e))
                console.print(error_msg)
                console.print(Text(traceback.format_exc(), style="dim"))
                sys.exit(1)

            # Show LangSmith thread link for threads with checkpointed
            # content (same table that backs the `/threads` listing).
            if thread_id:
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

            # Warn about available update on exit
            try:
                if result.update_available[0]:
                    from deepagents_cli.update_check import (
                        is_auto_update_enabled,
                        upgrade_command,
                    )

                    latest = result.update_available[1]
                    console.print()
                    update_msg = Text("Update available: ", style="yellow bold")
                    update_msg.append(f"v{latest}", style="yellow")
                    console.print(update_msg)
                    cmd_hint = Text("Run: ", style="dim")
                    cmd_hint.append(upgrade_command(), style="cyan")
                    console.print(cmd_hint)
                    if not is_auto_update_enabled():
                        auto_hint = Text("Enable auto-updates: ", style="dim")
                        auto_hint.append("/auto-update", style="cyan")
                        console.print(auto_hint)
            except Exception:
                logger.debug("Failed to display exit update banner", exc_info=True)
    except KeyboardInterrupt:
        # Clean exit on Ctrl+C — suppress ugly traceback.
        # `console` may not be bound if Ctrl+C arrives during config import.
        try:
            console.print("\n\n[yellow]Interrupted[/yellow]")
        except NameError:
            sys.stderr.write("\n\nInterrupted\n")
        sys.exit(0)


if __name__ == "__main__":
    cli_main()
