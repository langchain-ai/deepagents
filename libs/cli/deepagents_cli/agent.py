"""Agent management and creation for the CLI."""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, LocalShellBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware import MemoryMiddleware, SkillsMiddleware

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from deepagents.backends.sandbox import SandboxBackendProtocol
    from deepagents.middleware.async_subagents import AsyncSubAgent
    from deepagents.middleware.subagents import CompiledSubAgent, SubAgent
    from langchain.agents.middleware import InterruptOnConfig
    from langchain.agents.middleware.types import AgentState
    from langchain.messages import ToolCall
    from langchain.tools import BaseTool
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import ToolMessage
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.prebuilt.tool_node import ToolCallRequest
    from langgraph.pregel import Pregel
    from langgraph.runtime import Runtime
    from langgraph.types import Command

    from deepagents_cli.mcp_tools import MCPServerInfo
    from deepagents_cli.output import OutputFormat

from langchain.agents.middleware.types import AgentMiddleware

from deepagents_cli import theme
from deepagents_cli.config import (
    _ShellAllowAll,
    config,
    console,
    get_default_coding_instructions,
    get_glyphs,
    settings,
)
from deepagents_cli.configurable_model import ConfigurableModelMiddleware
from deepagents_cli.integrations.sandbox_factory import get_default_working_dir
from deepagents_cli.local_context import (
    LocalContextMiddleware,
    _AsyncExecutableBackend,
    _ExecutableBackend,
)
from deepagents_cli.project_utils import ProjectContext, get_server_project_context
from deepagents_cli.subagents import list_subagents
from deepagents_cli.unicode_security import (
    check_url_safety,
    detect_dangerous_unicode,
    format_warning_detail,
    render_with_unicode_markers,
    strip_dangerous_unicode,
    summarize_issues,
)

logger = logging.getLogger(__name__)

DEFAULT_AGENT_NAME = "agent"
"""The default agent name used when no `-a` flag is provided."""

REQUIRE_COMPACT_TOOL_APPROVAL: bool = True
"""When `True`, `compact_conversation` requires HITL approval like other gated tools."""


class ShellAllowListMiddleware(AgentMiddleware):
    """Validate shell commands against an allow-list without HITL interrupts.

    When the agent invokes a shell tool (any tool in `SHELL_TOOL_NAMES`),
    this middleware checks the command against the configured allow-list
    **before execution**. Rejected commands are returned as error `ToolMessage`
    objects — the graph never pauses, so LangSmith traces stay as a single
    continuous run.

    Use this middleware in non-interactive mode to avoid the
    interrupt/resume cycle that fragments traces.
    """

    def __init__(self, allow_list: list[str]) -> None:
        """Initialize with the shell allow-list to validate commands against.

        Args:
            allow_list: Allowed command names (e.g. `["ls", "cat", "grep"]`).
                Must be a non-empty restrictive list — not `SHELL_ALLOW_ALL`.

        Raises:
            ValueError: If `allow_list` is empty.
            TypeError: If `allow_list` is the `SHELL_ALLOW_ALL` sentinel.
        """
        from deepagents_cli.config import SHELL_ALLOW_ALL

        super().__init__()
        if not allow_list:
            msg = "allow_list must not be empty; disable shell access instead"
            raise ValueError(msg)
        if isinstance(allow_list, type(SHELL_ALLOW_ALL)):
            msg = (
                "SHELL_ALLOW_ALL should not be used with "
                "ShellAllowListMiddleware; use auto_approve=True instead"
            )
            raise TypeError(msg)
        self._allow_list = list(allow_list)

    def _validate_tool_call(self, request: ToolCallRequest) -> ToolMessage | None:
        """Return an error tool message when a shell command is not allowed.

        Args:
            request: The tool call request being processed.

        Returns:
            An error `ToolMessage` when the shell command should be rejected,
            otherwise `None`.
        """
        from langchain_core.messages import ToolMessage as LCToolMessage

        from deepagents_cli.config import SHELL_TOOL_NAMES, is_shell_command_allowed

        tool_name = request.tool_call["name"]
        if tool_name not in SHELL_TOOL_NAMES:
            return None

        args = request.tool_call.get("args") or {}
        command = args.get("command", "")
        if is_shell_command_allowed(command, self._allow_list):
            logger.debug("Shell command allowed: %r", command)
            return None

        logger.warning("Shell command rejected by allow-list: %r", command)
        allowed_str = ", ".join(self._allow_list)
        return LCToolMessage(
            content=(
                f"Shell command rejected: `{command}` is not in the allow-list. "
                f"Allowed commands: {allowed_str}. "
                f"Please use an allowed command or try another approach."
            ),
            name=tool_name,
            tool_call_id=request.tool_call["id"],
            status="error",
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Reject disallowed shell commands; pass everything else through.

        Args:
            request: The tool call request being processed.
            handler: The next handler in the middleware chain.

        Returns:
            The tool execution result, or an error `ToolMessage` for rejected
            shell commands.
        """
        if (rejection := self._validate_tool_call(request)) is not None:
            return rejection
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Reject disallowed shell commands; pass everything else through.

        Args:
            request: The tool call request being processed.
            handler: The next handler in the middleware chain.

        Returns:
            The tool execution result, or an error `ToolMessage` for rejected
            shell commands.
        """
        if (rejection := self._validate_tool_call(request)) is not None:
            return rejection
        return await handler(request)


def load_async_subagents(config_path: Path | None = None) -> list[AsyncSubAgent]:
    """Load async subagent definitions from `config.toml`.

    Reads the `[async_subagents]` section where each sub-table defines a remote
    LangGraph deployment:

    ```toml
    [async_subagents.researcher]
    description = "Research agent"
    url = "https://my-deployment.langsmith.dev"
    graph_id = "agent"
    ```

    Args:
        config_path: Path to config file.

            Defaults to `~/.deepagents/config.toml`.

    Returns:
        List of `AsyncSubAgent` specs (empty if section is absent or invalid).
    """
    if config_path is None:
        config_path = Path.home() / ".deepagents" / "config.toml"

    if not config_path.exists():
        return []

    try:
        with config_path.open("rb") as f:
            data = tomllib.load(f)
    except (tomllib.TOMLDecodeError, PermissionError, OSError) as e:
        logger.warning("Could not read async subagents from %s: %s", config_path, e)
        console.print(
            f"[bold yellow]Warning:[/bold yellow] Could not read async subagents "
            f"from {config_path}: {e}",
        )
        return []

    section = data.get("async_subagents")
    if not isinstance(section, dict):
        return []

    required = {"description", "graph_id"}
    agents: list[AsyncSubAgent] = []
    for name, spec in section.items():
        if not isinstance(spec, dict):
            logger.warning("Skipping async subagent '%s': expected a table", name)
            continue
        missing = required - spec.keys()
        if missing:
            logger.warning(
                "Skipping async subagent '%s': missing fields %s", name, missing
            )
            continue
        agent: AsyncSubAgent = {
            "name": name,
            "description": spec["description"],
            "graph_id": spec["graph_id"],
        }
        if "url" in spec and isinstance(spec["url"], str):
            agent["url"] = spec["url"]
        if "headers" in spec and isinstance(spec["headers"], dict):
            agent["headers"] = spec["headers"]
        agents.append(agent)

    return agents


def list_agents(*, output_format: OutputFormat = "text") -> None:
    """List all available agents.

    Args:
        output_format: Output format — `'text'` (Rich) or `'json'`.
    """
    agents_dir = settings.user_deepagents_dir

    if not agents_dir.exists() or not any(agents_dir.iterdir()):
        if output_format == "json":
            from deepagents_cli.output import write_json

            write_json("list", [])
            return
        console.print("[yellow]No agents found.[/yellow]")
        console.print(
            "[dim]Agents will be created in ~/.deepagents/ "
            "when you first use them.[/dim]",
            style=theme.MUTED,
        )
        return

    if output_format == "json":
        from deepagents_cli.output import write_json

        agents = []
        for agent_path in sorted(agents_dir.iterdir()):
            if agent_path.is_dir():
                agent_name = agent_path.name
                agents.append(
                    {
                        "name": agent_name,
                        "path": str(agent_path),
                        "has_agents_md": (agent_path / "AGENTS.md").exists(),
                        "is_default": agent_name == DEFAULT_AGENT_NAME,
                    }
                )
        write_json("list", agents)
        return

    from rich.markup import escape as escape_markup

    console.print("\n[bold]Available Agents:[/bold]\n", style=theme.PRIMARY)

    for agent_path in sorted(agents_dir.iterdir()):
        if agent_path.is_dir():
            agent_name = escape_markup(agent_path.name)
            agent_md = agent_path / "AGENTS.md"
            is_default = agent_path.name == DEFAULT_AGENT_NAME
            default_label = " [dim](default)[/dim]" if is_default else ""

            bullet = get_glyphs().bullet
            if agent_md.exists():
                console.print(
                    f"  {bullet} [bold]{agent_name}[/bold]{default_label}",
                    style=theme.PRIMARY,
                )
                console.print(
                    f"    {escape_markup(str(agent_path))}",
                    style=theme.MUTED,
                )
            else:
                console.print(
                    f"  {bullet} [bold]{agent_name}[/bold]{default_label}"
                    " [dim](incomplete)[/dim]",
                    style=theme.WARNING,
                )
                console.print(
                    f"    {escape_markup(str(agent_path))}",
                    style=theme.MUTED,
                )

    console.print()


def reset_agent(
    agent_name: str,
    source_agent: str | None = None,
    *,
    dry_run: bool = False,
    output_format: OutputFormat = "text",
) -> None:
    """Reset an agent to default or copy from another agent.

    Args:
        agent_name: Name of the agent to reset.
        source_agent: Copy AGENTS.md from this agent instead of default.
        dry_run: If `True`, print what would happen without making changes.
        output_format: Output format — `'text'` (Rich) or `'json'`.

    Raises:
        SystemExit: If the source agent is not found.
    """
    agents_dir = settings.user_deepagents_dir
    agent_dir = agents_dir / agent_name

    if source_agent:
        source_dir = agents_dir / source_agent
        source_md = source_dir / "AGENTS.md"

        if not source_md.exists():
            console.print(
                f"[bold red]Error:[/bold red] Source agent '{source_agent}' not found "
                "or has no AGENTS.md\n"
                "  Available agents: deepagents agents list"
            )
            raise SystemExit(1)

        source_content = source_md.read_text()
        action_desc = f"contents of agent '{source_agent}'"
    else:
        source_content = get_default_coding_instructions()
        action_desc = "default"

    if dry_run:
        if output_format == "json":
            from deepagents_cli.output import write_json

            write_json(
                "reset",
                {
                    "agent": agent_name,
                    "reset_to": source_agent or "default",
                    "path": str(agent_dir),
                    "dry_run": True,
                },
            )
            return
        exists = "remove and recreate" if agent_dir.exists() else "create"
        console.print(f"Would {exists} {agent_dir} with {action_desc} prompt.")
        console.print("No changes made.", style=theme.MUTED)
        return

    if agent_dir.exists():
        shutil.rmtree(agent_dir)
        if output_format != "json":
            console.print(
                f"Removed existing agent directory: {agent_dir}", style=theme.WARNING
            )

    agent_dir.mkdir(parents=True, exist_ok=True)
    agent_md = agent_dir / "AGENTS.md"
    agent_md.write_text(source_content)

    if output_format == "json":
        from deepagents_cli.output import write_json

        write_json(
            "reset",
            {
                "agent": agent_name,
                "reset_to": source_agent or "default",
                "path": str(agent_dir),
            },
        )
        return

    console.print(
        f"{get_glyphs().checkmark} Agent '{agent_name}' reset to {action_desc}",
        style=theme.PRIMARY,
    )
    console.print(f"Location: {agent_dir}\n", style=theme.MUTED)


MODEL_IDENTITY_RE = re.compile(r"### Model Identity\n\n.*?(?=###|\Z)", re.DOTALL)
"""Matches the `### Model Identity` section in the system prompt, up to the
next heading or end of string."""


def build_model_identity_section(
    name: str | None,
    provider: str | None = None,
    context_limit: int | None = None,
) -> str:
    """Build the `### Model Identity` section for the system prompt.

    Args:
        name: Model identifier (e.g. `claude-opus-4-6`).
        provider: Provider identifier (e.g. `anthropic`).
        context_limit: Max input tokens from the model profile.

    Returns:
        The section text including the heading and trailing newline,
        or an empty string if `name` is falsy.
    """
    if not name:
        return ""
    section = f"### Model Identity\n\nYou are running as model `{name}`"
    if provider:
        section += f" (provider: {provider})"
    section += ".\n"
    if context_limit:
        section += f"Your context window is {context_limit:,} tokens.\n"
    section += "\n"
    return section


def get_system_prompt(
    assistant_id: str,
    sandbox_type: str | None = None,
    *,
    interactive: bool = True,
    cwd: str | Path | None = None,
) -> str:
    """Get the base system prompt for the agent.

    Loads the base system prompt template from `system_prompt.md` and
    interpolates dynamic sections (model identity, working directory,
    skills path, execution mode).

    Args:
        assistant_id: The agent identifier for path references
        sandbox_type: Type of sandbox provider
            (`'agentcore'`, `'daytona'`, `'langsmith'`, `'modal'`, `'runloop'`).

            If `None`, agent is operating in local mode.
        interactive: When `False`, the prompt is tailored for headless
            non-interactive execution (no human in the loop).
        cwd: Override the working directory shown in the prompt.

    Returns:
        The system prompt string

    Example:
        ```txt
        You are running as model {MODEL} (provider: {PROVIDER}).

        Your context window is {CONTEXT_WINDOW} tokens.

        ... {CONDITIONAL SECTIONS} ...
        ```
    """
    template = (Path(__file__).parent / "system_prompt.md").read_text()

    skills_path = f"~/.deepagents/{assistant_id}/skills"

    if interactive:
        mode_description = "an interactive CLI on the user's computer"
        interactive_preamble = (
            "The user sends you messages and you respond with text and tool "
            "calls. Your tools run on the user's machine. The user can see "
            "your responses and tool outputs in real time, so keep them "
            "informed — but don't over-explain."
        )
        ambiguity_guidance = (
            "- If the request is ambiguous, ask questions before acting.\n"
            "- If asked how to approach something, explain first, then act."
        )
    else:
        mode_description = (
            "non-interactive (headless) mode — there is no human operator "
            "monitoring your output in real time"
        )
        interactive_preamble = (
            "You received a single task and must complete it fully and "
            "autonomously. There is no human available to answer follow-up "
            "questions, so do NOT ask for clarification — make reasonable "
            "assumptions and proceed."
        )
        ambiguity_guidance = (
            "- Do NOT ask clarifying questions — there is no human to answer "
            "them. Make reasonable assumptions and proceed.\n"
            "- If you encounter ambiguity, choose the most reasonable "
            "interpretation and note your assumption briefly.\n"
            "- Always use non-interactive command variants — no human is "
            "available to respond to prompts. Examples: `npm init -y` not "
            "`npm init`, `apt-get install -y` not `apt-get install`, "
            "`yes |` or `--no-input`/`--non-interactive` flags where "
            "available. Never run commands that block waiting for stdin."
        )

    model_identity_section = build_model_identity_section(
        settings.model_name,
        provider=settings.model_provider,
        context_limit=settings.model_context_limit,
    )

    # Build working directory section (local vs sandbox)
    if sandbox_type:
        working_dir = get_default_working_dir(sandbox_type)
        working_dir_section = (
            f"### Current Working Directory\n\n"
            f"You are operating in a **remote Linux sandbox** at `{working_dir}`.\n\n"
            f"All code execution and file operations happen in this sandbox "
            f"environment.\n\n"
            f"**Important:**\n"
            f"- The CLI is running locally on the user's machine, but you execute "
            f"code remotely\n"
            f"- Use `{working_dir}` as your working directory for all operations\n"
            f"- **You do NOT have access to the user's local filesystem.** Paths "
            f"like `/Users/...`, `/home/<local-user>/...`, `C:\\...`, etc. do not "
            f"exist in this sandbox. Never reference or attempt to read/write local "
            f"paths — all files must be within the sandbox at `{working_dir}`\n"
            f"- When delegating to subagents, ensure they also use sandbox paths "
            f"(`{working_dir}/...`), not local paths\n\n"
        )
    else:
        if cwd is not None:
            resolved_cwd = Path(cwd)
        else:
            try:
                resolved_cwd = Path.cwd()
            except OSError:
                logger.warning(
                    "Could not determine working directory for system prompt",
                    exc_info=True,
                )
                resolved_cwd = Path()
        cwd = resolved_cwd
        working_dir_section = (
            f"### Current Working Directory\n\n"
            f"The filesystem backend is currently operating in: `{cwd}`\n\n"
            f"### File System and Paths\n\n"
            f"**IMPORTANT - Path Handling:**\n"
            f"- All file paths must be absolute paths (e.g., `{cwd}/file.txt`)\n"
            f"- Use the working directory to construct absolute paths\n"
            f"- Example: To create a file in your working directory, "
            f"use `{cwd}/research_project/file.md`\n"
            f"- Never use relative paths - always construct full absolute paths\n\n"
        )

    result = (
        template.replace("{mode_description}", mode_description)
        .replace("{interactive_preamble}", interactive_preamble)
        .replace("{ambiguity_guidance}", ambiguity_guidance)
        .replace("{model_identity_section}", model_identity_section)
        .replace("{working_dir_section}", working_dir_section)
        .replace("{skills_path}", skills_path)
    )

    # Detect unreplaced placeholders (defense-in-depth for template typos)
    unreplaced = re.findall(r"\{[a-z_]+\}", result)
    if unreplaced:
        logger.warning("System prompt contains unreplaced placeholders: %s", unreplaced)

    return result


def _format_write_file_description(
    tool_call: ToolCall, _state: AgentState[Any], _runtime: Runtime[Any]
) -> str:
    """Format write_file tool call for approval prompt.

    Returns:
        Formatted description string for the write_file tool call.
    """
    args = tool_call["args"]
    file_path = args.get("file_path", "unknown")

    action = "Overwrite" if Path(file_path).exists() else "Create"

    return f"Action: {action} file"


def _format_edit_file_description(
    tool_call: ToolCall, _state: AgentState[Any], _runtime: Runtime[Any]
) -> str:
    """Format edit_file tool call for approval prompt.

    Returns:
        Formatted description string for the edit_file tool call.
    """
    args = tool_call["args"]
    replace_all = bool(args.get("replace_all", False))

    scope = "all occurrences" if replace_all else "single occurrence"
    return f"Action: Replace text ({scope})"


def _format_web_search_description(
    tool_call: ToolCall, _state: AgentState[Any], _runtime: Runtime[Any]
) -> str:
    """Format web_search tool call for approval prompt.

    Returns:
        Formatted description string for the web_search tool call.
    """
    args = tool_call["args"]
    query = args.get("query", "unknown")
    max_results = args.get("max_results", 5)

    return (
        f"Query: {query}\nMax results: {max_results}\n\n"
        f"{get_glyphs().warning}  This will use Tavily API credits"
    )


def _format_fetch_url_description(
    tool_call: ToolCall, _state: AgentState[Any], _runtime: Runtime[Any]
) -> str:
    """Format fetch_url tool call for approval prompt.

    Returns:
        Formatted description string for the fetch_url tool call.
    """
    args = tool_call["args"]
    url = str(args.get("url", "unknown"))
    display_url = strip_dangerous_unicode(url)
    timeout = args.get("timeout", 30)
    safety = check_url_safety(url)

    warning_lines: list[str] = []
    if not safety.safe:
        detail = format_warning_detail(safety.warnings)
        warning_lines.append(f"{get_glyphs().warning}  URL warning: {detail}")
    if safety.decoded_domain:
        warning_lines.append(
            f"{get_glyphs().warning}  Decoded domain: {safety.decoded_domain}"
        )

    warning_block = "\n".join(warning_lines)
    if warning_block:
        warning_block = f"\n{warning_block}"

    return (
        f"URL: {display_url}\nTimeout: {timeout}s\n\n"
        f"{get_glyphs().warning}  Will fetch and convert web content to markdown"
        f"{warning_block}"
    )


def _format_task_description(
    tool_call: ToolCall, _state: AgentState[Any], _runtime: Runtime[Any]
) -> str:
    """Format task (subagent) tool call for approval prompt.

    The task tool signature is: task(description: str, subagent_type: str)
    The description contains all instructions that will be sent to the subagent.

    Returns:
        Formatted description string for the task tool call.
    """
    args = tool_call["args"]
    description = args.get("description", "unknown")
    subagent_type = args.get("subagent_type", "unknown")

    # Truncate description if too long for display
    description_preview = description
    if len(description) > 500:  # noqa: PLR2004  # Subagent description length threshold
        description_preview = description[:500] + "..."

    glyphs = get_glyphs()
    separator = glyphs.box_horizontal * 40
    warning_msg = "Subagent will have access to file operations and shell commands"
    return (
        f"Subagent Type: {subagent_type}\n\n"
        f"{glyphs.warning} {warning_msg} {glyphs.warning}\n\n"
        f"Task Instructions:\n"
        f"{separator}\n"
        f"{description_preview}"
    )


def _format_execute_description(
    tool_call: ToolCall, _state: AgentState[Any], _runtime: Runtime[Any]
) -> str:
    """Format execute tool call for approval prompt.

    Returns:
        Formatted description string for the execute tool call.
    """
    args = tool_call["args"]
    command_raw = str(args.get("command", "N/A"))
    command = strip_dangerous_unicode(command_raw)
    project_context = get_server_project_context()
    effective_cwd = (
        str(project_context.user_cwd)
        if project_context is not None
        else str(Path.cwd())
    )
    lines = [f"Execute Command: {command}", f"Working Directory: {effective_cwd}"]

    issues = detect_dangerous_unicode(command_raw)
    if issues:
        summary = summarize_issues(issues)
        lines.append(f"{get_glyphs().warning}  Hidden Unicode detected: {summary}")
        raw_marked = render_with_unicode_markers(command_raw)
        if len(raw_marked) > 220:  # noqa: PLR2004  # UI display truncation threshold
            raw_marked = raw_marked[:220] + "..."
        lines.append(f"Raw: {raw_marked}")

    return "\n".join(lines)


def _add_interrupt_on() -> dict[str, InterruptOnConfig]:
    """Configure human-in-the-loop interrupt settings for all gated tools.

    Every tool that can have side effects or access external resources
    (shell execution, file writes/edits, web search, URL fetch, task
    delegation) is gated behind an approval prompt unless auto-approve
    is enabled.

    Returns:
        Dictionary mapping tool names to their interrupt configuration.
    """
    execute_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_execute_description,  # type: ignore[typeddict-item]  # Callable description narrower than TypedDict expects
    }

    write_file_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_write_file_description,  # type: ignore[typeddict-item]  # Callable description narrower than TypedDict expects
    }

    edit_file_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_edit_file_description,  # type: ignore[typeddict-item]  # Callable description narrower than TypedDict expects
    }

    web_search_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_web_search_description,  # type: ignore[typeddict-item]  # Callable description narrower than TypedDict expects
    }

    fetch_url_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_fetch_url_description,  # type: ignore[typeddict-item]  # Callable description narrower than TypedDict expects
    }

    task_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_task_description,  # type: ignore[typeddict-item]  # Callable description narrower than TypedDict expects
    }

    async_subagent_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": "Launch, update, or cancel a remote async subagent.",
    }

    interrupt_map: dict[str, InterruptOnConfig] = {
        "execute": execute_interrupt_config,
        "write_file": write_file_interrupt_config,
        "edit_file": edit_file_interrupt_config,
        "web_search": web_search_interrupt_config,
        "fetch_url": fetch_url_interrupt_config,
        "task": task_interrupt_config,
        "launch_async_subagent": async_subagent_interrupt_config,
        "update_async_subagent": async_subagent_interrupt_config,
        "cancel_async_subagent": async_subagent_interrupt_config,
    }

    if REQUIRE_COMPACT_TOOL_APPROVAL:
        interrupt_map["compact_conversation"] = {
            "allowed_decisions": ["approve", "reject"],
            "description": (
                "Offloads older messages to backend storage and "
                "replaces them with a summary, freeing context "
                "window space. Recent messages are kept as-is. "
                "Full history remains available for retrieval."
            ),
        }

    return interrupt_map


def create_cli_agent(
    model: str | BaseChatModel,
    assistant_id: str,
    *,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    sandbox: SandboxBackendProtocol | None = None,
    sandbox_type: str | None = None,
    system_prompt: str | None = None,
    interactive: bool = True,
    auto_approve: bool = False,
    interrupt_shell_only: bool = False,
    shell_allow_list: list[str] | None = None,
    enable_ask_user: bool = True,
    enable_memory: bool = True,
    enable_skills: bool = True,
    enable_shell: bool = True,
    checkpointer: BaseCheckpointSaver | None = None,
    mcp_server_info: list[MCPServerInfo] | None = None,
    cwd: str | Path | None = None,
    project_context: ProjectContext | None = None,
    async_subagents: list[AsyncSubAgent] | None = None,
) -> tuple[Pregel, CompositeBackend]:
    """Create a CLI-configured agent with flexible options.

    This is the main entry point for creating a deepagents CLI agent, usable
    both internally and from external code (e.g., benchmarking frameworks).

    Args:
        model: LLM model to use (e.g., `'anthropic:claude-sonnet-4-6'`)
        assistant_id: Agent identifier for memory/state storage
        tools: Additional tools to provide to agent
        sandbox: Optional sandbox backend for remote execution
            (e.g., `ModalSandbox`).

            If `None`, uses local filesystem + shell.
        sandbox_type: Type of sandbox provider
            (`'agentcore'`, `'daytona'`, `'langsmith'`, `'modal'`, `'runloop'`).
            Used for system prompt generation.
        system_prompt: Override the default system prompt.

            If `None`, generates one based on `sandbox_type`, `assistant_id`,
            and `interactive`.
        interactive: When `False`, the auto-generated system prompt is
            tailored for headless non-interactive execution. Ignored when
            `system_prompt` is provided explicitly.
        auto_approve: If `True`, no tools trigger human-in-the-loop
            interrupts — all calls (shell execution, file writes/edits,
            web search, URL fetch) run automatically.

            If `False`, tools pause for user confirmation via the approval menu.
            See `_add_interrupt_on` for the full list of gated tools.
        interrupt_shell_only: If `True`, all HITL interrupts are disabled;
            shell commands are validated inline by `ShellAllowListMiddleware`
            against the configured allow-list instead.

            Used in non-interactive mode with a restrictive shell allow-list
            to avoid splitting traces into multiple LangSmith runs.

            Has no effect when `auto_approve` is `True` (interrupts are already
            disabled) or when `shell_allow_list` is `SHELL_ALLOW_ALL`.
        shell_allow_list: Explicit restrictive shell allow-list forwarded from
            the CLI process. When provided (and `interrupt_shell_only` is
            `True`), used directly instead of reading `settings.shell_allow_list`
            (which may not be set in the server subprocess environment).
        enable_ask_user: Enable `AskUserMiddleware` so the agent can ask
            clarifying questions.

            Disabled in non-interactive mode.
        enable_memory: Enable `MemoryMiddleware` for persistent memory
        enable_skills: Enable `SkillsMiddleware` for custom agent skills
        enable_shell: Enable shell execution via `LocalShellBackend`
            (only in local mode). When enabled, the `execute` tool is available.
        checkpointer: Optional checkpointer for session persistence.
            When `None`, the graph is compiled without a checkpointer.
        mcp_server_info: MCP server metadata to surface in the system prompt.
        cwd: Override the working directory for the agent's filesystem backend
            and system prompt.
        project_context: Explicit project path context for project-sensitive
            behavior such as project `AGENTS.md` files, skills, subagents, and
            MCP trust.
        async_subagents: Remote LangGraph deployments to expose as async subagent tools.

            Loaded from `[async_subagents]` in `config.toml` or passed directly.

    Returns:
        2-tuple of `(agent_graph, backend)`

            - `agent_graph`: Configured LangGraph Pregel instance ready
                for execution
            - `composite_backend`: `CompositeBackend` for file operations
    """
    tools = tools or []
    effective_cwd = (
        Path(cwd)
        if cwd is not None
        else (project_context.user_cwd if project_context is not None else None)
    )

    # Setup agent directory for persistent memory (if enabled)
    if enable_memory or enable_skills:
        agent_dir = settings.ensure_agent_dir(assistant_id)
        agent_md = agent_dir / "AGENTS.md"
        if not agent_md.exists():
            # Create empty file for user customizations
            # Base instructions are loaded fresh from get_system_prompt()
            agent_md.touch()

    # Skills directories (if enabled)
    skills_dir = None
    user_agent_skills_dir = None
    project_skills_dir = None
    project_agent_skills_dir = None
    if enable_skills:
        skills_dir = settings.ensure_user_skills_dir(assistant_id)
        user_agent_skills_dir = settings.get_user_agent_skills_dir()
        project_skills_dir = (
            project_context.project_skills_dir()
            if project_context is not None
            else settings.get_project_skills_dir()
        )
        project_agent_skills_dir = (
            project_context.project_agent_skills_dir()
            if project_context is not None
            else settings.get_project_agent_skills_dir()
        )

    # Load custom subagents from filesystem
    custom_subagents: list[SubAgent | CompiledSubAgent] = []
    restrictive_shell_allow_list: list[str] | None = None
    if interrupt_shell_only and not auto_approve:
        # Prefer the explicitly forwarded allow-list (set by the CLI process
        # and passed through ServerConfig).  Fall back to settings only for
        # direct callers (e.g. benchmarking frameworks) that don't go through
        # the server subprocess path.
        if shell_allow_list:
            restrictive_shell_allow_list = list(shell_allow_list)
        elif settings.shell_allow_list and not isinstance(
            settings.shell_allow_list, _ShellAllowAll
        ):
            restrictive_shell_allow_list = list(settings.shell_allow_list)
        else:
            logger.warning(
                "interrupt_shell_only=True but no restrictive shell allow-list "
                "available; falling back to standard HITL interrupts"
            )

    user_agents_dir = settings.get_user_agents_dir(assistant_id)
    project_agents_dir = (
        project_context.project_agents_dir()
        if project_context is not None
        else settings.get_project_agents_dir()
    )

    for subagent_meta in list_subagents(
        user_agents_dir=user_agents_dir,
        project_agents_dir=project_agents_dir,
    ):
        subagent: SubAgent = {
            "name": subagent_meta["name"],
            "description": subagent_meta["description"],
            "system_prompt": subagent_meta["system_prompt"],
        }
        if subagent_meta["model"]:
            subagent["model"] = subagent_meta["model"]
        if restrictive_shell_allow_list is not None:
            subagent["middleware"] = [
                ShellAllowListMiddleware(restrictive_shell_allow_list)
            ]
        custom_subagents.append(subagent)

    if restrictive_shell_allow_list is not None:
        from deepagents.middleware.subagents import (
            GENERAL_PURPOSE_SUBAGENT,
            SubAgent as RuntimeSubAgent,
        )

        if not any(
            subagent["name"] == GENERAL_PURPOSE_SUBAGENT["name"]
            for subagent in custom_subagents
        ):
            general_purpose_subagent: RuntimeSubAgent = {
                "name": GENERAL_PURPOSE_SUBAGENT["name"],
                "description": GENERAL_PURPOSE_SUBAGENT["description"],
                "system_prompt": GENERAL_PURPOSE_SUBAGENT["system_prompt"],
                "middleware": [ShellAllowListMiddleware(restrictive_shell_allow_list)],
            }
            custom_subagents.append(general_purpose_subagent)

    # Build middleware stack based on enabled features
    agent_middleware = []
    agent_middleware.append(ConfigurableModelMiddleware())

    # Add ask_user middleware (must be early so its tool is available)
    if enable_ask_user:
        from deepagents_cli.ask_user import AskUserMiddleware

        agent_middleware.append(AskUserMiddleware())

    # Add memory middleware
    if enable_memory:
        memory_sources = [str(settings.get_user_agent_md_path(assistant_id))]
        project_agent_md_paths = (
            project_context.project_agent_md_paths()
            if project_context is not None
            else settings.get_project_agent_md_path()
        )
        memory_sources.extend(str(p) for p in project_agent_md_paths)

        agent_middleware.append(
            MemoryMiddleware(
                backend=FilesystemBackend(),
                sources=memory_sources,
            )
        )

    # Add skills middleware
    if enable_skills:
        # Lowest to highest precedence:
        # built-in -> user .deepagents -> user .agents
        # -> project .deepagents -> project .agents
        # -> user .claude (experimental) -> project .claude (experimental)
        sources = [str(settings.get_built_in_skills_dir())]
        sources.extend([str(skills_dir), str(user_agent_skills_dir)])
        if project_skills_dir:
            sources.append(str(project_skills_dir))
        if project_agent_skills_dir:
            sources.append(str(project_agent_skills_dir))

        # Experimental: Claude Code skill directories
        user_claude_skills_dir = settings.get_user_claude_skills_dir()
        if user_claude_skills_dir.exists():
            sources.append(str(user_claude_skills_dir))
        project_claude_skills_dir = settings.get_project_claude_skills_dir()
        if project_claude_skills_dir:
            sources.append(str(project_claude_skills_dir))

        agent_middleware.append(
            SkillsMiddleware(
                backend=FilesystemBackend(),
                sources=sources,
            )
        )

    # CONDITIONAL SETUP: Local vs Remote Sandbox
    if sandbox is None:
        # ========== LOCAL MODE ==========
        root_dir = effective_cwd if effective_cwd is not None else Path.cwd()
        if enable_shell:
            # Create environment for shell commands
            # Restore user's original LANGSMITH_PROJECT so their code traces separately
            shell_env = os.environ.copy()
            if settings.user_langchain_project:
                shell_env["LANGSMITH_PROJECT"] = settings.user_langchain_project

            # Use LocalShellBackend for filesystem + shell execution.
            # The SDK's FilesystemMiddleware exposes per-command timeout
            # on the execute tool natively.
            backend = LocalShellBackend(
                root_dir=root_dir,
                inherit_env=True,
                env=shell_env,
            )
        else:
            # No shell access - use plain FilesystemBackend
            backend = FilesystemBackend(root_dir=root_dir)
    else:
        # ========== REMOTE SANDBOX MODE ==========
        backend = sandbox  # Remote sandbox (ModalSandbox, etc.)
        # Note: Shell middleware not used in sandbox mode
        # File operations and execute tool are provided by the sandbox backend

    # Local context middleware (git info, directory tree, etc.).
    if isinstance(backend, (_ExecutableBackend, _AsyncExecutableBackend)):
        agent_middleware.append(
            LocalContextMiddleware(backend=backend, mcp_server_info=mcp_server_info)
        )

    # Add shell allow-list middleware when interrupt_shell_only is active.
    shell_middleware_added = False
    if restrictive_shell_allow_list is not None:
        agent_middleware.append(ShellAllowListMiddleware(restrictive_shell_allow_list))
        shell_middleware_added = True

    # Get or use custom system prompt
    if system_prompt is None:
        system_prompt = get_system_prompt(
            assistant_id=assistant_id,
            sandbox_type=sandbox_type,
            interactive=interactive,
            cwd=effective_cwd,
        )

    # Configure interrupt_on based on auto_approve / shell_middleware_added
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None
    if auto_approve or shell_middleware_added:  # noqa: SIM108  # if-else clearer than ternary for dual-path config
        # No HITL interrupts — tools run automatically.
        # When shell_middleware_added is True, shell validation is handled by
        # ShellAllowListMiddleware (added above) which rejects disallowed
        # commands inline as error ToolMessages, keeping the entire run in
        # a single LangSmith trace.
        interrupt_on = {}
    else:
        # Full HITL for destructive operations
        interrupt_on = _add_interrupt_on()  # type: ignore[assignment]  # InterruptOnConfig is compatible at runtime

    # Set up composite backend with routing
    # For local FilesystemBackend, route large tool results to /tmp to avoid polluting
    # the working directory. For sandbox backends, no special routing is needed.
    if sandbox is None:
        # Local mode: Route large results to a unique temp directory
        large_results_backend = FilesystemBackend(
            root_dir=tempfile.mkdtemp(prefix="deepagents_large_results_"),
            virtual_mode=True,
        )
        conversation_history_backend = FilesystemBackend(
            root_dir=tempfile.mkdtemp(prefix="deepagents_conversation_history_"),
            virtual_mode=True,
        )
        composite_backend = CompositeBackend(
            default=backend,
            routes={
                "/large_tool_results/": large_results_backend,
                "/conversation_history/": conversation_history_backend,
            },
        )
    else:
        # Sandbox mode: No special routing needed
        composite_backend = CompositeBackend(
            default=backend,
            routes={},
        )

    from deepagents.middleware.summarization import create_summarization_tool_middleware

    agent_middleware.append(
        create_summarization_tool_middleware(model, composite_backend)
    )

    # Create the agent
    all_subagents: list[SubAgent | CompiledSubAgent | AsyncSubAgent] = [
        *custom_subagents,
        *(async_subagents or []),
    ]
    agent = create_deep_agent(
        model=model,
        system_prompt=system_prompt,
        tools=tools,
        backend=composite_backend,
        middleware=agent_middleware,
        interrupt_on=interrupt_on,
        checkpointer=checkpointer,
        subagents=all_subagents or None,
    ).with_config(config)
    return agent, composite_backend
