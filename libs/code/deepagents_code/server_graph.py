"""Server-side graph entry point for `langgraph dev`.

This module is referenced by the generated `langgraph.json` and exposes a graph
factory that the LangGraph server can load and serve.

The graph is created by `make_graph()`, which reads configuration from
`ServerConfig.from_env()` — the same dataclass the CLI uses to *write* the
configuration via `ServerConfig.to_env()`. This shared schema ensures the two
sides stay in sync.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import sys
from typing import TYPE_CHECKING, Any

from deepagents_code._server_config import ServerConfig
from deepagents_code._startup_error import (
    STARTUP_ERROR_MARKER as _STARTUP_ERROR_MARKER,
    emit_startup_failure,
)
from deepagents_code.project_utils import ProjectContext, get_server_project_context

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

_sandbox_cm: Any = None
_sandbox_backend: Any = None
_mcp_session_manager: Any = None


def _print_startup_error(message: str) -> None:
    """Print a startup error for both humans and the parent app process.

    Args:
        message: Concise startup failure to surface in the parent process.
    """
    print(message, file=sys.stderr)  # noqa: T201  # stderr fallback for logs
    print(  # noqa: T201  # machine-readable marker consumed by server.py
        f"{_STARTUP_ERROR_MARKER}{message}",
        file=sys.stderr,
    )


def _get_mcp_session_manager() -> Any:  # noqa: ANN401
    """Return the process-wide MCP session manager singleton.

    Sessions are bound to the langgraph dev server's event loop. Cleanup
    therefore belongs to that loop's normal shutdown path, not `atexit` —
    an atexit handler runs after the loop is already closed and cannot
    await `AsyncExitStack.aclose()` safely. Subprocess handles held by
    stdio transports are released when the Python process exits.
    """
    global _mcp_session_manager  # noqa: PLW0603

    if _mcp_session_manager is None:
        from deepagents_code.mcp_tools import MCPSessionManager

        _mcp_session_manager = MCPSessionManager()

    return _mcp_session_manager


async def _build_tools(
    config: ServerConfig,
    project_context: ProjectContext | None,
) -> tuple[list[Any], list[Any] | None, list[Any]]:
    """Assemble the tool list based on server config.

    Loads built-in tools (conditionally including web search when Tavily is
    available) and MCP tools when enabled.

    MCP discovery is awaited on the server's event loop: LangGraph invokes this
    async factory on its running loop, so discovery must use `await` rather than
    `asyncio.run` (which raises inside a running loop). `stateless=True` ensures
    discovery only uses throwaway sessions, while the shared runtime session
    manager binds real sessions lazily inside the server loop on first tool
    invocation. MCP adapter imports are warmed in a worker thread inside
    `_load_tools_from_config` (only when active servers exist) because first
    import can perform blocking package-resource scans.

    Args:
        config: Deserialized server configuration.
        project_context: Resolved project context for MCP discovery.

    Returns:
        Tuple of `(tools, mcp_server_info, mcp_tools)`.

    Raises:
        FileNotFoundError: If the MCP config file is not found.
        RuntimeError: If MCP tool loading fails.
    """
    from deepagents_code.config import settings
    from deepagents_code.tools import fetch_url, get_current_thread_id, web_search

    tools: list[Any] = [fetch_url, get_current_thread_id]
    if settings.has_tavily:
        tools.append(web_search)

    mcp_server_info: list[Any] | None = None
    mcp_tools: list[Any] = []
    if not config.no_mcp:
        from deepagents_code.mcp_tools import resolve_and_load_mcp_tools
        from deepagents_code.plugins.adapters.mcp import discover_plugin_mcp_configs

        project_dir = (
            project_context.project_root or project_context.user_cwd
            if project_context is not None
            else None
        )
        # Offload plugin discovery: it does blocking disk IO (`os.mkdir` for
        # per-plugin data dirs, plus state/manifest reads) that `blockbuster`
        # rejects on the server event loop.
        plugin_mcp_configs = await asyncio.to_thread(
            discover_plugin_mcp_configs, project_dir=project_dir
        )
        try:
            mcp_tools, _, mcp_server_info = await resolve_and_load_mcp_tools(
                explicit_config_path=config.mcp_config_path,
                no_mcp=config.no_mcp,
                trust_project_mcp=config.trust_project_mcp,
                project_context=project_context,
                additional_configs=plugin_mcp_configs,
                stateless=True,
                session_manager=_get_mcp_session_manager(),
            )
        except FileNotFoundError:
            logger.exception("MCP config file not found: %s", config.mcp_config_path)
            raise
        except RuntimeError:
            logger.exception(
                "Failed to load MCP tools (config: %s)", config.mcp_config_path
            )
            raise

        tools.extend(mcp_tools)
        if mcp_tools:
            logger.info("Loaded %d MCP tool(s)", len(mcp_tools))

    return tools, mcp_server_info, mcp_tools


def _criteria_context_tools(
    tools: list[Any],
    mcp_tools: list[Any],
) -> list[Any]:
    """Select external context tools from the normal agent tool list.

    Args:
        tools: Main agent tools in execution order.
        mcp_tools: Exact tool objects returned by MCP discovery.

    Returns:
        External context tools available to criteria generation. MCP tools are
        included only when their protocol annotations explicitly declare them
        read-only.
    """
    from deepagents_code.tools import fetch_url, web_search

    allowed_ids = {id(fetch_url), id(web_search)}
    allowed_ids.update(
        id(tool) for tool in mcp_tools if _mcp_tool_is_explicitly_read_only(tool)
    )
    return [tool for tool in tools if id(tool) in allowed_ids]


def _mcp_tool_is_explicitly_read_only(tool: Any) -> bool:  # noqa: ANN401
    """Return whether a wrapped MCP tool is unambiguously read-only.

    MCP `ToolAnnotations.readOnlyHint` is serialized by the installed adapter
    into the LangChain tool's metadata as the camel-case `readOnlyHint` key.
    Require the literal boolean `True` and reject a contradictory destructive
    hint so absent, malformed, or ambiguous annotations fail closed.

    Returns:
        `True` only for an explicitly and consistently read-only MCP tool.
    """
    from deepagents_code.auto_mode import mcp_tool_is_coherently_read_only

    return mcp_tool_is_coherently_read_only(tool)


async def _make_graph() -> Any:  # noqa: ANN401
    """Create the agent graph from environment-based configuration.

    Reads `DEEPAGENTS_CODE_SERVER_*` env vars via `ServerConfig.from_env()`
    (the inverse of `ServerConfig.to_env()` used by the app process), resolves a
    model, assembles tools, and compiles the agent graph.

    Returns:
        Compiled LangGraph agent graph.
    """
    config = ServerConfig.from_env()
    project_context = get_server_project_context()

    from deepagents_code.agent import create_cli_agent, load_async_subagents
    from deepagents_code.config import (
        configure_langsmith_secret_redaction,
        create_model,
        is_memory_auto_save_enabled,
        settings,
    )

    if project_context is not None:
        settings.reload_from_environment(start_path=project_context.user_cwd)
    configure_langsmith_secret_redaction()

    # Offload to a worker thread: `create_model` does blocking disk IO for some
    # providers (e.g. the `openai_codex` token store currently acquires a file
    # lock via `langchain-openai` that calls `os.mkdir`), which `blockbuster`
    # rejects on the server event loop.
    result = await asyncio.to_thread(
        create_model,
        config.model,
        extra_kwargs=config.model_params,
        profile_overrides=config.profile_overrides,
    )
    result.apply_to_settings()

    tools, mcp_server_info, mcp_tools = await _build_tools(config, project_context)

    # Create sandbox backend if a sandbox provider is configured.
    # The context manager is created here in the factory, but its reference is
    # stored in a module-level global (and cleaned up via atexit) so the sandbox
    # lives for the entire server process lifetime. `make_graph` caches the built
    # graph, so this runs once per process despite LangGraph's per-run factory
    # invocation.
    global _sandbox_cm, _sandbox_backend  # noqa: PLW0603
    sandbox_backend = None
    if config.sandbox_type:
        from deepagents_code.integrations.sandbox_factory import create_sandbox

        try:
            _sandbox_cm = create_sandbox(
                config.sandbox_type,
                sandbox_id=config.sandbox_id,
                snapshot_name=config.sandbox_snapshot_name,
                setup_script_path=config.sandbox_setup,
            )
            _sandbox_backend = _sandbox_cm.__enter__()  # noqa: PLC2801  # Context manager kept open for server process lifetime
            sandbox_backend = _sandbox_backend

            def _cleanup_sandbox() -> None:
                if _sandbox_cm is not None:
                    _sandbox_cm.__exit__(None, None, None)

            atexit.register(_cleanup_sandbox)
        except ImportError:
            logger.exception(
                "Sandbox provider '%s' is not installed", config.sandbox_type
            )
            _print_startup_error(
                f"Sandbox provider '{config.sandbox_type}' is not installed"
            )
            sys.exit(1)
        except NotImplementedError:
            logger.exception("Sandbox type '%s' is not supported", config.sandbox_type)
            _print_startup_error(
                f"Sandbox type '{config.sandbox_type}' is not supported"
            )
            sys.exit(1)
        except ValueError as exc:
            logger.exception(
                "Invalid sandbox configuration for '%s'", config.sandbox_type
            )
            _print_startup_error(f"Invalid sandbox configuration: {exc}")
            sys.exit(1)
        except Exception as exc:
            logger.exception("Sandbox creation failed for '%s'", config.sandbox_type)
            _print_startup_error(
                f"Sandbox creation failed for '{config.sandbox_type}': {exc}"
            )
            sys.exit(1)

    def _create_cli_agent_sync() -> Any:  # noqa: ANN401
        async_subagents = load_async_subagents() or None
        from deepagents_code._env_vars import EXPERIMENTAL, is_env_truthy

        auto_mode_enabled = (
            config.interactive
            and sandbox_backend is None
            and is_env_truthy(EXPERIMENTAL)
        )

        # These process-global settings writes are safe here because `make_graph`
        # is lock-serialized and caches one graph for the server process lifetime.
        if config.interpreter_ptc is not None:
            settings.interpreter_ptc = config.interpreter_ptc
        if config.interpreter_ptc_acknowledge_unsafe:
            settings.interpreter_ptc_acknowledge_unsafe = True
        if config.enable_interpreter:
            settings.enable_interpreter = True

        agent, _composite_backend = create_cli_agent(
            model=result.model,
            assistant_id=config.assistant_id,
            tools=tools,
            mcp_tools=mcp_tools,
            sandbox=sandbox_backend,
            sandbox_type=config.sandbox_type,
            system_prompt=config.system_prompt,
            interactive=config.interactive,
            auto_approve=config.auto_approve,
            auto_mode_enabled=auto_mode_enabled,
            interrupt_shell_only=config.interrupt_shell_only,
            shell_allow_list=config.shell_allow_list,
            fs_tools=config.allow_fs_tools,
            enable_ask_user=config.enable_ask_user,
            enable_memory=config.enable_memory,
            memory_auto_save=is_memory_auto_save_enabled(),
            enable_skills=config.enable_skills,
            enable_shell=config.enable_shell,
            enable_interpreter=config.enable_interpreter,
            rubric_model=config.rubric_model,
            rubric_max_iterations=config.rubric_max_iterations,
            mcp_server_info=mcp_server_info,
            cwd=project_context.user_cwd if project_context is not None else config.cwd,
            project_context=project_context,
            async_subagents=async_subagents,
            goal_criteria_tools=_criteria_context_tools(tools, mcp_tools),
        )
        return agent

    return await asyncio.to_thread(_create_cli_agent_sync)


def _build_graph_factory(
    builder: Callable[[], Awaitable[Any]] | None = None,
) -> Callable[[], Awaitable[Any]]:
    """Build the cached async graph factory exposed to `langgraph dev`.

    The returned coroutine function is what `langgraph.json` references. It keeps
    its cache and lock in this closure rather than in module-level globals, so
    importing the module (e.g. for import-only checks) introduces no shared
    mutable state.

    Args:
        builder: Optional alternate graph builder.

    Returns:
        A zero-arg async factory that builds the graph once and returns the
        cached instance on every subsequent call.
    """
    missing = object()
    graph: Any = missing
    lock = asyncio.Lock()

    async def make_graph() -> Any:  # noqa: ANN401
        """Create (or return the cached) agent graph for `langgraph dev`.

        LangGraph loads this async factory from the generated `langgraph.json`
        and invokes it lazily on its event loop — and again on every run. The
        built graph is cached for the process lifetime so MCP discovery, sandbox
        creation, and `atexit` registration each happen exactly once; re-running
        them per request would re-discover MCP servers, leak sandbox sessions,
        and stack duplicate `atexit` handlers. Any construction failure is
        converted into a startup-error marker (scraped by the parent app
        process) before exiting.

        Returns:
            Compiled LangGraph agent graph.
        """
        nonlocal graph
        if graph is not missing:
            return graph
        async with lock:
            if graph is missing:
                try:
                    graph = await (builder or _make_graph)()
                except Exception as exc:  # noqa: BLE001  # top-level barrier: any construction failure must surface to the parent as a marker
                    emit_startup_failure(exc)
                    sys.exit(1)
            return graph

    return make_graph


make_graph = _build_graph_factory()
