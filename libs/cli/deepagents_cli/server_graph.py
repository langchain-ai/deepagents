"""Server-side graph entry point for `langgraph dev`.

This module is referenced by the generated `langgraph.json` and exposes the CLI
agent graph as a module-level variable that the LangGraph server can load
and serve.

The graph is created at module import time via `make_graph()`, which reads
configuration from `ServerConfig.from_env()` — the same dataclass the CLI uses
to *write* the configuration via `ServerConfig.to_env()`. This shared schema
ensures the two sides stay in sync.
"""

from __future__ import annotations

import atexit
import logging
import sys
import traceback
from typing import Any

from deepagents_cli._server_config import ServerConfig
from deepagents_cli.project_utils import ProjectContext, get_server_project_context

logger = logging.getLogger(__name__)

# Module-level state kept alive for the server process lifetime.
_sandbox_cm: Any = None
_sandbox_backend: Any = None
_mcp_session_manager: Any = None
"""MCP session manager held at module level so its sessions stay open.

MCP tools capture a reference to a live `ClientSession`. That session is tied
to an async exit stack inside `MCPSessionManager`. If the manager is garbage-
collected (or its event loop torn down), the underlying stream is closed and
every subsequent tool call raises `ClosedResourceError`.

Keeping the manager here ensures it outlives the `asyncio.run()` call used to
bootstrap it and is only cleaned up at process exit via the registered atexit
handler.
"""


def _build_tools(
    config: ServerConfig,
    project_context: ProjectContext | None,
) -> tuple[list[Any], list[Any] | None]:
    """Assemble the tool list based on server config.

    Loads built-in tools (conditionally including web search when Tavily is
    available) and MCP tools when enabled.

    MCP discovery runs synchronously via `asyncio.run` because this function is
    called during module-level graph construction (before the server's async
    event loop is available). The returned `MCPSessionManager` is stored at
    module level so its sessions remain open for the server process lifetime.

    Args:
        config: Deserialized server configuration.
        project_context: Resolved project context for MCP discovery.

    Returns:
        Tuple of `(tools, mcp_server_info)`.

    Raises:
        FileNotFoundError: If the MCP config file is not found.
        RuntimeError: If MCP tool loading fails.
    """
    global _mcp_session_manager  # noqa: PLW0603  # Module-level lifetime management requires global

    from deepagents_cli.config import settings
    from deepagents_cli.tools import fetch_url, web_search

    tools: list[Any] = [fetch_url]
    if settings.has_tavily:
        tools.append(web_search)

    mcp_server_info: list[Any] | None = None
    if not config.no_mcp:
        import asyncio

        from deepagents_cli.mcp_tools import resolve_and_load_mcp_tools

        try:
            mcp_tools, mcp_session_manager, mcp_server_info = asyncio.run(
                resolve_and_load_mcp_tools(
                    explicit_config_path=config.mcp_config_path,
                    no_mcp=config.no_mcp,
                    trust_project_mcp=config.trust_project_mcp,
                    project_context=project_context,
                )
            )
        except FileNotFoundError:
            logger.exception("MCP config file not found: %s", config.mcp_config_path)
            raise
        except RuntimeError:
            logger.exception(
                "Failed to load MCP tools (config: %s)", config.mcp_config_path
            )
            raise

        if mcp_session_manager is not None:
            _mcp_session_manager = mcp_session_manager

            def _cleanup_mcp() -> None:
                if _mcp_session_manager is not None:
                    asyncio.run(_mcp_session_manager.cleanup())

            atexit.register(_cleanup_mcp)

        tools.extend(mcp_tools)
        if mcp_tools:
            logger.info("Loaded %d MCP tool(s)", len(mcp_tools))

    return tools, mcp_server_info


def make_graph() -> Any:  # noqa: ANN401
    """Create the CLI agent graph from environment-based configuration.

    Reads `DEEPAGENTS_CLI_SERVER_*` env vars via `ServerConfig.from_env()`
    (the inverse of `ServerConfig.to_env()` used by the CLI process), resolves a
    model, assembles tools, and compiles the agent graph.

    Returns:
        Compiled LangGraph agent graph.
    """
    config = ServerConfig.from_env()
    project_context = get_server_project_context()

    from deepagents_cli.agent import create_cli_agent, load_async_subagents
    from deepagents_cli.config import create_model, settings

    if project_context is not None:
        settings.reload_from_environment(start_path=project_context.user_cwd)

    result = create_model(config.model, extra_kwargs=config.model_params)
    result.apply_to_settings()

    tools, mcp_server_info = _build_tools(config, project_context)

    # Create sandbox backend if a sandbox provider is configured.
    # The context manager is held open at module level and cleaned up via
    # atexit so the sandbox lives for the entire server process lifetime.
    global _sandbox_cm, _sandbox_backend  # noqa: PLW0603
    sandbox_backend = None
    if config.sandbox_type:
        from deepagents_cli.integrations.sandbox_factory import create_sandbox

        try:
            _sandbox_cm = create_sandbox(
                config.sandbox_type,
                sandbox_id=config.sandbox_id,
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
            print(  # noqa: T201  # stderr fallback — logger may not reach parent process
                f"Sandbox provider '{config.sandbox_type}' is not installed",
                file=sys.stderr,
            )
            sys.exit(1)
        except NotImplementedError:
            logger.exception("Sandbox type '%s' is not supported", config.sandbox_type)
            print(  # noqa: T201  # stderr fallback — logger may not reach parent process
                f"Sandbox type '{config.sandbox_type}' is not supported",
                file=sys.stderr,
            )
            sys.exit(1)
        except Exception as exc:
            logger.exception("Sandbox creation failed for '%s'", config.sandbox_type)
            print(  # noqa: T201  # stderr fallback — logger may not reach parent process
                f"Sandbox creation failed for '{config.sandbox_type}': {exc}",
                file=sys.stderr,
            )
            sys.exit(1)

    async_subagents = load_async_subagents() or None

    agent, _ = create_cli_agent(
        model=result.model,
        assistant_id=config.assistant_id,
        tools=tools,
        sandbox=sandbox_backend,
        sandbox_type=config.sandbox_type,
        system_prompt=config.system_prompt,
        interactive=config.interactive,
        auto_approve=config.auto_approve,
        interrupt_shell_only=config.interrupt_shell_only,
        shell_allow_list=config.shell_allow_list,
        enable_ask_user=config.enable_ask_user,
        enable_memory=config.enable_memory,
        enable_skills=config.enable_skills,
        enable_shell=config.enable_shell,
        mcp_server_info=mcp_server_info,
        cwd=project_context.user_cwd if project_context is not None else config.cwd,
        project_context=project_context,
        async_subagents=async_subagents,
    )
    return agent


try:
    graph = make_graph()
except Exception as exc:
    logger.critical("Failed to initialize server graph", exc_info=True)
    print(  # noqa: T201  # stderr fallback — logger may not reach parent process
        f"Failed to initialize server graph: {exc}\n{traceback.format_exc()}",
        file=sys.stderr,
    )
    sys.exit(1)
