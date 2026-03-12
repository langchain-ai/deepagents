"""Server-side graph entry point for `langgraph dev`.

This module is referenced by the generated `langgraph.json` and exposes the CLI
agent graph as a module-level variable that the LangGraph server can load
and serve.

The graph is created lazily on first access via `make_graph()`, which reads
configuration from `ServerConfig.from_env()` — the same dataclass the CLI uses
to *write* the configuration via `ServerConfig.to_env()`. This shared schema
ensures the two sides stay in sync.
"""

from __future__ import annotations

import logging
from typing import Any

from deepagents_cli._server_config import ServerConfig
from deepagents_cli.project_utils import ProjectContext, get_server_project_context

logger = logging.getLogger(__name__)


def _build_tools(
    config: ServerConfig,
    project_context: ProjectContext | None,
) -> tuple[list[Any], list[Any] | None]:
    """Assemble the tool list based on server config.

    Loads built-in tools unconditionally and MCP tools when enabled.

    MCP discovery runs synchronously via `asyncio.run` because this function is
    called during module-level graph construction (before the server's async
    event loop is available).

    Args:
        config: Deserialized server configuration.
        project_context: Resolved project context for MCP discovery.

    Returns:
        Tuple of `(tools, mcp_server_info)`.
    """
    from deepagents_cli.config import settings
    from deepagents_cli.tools import fetch_url, http_request, web_search

    tools: list[Any] = [http_request, fetch_url]
    if settings.has_tavily:
        tools.append(web_search)

    mcp_server_info: list[Any] | None = None
    if not config.no_mcp:
        import asyncio

        from deepagents_cli.mcp_tools import resolve_and_load_mcp_tools

        mcp_tools, _, mcp_server_info = asyncio.run(
            resolve_and_load_mcp_tools(
                explicit_config_path=config.mcp_config_path,
                no_mcp=config.no_mcp,
                trust_project_mcp=config.trust_project_mcp,
                project_context=project_context,
            )
        )
        tools.extend(mcp_tools)

    return tools, mcp_server_info


def make_graph() -> Any:  # noqa: ANN401
    """Create the CLI agent graph from environment-based configuration.

    Reads `DA_SERVER_*` env vars via `ServerConfig.from_env()` (the inverse of
    `ServerConfig.to_env()` used by the CLI process), resolves a model,
    assembles tools, and compiles the agent graph.

    Returns:
        Compiled LangGraph agent graph.
    """
    config = ServerConfig.from_env()
    project_context = get_server_project_context()

    from deepagents_cli.agent import create_cli_agent
    from deepagents_cli.config import create_model, settings

    if project_context is not None:
        settings.reload_from_environment(start_path=project_context.user_cwd)

    result = create_model(config.model, extra_kwargs=config.model_params)
    result.apply_to_settings()

    tools, mcp_server_info = _build_tools(config, project_context)

    agent, _ = create_cli_agent(
        model=result.model,
        assistant_id=config.assistant_id,
        tools=tools,
        sandbox_type=config.sandbox_type,
        system_prompt=config.system_prompt,
        interactive=config.interactive,
        auto_approve=config.auto_approve,
        enable_memory=config.enable_memory,
        enable_skills=config.enable_skills,
        enable_shell=config.enable_shell,
        enable_ask_user=config.enable_ask_user,
        mcp_server_info=mcp_server_info,
        cwd=project_context.user_cwd if project_context is not None else config.cwd,
        project_context=project_context,
    )
    return agent


graph = make_graph()
