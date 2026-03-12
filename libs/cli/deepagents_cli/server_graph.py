"""Server-side graph entry point for `langgraph dev`.

This module is referenced by the generated `langgraph.json` and exposes
the CLI agent graph as a module-level variable that the LangGraph server
can load and serve.

The graph is created lazily on first access via `make_graph()`, which
reads configuration from environment variables set by the CLI before
starting the server process.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from deepagents_cli._server_constants import ENV_PREFIX as _ENV_PREFIX

logger = logging.getLogger(__name__)


def _read_env_json(key: str) -> Any:  # noqa: ANN401
    """Read a JSON-encoded environment variable.

    Args:
        key: Environment variable name.

    Returns:
        Parsed JSON value, or `None` if not set.
    """
    raw = os.environ.get(key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Failed to parse env var %s as JSON", key)
        return None


def make_graph() -> Any:  # noqa: ANN401
    """Create the CLI agent graph from environment-based configuration.

    Environment variables (set by the CLI before server start):
        DA_SERVER_MODEL: Model spec string (e.g., `anthropic:claude-sonnet-4-6`).
        DA_SERVER_MODEL_PARAMS: JSON-encoded extra model kwargs.
        DA_SERVER_ASSISTANT_ID: Agent identifier.
        DA_SERVER_SYSTEM_PROMPT: Optional system prompt override.
        DA_SERVER_AUTO_APPROVE: `"true"` to auto-approve all tools.
        DA_SERVER_INTERACTIVE: `"true"` to enable interactive mode.
        DA_SERVER_CWD: Working directory for the agent.
        DA_SERVER_SANDBOX_TYPE: Sandbox type string.
        DA_SERVER_ENABLE_MEMORY: `"true"` to enable memory middleware.
        DA_SERVER_ENABLE_SKILLS: `"true"` to enable skills middleware.
        DA_SERVER_ENABLE_SHELL: `"true"` to enable shell execution.
        DA_SERVER_ENABLE_ASK_USER: `"true"` to enable ask_user tool.
        DA_SERVER_MCP_CONFIG_PATH: Path to MCP config file.
        DA_SERVER_NO_MCP: `"true"` to disable all MCP tool loading.
        DA_SERVER_TRUST_PROJECT_MCP: `"true"` or `"false"` to control
            project-level MCP server trust.

    Returns:
        Compiled LangGraph agent graph.
    """
    from deepagents_cli.agent import DEFAULT_AGENT_NAME, create_cli_agent
    from deepagents_cli.config import create_model, settings
    from deepagents_cli.tools import fetch_url, http_request, web_search

    model_spec = os.environ.get(f"{_ENV_PREFIX}MODEL")
    assistant_id = os.environ.get(f"{_ENV_PREFIX}ASSISTANT_ID", DEFAULT_AGENT_NAME)
    system_prompt = os.environ.get(f"{_ENV_PREFIX}SYSTEM_PROMPT")
    auto_approve = os.environ.get(f"{_ENV_PREFIX}AUTO_APPROVE", "").lower() == "true"
    sandbox_type = os.environ.get(f"{_ENV_PREFIX}SANDBOX_TYPE")
    enable_memory = (
        os.environ.get(f"{_ENV_PREFIX}ENABLE_MEMORY", "true").lower() == "true"
    )
    enable_skills = (
        os.environ.get(f"{_ENV_PREFIX}ENABLE_SKILLS", "true").lower() == "true"
    )
    enable_shell = (
        os.environ.get(f"{_ENV_PREFIX}ENABLE_SHELL", "true").lower() == "true"
    )
    enable_ask_user = (
        os.environ.get(f"{_ENV_PREFIX}ENABLE_ASK_USER", "").lower() == "true"
    )
    interactive_str = os.environ.get(f"{_ENV_PREFIX}INTERACTIVE", "true")
    interactive = interactive_str.lower() == "true"
    cwd = os.environ.get(f"{_ENV_PREFIX}CWD")

    model_params = _read_env_json(f"{_ENV_PREFIX}MODEL_PARAMS")
    result = create_model(model_spec, extra_kwargs=model_params)
    model = result.model
    result.apply_to_settings()

    tools: list[Any] = [http_request, fetch_url]
    if settings.has_tavily:
        tools.append(web_search)

    no_mcp = os.environ.get(f"{_ENV_PREFIX}NO_MCP", "").lower() == "true"
    trust_project_mcp_raw = os.environ.get(f"{_ENV_PREFIX}TRUST_PROJECT_MCP")
    trust_project_mcp = (
        trust_project_mcp_raw.lower() == "true" if trust_project_mcp_raw else None
    )

    mcp_server_info = None
    mcp_config_path = os.environ.get(f"{_ENV_PREFIX}MCP_CONFIG_PATH")
    if mcp_config_path and not no_mcp:
        import asyncio

        from deepagents_cli.mcp_tools import resolve_and_load_mcp_tools

        mcp_tools, _, mcp_server_info = asyncio.run(
            resolve_and_load_mcp_tools(
                explicit_config_path=mcp_config_path,
                no_mcp=no_mcp,
                trust_project_mcp=(
                    trust_project_mcp if trust_project_mcp is not None else True
                ),
            )
        )
        tools.extend(mcp_tools)

    agent, _ = create_cli_agent(
        model=model,
        assistant_id=assistant_id,
        tools=tools,
        sandbox_type=sandbox_type if sandbox_type and sandbox_type != "none" else None,
        system_prompt=system_prompt,
        interactive=interactive,
        auto_approve=auto_approve,
        enable_memory=enable_memory,
        enable_skills=enable_skills,
        enable_shell=enable_shell,
        enable_ask_user=enable_ask_user,
        checkpointer=False,
        mcp_server_info=mcp_server_info,
        cwd=cwd,
    )
    return agent


graph = make_graph()
