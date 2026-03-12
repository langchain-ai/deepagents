"""Server lifecycle orchestration for the CLI.

Provides `start_server_and_get_agent` which handles the full flow of:

1. Setting up environment variables for the server graph
2. Generating `langgraph.json`
3. Copying the server graph entry point
4. Starting the `langgraph dev` server
5. Returning a `RemoteAgent` client
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from deepagents_cli.mcp_tools import MCPSessionManager
    from deepagents_cli.remote_client import RemoteAgent
    from deepagents_cli.server import ServerProcess

from deepagents_cli._server_constants import ENV_PREFIX as _ENV_PREFIX
from deepagents_cli.project_utils import ProjectContext

logger = logging.getLogger(__name__)


def _set_or_clear_server_env(name: str, value: str | None) -> None:
    """Set or clear a server environment variable.

    Args:
        name: Suffix after `DA_SERVER_`.
        value: String value to set, or `None` to clear the variable.
    """
    key = f"{_ENV_PREFIX}{name}"
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


def _capture_server_project_context() -> ProjectContext | None:
    """Capture explicit user/project path context for the server process.

    Returns:
        Explicit project context, or `None` when cwd cannot be determined.
    """
    try:
        return ProjectContext.from_user_cwd(Path.cwd())
    except OSError:
        logger.warning("Could not determine working directory for server")
        return None


async def start_server_and_get_agent(
    *,
    assistant_id: str,
    model_name: str | None = None,
    model_params: dict[str, Any] | None = None,
    auto_approve: bool = False,
    sandbox_type: str = "none",
    enable_shell: bool = True,
    enable_ask_user: bool = False,
    mcp_config_path: str | None = None,
    no_mcp: bool = False,
    trust_project_mcp: bool | None = None,
    interactive: bool = True,
    host: str = "127.0.0.1",
    port: int = 2024,
) -> tuple[RemoteAgent, ServerProcess, MCPSessionManager | None]:
    """Start a LangGraph server and return a connected remote agent client.

    Args:
        assistant_id: Agent identifier.
        model_name: Model spec string.
        model_params: Extra model kwargs.
        auto_approve: Auto-approve all tools.
        sandbox_type: Sandbox type.
        enable_shell: Enable shell execution tools.
        enable_ask_user: Enable ask_user tool.
        mcp_config_path: Path to MCP config.
        no_mcp: Disable MCP.
        trust_project_mcp: Trust project MCP servers.
        interactive: Whether the agent is interactive.
        host: Server host.
        port: Server port.

    Returns:
        Tuple of `(remote_agent, server_process, mcp_session_manager)`.
    """
    from deepagents_cli.remote_client import RemoteAgent
    from deepagents_cli.server import ServerProcess, generate_langgraph_json

    work_dir = Path(tempfile.mkdtemp(prefix="deepagents_server_"))
    project_context = _capture_server_project_context()

    _set_server_env(
        project_context=project_context,
        model_name=model_name,
        model_params=model_params,
        assistant_id=assistant_id,
        auto_approve=auto_approve,
        sandbox_type=sandbox_type,
        enable_shell=enable_shell,
        enable_ask_user=enable_ask_user,
        mcp_config_path=mcp_config_path,
        no_mcp=no_mcp,
        trust_project_mcp=trust_project_mcp,
        interactive=interactive,
    )

    server_graph_src = Path(__file__).parent / "server_graph.py"
    server_graph_dst = work_dir / "server_graph.py"
    shutil.copy2(server_graph_src, server_graph_dst)

    checkpointer_path = work_dir / "checkpointer.py"

    _write_checkpointer(work_dir)
    _write_pyproject(work_dir)

    generate_langgraph_json(
        work_dir,
        graph_ref=f"{server_graph_dst.resolve()}:graph",
        checkpointer_path=f"{checkpointer_path.resolve()}:create_checkpointer",
    )

    server = ServerProcess(
        host=host, port=port, config_dir=work_dir, owns_config_dir=True
    )
    await server.start()

    agent = RemoteAgent(
        url=server.url,
        graph_name="agent",
    )

    return agent, server, None


def _set_server_env(
    *,
    project_context: ProjectContext | None,
    model_name: str | None,
    model_params: dict[str, Any] | None,
    assistant_id: str,
    auto_approve: bool,
    sandbox_type: str,
    enable_shell: bool,
    enable_ask_user: bool,
    mcp_config_path: str | None,
    no_mcp: bool,
    trust_project_mcp: bool | None,
    interactive: bool,
) -> None:
    """Set environment variables for the server graph process.

    The server graph runs in a separate Python interpreter, so env vars are the
    communication channel for CLI configuration.

    Args:
        project_context: Explicit user/project path context for the server.
        model_name: Model spec.
        model_params: Extra model kwargs.
        assistant_id: Agent identifier.
        auto_approve: Auto-approve flag.
        sandbox_type: Sandbox type.
        enable_shell: Enable shell execution tools.
        enable_ask_user: Ask user flag.
        mcp_config_path: MCP config path.
        no_mcp: Disable MCP.
        trust_project_mcp: Trust project MCP servers.
        interactive: Interactive mode flag.
    """
    normalized_mcp_config_path: str | None = None
    if mcp_config_path:
        try:
            if project_context is not None:
                normalized_mcp_config_path = str(
                    project_context.resolve_user_path(mcp_config_path)
                )
            else:
                normalized_mcp_config_path = str(
                    Path(mcp_config_path).expanduser().resolve()
                )
        except OSError:
            logger.warning("Could not normalize MCP config path %s", mcp_config_path)
            normalized_mcp_config_path = mcp_config_path

    _set_or_clear_server_env("MODEL", model_name)
    _set_or_clear_server_env("ASSISTANT_ID", assistant_id)
    _set_or_clear_server_env("AUTO_APPROVE", str(auto_approve).lower())
    _set_or_clear_server_env("INTERACTIVE", str(interactive).lower())
    _set_or_clear_server_env("ENABLE_SHELL", str(enable_shell).lower())
    _set_or_clear_server_env("ENABLE_ASK_USER", str(enable_ask_user).lower())
    _set_or_clear_server_env("NO_MCP", str(no_mcp).lower())
    _set_or_clear_server_env(
        "TRUST_PROJECT_MCP",
        str(trust_project_mcp).lower() if trust_project_mcp is not None else None,
    )
    _set_or_clear_server_env(
        "SANDBOX_TYPE",
        sandbox_type if sandbox_type and sandbox_type != "none" else None,
    )
    _set_or_clear_server_env("MCP_CONFIG_PATH", normalized_mcp_config_path)
    _set_or_clear_server_env(
        "MODEL_PARAMS",
        json.dumps(model_params) if model_params is not None else None,
    )
    _set_or_clear_server_env(
        "CWD",
        str(project_context.user_cwd) if project_context is not None else None,
    )
    _set_or_clear_server_env(
        "PROJECT_ROOT",
        (
            str(project_context.project_root)
            if project_context is not None and project_context.project_root is not None
            else None
        ),
    )


def _write_checkpointer(work_dir: Path) -> None:
    """Write a checkpointer module that persists to ~/.deepagents/sessions.db.

    This makes the LangGraph server store checkpoints on disk so thread history
    survives server restarts and `/threads` / `-r` work correctly.

    Args:
        work_dir: Server working directory.
    """
    from deepagents_cli.sessions import get_db_path

    db_path = str(get_db_path())
    content = f'''\
"""Persistent SQLite checkpointer for the LangGraph dev server."""

from contextlib import asynccontextmanager


@asynccontextmanager
async def create_checkpointer():
    """Yield an AsyncSqliteSaver connected to the CLI sessions DB."""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    async with AsyncSqliteSaver.from_conn_string(
        {db_path!r}
    ) as saver:
        yield saver
'''
    (work_dir / "checkpointer.py").write_text(content)


def _write_pyproject(work_dir: Path) -> None:
    """Write a minimal pyproject.toml for the server working directory.

    The `langgraph dev` server needs to install the project dependencies.
    We point it at the CLI package which transitively pulls in the SDK.

    Args:
        work_dir: Server working directory.
    """
    cli_dir = Path(__file__).parent.parent
    content = f"""[project]
name = "deepagents-server-runtime"
version = "0.0.1"
requires-python = ">=3.11"
dependencies = [
    "deepagents-cli @ file://{cli_dir}",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
"""
    (work_dir / "pyproject.toml").write_text(content)
