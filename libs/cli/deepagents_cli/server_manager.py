"""Server lifecycle orchestration for the CLI.

Provides `start_server_and_get_agent` which handles the full flow of:
1. Setting up environment variables for the server graph
2. Generating `langgraph.json`
3. Copying the server graph entry point
4. Starting the `langgraph dev` server
5. Returning a `RemoteAgent` client
"""

from __future__ import annotations

import contextlib
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

logger = logging.getLogger(__name__)

_ENV_PREFIX = "DA_SERVER_"


async def start_server_and_get_agent(
    *,
    assistant_id: str,
    model_name: str | None = None,
    model_params: dict[str, Any] | None = None,
    auto_approve: bool = False,
    sandbox_type: str = "none",
    enable_ask_user: bool = False,
    mcp_config_path: str | None = None,
    no_mcp: bool = False,  # noqa: ARG001
    trust_project_mcp: bool | None = None,  # noqa: ARG001
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
        enable_ask_user: Enable ask_user tool.
        mcp_config_path: Path to MCP config.
        no_mcp: Disable MCP.
        trust_project_mcp: Trust project MCP servers.
        interactive: Whether the agent is interactive.
        host: Server host.
        port: Server port.

    Returns:
        Tuple of (remote_agent, server_process, mcp_session_manager).
    """
    from deepagents_cli.remote_client import RemoteAgent
    from deepagents_cli.server import ServerProcess, generate_langgraph_json

    work_dir = tempfile.mkdtemp(prefix="deepagents_server_")

    _set_server_env(
        model_name=model_name,
        model_params=model_params,
        assistant_id=assistant_id,
        auto_approve=auto_approve,
        sandbox_type=sandbox_type,
        enable_ask_user=enable_ask_user,
        mcp_config_path=mcp_config_path,
        interactive=interactive,
    )

    server_graph_src = Path(__file__).parent / "server_graph.py"
    server_graph_dst = Path(work_dir) / "server_graph.py"
    shutil.copy2(server_graph_src, server_graph_dst)

    _write_pyproject(Path(work_dir))

    generate_langgraph_json(
        work_dir,
        graph_ref="./server_graph.py:graph",
    )

    server = ServerProcess(host=host, port=port, config_dir=work_dir)
    await server.start()

    agent = RemoteAgent(
        url=server.url,
        assistant_id=assistant_id,
        graph_name="agent",
    )

    return agent, server, None


def _set_server_env(
    *,
    model_name: str | None,
    model_params: dict[str, Any] | None,
    assistant_id: str,
    auto_approve: bool,
    sandbox_type: str,
    enable_ask_user: bool,
    mcp_config_path: str | None,
    interactive: bool,
) -> None:
    """Set environment variables for the server graph process.

    Args:
        model_name: Model spec.
        model_params: Extra model kwargs.
        assistant_id: Agent identifier.
        auto_approve: Auto-approve flag.
        sandbox_type: Sandbox type.
        enable_ask_user: Ask user flag.
        mcp_config_path: MCP config path.
        interactive: Interactive mode flag.
    """
    if model_name:
        os.environ[f"{_ENV_PREFIX}MODEL"] = model_name
    os.environ[f"{_ENV_PREFIX}ASSISTANT_ID"] = assistant_id
    os.environ[f"{_ENV_PREFIX}AUTO_APPROVE"] = str(auto_approve).lower()
    os.environ[f"{_ENV_PREFIX}INTERACTIVE"] = str(interactive).lower()
    os.environ[f"{_ENV_PREFIX}ENABLE_ASK_USER"] = str(enable_ask_user).lower()

    if sandbox_type and sandbox_type != "none":
        os.environ[f"{_ENV_PREFIX}SANDBOX_TYPE"] = sandbox_type
    if mcp_config_path:
        os.environ[f"{_ENV_PREFIX}MCP_CONFIG_PATH"] = mcp_config_path
    if model_params:
        os.environ[f"{_ENV_PREFIX}MODEL_PARAMS"] = json.dumps(model_params)

    with contextlib.suppress(OSError):
        os.environ[f"{_ENV_PREFIX}CWD"] = str(Path.cwd())


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
    "langgraph-cli[inmem]>=0.1.55",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
"""
    (work_dir / "pyproject.toml").write_text(content)
