"""Server lifecycle orchestration for the app.

Provides `start_server_and_get_agent` which handles the full flow of:

1. Building a `ServerConfig` from application arguments
2. Writing config to env vars via `ServerConfig.to_env()`
3. Scaffolding a workspace (langgraph.json, checkpointer, pyproject)
4. Starting the `langgraph dev` server
5. Returning a `RemoteAgent` client

Also provides `server_session`, an async context manager that wraps
server startup and guaranteed cleanup so callers don't need to
duplicate try/finally teardown.
"""

from __future__ import annotations

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from deepagents_code.client.launch.server import ServerProcess
    from deepagents_code.client.remote_client import RemoteAgent
    from deepagents_code.mcp_tools import MCPSessionManager

from deepagents_code._env_vars import SERVER_ENV_PREFIX
from deepagents_code._server_config import ServerConfig
from deepagents_code.client.launch.server import _EPHEMERAL_PORT
from deepagents_code.project_utils import ProjectContext

logger = logging.getLogger(__name__)
_DISTRIBUTION_NAME = "deepagents-code"


def _set_or_clear_server_env(name: str, value: str | None) -> None:
    """Set or clear a `DEEPAGENTS_CODE_SERVER_*` environment variable.

    Args:
        name: Suffix after `DEEPAGENTS_CODE_SERVER_`.
        value: String value to set, or `None` to clear the variable.
    """
    key = f"{SERVER_ENV_PREFIX}{name}"
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


def _apply_server_config(config: ServerConfig) -> None:
    """Write a `ServerConfig` to `DEEPAGENTS_CODE_SERVER_*` env vars.

    Uses `ServerConfig.to_env()` so that the set of variables and their
    serialization format are defined in one place (the `ServerConfig` dataclass)
    rather than maintained independently here and in the
    reader (`ServerConfig.from_env()`).

    Args:
        config: Fully resolved server configuration.
    """
    for suffix, value in config.to_env().items():
        _set_or_clear_server_env(suffix, value)


def _capture_project_context() -> ProjectContext | None:
    """Capture the user's project context for the server subprocess.

    Returns:
        Explicit project context, or `None` when cwd cannot be determined.
    """
    try:
        return ProjectContext.from_user_cwd(Path.cwd())
    except OSError:
        logger.warning("Could not determine working directory for server")
        return None


# ------------------------------------------------------------------
# Workspace scaffolding
# ------------------------------------------------------------------


def _scaffold_workspace(work_dir: Path) -> None:
    """Prepare the server working directory with all required files.

    Generates the auxiliary files (checkpointer module, `pyproject.toml`,
    `langgraph.json`) that `langgraph dev` needs to boot. The generated
    graph reference imports the installed `deepagents_code` package directly.

    Args:
        work_dir: Temporary directory that will become the server's cwd.
    """
    from deepagents_code.client.launch.server import generate_langgraph_json

    _write_checkpointer(work_dir)
    _write_pyproject(work_dir)

    # `graph_ref` is a dotted import of the installed `deepagents_code` package,
    # but `checkpointer_path` stays cwd-relative: checkpointer.py is generated
    # fresh into work_dir (which `ServerProcess.start()` sets as the subprocess
    # cwd) and is not an importable package module. Don't "unify" these — a
    # dotted ref for the checkpointer would fail to resolve.
    generate_langgraph_json(
        work_dir,
        graph_ref="deepagents_code.server_graph:make_graph",
        checkpointer_path="./checkpointer.py:create_checkpointer",
    )


def _write_checkpointer(work_dir: Path) -> None:
    """Write a checkpointer module that reads its DB path from the environment.

    The generated module reads the DB path env var at runtime so the path
    is never baked into generated source. This is consistent with the
    `DEEPAGENTS_CODE_SERVER_*` env-var communication pattern used elsewhere.

    Args:
        work_dir: Server working directory.
    """
    from deepagents_code.sessions import get_db_path

    # Set the env var that the generated module will read at import time.
    os.environ[f"{SERVER_ENV_PREFIX}DB_PATH"] = str(get_db_path())

    db_path_var = f"{SERVER_ENV_PREFIX}DB_PATH"
    content = f'''\
"""Persistent SQLite checkpointer for the LangGraph dev server."""

import os
from contextlib import asynccontextmanager


@asynccontextmanager
async def create_checkpointer():
    """Yield an AsyncSqliteSaver connected to the app's sessions DB.

    The database path is read from the `{db_path_var}` env var
    (set by the app before server startup) rather than hard-coded, so
    the checkpointer module works without code generation.
    """
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    db_path = os.environ.get("{db_path_var}")
    if not db_path:
        raise RuntimeError(
            "{db_path_var} not set. The app must set this "
            "env var before server startup."
        )
    async with AsyncSqliteSaver.from_conn_string(db_path) as saver:
        yield saver
'''
    (work_dir / "checkpointer.py").write_text(content)


def _write_pyproject(work_dir: Path) -> None:
    """Write a minimal pyproject.toml for the server working directory.

    The `langgraph dev` server needs to install the project dependencies.
    We point it at the app package which transitively pulls in the SDK.

    Args:
        work_dir: Server working directory.
    """
    content = f"""[project]
name = "deepagents-server-runtime"
version = "0.0.1"
requires-python = ">=3.11"
dependencies = [
    "{_runtime_package_dependency()}",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
"""
    (work_dir / "pyproject.toml").write_text(content)


def _default_package_project_root() -> Path | None:
    """Return the project root that contains the top-level package.

    Returns:
        The directory above `deepagents_code` — the editable project root in
            source checkouts and `site-packages` for installed wheels — or
            `None` when the package location cannot be determined (e.g. a frozen
            or zipimport build where `__file__` is unset). Returning `None`
            (rather than guessing `Path.cwd()`) lets the caller fall back to
            the installed distribution version instead of pointing at whatever
            unrelated project happens to sit in the launch directory.
    """
    import deepagents_code

    # `getattr` with a default: `__file__` is unset on frozen/zipimport builds
    # and namespace packages, so it is not guaranteed to exist at runtime.
    package_init = getattr(deepagents_code, "__file__", None)
    if package_init is None:
        return None
    return Path(package_init).resolve().parent.parent


def _runtime_package_dependency(package_root: Path | None = None) -> str:
    """Return the dependency spec for the app package in the server runtime.

    Editable source checkouts can use a local path dependency so the generated
    runtime sees the working tree. Installed wheels cannot: the package parent is
    `site-packages`, which is not an installable project. In that case, depend on
    the installed distribution version instead.

    Args:
        package_root: Optional package project root for tests.

    Returns:
        Requirement string for the generated runtime `pyproject.toml`.
    """
    root = package_root or _default_package_project_root()
    if root is not None and (root / "pyproject.toml").is_file():
        return f"{_DISTRIBUTION_NAME} @ {root.as_uri()}"

    try:
        installed_version = version(_DISTRIBUTION_NAME)
    except PackageNotFoundError:
        return _DISTRIBUTION_NAME
    return f"{_DISTRIBUTION_NAME}=={installed_version}"


# ------------------------------------------------------------------
# MCP pre-flight validation
# ------------------------------------------------------------------


def _preflight_validate_mcp_config(
    *,
    mcp_config_path: str | None,
    no_mcp: bool,
) -> None:
    """Validate the explicit `--mcp-config` path before spawning the server.

    Catches the common failure mode of passing a malformed MCP config: a
    `ValueError` raised inside the server subprocess otherwise surfaces as an
    opaque truncated log dump in `wait_for_server_healthy`. Running the same
    validation in the parent process lets the TUI display a clean, actionable
    message with the offending path and reason.

    Project-level and user-level configs discovered by `resolve_and_load_mcp_tools`
    are not validated here; their errors are already handled leniently via
    `load_mcp_config_with_error` and surface as errored entries in the
    `/mcp` viewer rather than as a fatal startup failure.

    Args:
        mcp_config_path: Explicit path passed via `--mcp-config`, or `None`.
        no_mcp: When `True`, MCP is disabled and validation is skipped.

    Raises:
        MCPConfigError: If the config file is malformed or missing required
            fields. Message includes the offending path for context.
    """
    if no_mcp or not mcp_config_path:
        return

    from deepagents_code.mcp_tools import MCPConfigError, load_mcp_config

    try:
        load_mcp_config(mcp_config_path)
    except MCPConfigError:
        raise
    except FileNotFoundError as exc:
        msg = f"MCP config file not found: {mcp_config_path}"
        raise MCPConfigError(msg) from exc
    except (ValueError, TypeError) as exc:
        # `ValueError` covers `json.JSONDecodeError` (subclass) and the
        # shape/field validators in `_validate_server_config`; `TypeError`
        # covers the wrong-type branches. Bare `RuntimeError` is
        # deliberately NOT caught — it would mask unrelated bugs
        # (recursion, reentrancy, stdlib internals) as config errors.
        msg = f"Invalid MCP config at {mcp_config_path}: {exc}"
        raise MCPConfigError(msg) from exc


# ------------------------------------------------------------------
# Server startup
# ------------------------------------------------------------------


async def start_server_and_get_agent(
    *,
    assistant_id: str,
    model_name: str | None = None,
    model_params: dict[str, Any] | None = None,
    auto_approve: bool = False,
    interrupt_shell_only: bool = False,
    shell_allow_list: list[str] | None = None,
    sandbox_type: str = "none",
    sandbox_id: str | None = None,
    sandbox_snapshot_name: str | None = None,
    sandbox_setup: str | None = None,
    enable_shell: bool = True,
    enable_ask_user: bool = False,
    enable_interpreter: bool | None = None,
    interpreter_ptc: str | list[str] | None = None,
    interpreter_ptc_acknowledge_unsafe: bool = False,
    rubric_model: str | None = None,
    rubric_max_iterations: int | None = None,
    mcp_config_path: str | None = None,
    no_mcp: bool = False,
    trust_project_mcp: bool | None = None,
    interactive: bool = True,
    host: str = "127.0.0.1",
    port: int = _EPHEMERAL_PORT,
) -> tuple[RemoteAgent, ServerProcess, MCPSessionManager | None]:
    """Start a LangGraph server and return a connected remote agent client.

    Args:
        assistant_id: Agent identifier.
        model_name: Model spec string.
        model_params: Extra model kwargs.
        auto_approve: Auto-approve all tools.
        interrupt_shell_only: Validate shell commands via middleware instead of HITL.
        shell_allow_list: Restrictive shell allow-list for `ShellAllowListMiddleware`.
        sandbox_type: Sandbox type.
        sandbox_id: Existing sandbox ID to reuse.
        sandbox_snapshot_name: Snapshot (langsmith) or blueprint (runloop) name.
        sandbox_setup: Path to setup script for the sandbox.
        enable_shell: Enable shell execution tools.
        enable_ask_user: Enable ask_user tool.
        enable_interpreter: Enable the JS interpreter (`js_eval`) middleware on
            the main agent. `None` uses the sandbox-aware default.
        interpreter_ptc: Override for `settings.interpreter_ptc` (PTC allowlist).
        interpreter_ptc_acknowledge_unsafe: Explicit acknowledgement for
            `interpreter_ptc="all"` outside of `auto_approve`.
        rubric_model: Grader model spec; `None` reuses the main model.
        rubric_max_iterations: Explicit grader iterations per rubric attempt;
            `None` uses the SDK default.
        mcp_config_path: Path to MCP config.
        no_mcp: Disable MCP.
        trust_project_mcp: Trust project MCP servers.
        interactive: Whether the agent is interactive.
        host: Server host.
        port: Server port. Defaults to `_EPHEMERAL_PORT` (0), letting the server
            pick a free ephemeral port instead of the well-known `langgraph dev`
            port 2024.

    Returns:
        Tuple of `(remote_agent, server_process, mcp_session_manager)`.
            The `mcp_session_manager` is currently always `None` (MCP lifecycle
            is handled server-side).

    Raises:
        MCPConfigError: The explicit `--mcp-config` path is malformed,
            missing, or references contradictory transport fields. Raised
            from the pre-flight validator before any subprocess is spawned.
    """  # noqa: DOC502 - `_preflight_validate_mcp_config()` raises indirectly
    from deepagents_code.client.launch.server import ServerProcess
    from deepagents_code.client.remote_client import RemoteAgent

    project_context = _capture_project_context()

    _preflight_validate_mcp_config(
        mcp_config_path=mcp_config_path,
        no_mcp=no_mcp,
    )

    config = ServerConfig.from_cli_args(
        project_context=project_context,
        model_name=model_name,
        model_params=model_params,
        assistant_id=assistant_id,
        auto_approve=auto_approve,
        interrupt_shell_only=interrupt_shell_only,
        shell_allow_list=shell_allow_list,
        sandbox_type=sandbox_type,
        sandbox_id=sandbox_id,
        sandbox_snapshot_name=sandbox_snapshot_name,
        sandbox_setup=sandbox_setup,
        enable_shell=enable_shell,
        enable_ask_user=enable_ask_user,
        enable_interpreter=enable_interpreter,
        interpreter_ptc=interpreter_ptc,
        interpreter_ptc_acknowledge_unsafe=interpreter_ptc_acknowledge_unsafe,
        rubric_model=rubric_model,
        rubric_max_iterations=rubric_max_iterations,
        mcp_config_path=mcp_config_path,
        no_mcp=no_mcp,
        trust_project_mcp=trust_project_mcp,
        interactive=interactive,
    )
    _apply_server_config(config)

    work_dir = Path(tempfile.mkdtemp(prefix="deepagents_server_"))
    _scaffold_workspace(work_dir)

    server = ServerProcess(
        host=host,
        port=port,
        config_dir=work_dir,
        owns_config_dir=True,
        scaffold=_scaffold_workspace,
    )
    try:
        await server.start()
        await server.wait_for_graph_ready("agent")
    except Exception:
        server.stop()
        raise

    agent = RemoteAgent(
        url=server.url,
        graph_name="agent",
    )

    return agent, server, None


# ------------------------------------------------------------------
# Session context manager
# ------------------------------------------------------------------


@asynccontextmanager
async def server_session(
    *,
    assistant_id: str,
    model_name: str | None = None,
    model_params: dict[str, Any] | None = None,
    auto_approve: bool = False,
    interrupt_shell_only: bool = False,
    shell_allow_list: list[str] | None = None,
    sandbox_type: str = "none",
    sandbox_id: str | None = None,
    sandbox_snapshot_name: str | None = None,
    sandbox_setup: str | None = None,
    enable_shell: bool = True,
    enable_ask_user: bool = False,
    enable_interpreter: bool | None = None,
    interpreter_ptc: str | list[str] | None = None,
    interpreter_ptc_acknowledge_unsafe: bool = False,
    rubric_model: str | None = None,
    rubric_max_iterations: int | None = None,
    mcp_config_path: str | None = None,
    no_mcp: bool = False,
    trust_project_mcp: bool | None = None,
    interactive: bool = True,
    host: str = "127.0.0.1",
    port: int = _EPHEMERAL_PORT,
) -> AsyncIterator[tuple[RemoteAgent, ServerProcess]]:
    """Async context manager that starts a server and guarantees cleanup.

    Wraps `start_server_and_get_agent` so callers don't need to duplicate the
    try/finally pattern for stopping the server.

    Args:
        assistant_id: Agent identifier.
        model_name: Model spec string.
        model_params: Extra model kwargs.
        auto_approve: Auto-approve all tools.
        interrupt_shell_only: Validate shell commands via middleware instead of HITL.
        shell_allow_list: Restrictive shell allow-list for `ShellAllowListMiddleware`.
        sandbox_type: Sandbox type.
        sandbox_id: Existing sandbox ID to reuse.
        sandbox_snapshot_name: Snapshot (langsmith) or blueprint (runloop) name.
        sandbox_setup: Path to setup script for the sandbox.
        enable_shell: Enable shell execution tools.
        enable_ask_user: Enable ask_user tool.
        enable_interpreter: Enable the JS interpreter (`js_eval`) middleware on
            the main agent. `None` uses the sandbox-aware default.
        interpreter_ptc: Override for `settings.interpreter_ptc` (PTC allowlist).
        interpreter_ptc_acknowledge_unsafe: Explicit acknowledgement for
            `interpreter_ptc="all"` outside of `auto_approve`.
        rubric_model: Grader model spec; `None` reuses the main model.
        rubric_max_iterations: Explicit grader iterations per rubric attempt;
            `None` uses the SDK default.
        mcp_config_path: Path to MCP config.
        no_mcp: Disable MCP.
        trust_project_mcp: Trust project MCP servers.
        interactive: Whether the agent is interactive.
        host: Server host.
        port: Server port. Defaults to `_EPHEMERAL_PORT` (0), letting the server
            pick a free ephemeral port instead of the well-known `langgraph dev`
            port 2024.

    Yields:
        Tuple of `(remote_agent, server_process)`.
    """
    server_proc: ServerProcess | None = None
    mcp_session_manager: MCPSessionManager | None = None
    try:
        agent, server_proc, mcp_session_manager = await start_server_and_get_agent(
            assistant_id=assistant_id,
            model_name=model_name,
            model_params=model_params,
            auto_approve=auto_approve,
            interrupt_shell_only=interrupt_shell_only,
            shell_allow_list=shell_allow_list,
            sandbox_type=sandbox_type,
            sandbox_id=sandbox_id,
            sandbox_snapshot_name=sandbox_snapshot_name,
            sandbox_setup=sandbox_setup,
            enable_shell=enable_shell,
            enable_ask_user=enable_ask_user,
            enable_interpreter=enable_interpreter,
            interpreter_ptc=interpreter_ptc,
            interpreter_ptc_acknowledge_unsafe=interpreter_ptc_acknowledge_unsafe,
            rubric_model=rubric_model,
            rubric_max_iterations=rubric_max_iterations,
            mcp_config_path=mcp_config_path,
            no_mcp=no_mcp,
            trust_project_mcp=trust_project_mcp,
            interactive=interactive,
            host=host,
            port=port,
        )
        yield agent, server_proc
    finally:
        if mcp_session_manager is not None:
            try:
                await mcp_session_manager.cleanup()
            except Exception:
                logger.warning("MCP session cleanup failed", exc_info=True)
        if server_proc is not None:
            server_proc.stop()
