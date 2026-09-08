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
import dataclasses
import logging
import sys
from collections import OrderedDict
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NamedTuple

# Imported at runtime rather than under TYPE_CHECKING: the LangGraph server
# classifies `make_graph` by resolving its annotations with
# `typing.get_type_hints` at graph-load time. A name that only type checkers
# can see fails to resolve, and the server then refuses to load the graph.
from langgraph_sdk.runtime import ServerRuntime as LangGraphServerRuntime  # noqa: TC002

from deepagents_code._cli_context import CLIContextSchema
from deepagents_code._server_config import ServerConfig
from deepagents_code._startup_error import (
    STARTUP_ERROR_MARKER as _STARTUP_ERROR_MARKER,
    emit_startup_failure,
)
from deepagents_code.configuration.interpreter import InterpreterConfig
from deepagents_code.configuration.resolver import get_config_resolver
from deepagents_code.project_utils import ProjectContext, get_server_project_context
from deepagents_code.workspace import WorkspaceConflictError, resolve_workspace

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping
    from contextlib import AbstractContextManager

    from deepagents.backends.composite import CompositeBackend

    EnvironmentContext = Callable[
        [Mapping[str, str] | None], AbstractContextManager[None]
    ]

    from deepagents_code.config import CredentialsSnapshot
    from deepagents_code.extensions.registry import ExtensionRegistry
    from deepagents_code.offload_middleware import OffloadOperation
    from deepagents_code.workspace import WorkspaceBinding

logger = logging.getLogger(__name__)

_sandbox_cm: Any = None
_sandbox_backend: Any = None
_mcp_session_manager: Any = None
_server_tracing_settings: tuple[dict[str, str | None], bool] | None = None
_server_tracing_initialized = False


def _configure_server_tracing(environ: Mapping[str, str], *, redact: bool) -> None:
    """Pin tracing for the server lifetime before any runtime can execute.

    LangSmith uses process-wide env caches and a default client. Replacing
    them, even under a build lock, reroutes cached and concurrently executing
    runtimes. Only workspaces with matching tracing settings can share this
    process. Keep the reservation across failed builds and cache eviction.

    Called on the server loop with no suspension between claim and setup.
    """
    from deepagents_code.config import (
        _tracing_environment_values,
        configure_langsmith_secret_redaction,
        reconcile_tracing_environment,
    )

    global _server_tracing_settings, _server_tracing_initialized  # noqa: PLW0603  # process-lifetime policy
    settings = (_tracing_environment_values(environ), redact)
    if _server_tracing_settings is not None and settings != _server_tracing_settings:
        reason = (
            "its LangSmith tracing settings differ from this server's; "
            "start a separate server for this workspace"
        )
        conflict = WorkspaceConflictError.from_reason(reason)
        raise conflict
    _server_tracing_settings = settings
    if not _server_tracing_initialized:
        reconcile_tracing_environment(environ)
        # Keep redaction on the server task: its fail-closed disable must
        # reach this task's LangSmith ContextVar, not a worker's copied context.
        configure_langsmith_secret_redaction()
        _server_tracing_initialized = True


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
    *,
    tavily_api_key: str | None,
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
        tavily_api_key: Workspace Tavily key, or `None` when the workspace
            configures none. An empty string still binds the tool, which then
            reports the key as unconfigured.

    Returns:
        Tuple of `(tools, mcp_server_info, mcp_tools)`.

    Raises:
        FileNotFoundError: If the MCP config file is not found.
        RuntimeError: If MCP tool loading fails.
    """
    from deepagents_code.tools import (
        create_web_search_tool,
        fetch_url,
        get_current_thread_id,
    )

    tools: list[Any] = [fetch_url, get_current_thread_id]
    if tavily_api_key is not None:
        tools.append(create_web_search_tool(tavily_api_key))

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
    """Select read-only external tools for criteria drafting and rubric grading.

    Args:
        tools: Main agent tools in execution order.
        mcp_tools: Exact tool objects returned by MCP discovery.

    Returns:
        External context tools available to criteria generation and grading.
        MCP tools are included only when their protocol annotations explicitly
        declare them read-only.
    """
    from deepagents_code.tools import fetch_url, is_web_search_tool

    allowed_ids = {id(fetch_url)}
    allowed_ids.update(id(tool) for tool in tools if is_web_search_tool(tool))
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


class ServerRuntime(NamedTuple):
    """The one-per-process result of building this server's agent.

    A named tuple rather than a bare tuple so the three slots are addressed by
    name: `agent` is structurally opaque to the type checker (the SDK exposes no
    usable compiled-graph type here), so a positional transposition would hand
    LangGraph the backend as its compiled graph with no complaint.
    """

    agent: Any
    """Compiled LangGraph agent graph served as `agent`."""

    backend: CompositeBackend
    """Composite backend the agent and its operations were built with."""

    offload: OffloadOperation
    """Server-owned thread offload operation bound to `backend`."""


async def _make_graphs(
    *,
    config_override: ServerConfig | None = None,
    project_context_override: ProjectContext | None = None,
) -> ServerRuntime:
    """Create the agent graph and the backend carrying its shared resources.

    Reads `DEEPAGENTS_CODE_SERVER_*` env vars via `ServerConfig.from_env()`
    (the inverse of `ServerConfig.to_env()` used by the app process), resolves a
    model, assembles tools, and compiles the agent graph.

    Returns:
        The agent graph, its configured composite backend, and the server-owned
            offload operation bound to that backend.
    """
    config = config_override or ServerConfig.from_env()
    workspace_path = (
        project_context_override.user_cwd
        if project_context_override is not None
        else Path(config.cwd)
        if config.cwd is not None
        else None
    )

    # Offload the workspace environment snapshot off the event loop. Dotenv
    # discovery walks parent directories (`Path.resolve()`, `is_file()`) and
    # reads up to three files, and `snapshot_from_environment` adds
    # `find_project_root()` -> `Path.cwd()` — all of which `blockbuster`
    # rejects when invoked directly from the server loop (see issue #5043),
    # for the same reason as the offload in `_make_graphs_in_environment`.
    def _resolve_workspace_environment() -> tuple[
        Mapping[str, str], CredentialsSnapshot, EnvironmentContext, bool
    ]:
        from deepagents_code.config import (
            Credentials,
            _ensure_bootstrap,
            _preview_dotenv_environ,
            is_langsmith_redaction_enabled,
            use_environment,
        )

        # Finish the one-time global credential publication before pinning
        # tracing. A later lazy import of `agent` must not overwrite the pin.
        _ensure_bootstrap()
        environ = MappingProxyType(_preview_dotenv_environ(start_path=workspace_path))
        with use_environment(environ):
            redact = is_langsmith_redaction_enabled()
        return (
            environ,
            Credentials.snapshot_from_environment(
                start_path=workspace_path,
                environ=environ,
            ),
            use_environment,
            redact,
        )

    (
        workspace_env,
        workspace_credentials,
        use_environment,
        redact,
    ) = await asyncio.to_thread(_resolve_workspace_environment)

    with use_environment(workspace_env):
        _configure_server_tracing(workspace_env, redact=redact)
        return await _make_graphs_in_environment(
            config=config,
            project_context_override=project_context_override,
            workspace_env=workspace_env,
            workspace_credentials=workspace_credentials,
        )


async def _make_graphs_in_environment(
    *,
    config: ServerConfig,
    project_context_override: ProjectContext | None,
    workspace_env: Mapping[str, str],
    workspace_credentials: CredentialsSnapshot,
) -> ServerRuntime:
    """Build one runtime while its immutable workspace environment is active.

    Returns:
        Agent graph and its workspace-bound resources.
    """

    # Offload cwd/path resolution and the lazy settings bootstrap off the event
    # loop. On Windows, `Path.resolve()` / `Path.cwd()` call `os.getcwd()`, which
    # `blockbuster` rejects when invoked directly from the server loop (see
    # issue #5043). Importing `deepagents_code.agent` / first `settings` access
    # can also trigger `find_project_root()` -> `Path.cwd()`.
    def _resolve_project_context_and_settings() -> tuple[
        ProjectContext | None,
        Any,
        Any,
        Any,
        Any,
        Any,
    ]:
        project_context = project_context_override or get_server_project_context()

        from deepagents_code.agent import create_cli_agent, load_async_subagents
        from deepagents_code.config import (
            create_model,
            is_memory_auto_save_enabled,
            resolve_auto_classifier_model_for_provider,
        )

        return (
            project_context,
            create_cli_agent,
            load_async_subagents,
            create_model,
            is_memory_auto_save_enabled,
            resolve_auto_classifier_model_for_provider,
        )

    (
        project_context,
        create_cli_agent,
        load_async_subagents,
        create_model,
        is_memory_auto_save_enabled,
        resolve_auto_classifier_model_for_provider,
    ) = await asyncio.to_thread(_resolve_project_context_and_settings)
    # Offload to a worker thread: `create_model` does blocking disk IO for some
    # providers (e.g. the `openai_codex` token store currently acquires a file
    # lock via `langchain-openai` that calls `os.mkdir`), which `blockbuster`
    # rejects on the server event loop.
    result = await asyncio.to_thread(
        create_model,
        config.model,
        extra_kwargs=config.model_params,
        profile_overrides=config.profile_overrides,
        cli_max_retries=config.cli_max_retries,
    )
    result.apply_to_runtime_state()

    tools, mcp_server_info, mcp_tools = await _build_tools(
        config,
        project_context,
        tavily_api_key=workspace_credentials.tavily_api_key,
    )
    read_only_context_tools = _criteria_context_tools(tools, mcp_tools)

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

    extension_registry: ExtensionRegistry | None = None

    def _create_cli_graphs_sync() -> ServerRuntime:
        async_subagents = load_async_subagents() or None
        auto_mode_enabled = config.interactive and sandbox_backend is None

        interpreter_config = (
            InterpreterConfig.from_resolver(
                get_config_resolver(),
                ptc=config.interpreter_ptc,
                ptc_acknowledge_unsafe=config.interpreter_ptc_acknowledge_unsafe,
            )
            if config.enable_interpreter
            else None
        )

        agent, composite_backend = create_cli_agent(
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
            interpreter_config=interpreter_config,
            rubric_model=config.rubric_model,
            rubric_max_iterations=config.rubric_max_iterations,
            auto_classifier_model=resolve_auto_classifier_model_for_provider(
                result.provider,
                config.auto_classifier_model,
            ),
            recursion_limit=config.recursion_limit,
            mcp_server_info=mcp_server_info,
            cwd=project_context.user_cwd if project_context is not None else config.cwd,
            project_context=project_context,
            async_subagents=async_subagents,
            goal_criteria_tools=read_only_context_tools,
            rubric_grader_tools=read_only_context_tools,
            model_retries=result.model_retries,
            cli_max_retries=result.cli_max_retries,
            summarization_model=config.summarization_model,
            extension_registry=extension_registry,
            environ=workspace_env,
            credentials_snapshot=workspace_credentials,
            model_result=result,
        )
        from deepagents_code.offload_middleware import offload_operation_from

        offload = offload_operation_from(composite_backend)
        if offload is None:
            msg = (
                "Agent backend did not publish its offload operation; "
                "/offload has no server implementation."
            )
            raise RuntimeError(msg)
        return ServerRuntime(
            agent=agent,
            backend=composite_backend,
            offload=offload,
        )

    from deepagents_code._env_vars import EXPERIMENTAL, is_env_truthy

    if is_env_truthy(EXPERIMENTAL, environ=workspace_env):
        from deepagents_code.extensions import ExtensionMode, load_extensions
        from deepagents_code.extensions.runtime import bind_server_extensions

        extension_result = await load_extensions(
            cwd=(
                project_context.user_cwd
                if project_context is not None
                else Path(config.cwd)
                if config.cwd is not None
                else None
            ),
            mode=(
                ExtensionMode.INTERACTIVE
                if config.interactive
                else ExtensionMode.HEADLESS
            ),
            project_root=(
                project_context.project_root or project_context.user_cwd
                if project_context is not None
                else None
            ),
            project_trust_granted=config.trust_project_extensions,
            cli_paths=tuple(Path(path) for path in config.extension_paths),
        )
        for message in extension_result.errors:
            logger.warning("Extension not loaded: %s", message)
        if extension_result.active:
            extension_registry = extension_result.registry
            bind_server_extensions(extension_result)
    try:
        return await asyncio.to_thread(_create_cli_graphs_sync)
    except BaseException:
        if extension_registry is not None:
            from deepagents_code.extensions.runtime import shutdown_server_extensions

            await shutdown_server_extensions()
        raise


def _build_runtime_factory(
    builder: Callable[[], Awaitable[ServerRuntime]] | None = None,
) -> Callable[[], Awaitable[ServerRuntime]]:
    """Build the cached factory for all server-owned runtime resources.

    The cache is load-bearing, not an optimization: MCP discovery, sandbox
    creation, and `atexit` registration each must happen exactly once. Building
    per request would re-discover MCP servers, leak sandbox sessions, and stack
    duplicate `atexit` handlers. Two consumers now share this cache -- the
    interactive graph and the offload HTTP route -- so both must resolve the
    *same* agent, backend, and compaction policy for a server-side archive to be
    readable by the agent.

    The cache and its lock live in this closure rather than in module-level
    globals, so importing this module introduces no shared mutable state; the
    single process-wide instance is created explicitly at the bottom of the
    module.

    Args:
        builder: Optional alternate builder used by unit tests.

    Returns:
        Async runtime factory shared by the graph and custom operation API.
    """
    runtime: ServerRuntime | None = None
    lock = asyncio.Lock()

    async def get_runtime() -> ServerRuntime:
        """Return the cached interactive graph and operation resources."""
        nonlocal runtime
        if runtime is None:
            async with lock:
                if runtime is None:
                    try:
                        from deepagents_code.configuration.service import (
                            require_healthy_managed_config,
                        )

                        await asyncio.to_thread(
                            require_healthy_managed_config,
                            refresh=True,
                        )
                        runtime = await (builder or _make_graphs)()
                    except Exception as exc:  # noqa: BLE001  # startup barrier
                        emit_startup_failure(exc)
                        sys.exit(1)
        return runtime

    return get_runtime


def _build_graph_factory(
    builder: Callable[[], Awaitable[ServerRuntime]] | None = None,
) -> Callable[[], Awaitable[Any]]:
    """Build a cached graph factory, for tests.

    `langgraph.json` references the module-level `make_graph`, which delegates to
    `get_server_runtime`; nothing in production calls this. It survives so unit
    tests can inject a builder.

    Args:
        builder: Optional alternate runtime builder used by unit tests.

    Returns:
        Async graph factory for the interactive `agent` graph.
    """
    get_runtime = _build_runtime_factory(builder)

    async def make_graph() -> Any:  # noqa: ANN401
        """Create or return the cached agent graph for `langgraph dev`.

        Returns:
            Compiled LangGraph agent graph.
        """
        return (await get_runtime()).agent

    return make_graph


_get_runtime = _build_runtime_factory()
_MAX_WORKSPACE_RUNTIMES = 32
_workspace_runtimes: OrderedDict[str, ServerRuntime] = OrderedDict()
_workspace_runtime_lock = asyncio.Lock()
_sandbox_workspace_id: str | None = None


def _cached_workspace_runtime(binding: WorkspaceBinding) -> ServerRuntime | None:
    """Return and refresh a cached runtime for one workspace binding."""
    cached = _workspace_runtimes.get(binding.resource_key)
    if cached is None:
        return None
    _workspace_runtimes.move_to_end(binding.resource_key)
    return cached


def _claim_sandbox_workspace(
    sandbox_type: str | None,
    binding: WorkspaceBinding,
) -> None:
    """Reserve the process-wide sandbox for the first requesting workspace."""
    global _sandbox_workspace_id  # noqa: PLW0603  # process-lifetime ownership
    if not sandbox_type:
        return
    if _sandbox_workspace_id is None:
        _sandbox_workspace_id = binding.workspace_id
        return
    if _sandbox_workspace_id == binding.workspace_id:
        return
    reason = (
        "a runtime for another workspace already exists and the configured "
        "sandbox is process-wide"
    )
    # Built into a local first: `raise X.from_reason(...)` reads as a
    # `from_reason` raise to ruff's DOC501.
    conflict = WorkspaceConflictError.from_reason(reason)
    raise conflict


def _remember_workspace_runtime(
    binding: WorkspaceBinding,
    runtime: ServerRuntime,
) -> None:
    """Cache one workspace runtime and enforce the bounded LRU size."""
    _workspace_runtimes[binding.resource_key] = runtime
    if len(_workspace_runtimes) > _MAX_WORKSPACE_RUNTIMES:
        _workspace_runtimes.popitem(last=False)


async def _default_workspace_binding(config: ServerConfig) -> WorkspaceBinding | None:
    """Resolve the launch workspace represented by the server configuration.

    Returns:
        The canonical launch binding, or `None` without a configured workspace.
    """
    if config.cwd is None:
        return None
    return await asyncio.to_thread(
        resolve_workspace,
        config.cwd,
        config.to_workspace_payload(),
        config_fingerprint=config.workspace_fingerprint(),
    )


async def _workspace_runtime(binding: WorkspaceBinding) -> ServerRuntime:
    """Build or reuse a runtime from the persisted workspace resource policy.

    Returns:
        The runtime selected by the binding's immutable resource key.
    """
    cached = _cached_workspace_runtime(binding)
    if cached is not None:
        return cached
    async with _workspace_runtime_lock:
        cached = _cached_workspace_runtime(binding)
        if cached is not None:
            return cached
        config = ServerConfig.from_env()
        current_config = dataclasses.replace(
            config,
            cwd=binding.cwd,
            project_root=binding.project_root,
        )
        if (
            current_config.workspace_fingerprint() != binding.config_fingerprint
            or current_config.to_workspace_payload() != binding.workspace_config()
        ):
            reason = "the server configuration changed after this workspace was bound"
            # Built into a local first: `raise X.from_reason(...)` reads as a
            # `from_reason` raise to ruff's DOC501.
            conflict = WorkspaceConflictError.from_reason(reason)
            raise conflict
        _claim_sandbox_workspace(current_config.sandbox_type, binding)
        project_context = ProjectContext(
            user_cwd=Path(binding.cwd),
            project_root=Path(binding.project_root) if binding.project_root else None,
        )
        runtime = await _make_graphs(
            config_override=current_config,
            project_context_override=project_context,
        )
        _remember_workspace_runtime(binding, runtime)
        return runtime


async def get_server_runtime() -> ServerRuntime:
    """Return resources shared by the graph and dcode operation routes.

    Builds once and caches. A construction failure is converted into a
    startup-error marker (scraped by the parent app process) before
    `sys.exit(1)`, which is right for the `langgraph.json` graph factory at
    startup. Callers in request scope must contain that exit -- `SystemExit` is a
    `BaseException` -- as `offload_api._execute_offload` does, mapping it to a 503
    rather than killing the server mid-request.

    Returns:
        The cached server runtime.
    """
    # Resolving the launch binding touches the filesystem and can raise, and
    # claiming the sandbox can refuse. Both run before `_get_runtime`, so they
    # sit outside its startup barrier and would exit without the marker the
    # parent app process scrapes. Emit it here instead.
    try:
        config = ServerConfig.from_env()
        binding = await _default_workspace_binding(config)
    except Exception as exc:  # noqa: BLE001  # startup barrier
        emit_startup_failure(exc)
        sys.exit(1)
    async with _workspace_runtime_lock:
        if binding is None:
            return await _get_runtime()
        cached = _cached_workspace_runtime(binding)
        if cached is not None:
            return cached
        _claim_sandbox_workspace(config.sandbox_type, binding)
        runtime = await _get_runtime()
        _remember_workspace_runtime(binding, runtime)
        return runtime


async def make_graph(
    config: dict[str, Any] | None = None,
    runtime: LangGraphServerRuntime[CLIContextSchema] | None = None,
) -> Any:  # noqa: ANN401
    """Return the graph after validating execution workspace context.

    Raises:
        ValueError: If execution context is missing or malformed.
    """
    execution = runtime.execution_runtime if runtime is not None else None
    if execution is not None:
        context = CLIContextSchema.from_payload(execution.context)
        thread_id = (config or {}).get("configurable", {}).get("thread_id")
        if context is None or not isinstance(thread_id, str) or not thread_id:
            msg = "A thread id and workspace context are required for execution."
            raise ValueError(msg)
        from deepagents_code.workspace import require_thread_workspace

        binding = await require_thread_workspace(thread_id, context.workspace)
        return (await _workspace_runtime(binding)).agent
    return (await get_server_runtime()).agent
