"""Server-side graph entry point for ``deepagents deploy``.

This module is referenced by the generated ``langgraph.json`` and exposes the
agent graph as a module-level variable that the LangGraph server loads and
serves.

Unlike the local ``server_graph.py`` (which reads ``DA_SERVER_*`` env vars),
this graph reads a co-located ``deploy_config.json`` that was bundled at deploy
time. It configures:

- Model and prompt
- Backend with scoped namespaces (Store backend by default)
- Sandbox with scoped lifecycle
- Custom tools
- Memory and skills
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Namespace resolution
# ---------------------------------------------------------------------------


def _resolve_scope(scope: str, config: dict[str, Any]) -> tuple[str, ...]:
    """Resolve a scope string to namespace components from runtime config.

    Args:
        scope: One of "assistant", "user", "thread", "user+thread".
        config: The full LangGraph runtime config dict.

    Returns:
        Tuple of scope identifiers.
    """
    c = config.get("configurable", {})

    if scope == "assistant":
        return (c.get("assistant_id", "default"),)
    elif scope == "user":
        auth_user = c.get("langgraph_auth_user", {})
        user_id = auth_user.get("identity", "default")
        return (user_id,)
    elif scope == "thread":
        return (c.get("thread_id", "default"),)
    elif scope == "user+thread":
        auth_user = c.get("langgraph_auth_user", {})
        user_id = auth_user.get("identity", "default")
        return (user_id, c.get("thread_id", "default"))
    else:
        logger.warning("Unknown scope %r, falling back to assistant", scope)
        return (c.get("assistant_id", "default"),)


def _make_namespace_factory(scope: str, prefix: str) -> Any:
    """Create a namespace factory for the StoreBackend.

    Returns a callable ``(BackendContext) -> tuple[str, ...]`` that resolves
    the namespace at runtime based on the configured scope.
    """

    def namespace_factory(ctx: Any) -> tuple[str, ...]:
        from langgraph.config import get_config

        config = get_config()
        scope_parts = _resolve_scope(scope, config)
        return (*scope_parts, prefix)

    return namespace_factory


# ---------------------------------------------------------------------------
# Tool loading
# ---------------------------------------------------------------------------


def _load_tools(deploy_config: dict[str, Any]) -> list[Any]:
    """Load tools based on deploy configuration."""
    tools_config = deploy_config.get("tools", {})
    tools: list[Any] = []

    if tools_config.get("http_request", True):
        from deepagents_cli.tools import http_request

        tools.append(http_request)

    if tools_config.get("fetch_url", True):
        from deepagents_cli.tools import fetch_url

        tools.append(fetch_url)

    if tools_config.get("web_search", True):
        try:
            from deepagents_cli.tools import web_search

            tools.append(web_search)
        except Exception:
            logger.warning("web_search tool not available (missing TAVILY_API_KEY?)")

    # Load custom tools from bundled module
    bundled_custom = tools_config.get("_bundled_custom")
    if bundled_custom:
        module_ref, var_name = bundled_custom.rsplit(":", 1)
        module_path = module_ref.lstrip("./")
        if module_path.endswith(".py"):
            module_path = module_path[:-3]
        try:
            mod = importlib.import_module(module_path)
            custom_tools = getattr(mod, var_name)
            if callable(custom_tools):
                custom_tools = custom_tools()
            tools.extend(custom_tools)
            logger.info("Loaded %d custom tool(s) from %s", len(custom_tools), bundled_custom)
        except Exception:
            logger.exception("Failed to load custom tools from %s", bundled_custom)

    return tools


# ---------------------------------------------------------------------------
# Sandbox setup
# ---------------------------------------------------------------------------


def _create_sandbox_backend(deploy_config: dict[str, Any]) -> Any | None:
    """Create a sandbox backend if configured.

    Returns a sandbox backend instance, or None if sandbox is disabled.
    """
    sandbox_config = deploy_config.get("sandbox")
    if sandbox_config is None:
        return None

    provider = sandbox_config.get("provider", "langsmith")

    try:
        from deepagents_cli.integrations.sandbox_factory import create_sandbox

        # TODO: Implement scoped sandbox lifecycle.
        #
        # Currently creates a single sandbox for the server process lifetime
        # (same as the local CLI). For proper scoping (thread/user/assistant),
        # the sandbox ID needs to be stored in the LangGraph Store keyed by
        # the scope, and reconnected on subsequent runs.
        #
        # The create-or-reconnect pattern:
        #   1. Resolve scope key from runtime config
        #   2. Look up existing sandbox ID in Store
        #   3. If found, reconnect; if not, create new and store ID
        #   4. For thread scope, clean up when thread is deleted
        sandbox_cm = create_sandbox(
            provider,
            sandbox_id=None,
            setup_script_path=sandbox_config.get("setup_script"),
        )
        backend = sandbox_cm.__enter__()  # noqa: PLC2801

        import atexit

        def _cleanup() -> None:
            sandbox_cm.__exit__(None, None, None)

        atexit.register(_cleanup)
        return backend
    except ImportError:
        logger.exception("Sandbox provider '%s' is not installed", provider)
        return None
    except Exception:
        logger.exception("Failed to create sandbox for provider '%s'", provider)
        return None


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def make_graph() -> Any:  # noqa: ANN401
    """Create the deployed agent graph from ``deploy_config.json``.

    Uses ``create_deep_agent`` directly (rather than ``create_cli_agent``)
    so we can pass a ``StoreBackend`` with the configured namespace factory
    for scoped file storage.
    """
    config_path = Path(__file__).parent / "deploy_config.json"
    if not config_path.exists():
        msg = f"deploy_config.json not found at {config_path}. Was the deploy bundle created correctly?"
        raise FileNotFoundError(msg)

    deploy_config = json.loads(config_path.read_text())

    # --- Model ---
    from deepagents._models import resolve_model

    model = resolve_model(deploy_config.get("model", "anthropic:claude-sonnet-4-6"))

    # --- Tools ---
    tools = _load_tools(deploy_config)

    # --- Sandbox ---
    sandbox_backend = _create_sandbox_backend(deploy_config)

    # --- Backend ---
    backend_config = deploy_config.get("backend", {})
    backend_type = backend_config.get("type", "store")
    ns_config = backend_config.get("namespace", {})
    ns_scope = ns_config.get("scope", "assistant")
    ns_prefix = ns_config.get("prefix", "filesystem")

    from deepagents.backends.store import StoreBackend

    if backend_type == "store":
        # StoreBackend with scoped namespace factory
        backend: Any = lambda rt: StoreBackend(  # noqa: E731
            rt,
            namespace=_make_namespace_factory(ns_scope, ns_prefix),
        )
    elif backend_type == "sandbox" and sandbox_backend is not None:
        # Sandbox IS the backend — file ops go through the sandbox
        backend = sandbox_backend
    elif backend_type == "custom":
        custom_path = backend_config.get("path")
        if not custom_path:
            msg = "Custom backend requires a 'path' field"
            raise ValueError(msg)
        module_ref, var_name = custom_path.rsplit(":", 1)
        module_path = module_ref.lstrip("./")
        if module_path.endswith(".py"):
            module_path = module_path[:-3]
        mod = importlib.import_module(module_path)
        backend = getattr(mod, var_name)
    else:
        # Fallback to store
        backend = lambda rt: StoreBackend(  # noqa: E731
            rt,
            namespace=_make_namespace_factory(ns_scope, ns_prefix),
        )

    # --- Memory sources ---
    memory_config = deploy_config.get("memory", {})
    memory_sources = memory_config.get("_bundled_sources", memory_config.get("sources", []))
    resolved_memory: list[str] = []
    for source in memory_sources:
        source_path = Path(__file__).parent / source
        if source_path.exists():
            resolved_memory.append(str(source_path))
        else:
            logger.warning("Memory source not found: %s", source_path)

    # --- Skills sources ---
    skills_config = deploy_config.get("skills", {})
    bundled_skills = skills_config.get("_bundled_sources", [])
    resolved_skills: list[str] = []
    for source in bundled_skills:
        source_path = Path(__file__).parent / source
        if source_path.exists():
            resolved_skills.append(str(source_path))
        else:
            logger.warning("Skills source not found: %s", source_path)

    # --- System prompt ---
    system_prompt = deploy_config.get("prompt")

    # --- Build the agent ---
    from deepagents.graph import create_deep_agent

    agent = create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        backend=backend,
        memory=resolved_memory or None,
        skills=resolved_skills or None,
        interrupt_on={},  # No HITL in deployment — auto-approve everything
    )

    return agent


try:
    graph = make_graph()
except Exception as exc:
    import traceback

    logger.critical("Failed to initialize deploy graph", exc_info=True)
    print(  # noqa: T201
        f"Failed to initialize deploy graph: {exc}\n{traceback.format_exc()}",
        file=sys.stderr,
    )
    sys.exit(1)
