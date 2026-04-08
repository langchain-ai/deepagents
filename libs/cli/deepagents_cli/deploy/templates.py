"""String templates for generated deployment artifacts.

These templates are rendered by the bundler with values from
:class:`~deepagents_cli.deploy.config.DeployConfig`.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# deploy_graph.py — the server entry point that runs in LangGraph Platform
# ---------------------------------------------------------------------------

DEPLOY_GRAPH_TEMPLATE = '''\
"""Auto-generated deepagents deploy entry point.

Created by ``deepagents deploy``. Do not edit manually — changes will be
overwritten on the next deploy.
"""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from deepagents import create_deep_agent
from deepagents.backends.store import StoreBackend

if TYPE_CHECKING:
    from langgraph_sdk.runtime import ServerRuntime

logger = logging.getLogger(__name__)

BUNDLE_PATH = Path(__file__).parent / "_bundle.json"
AGENT_NAME = {agent_name!r}
MEMORY_SCOPE = {memory_scope!r}  # "agent" = shared, "user" = per-user

_store_seeded = False


def _get_namespace(assistant_id, user_identity=None):
    """Build the store namespace.

    - agent scope: ``(assistant_id, "filesystem")``
    - user scope: ``(assistant_id, user_identity, "filesystem")``
    """
    if MEMORY_SCOPE == "user" and user_identity:
        return (assistant_id, str(user_identity), "filesystem")
    return (assistant_id, "filesystem")


async def _seed_store(runtime: "ServerRuntime"):
    """Seed the store from _bundle.json (idempotent, runs once)."""
    global _store_seeded
    if _store_seeded:
        return
    _store_seeded = True

    if not BUNDLE_PATH.exists():
        logger.warning("No bundle file found at %s", BUNDLE_PATH)
        return

    store = runtime.store
    if store is None:
        logger.warning("No store available, skipping seed")
        return

    # Derive namespace from the default assistant_id
    namespace = (AGENT_NAME, "filesystem")
    bundle = json.loads(BUNDLE_PATH.read_text())
    for path, content in bundle.items():
        existing = await store.aget(namespace, path)
        if existing is None:
            await store.aput(namespace, path, {{"content": content, "encoding": "utf-8"}})
            logger.info("Seeded store: %s", path)
    logger.info("Store seeding complete (%d entries checked)", len(bundle))


def _namespace_factory(ctx):
    """Namespace factory for StoreBackend — called at runtime."""
    user_identity = None
    if MEMORY_SCOPE == "user" and ctx is not None:
        try:
            user = ctx.runtime.server_info.user
            if user is not None:
                user_identity = user.identity
        except AttributeError:
            logger.warning("server_info.user not available; falling back to agent scope")
    return _get_namespace(AGENT_NAME, user_identity)


{tools_import_block}

{mcp_tools_block}

{before_agent_block}

async def make_graph(runtime: "ServerRuntime"):
    """Graph factory — called by LangGraph Platform for each access context.

    Seeds the store from the bundled _bundle.json on first call,
    then assembles the deep agent graph. For execution contexts with
    sandbox enabled, creates a thread-scoped sandbox as the backend
    (matching the OpenSWE pattern).
    """
    await _seed_store(runtime)

    # Default backend: store for memory/skills
    backend = StoreBackend(runtime=None, namespace=_namespace_factory)

    {sandbox_backend_block}

    tools = []
    {tools_load_call}
    # Load MCP tools only during execution (they require async + network)
    if runtime.execution_runtime is not None:
        {mcp_tools_load_call}
        pass  # noqa: ensure block is never empty

    return create_deep_agent(
        model={model!r},
        system_prompt={system_prompt!r},
        memory={memory_sources!r},
        skills={skills_sources!r},
        tools=tools,
        backend=backend,
    )


graph = make_graph
'''

# ---------------------------------------------------------------------------
# Tools import block — included when [tools].python_file is set
# ---------------------------------------------------------------------------

TOOLS_IMPORT_TEMPLATE = '''\
def _load_tools():
    """Load custom tools from bundled Python file."""
    from {module_name} import {function_imports}
    return [{function_list}]
'''

TOOLS_IMPORT_AUTODISCOVER_TEMPLATE = '''\
def _load_tools():
    """Auto-discover @tool functions from bundled Python file."""
    import importlib
    from langchain_core.tools import BaseTool

    mod = importlib.import_module({module_name!r})
    tools = []
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, BaseTool):
            tools.append(obj)
    return tools
'''

# ---------------------------------------------------------------------------
# MCP tools block — included when [mcp].config is set
# ---------------------------------------------------------------------------

MCP_TOOLS_TEMPLATE = '''async def _load_mcp_tools():
    """Load MCP tools from bundled config (http/sse only)."""
    import json
    from pathlib import Path

    mcp_path = Path(__file__).parent / "_mcp.json"
    raw = json.loads(mcp_path.read_text())
    servers = raw.get("mcpServers", {})

    # Convert to langchain_mcp_adapters connection format
    connections = {}
    for name, cfg in servers.items():
        transport = cfg.get("type", cfg.get("transport", "stdio"))
        if transport in ("http", "sse"):
            conn = {"transport": transport, "url": cfg["url"]}
            if "headers" in cfg:
                conn["headers"] = cfg["headers"]
            connections[name] = conn

    if not connections:
        return []

    # Import lazily to avoid blocking scandir in dev server
    import asyncio
    loop = asyncio.get_running_loop()
    from langchain_mcp_adapters.client import MultiServerMCPClient
    client = MultiServerMCPClient(connections)
    return await client.get_tools()
'''

# ---------------------------------------------------------------------------
# Hub-backed deploy graph
#
# Used when ``[sandbox].provider != "none"`` and the project pushes its
# agent files to a LangSmith Prompt Hub repo at deploy time. The runtime
# composition mirrors ``examples/deploy-coding-agent/coding_agent.py``:
#
#   - ``HubBackend`` mounted at ``/longterm/`` for skills + AGENTS.md
#   - ``LangSmithSandbox`` as the default backend, cached by ``thread_id``
#     so each thread gets its own sandbox
#   - ``CompositeBackend`` routing the two above
#
# Skills + memory + .mcp.json are NOT bundled into the deployment image —
# they live in the hub repo, pushed by ``deepagents deploy`` itself.
# ---------------------------------------------------------------------------

DEPLOY_GRAPH_HUB_TEMPLATE = '''\
"""Auto-generated deepagents deploy entry point (hub-backed).

Created by ``deepagents deploy``. Do not edit manually — changes will be
overwritten on the next deploy.
"""

import logging
import os
from typing import TYPE_CHECKING

from deepagents import create_deep_agent
from deepagents.backends.composite import CompositeBackend
from deepagents.backends.langsmith import LangSmithSandbox

# ``HubBackend`` is bundled by ``deepagents deploy`` as a sibling module
# until it ships in the published deepagents package on PyPI.
from _deepagents_hub import HubBackend  # type: ignore[import-not-found]

if TYPE_CHECKING:
    from langchain.tools import ToolRuntime

    from deepagents.backends.protocol import BackendProtocol

logger = logging.getLogger(__name__)

AGENT_NAME = {agent_name!r}
HUB_REPO = {hub_repo!r}
SANDBOX_TEMPLATE = {sandbox_template!r}
SANDBOX_IMAGE = {sandbox_image!r}

# Mount points inside the composite backend.
HUB_PREFIX = "/longterm/"

_hub: HubBackend | None = None


def _get_hub() -> HubBackend:
    global _hub
    if _hub is None:
        # ``LANGSMITH_API_KEY`` is reserved by LangGraph Platform and stripped
        # from the deployed env, so prefer the explicit sandbox key.
        api_key = (
            os.environ.get("LANGSMITH_SANDBOX_API_KEY")
            or os.environ.get("LANGSMITH_API_KEY")
        )
        _hub = HubBackend.from_env(HUB_REPO, api_key=api_key)
    return _hub


# Process-local sandbox cache keyed by thread_id.
_SANDBOXES: dict = {{}}


def _get_or_create_sandbox(thread_id: str) -> LangSmithSandbox:
    """Get or create a LangSmith sandbox cached by ``thread_id``."""
    if thread_id in _SANDBOXES:
        return _SANDBOXES[thread_id]

    from langsmith.sandbox import ResourceNotFoundError, SandboxClient

    api_key = os.environ.get("LANGSMITH_SANDBOX_API_KEY") or os.environ["LANGSMITH_API_KEY"]
    client = SandboxClient(api_key=api_key)

    try:
        client.get_template(SANDBOX_TEMPLATE)
    except ResourceNotFoundError:
        client.create_template(name=SANDBOX_TEMPLATE, image=SANDBOX_IMAGE)

    sandbox = client.create_sandbox(template_name=SANDBOX_TEMPLATE)
    backend = LangSmithSandbox(sandbox)
    _SANDBOXES[thread_id] = backend
    logger.info("Created sandbox %s for thread %s", sandbox.name, thread_id)
    return backend


def _build_backend(runtime):
    from langgraph.config import get_config

    thread_id = get_config().get("configurable", {{}}).get("thread_id", "local")
    sandbox = _get_or_create_sandbox(str(thread_id))
    hub = _get_hub()
    return CompositeBackend(
        default=sandbox,
        routes={{
            HUB_PREFIX: hub,
        }},
    )


graph = create_deep_agent(
    model={model!r},
    system_prompt={system_prompt!r},
    memory={memory_sources!r},
    skills=[f"{{HUB_PREFIX}}skills/"],
    backend=_build_backend,
)
'''


# ---------------------------------------------------------------------------
# Before-agent middleware for sandbox + skills copying (coding agent pattern)
# ---------------------------------------------------------------------------

BEFORE_AGENT_SANDBOX_TEMPLATE = '''# Thread-scoped sandbox cache (matches OpenSWE pattern)
_SANDBOX_BACKENDS = {{}}


def _get_or_create_sandbox(thread_id):
    """Get or create a thread-scoped LangSmith sandbox.

    Sandboxes are cached per thread_id. Each thread gets its own
    isolated execution environment.
    """
    if thread_id in _SANDBOX_BACKENDS:
        return _SANDBOX_BACKENDS[thread_id]

    import os
    from deepagents.backends.langsmith import LangSmithSandbox
    from langsmith.sandbox import SandboxClient

    api_key = (
        os.environ.get("LANGSMITH_SANDBOX_API_KEY")
        or os.environ.get("LANGSMITH_API_KEY")
    )
    if not api_key:
        logger.warning("No LangSmith API key for sandbox creation")
        return None

    client = SandboxClient(api_key=api_key)
    template_name = {sandbox_template!r}

    # Ensure template exists
    from langsmith.sandbox import ResourceNotFoundError
    try:
        client.get_template(template_name)
    except ResourceNotFoundError:
        client.create_template(name=template_name, image={sandbox_image!r})

    sandbox = client.create_sandbox(template_name=template_name)
    backend = LangSmithSandbox(sandbox)
    logger.info("Created sandbox %s for thread %s", sandbox.name, thread_id)

    # Upload skills and memory into the sandbox
    if BUNDLE_PATH.exists():
        bundle = json.loads(BUNDLE_PATH.read_text())
        files_to_upload = [
            (path, content.encode("utf-8"))
            for path, content in bundle.items()
        ]
        if files_to_upload:
            backend.upload_files(files_to_upload)
            logger.info("Uploaded %d files to sandbox", len(files_to_upload))

    _SANDBOX_BACKENDS[thread_id] = backend
    return backend
'''

# ---------------------------------------------------------------------------
# langgraph.json
# ---------------------------------------------------------------------------

LANGGRAPH_JSON_TEMPLATE = '''\
{{
  "dependencies": ["."],
  "graphs": {{
    "agent": "./deploy_graph.py:graph"
  }},
  {env_line}
  "python_version": {python_version!r}
}}
'''

# ---------------------------------------------------------------------------
# pyproject.toml
# ---------------------------------------------------------------------------

PYPROJECT_TEMPLATE = '''\
[project]
name = {agent_name!r}
version = "0.1.0"
requires-python = ">={python_version}"
dependencies = [
    "deepagents==0.5.0a4",
{extra_deps}]

[tool.setuptools]
py-modules = []
'''
