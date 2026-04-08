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
    """Load MCP tools from bundled config (http/sse only).

    Wrapped in a try/except so that an unreachable or misconfigured MCP
    server logs a warning and returns ``[]`` rather than crashing the
    entire graph factory. Individual server failures degrade gracefully.
    """
    import json
    from pathlib import Path

    mcp_path = Path(__file__).parent / "_mcp.json"
    if not mcp_path.exists():
        return []

    try:
        raw = json.loads(mcp_path.read_text())
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to parse _mcp.json: %s", exc)
        return []

    servers = raw.get("mcpServers", {})
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

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient(connections)
        return await client.get_tools()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to load MCP tools from %d server(s): %s",
            len(connections),
            exc,
        )
        return []
'''

# ---------------------------------------------------------------------------
# Per-provider sandbox creation blocks (hub template)
#
# Each block defines ``_get_or_create_sandbox(thread_id) -> BackendProtocol``
# using the canonical SDK init for that provider. Picked by the bundler
# from ``[sandbox].provider``.
# ---------------------------------------------------------------------------

SANDBOX_BLOCK_LANGSMITH = '''\
from deepagents.backends.langsmith import LangSmithSandbox

_SANDBOXES: dict = {}


def _get_or_create_sandbox(thread_id):
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
    logger.info("Created LangSmith sandbox %s for thread %s", sandbox.name, thread_id)
    return backend
'''

SANDBOX_BLOCK_DAYTONA = '''\
from langchain_daytona import DaytonaSandbox

_SANDBOXES: dict = {}


def _get_or_create_sandbox(thread_id):
    """Get or create a Daytona sandbox cached by ``thread_id``."""
    if thread_id in _SANDBOXES:
        return _SANDBOXES[thread_id]

    from daytona import Daytona, CreateSandboxFromImageParams

    client = Daytona()  # reads DAYTONA_API_KEY / DAYTONA_TARGET from env
    sandbox = client.create(CreateSandboxFromImageParams(image=SANDBOX_IMAGE))
    backend = DaytonaSandbox(sandbox=sandbox)
    _SANDBOXES[thread_id] = backend
    logger.info("Created Daytona sandbox %s for thread %s", sandbox.id, thread_id)
    return backend
'''

SANDBOX_BLOCK_MODAL = '''\
from langchain_modal import ModalSandbox

_SANDBOXES: dict = {}


def _get_or_create_sandbox(thread_id):
    """Get or create a Modal sandbox cached by ``thread_id``."""
    if thread_id in _SANDBOXES:
        return _SANDBOXES[thread_id]

    import modal

    image = modal.Image.from_registry(SANDBOX_IMAGE)
    sb = modal.Sandbox.create(image=image)
    backend = ModalSandbox(sandbox=sb)
    _SANDBOXES[thread_id] = backend
    logger.info("Created Modal sandbox for thread %s", thread_id)
    return backend
'''

SANDBOX_BLOCK_RUNLOOP = '''\
from langchain_runloop import RunloopSandbox

_SANDBOXES: dict = {}


def _get_or_create_sandbox(thread_id):
    """Get or create a Runloop devbox cached by ``thread_id``."""
    if thread_id in _SANDBOXES:
        return _SANDBOXES[thread_id]

    from runloop_api_client import Runloop

    client = Runloop()  # reads RUNLOOP_API_KEY from env
    devbox = client.devboxes.create_and_await_running()
    backend = RunloopSandbox(devbox=devbox)
    _SANDBOXES[thread_id] = backend
    logger.info("Created Runloop devbox %s for thread %s", devbox.id, thread_id)
    return backend
'''

SANDBOX_BLOCK_NONE = '''\
from deepagents.backends.state import StateBackend

_STATE_BACKEND: StateBackend | None = None


def _get_or_create_sandbox(thread_id):  # noqa: ARG001  # thread_id unused
    """No sandbox configured — fall back to a process-wide ``StateBackend``."""
    global _STATE_BACKEND
    if _STATE_BACKEND is None:
        _STATE_BACKEND = StateBackend()
    return _STATE_BACKEND
'''

# Map of provider -> (sandbox_block, requires_partner_package).
# ``requires_partner_package`` is added to the generated pyproject.toml.
SANDBOX_BLOCKS = {
    "langsmith": (SANDBOX_BLOCK_LANGSMITH, None),
    "daytona": (SANDBOX_BLOCK_DAYTONA, "langchain-daytona"),
    "modal": (SANDBOX_BLOCK_MODAL, "langchain-modal"),
    "runloop": (SANDBOX_BLOCK_RUNLOOP, "langchain-runloop"),
    "none": (SANDBOX_BLOCK_NONE, None),
}


# ---------------------------------------------------------------------------
# Generated deploy graph
#
# Composes the runtime backend graph from a ``deepagents.toml`` config:
#
#   - Composite default = sandbox (per provider) or StateBackend if
#     ``[sandbox].provider = "none"``.
#   - ``/agent_memories/`` mount = HubBackend or StoreBackend depending
#     on ``[agent_memories].backend``. Holds shared, agent-scoped files
#     (``AGENTS.md``, ``skills/...``).
#   - ``/user_memories/`` mount = StoreBackend with a ``(agent, user_id,
#     "user_memories")`` namespace. Holds per-user mutable files like
#     coding preferences.
#
# The graph is exported as an ``async make_graph`` factory so MCP tool
# loading can ``await`` cleanly. Tools loading and MCP loading are
# conditional blocks injected by the bundler.
# ---------------------------------------------------------------------------

DEPLOY_GRAPH_HUB_TEMPLATE = '''\
"""Auto-generated deepagents deploy entry point.

Created by ``deepagents deploy``. Do not edit manually — changes will be
overwritten on the next deploy.
"""

import logging
import os
from typing import TYPE_CHECKING

from deepagents import create_deep_agent
from deepagents.backends.composite import CompositeBackend
from deepagents.backends.store import StoreBackend

# ``HubBackend`` is bundled by ``deepagents deploy`` as a sibling module
# until it ships in the published deepagents package on PyPI.
from _deepagents_hub import HubBackend  # type: ignore[import-not-found]

if TYPE_CHECKING:
    from langgraph_sdk.runtime import ServerRuntime

logger = logging.getLogger(__name__)

AGENT_NAME = {agent_name!r}
HUB_REPO = {hub_repo!r}
SANDBOX_TEMPLATE = {sandbox_template!r}
SANDBOX_IMAGE = {sandbox_image!r}

# Mount points inside the composite backend.
AGENT_MEMORIES_PREFIX = "/agent_memories/"
USER_MEMORIES_PREFIX = "/user_memories/"

# Backend selection for /agent_memories/. Either "hub" (a LangSmith
# Prompt Hub repo, agent-scoped, versioned in the UI) or "store" (the
# LangGraph store with an agent-scoped namespace).
AGENT_MEMORIES_BACKEND = {agent_memories_backend!r}


def _agent_memories_namespace(runtime):  # noqa: ARG001
    """Agent-scoped namespace for /agent_memories/ when store-backed."""
    return (AGENT_NAME, "agent_memories")


def _user_memories_namespace(runtime):
    """Per-user namespace: ``(agent_name, user_id, "user_memories")``."""
    user_id = "anonymous"
    try:
        user = runtime.server_info.user
        if user is not None and user.identity:
            user_id = str(user.identity)
    except AttributeError:
        pass
    return (AGENT_NAME, user_id, "user_memories")


_hub: HubBackend | None = None


def _get_hub() -> HubBackend:
    global _hub
    if _hub is None:
        # ``LANGSMITH_API_KEY`` may be reserved by LangGraph Platform and
        # stripped from the deployed env, so prefer the explicit sandbox key.
        api_key = (
            os.environ.get("LANGSMITH_SANDBOX_API_KEY")
            or os.environ.get("LANGSMITH_API_KEY")
        )
        _hub = HubBackend.from_env(HUB_REPO, api_key=api_key)
    return _hub


def _get_agent_memories_backend():
    """Resolve /agent_memories/ to a hub or store backend based on config."""
    if AGENT_MEMORIES_BACKEND == "hub":
        return _get_hub()
    return StoreBackend(namespace=_agent_memories_namespace)


# The bundler ships an ``_agent_memories_seed.json`` artifact next to
# this file. Each entry is a ``{{path: content}}`` pair that gets uploaded
# into the configured agent_memories backend (hub or store) on first
# invocation via ``backend.upload_files`` — one code path regardless of
# the chosen backend. ``_AGENT_MEMORIES_SEEDED`` gates the work per
# process so repeated runs don't re-upload.
_AGENT_MEMORIES_SEEDED = False


async def _seed_agent_memories_if_needed() -> None:
    """Upload the bundled agent_memories seed into the configured backend.

    Resolves the agent_memories backend via `_get_agent_memories_backend`,
    reads ``_agent_memories_seed.json`` from the build dir, and calls
    ``backend.aupload_files`` once per process. Same code path for hub
    and store backends — both implement the async upload contract.

    No-op when the seed file is missing, the seed is empty, or the
    process has already seeded.
    """
    global _AGENT_MEMORIES_SEEDED
    if _AGENT_MEMORIES_SEEDED:
        return

    import json
    from pathlib import Path

    seed_path = Path(__file__).parent / "_agent_memories_seed.json"
    if not seed_path.exists():
        _AGENT_MEMORIES_SEEDED = True
        return

    try:
        seed = json.loads(seed_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to parse _agent_memories_seed.json: %s", exc)
        _AGENT_MEMORIES_SEEDED = True
        return

    if not seed:
        _AGENT_MEMORIES_SEEDED = True
        return

    backend = _get_agent_memories_backend()
    files = [(path, content.encode("utf-8")) for path, content in seed.items()]
    try:
        responses = await backend.aupload_files(files)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to seed /agent_memories/ via %s: %s",
            AGENT_MEMORIES_BACKEND,
            exc,
        )
        _AGENT_MEMORIES_SEEDED = True
        return

    failed = [r for r in responses if getattr(r, "error", None)]
    logger.info(
        "Seeded %d/%d /agent_memories/ entries into %s backend (%d failed)",
        len(responses) - len(failed),
        len(seed),
        AGENT_MEMORIES_BACKEND,
        len(failed),
    )
    _AGENT_MEMORIES_SEEDED = True


{sandbox_block}

def _build_backend(runtime):
    from langgraph.config import get_config

    thread_id = get_config().get("configurable", {{}}).get("thread_id", "local")
    sandbox_backend = _get_or_create_sandbox(str(thread_id))
    return CompositeBackend(
        default=sandbox_backend,
        routes={{
            AGENT_MEMORIES_PREFIX: _get_agent_memories_backend(),
            USER_MEMORIES_PREFIX: StoreBackend(namespace=_user_memories_namespace),
        }},
    )


{tools_import_block}

{mcp_tools_block}


async def make_graph(runtime):
    """Async graph factory.

    Built async so MCP tool loading can ``await`` cleanly. Static tools
    are loaded synchronously above; everything else is composed here
    per-invocation (cheap — backends and seed state are cached).
    """
    await _seed_agent_memories_if_needed()

    tools: list = []
    {tools_load_call}
    {mcp_tools_load_call}

    return create_deep_agent(
        model={model!r},
        system_prompt={system_prompt!r},
        memory={memory_sources!r},
        skills=[f"{{AGENT_MEMORIES_PREFIX}}skills/"],
        tools=tools,
        backend=_build_backend,
    )


graph = make_graph
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
