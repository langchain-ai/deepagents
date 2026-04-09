"""String templates for generated deployment artifacts.

These templates are rendered by the bundler with values from
`~deepagents_cli.deploy.config.DeployConfig`.

The generated `deploy_graph.py` uses a `CompositeBackend` with two
read-only `StoreBackend` routes (memories and skills) and the configured sandbox
as the default writable backend.

There is no hub path and no custom Python tools.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Per-provider sandbox creation blocks
#
# Each block defines `_get_or_create_sandbox(cache_key) -> BackendProtocol`.
# The caller builds the cache_key from either the thread_id or the
# assistant_id depending on `[sandbox].scope`.
# using the canonical SDK init for that provider.
# ---------------------------------------------------------------------------

SANDBOX_BLOCK_LANGSMITH = '''\
from deepagents.backends.langsmith import LangSmithSandbox

_SANDBOXES: dict = {}


def _get_or_create_sandbox(cache_key):
    """Get or create a LangSmith sandbox cached by ``cache_key``."""
    if cache_key in _SANDBOXES:
        return _SANDBOXES[cache_key]

    from langsmith.sandbox import ResourceNotFoundError, SandboxClient

    api_key = (
        os.environ.get("LANGSMITH_SANDBOX_API_KEY")
        or os.environ.get("LANGSMITH_API_KEY")
        or os.environ["LANGCHAIN_API_KEY"]
    )
    client = SandboxClient(api_key=api_key)

    try:
        client.get_template(SANDBOX_TEMPLATE)
    except ResourceNotFoundError:
        client.create_template(name=SANDBOX_TEMPLATE, image=SANDBOX_IMAGE)

    sandbox = client.create_sandbox(template_name=SANDBOX_TEMPLATE)
    backend = LangSmithSandbox(sandbox)
    _SANDBOXES[cache_key] = backend
    logger.info(
        "Created LangSmith sandbox %s for key %s",
        sandbox.name,
        cache_key,
    )
    return backend
'''

SANDBOX_BLOCK_DAYTONA = '''\
from langchain_daytona import DaytonaSandbox

_SANDBOXES: dict = {}


def _get_or_create_sandbox(cache_key):
    """Get or create a Daytona sandbox cached by ``cache_key``."""
    if cache_key in _SANDBOXES:
        return _SANDBOXES[cache_key]

    from daytona import Daytona, CreateSandboxFromImageParams

    client = Daytona()
    sandbox = client.create(CreateSandboxFromImageParams(image=SANDBOX_IMAGE))
    backend = DaytonaSandbox(sandbox=sandbox)
    _SANDBOXES[cache_key] = backend
    logger.info("Created Daytona sandbox %s for cache_key %s", sandbox.id, cache_key)
    return backend
'''

SANDBOX_BLOCK_MODAL = '''\
from langchain_modal import ModalSandbox

_SANDBOXES: dict = {}


def _get_or_create_sandbox(cache_key):
    """Get or create a Modal sandbox cached by ``cache_key``."""
    if cache_key in _SANDBOXES:
        return _SANDBOXES[cache_key]

    import modal

    image = modal.Image.from_registry(SANDBOX_IMAGE)
    sb = modal.Sandbox.create(image=image)
    backend = ModalSandbox(sandbox=sb)
    _SANDBOXES[cache_key] = backend
    logger.info("Created Modal sandbox for cache_key %s", cache_key)
    return backend
'''

SANDBOX_BLOCK_RUNLOOP = '''\
from langchain_runloop import RunloopSandbox

_SANDBOXES: dict = {}


def _get_or_create_sandbox(cache_key):
    """Get or create a Runloop devbox cached by ``cache_key``."""
    if cache_key in _SANDBOXES:
        return _SANDBOXES[cache_key]

    from runloop_api_client import Runloop

    client = Runloop()
    devbox = client.devboxes.create_and_await_running()
    backend = RunloopSandbox(devbox=devbox)
    _SANDBOXES[cache_key] = backend
    logger.info("Created Runloop devbox %s for cache_key %s", devbox.id, cache_key)
    return backend
'''

SANDBOX_BLOCK_NONE = '''\
from deepagents.backends.state import StateBackend

_STATE_BACKEND: StateBackend | None = None


def _get_or_create_sandbox(cache_key):  # noqa: ARG001
    """No sandbox configured — fall back to a process-wide StateBackend."""
    global _STATE_BACKEND
    if _STATE_BACKEND is None:
        _STATE_BACKEND = StateBackend()
    return _STATE_BACKEND
'''

SANDBOX_BLOCKS = {
    "langsmith": (SANDBOX_BLOCK_LANGSMITH, None),
    "daytona": (SANDBOX_BLOCK_DAYTONA, "langchain-daytona"),
    "modal": (SANDBOX_BLOCK_MODAL, "langchain-modal"),
    "runloop": (SANDBOX_BLOCK_RUNLOOP, "langchain-runloop"),
    "none": (SANDBOX_BLOCK_NONE, None),
}
"""Map of provider -> (sandbox_block, requires_partner_package)."""


# ---------------------------------------------------------------------------
# MCP tools loader (only emitted when mcp.json is present)
# ---------------------------------------------------------------------------

MCP_TOOLS_TEMPLATE = '''\
async def _load_mcp_tools():
    """Load MCP tools from bundled config (http/sse only)."""
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
# deploy_graph.py — the generated server entry point
#
# Store layout (CompositeBackend with sandbox default + two read-only routes):
#
#   Mount          Namespace                         Writable
#   -------------  --------------------------------  --------
#   /memories/     (assistant_id, "memories")        no
#   /skills/       (assistant_id, "skills")          no
#   default        sandbox (per `[sandbox].scope`)    yes
#
# `make_graph` takes the `RunnableConfig` at factory time, pulls
# `assistant_id` from `config["configurable"]`, and uses it as the
# top-level namespace component so different assistants built from the
# same graph have isolated memories and skills.
#
# The bundler ships `_seed.json` containing both payloads; the factory
# seeds each namespace once per (process, assistant_id).
# ---------------------------------------------------------------------------

DEPLOY_GRAPH_TEMPLATE = '''\
"""Auto-generated deepagents deploy entry point.

Created by `deepagents deploy`. Do not edit manually — changes will be
overwritten on the next deploy.
"""

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from deepagents import create_deep_agent
from deepagents.backends.composite import CompositeBackend
from deepagents.backends.protocol import EditResult, SandboxBackendProtocol, WriteResult
from deepagents.backends.store import StoreBackend
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
)
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolRuntime

if TYPE_CHECKING:
    from langgraph.runtime import Runtime
    from langgraph_sdk.runtime import ServerRuntime

logger = logging.getLogger(__name__)

SANDBOX_TEMPLATE = {sandbox_template!r}
SANDBOX_IMAGE = {sandbox_image!r}

# Mount points inside the composite backend.
MEMORIES_PREFIX = "/memories/"
SKILLS_PREFIX = "/skills/"

# What to seed into the store on first run.
SEED_PATH = Path(__file__).parent / "_seed.json"

# AGENTS.md is loaded at build time and baked into this module as the
# system prompt. The same content is seeded into the store under the
# ``memories`` namespace so the agent can read /memories/AGENTS.md at
# runtime — writes to /memories/ and /skills/ are blocked by
# ReadOnlyStoreBackend.
SYSTEM_PROMPT = {system_prompt!r}


class SandboxSyncMiddleware(AgentMiddleware):
    """Sync skill files from the store into the sandbox filesystem.

    Downloads all files under the configured skill sources from the composite
    backend (which routes /skills/ to the store) and uploads them directly
    into the sandbox so scripts can be executed.
    """

    def __init__(self, *, backend, sources):
        self._backend = backend
        self._sources = sources
        self._synced_keys: set = set()

    def _get_backend(self, state, runtime, config):
        if callable(self._backend):
            tool_runtime = ToolRuntime(
                state=state,
                context=runtime.context,
                stream_writer=runtime.stream_writer,
                store=runtime.store,
                config=config,
                tool_call_id=None,
            )
            return self._backend(tool_runtime)
        return self._backend

    async def _collect_files(self, backend, path):
        """Recursively list all files under *path* via ls (not glob)."""
        result = await backend.als(path)
        files = []
        for entry in result.entries or []:
            if entry.get("is_dir"):
                files.extend(await self._collect_files(backend, entry["path"]))
            else:
                files.append(entry["path"])
        return files

    async def abefore_agent(self, state, runtime, config):
        backend = self._get_backend(state, runtime, config)
        if not isinstance(backend, CompositeBackend):
            return None
        sandbox = backend.default
        if not isinstance(sandbox, SandboxBackendProtocol):
            return None

        # Only sync once per sandbox instance
        cache_key = id(sandbox)
        if cache_key in self._synced_keys:
            return None
        self._synced_keys.add(cache_key)

        files_to_upload = []
        for source in self._sources:
            paths = await self._collect_files(backend, source)
            if not paths:
                continue
            responses = await backend.adownload_files(paths)
            for resp in responses:
                if resp.content is not None:
                    files_to_upload.append((resp.path, resp.content))

        if files_to_upload:
            results = await sandbox.aupload_files(files_to_upload)
            uploaded = sum(1 for r in results if r.error is None)
            logger.info(
                "Synced %d/%d skill files into sandbox",
                uploaded,
                len(files_to_upload),
            )

        return None

    def wrap_model_call(self, request, handler):
        return handler(request)

    async def awrap_model_call(self, request, handler):
        return await handler(request)


class ReadOnlyStoreBackend(StoreBackend):
    """StoreBackend that rejects all writes and edits."""

    _READ_ONLY_MSG = (
        "This path is read-only. /memories/ and /skills/ are managed by "
        "the deployment config — they cannot be edited at runtime."
    )

    def write(self, file_path, content):  # noqa: ARG002
        return WriteResult(error=self._READ_ONLY_MSG)

    async def awrite(self, file_path, content):  # noqa: ARG002
        return WriteResult(error=self._READ_ONLY_MSG)

    def edit(  # noqa: ARG002, FBT002
        self, file_path, old_string, new_string, replace_all=False,
    ):
        return EditResult(error=self._READ_ONLY_MSG)

    async def aedit(  # noqa: ARG002, FBT002
        self, file_path, old_string, new_string, replace_all=False,
    ):
        return EditResult(error=self._READ_ONLY_MSG)


_SEED_CACHE: dict | None = None


def _load_seed() -> dict:
    """Load and cache the bundled seed payload."""
    global _SEED_CACHE
    if _SEED_CACHE is not None:
        return _SEED_CACHE
    if not SEED_PATH.exists():
        _SEED_CACHE = {{"memories": {{}}, "skills": {{}}}}
        return _SEED_CACHE
    try:
        _SEED_CACHE = json.loads(SEED_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to parse _seed.json: %s", exc)
        _SEED_CACHE = {{"memories": {{}}, "skills": {{}}}}
    return _SEED_CACHE


# Per-(process, assistant_id) gate.
_SEEDED_ASSISTANTS: set[str] = set()


async def _seed_store_if_needed(store, assistant_id: str) -> None:
    """Seed memories + skills under ``assistant_id`` once per process."""
    if assistant_id in _SEEDED_ASSISTANTS:
        return
    _SEEDED_ASSISTANTS.add(assistant_id)

    seed = _load_seed()

    memories_ns = (assistant_id, "memories")
    for path, content in seed.get("memories", {{}}).items():
        if await store.aget(memories_ns, path) is None:
            await store.aput(
                memories_ns,
                path,
                {{"content": content, "encoding": "utf-8"}},
            )

    skills_ns = (assistant_id, "skills")
    for path, content in seed.get("skills", {{}}).items():
        if await store.aget(skills_ns, path) is None:
            await store.aput(
                skills_ns,
                path,
                {{"content": content, "encoding": "utf-8"}},
            )


{sandbox_block}

{mcp_tools_block}


def _make_namespace_factory(assistant_id: str, section: str):
    """Return a namespace factory closed over an assistant id + section."""
    def _factory(ctx):  # noqa: ARG001
        return (assistant_id, section)
    return _factory


SANDBOX_SCOPE = {sandbox_scope!r}


def _build_backend_factory(assistant_id: str):
    """Return a backend factory that builds the composite per invocation."""
    def _factory(ctx):  # noqa: ARG001
        from langgraph.config import get_config

        if SANDBOX_SCOPE == "assistant":
            cache_key = f"assistant:{{assistant_id}}"
        else:
            thread_id = get_config().get("configurable", {{}}).get("thread_id", "local")
            cache_key = f"thread:{{thread_id}}"
        sandbox_backend = _get_or_create_sandbox(cache_key)
        return CompositeBackend(
            default=sandbox_backend,
            routes={{
                MEMORIES_PREFIX: ReadOnlyStoreBackend(
                    namespace=_make_namespace_factory(assistant_id, "memories"),
                ),
                SKILLS_PREFIX: ReadOnlyStoreBackend(
                    namespace=_make_namespace_factory(assistant_id, "skills"),
                ),
            }},
        )
    return _factory


async def make_graph(config: RunnableConfig, runtime: "ServerRuntime"):
    """Async graph factory.

    Accepts the invocation's ``RunnableConfig`` so we can pull the
    ``assistant_id`` out of ``configurable`` and scope all store reads
    and writes under it. Seeds the memories + skills namespaces once per
    (process, assistant_id), then assembles the deep agent graph.
    """
    configurable = (config or {{}}).get("configurable", {{}}) or {{}}
    assistant_id = str(configurable.get("assistant_id") or {default_assistant_id!r})

    store = getattr(runtime, "store", None)
    if store is not None:
        await _seed_store_if_needed(store, assistant_id)

    tools: list = []
    {mcp_tools_load_call}

    backend_factory = _build_backend_factory(assistant_id)

    return create_deep_agent(
        model={model!r},
        system_prompt=SYSTEM_PROMPT,
        memory=[f"{{MEMORIES_PREFIX}}AGENTS.md"],
        skills=[SKILLS_PREFIX],
        tools=tools,
        backend=backend_factory,
        middleware=[
            SandboxSyncMiddleware(backend=backend_factory, sources=[SKILLS_PREFIX]),
        ],
    )


graph = make_graph
'''


# ---------------------------------------------------------------------------
# pyproject.toml
# ---------------------------------------------------------------------------

PYPROJECT_TEMPLATE = """\
[project]
name = {agent_name!r}
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "deepagents==0.5.1",
{extra_deps}]

[tool.setuptools]
py-modules = []
"""
