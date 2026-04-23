"""String templates for generated deployment artifacts.

These templates are rendered by the bundler with values from
`~deepagents_cli.deploy.config.DeployConfig`.

The generated `deploy_graph.py` uses a `CompositeBackend` with all
managed content under `/memories/` — `/memories/AGENTS.md`,
`/memories/skills/`, and `/memories/user/` (per-user templates) —
backed by `StoreBackend` instances.  The configured sandbox is the
default writable backend.  Write access is controlled via
`FilesystemPermission` rules derived from each file's YAML frontmatter
`permissions` field.

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
_SANDBOX_FS_CAPACITY_BYTES = 16 * 1024**3


def _get_or_create_sandbox(cache_key):
    """Get or create a LangSmith sandbox cached by `cache_key`.

    Uses raw `os.environ` (not the CLI's `resolve_env_var`) because the
    deployed bundle cannot import `deepagents_cli` internals;
    `DEEPAGENTS_CLI_`-prefixed vars are not honored here.
    """
    if cache_key in _SANDBOXES:
        return _SANDBOXES[cache_key]

    from langsmith.sandbox import SandboxClient

    api_key = (
        os.environ.get("LANGSMITH_SANDBOX_API_KEY")
        or os.environ.get("LANGSMITH_API_KEY")
        or os.environ.get("LANGCHAIN_API_KEY")
    )
    if not api_key:
        raise RuntimeError(
            "No LangSmith sandbox API key found. Set "
            "LANGSMITH_SANDBOX_API_KEY, LANGSMITH_API_KEY, or LANGCHAIN_API_KEY."
        )
    client = SandboxClient(api_key=api_key)

    snapshot_id = os.environ.get("LANGSMITH_SANDBOX_SNAPSHOT_ID")
    if not snapshot_id:
        snapshot_name = (
            os.environ.get("LANGSMITH_SANDBOX_SNAPSHOT_NAME") or SANDBOX_SNAPSHOT
        )
        try:
            snapshots = client.list_snapshots()
        except Exception as e:
            raise RuntimeError(f"Failed to list snapshots: {e}") from e

        snapshot_id = None
        non_ready_status = None
        for snap in snapshots:
            if snap.name != snapshot_name:
                continue
            if snap.status == "ready":
                snapshot_id = snap.id
                break
            non_ready_status = snap.status

        if snapshot_id is None:
            if non_ready_status is not None:
                raise RuntimeError(
                    f"Snapshot {snapshot_name!r} exists but is "
                    f"in state {non_ready_status!r}. Wait for it to finish "
                    "building, or delete it to rebuild."
                )
            try:
                snapshot = client.create_snapshot(
                    name=snapshot_name,
                    docker_image=SANDBOX_IMAGE,
                    fs_capacity_bytes=_SANDBOX_FS_CAPACITY_BYTES,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to build snapshot {snapshot_name!r}: {e}"
                ) from e
            snapshot_id = snapshot.id

    try:
        sandbox = client.create_sandbox(snapshot_id=snapshot_id)
    except Exception as e:
        raise RuntimeError(
            f"Failed to create sandbox from snapshot {snapshot_id!r}: {e}"
        ) from e
    backend = LangSmithSandbox(sandbox)
    _SANDBOXES[cache_key] = backend
    logger.info(
        "Created LangSmith sandbox %s for key %s",
        sandbox.name,
        cache_key,
    )
    return backend
'''
"""Sandbox creation block for the LangSmith provider."""

SANDBOX_BLOCK_DAYTONA = '''\
from langchain_daytona import DaytonaSandbox

_SANDBOXES: dict = {}


def _get_or_create_sandbox(cache_key):
    """Get or create a Daytona sandbox cached by `cache_key`."""
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
"""Sandbox creation block for the Daytona provider."""

SANDBOX_BLOCK_MODAL = '''\
from langchain_modal import ModalSandbox

_SANDBOXES: dict = {}


def _get_or_create_sandbox(cache_key):
    """Get or create a Modal sandbox cached by `cache_key`."""
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
"""Sandbox creation block for the Modal provider."""

SANDBOX_BLOCK_RUNLOOP = '''\
from langchain_runloop import RunloopSandbox

_SANDBOXES: dict = {}


def _get_or_create_sandbox(cache_key):
    """Get or create a Runloop devbox cached by `cache_key`."""
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
"""Sandbox creation block for the Runloop provider."""

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
"""Fallback block used when no sandbox provider is configured."""

SANDBOX_BLOCKS = {
    "langsmith": (SANDBOX_BLOCK_LANGSMITH, None),
    "daytona": (SANDBOX_BLOCK_DAYTONA, "langchain-daytona"),
    "modal": (SANDBOX_BLOCK_MODAL, "langchain-modal"),
    "runloop": (SANDBOX_BLOCK_RUNLOOP, "langchain-runloop"),
    "none": (SANDBOX_BLOCK_NONE, None),
}
"""Map of `provider -> (sandbox_block, requires_partner_package)`."""

# ---------------------------------------------------------------------------
# Per-provider auth blocks
#
# Each block defines the `@auth.authenticate` handler for a provider.
# The shared `@auth.on` handler is appended to all providers automatically.
# ---------------------------------------------------------------------------

AUTH_ON_HANDLER = '''\


@auth.on.threads
async def add_owner(
    ctx: Auth.types.AuthContext,
    value: dict,
):
    """Scope all resources to the authenticated user."""
    if is_studio_user(ctx.user):
        return {}

    filters = {"owner": ctx.user.identity}
    metadata = value.setdefault("metadata", {})
    metadata.update(filters)
    return filters
'''

AUTH_BLOCK_SUPABASE = '''\
"""Supabase auth for LangGraph deploy.

Validates the Bearer token against Supabase's /auth/v1/user endpoint
and scopes resources per authenticated user.
"""

import os

import httpx
from langgraph_sdk import Auth
from langgraph_sdk.auth import is_studio_user

auth = Auth()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_PUBLISHABLE_DEFAULT_KEY = os.environ["SUPABASE_PUBLISHABLE_DEFAULT_KEY"]

_http_client = httpx.AsyncClient()


@auth.authenticate
async def get_current_user(
    authorization: str | None,
) -> Auth.types.MinimalUserDict:
    """Validate Supabase token and return user identity."""
    if not authorization or not authorization.startswith("Bearer "):
        raise Auth.exceptions.HTTPException(
            status_code=401, detail="Missing or invalid authorization header"
        )

    token = authorization.removeprefix("Bearer ").strip()

    response = await _http_client.get(
        f"{SUPABASE_URL}/auth/v1/user",
        headers={
            "Authorization": f"Bearer {token}",
            "apikey": SUPABASE_PUBLISHABLE_DEFAULT_KEY,
        },
    )

    if response.status_code != 200:
        raise Auth.exceptions.HTTPException(
            status_code=401, detail="Invalid or expired token"
        )

    user = response.json()
    return {
        "identity": user["id"],
        "display_name": user.get("email", ""),
    }
'''

AUTH_BLOCK_CLERK = '''\
"""Clerk auth for LangGraph deploy.

Fetches JWKS from Clerk's API, caches the signing keys, and verifies
the session JWT locally. Scopes resources per authenticated user.
"""

import os

import jwt as pyjwt
from jwt import PyJWKClient
from langgraph_sdk import Auth
from langgraph_sdk.auth import is_studio_user

auth = Auth()

CLERK_SECRET_KEY = os.environ["CLERK_SECRET_KEY"]

_jwks_client = PyJWKClient(
    "https://api.clerk.com/v1/jwks",
    headers={
        "Authorization": f"Bearer {CLERK_SECRET_KEY}",
        "User-Agent": "deepagents-deploy/1.0",
    },
)


@auth.authenticate
async def get_current_user(
    authorization: str | None,
) -> Auth.types.MinimalUserDict:
    """Validate Clerk session JWT and return user identity."""
    if not authorization or not authorization.startswith("Bearer "):
        raise Auth.exceptions.HTTPException(
            status_code=401, detail="Missing or invalid authorization header"
        )

    token = authorization.removeprefix("Bearer ").strip()

    try:
        signing_key = _jwks_client.get_signing_key_from_jwt(token)
        payload = pyjwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
        )
    except pyjwt.exceptions.PyJWTError as exc:
        raise Auth.exceptions.HTTPException(
            status_code=401, detail=f"Invalid token: {exc}"
        )

    return {
        "identity": payload["sub"],
        "display_name": payload.get("email", payload.get("name", "")),
    }
'''

AUTH_BLOCKS: dict[str, tuple[str, str | None]] = {
    "supabase": (AUTH_BLOCK_SUPABASE, None),
    "clerk": (AUTH_BLOCK_CLERK, "pyjwt"),
}
"""Map of auth provider -> (auth_block_template, optional_pip_dependency)."""


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
# Sync subagents loader (only emitted when sync subagents are present)
# ---------------------------------------------------------------------------

SYNC_SUBAGENTS_TEMPLATE = '''\
from deepagents.middleware.subagents import SubAgent


async def _build_sync_subagents(seed, store, assistant_id):
    """Build SubAgent dicts from seed data and seed their memories/skills."""
    subagents_data = seed.get("subagents", {})
    if not subagents_data:
        return []

    subagents = []
    for name, data in subagents_data.items():
        sa: SubAgent = {
            "name": data["config"]["name"],
            "description": data["config"]["description"],
            "system_prompt": data["memories"]["/AGENTS.md"],
        }
        if data["config"].get("model"):
            sa["model"] = data["config"]["model"]

        # Seed subagent memories and skills into store under subagent namespace.
        sa_ns = (assistant_id, "subagents", name)
        if store is not None:
            for path, content in data.get("memories", {}).items():
                if await store.aget(sa_ns, path) is None:
                    await store.aput(
                        sa_ns,
                        path,
                        {"content": content, "encoding": "utf-8"},
                    )
            for path, content in data.get("skills", {}).items():
                if await store.aget(sa_ns, path) is None:
                    await store.aput(
                        sa_ns,
                        path,
                        {"content": content, "encoding": "utf-8"},
                    )

        sa_prefix = f"/memories/subagents/{name}/"
        if data.get("skills"):
            sa["skills"] = [f"{sa_prefix}skills/"]

        if data.get("mcp"):
            sa["tools"] = await _load_subagent_mcp_tools(data["mcp"])

        # Restrict filesystem access to the subagent's own namespace.
        # Allow comes first (first-match wins); the deny rule blocks
        # everything else under /memories/ — parent AGENTS.md, skills, etc.
        sa["permissions"] = [
            FilesystemPermission(
                operations=["read", "write"],
                paths=[f"{sa_prefix}**"],
                mode="allow",
            ),
            FilesystemPermission(
                operations=["read", "write"],
                paths=["/memories/**"],
                mode="deny",
            ),
        ]

        subagents.append(sa)
    return subagents


async def _load_subagent_mcp_tools(mcp_config):
    """Load MCP tools for a subagent from its mcp config."""
    servers = mcp_config.get("mcpServers", {})
    connections = {}
    for sname, cfg in servers.items():
        transport = cfg.get("type", cfg.get("transport", "stdio"))
        if transport in ("http", "sse"):
            conn = {"transport": transport, "url": cfg["url"]}
            if "headers" in cfg:
                conn["headers"] = cfg["headers"]
            connections[sname] = conn

    if not connections:
        return []

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient(connections)
        return await client.get_tools()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load subagent MCP tools: %s", exc)
        return []
'''


# ---------------------------------------------------------------------------
# deploy_graph.py — the generated server entry point
#
# Store layout (CompositeBackend with sandbox default + routed stores):
#
#   Mount              Namespace                                   Writable
#   -----------------  ------------------------------------------  --------
#   /memories/user/    (assistant_id, user_id)   yes  [user AGENTS.md]
#   /memories/skills/  (assistant_id,)           no
#   /memories/         (assistant_id,)           no   [AGENTS.md]
#   default            sandbox (per scope)       yes
#
# `make_graph` takes the `RunnableConfig` at factory time, pulls
# `assistant_id` from `config["configurable"]`, and uses it as the
# top-level namespace component so different assistants built from the
# same graph have isolated memories and skills.
#
# User memories are namespaced per (assistant_id, user_id) so each
# user gets their own copy.  Template files are seeded on first access
# (only if not already present).  Write access is controlled per-file
# via frontmatter `permissions: read-write` declarations.
#
# The bundler ships `_seed.json` containing all payloads; the factory
# seeds each namespace once per (process, assistant_id) and user
# memories once per (process, assistant_id, user_id).
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
from deepagents.backends.protocol import SandboxBackendProtocol
from deepagents.backends.store import StoreBackend
from deepagents.middleware.permissions import FilesystemPermission
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

SANDBOX_SNAPSHOT = {sandbox_snapshot!r}
SANDBOX_IMAGE = {sandbox_image!r}

# Mount points inside the composite backend.
# Everything lives under /memories/ — longest-prefix-first routing
# ensures /memories/user/ and /memories/skills/ match before /memories/.
MEMORIES_PREFIX = "/memories/"
SKILLS_PREFIX = "/memories/skills/"
USER_PREFIX = "/memories/user/"

HAS_USER_MEMORIES = {has_user_memories!r}

# What to seed into the store on first run.
SEED_PATH = Path(__file__).parent / "_seed.json"


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


_SEED_CACHE: dict | None = None


def _load_seed() -> dict:
    """Load and cache the bundled seed payload."""
    global _SEED_CACHE
    if _SEED_CACHE is not None:
        return _SEED_CACHE
    if not SEED_PATH.exists():
        _SEED_CACHE = {{"memories": {{}}, "skills": {{}}, "user_memories": {{}}}}
        return _SEED_CACHE
    try:
        _SEED_CACHE = json.loads(SEED_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to parse _seed.json: %s", exc)
        _SEED_CACHE = {{"memories": {{}}, "skills": {{}}, "user_memories": {{}}}}
    return _SEED_CACHE


# Per-(process, assistant_id) gate.
_SEEDED_ASSISTANTS: set[str] = set()

# Per-(process, assistant_id, user_id) gate for user memories.
_SEEDED_USERS: set[tuple[str, str]] = set()


async def _seed_store_if_needed(store, assistant_id: str) -> None:
    """Seed memories + skills under `assistant_id` once per process."""
    if assistant_id in _SEEDED_ASSISTANTS:
        return
    _SEEDED_ASSISTANTS.add(assistant_id)

    seed = _load_seed()

    memories_ns = (assistant_id,)
    for path, content in seed.get("memories", {{}}).items():
        if await store.aget(memories_ns, path) is None:
            await store.aput(
                memories_ns,
                path,
                {{"content": content, "encoding": "utf-8"}},
            )

    skills_ns = (assistant_id,)
    for path, content in seed.get("skills", {{}}).items():
        if await store.aget(skills_ns, path) is None:
            await store.aput(
                skills_ns,
                path,
                {{"content": content, "encoding": "utf-8"}},
            )


async def _seed_user_memories_if_needed(
    store, assistant_id: str, user_id: str,
) -> None:
    """Seed user memory templates once per (assistant_id, user_id).

    Only writes entries that do not yet exist in the store, so
    user-modified memories are never overwritten.
    """
    key = (assistant_id, user_id)
    if key in _SEEDED_USERS:
        return
    _SEEDED_USERS.add(key)

    seed = _load_seed()
    user_memories = seed.get("user_memories", {{}})
    if not user_memories:
        return

    user_ns = (assistant_id, user_id)
    for path, content in user_memories.items():
        if await store.aget(user_ns, path) is None:
            await store.aput(
                user_ns,
                path,
                {{"content": content, "encoding": "utf-8"}},
            )
    logger.info(
        "Seeded %d user memory template(s) for user %s",
        len(user_memories),
        user_id,
    )


{sandbox_block}

{mcp_tools_block}

{sync_subagents_block}


def _make_namespace_factory(assistant_id: str, *extra: str):
    """Return a namespace factory closed over an assistant id + extra."""
    ns = (assistant_id, *extra)
    def _factory(ctx):  # noqa: ARG001
        return ns
    return _factory


def _make_user_namespace_factory(assistant_id: str):
    """Return a namespace factory that includes the user_id.

    Uses `rt.server_info.user.identity` from custom auth.  The platform
    always injects user_id from auth, so no configurable fallback is needed.
    """
    def _factory(rt):
        user = getattr(rt.server_info, "user", None) if rt.server_info else None
        identity = getattr(user, "identity", None) if user else None
        if not identity:
            raise ValueError(
                "user_id is required when user memories are enabled. "
                "Set it via custom auth (runtime.user.identity)."
            )
        return (assistant_id, str(identity))
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

        routes = {{
            MEMORIES_PREFIX: StoreBackend(
                namespace=_make_namespace_factory(assistant_id),
            ),
            SKILLS_PREFIX: StoreBackend(
                namespace=_make_namespace_factory(assistant_id),
            ),
        }}

        if HAS_USER_MEMORIES:
            routes[USER_PREFIX] = StoreBackend(
                namespace=_make_user_namespace_factory(assistant_id),
            )

        # Add subagent store routes for seeded sync subagents.
        seed = _load_seed()
        for sa_name in seed.get("subagents", {{}}):
            sa_prefix = f"{{MEMORIES_PREFIX}}subagents/{{sa_name}}/"
            routes[sa_prefix] = StoreBackend(
                namespace=_make_namespace_factory(assistant_id, "subagents", sa_name),
            )

        return CompositeBackend(
            default=sandbox_backend,
            routes=routes,
        )
    return _factory


async def make_graph(config: RunnableConfig, runtime: "ServerRuntime"):
    """Async graph factory.

    Accepts the invocation's `RunnableConfig` for `assistant_id` and
    the `ServerRuntime` for `store` and `user.identity`.  Seeds
    memories + skills once per (process, assistant_id), and user memories
    once per (process, assistant_id, user_id).  Gracefully skips user
    memory features when no user_id is available.
    """
    configurable = (config or {{}}).get("configurable", {{}}) or {{}}
    assistant_id = str(configurable.get("assistant_id") or {default_assistant_id!r})

    store = getattr(runtime, "store", None)
    user_id = None
    if HAS_USER_MEMORIES:
        user = getattr(runtime, "user", None)
        identity = getattr(user, "identity", None) if user else None
        user_id = str(identity) if identity else None
    if HAS_USER_MEMORIES and not user_id:
        logger.warning(
            "User memories are enabled but no user_id found "
            "(runtime.user.identity is empty). User memory features "
            "will be skipped for this invocation."
        )
    if store is not None:
        await _seed_store_if_needed(store, assistant_id)
        if HAS_USER_MEMORIES and user_id:
            await _seed_user_memories_if_needed(store, assistant_id, user_id)

    tools: list = []
    {mcp_tools_load_call}

    seed = _load_seed()
    all_subagents: list = []
    {sync_subagents_load_call}

    backend_factory = _build_backend_factory(assistant_id)

    # Preload AGENTS.md + user memory into the agent's context.
    memory_sources = [f"{{MEMORIES_PREFIX}}AGENTS.md"]
    if HAS_USER_MEMORIES and user_id:
        memory_sources.append(f"{{USER_PREFIX}}AGENTS.md")

    # AGENTS.md and skills are read-only; user memories are writable.
    permissions = [
        FilesystemPermission(
            operations=["write"],
            paths=[f"{{MEMORIES_PREFIX}}AGENTS.md", f"{{SKILLS_PREFIX}}**"],
            mode="deny",
        ),
    ]

    return create_deep_agent(
        model={model!r},
        memory=memory_sources,
        skills=[SKILLS_PREFIX],
        tools=tools,
        subagents=all_subagents or None,
        backend=backend_factory,
        permissions=permissions,
        middleware=[
            SandboxSyncMiddleware(backend=backend_factory, sources=[SKILLS_PREFIX]),
        ],
    )


graph = make_graph
'''
"""Generated `deploy_graph.py` source — the server entry point."""


# ---------------------------------------------------------------------------
# pyproject.toml
# ---------------------------------------------------------------------------

PYPROJECT_TEMPLATE = """\
[project]
name = {agent_name!r}
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "deepagents==0.5.3",
{extra_deps}]

[tool.setuptools]
py-modules = []
"""
"""Generated `pyproject.toml` source for the deployed bundle."""


# ---------------------------------------------------------------------------
# app.py (Starlette static mount)
# ---------------------------------------------------------------------------

APP_PY_TEMPLATE = '''\
"""Starlette app mounting the bundled chat UI on /app.

Generated by `deepagent deploy`. LangGraph Platform reads the `http.app`
key in `langgraph.json` and attaches this app alongside the graph.

Uses Starlette directly (not FastAPI) because Starlette is already a
transitive dep of langgraph-cli / langgraph-api in both the dev runtime
and the deployed runtime, whereas FastAPI would require an explicit
install step that `langgraph dev` does not perform.
"""

from __future__ import annotations

from pathlib import Path

from starlette.applications import Starlette
from starlette.responses import JSONResponse, RedirectResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

_FRONTEND_DIR = Path(__file__).parent / "frontend_dist"


async def healthz(_request):
    return JSONResponse({"ok": True})


async def app_root_redirect(_request):
    # Starlette's Mount at "/app" matches "/app/*" — a bare "/app" 404s
    # otherwise. Redirect so users typing the clean URL land correctly.
    return RedirectResponse(url="/app/", status_code=308)


app = Starlette(
    routes=[
        Route("/healthz", healthz),
        Route("/app", app_root_redirect),
        Mount(
            "/app",
            app=StaticFiles(directory=str(_FRONTEND_DIR), html=True),
            name="frontend",
        ),
    ],
)
'''
"""Generated `app.py` — a Starlette app that serves the frontend at /app."""
