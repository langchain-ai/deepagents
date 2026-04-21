"""Auto-generated deepagents deploy entry point.

Created by `deepagents deploy`. Do not edit manually — changes will be
overwritten on the next deploy.
"""

import json
import logging
import os
from typing import TYPE_CHECKING

from google.cloud import storage
from google.oauth2 import service_account

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

SANDBOX_SNAPSHOT = 'deepagents-deploy'
SANDBOX_IMAGE = 'python:3'

# Mount points inside the composite backend.
# Everything lives under /memories/ — longest-prefix-first routing
# ensures /memories/user/ and /memories/skills/ match before /memories/.
MEMORIES_PREFIX = "/memories/"
SKILLS_PREFIX = "/memories/skills/"
USER_PREFIX = "/memories/user/"

HAS_USER_MEMORIES = True

# What to seed into the store on first run — loaded from GCS.
# GCS_SEED_PATH is a gs://bucket/path/to/_seed.json URI.
# GCS_SERVICE_ACCOUNT_KEY is the JSON-encoded service account key.
GCS_SEED_PATH_ENV = "GCS_SEED_PATH"
GCS_SERVICE_ACCOUNT_KEY_ENV = "GCS_SERVICE_ACCOUNT_KEY"


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


def _empty_seed() -> dict:
    return {"memories": {}, "skills": {}, "user_memories": {}}


def _parse_gs_uri(uri: str) -> tuple[str, str]:
    """Parse a gs://bucket/object URI into (bucket, object)."""
    if not uri.startswith("gs://"):
        raise ValueError(f"GCS_SEED_PATH must start with gs://, got: {uri!r}")
    bucket, _, blob = uri[len("gs://"):].partition("/")
    if not bucket or not blob:
        raise ValueError(f"GCS_SEED_PATH must be gs://bucket/object, got: {uri!r}")
    return bucket, blob


def _load_seed() -> dict:
    """Load and cache the seed payload from Google Cloud Storage."""
    global _SEED_CACHE
    if _SEED_CACHE is not None:
        return _SEED_CACHE

    gcs_path = os.environ.get(GCS_SEED_PATH_ENV)
    key_json = os.environ.get(GCS_SERVICE_ACCOUNT_KEY_ENV)
    if not gcs_path or not key_json:
        logger.warning(
            "%s or %s not set; falling back to empty seed",
            GCS_SEED_PATH_ENV,
            GCS_SERVICE_ACCOUNT_KEY_ENV,
        )
        _SEED_CACHE = _empty_seed()
        return _SEED_CACHE

    try:
        key_info = json.loads(key_json)
        credentials = service_account.Credentials.from_service_account_info(key_info)
        client = storage.Client(
            project=key_info.get("project_id"),
            credentials=credentials,
        )
        bucket_name, blob_name = _parse_gs_uri(gcs_path)
        blob = client.bucket(bucket_name).blob(blob_name)
        content = blob.download_as_bytes()
        _SEED_CACHE = json.loads(content.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load seed from GCS (%s): %s", gcs_path, exc)
        _SEED_CACHE = _empty_seed()
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
    for path, content in seed.get("memories", {}).items():
        if await store.aget(memories_ns, path) is None:
            await store.aput(
                memories_ns,
                path,
                {"content": content, "encoding": "utf-8"},
            )

    skills_ns = (assistant_id,)
    for path, content in seed.get("skills", {}).items():
        if await store.aget(skills_ns, path) is None:
            await store.aput(
                skills_ns,
                path,
                {"content": content, "encoding": "utf-8"},
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
    user_memories = seed.get("user_memories", {})
    if not user_memories:
        return

    user_ns = (assistant_id, user_id)
    for path, content in user_memories.items():
        if await store.aget(user_ns, path) is None:
            await store.aput(
                user_ns,
                path,
                {"content": content, "encoding": "utf-8"},
            )
    logger.info(
        "Seeded %d user memory template(s) for user %s",
        len(user_memories),
        user_id,
    )


from deepagents.backends.state import StateBackend

_STATE_BACKEND: StateBackend | None = None


def _get_or_create_sandbox(cache_key):  # noqa: ARG001
    """No sandbox configured — fall back to a process-wide StateBackend."""
    global _STATE_BACKEND
    if _STATE_BACKEND is None:
        _STATE_BACKEND = StateBackend()
    return _STATE_BACKEND







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


SANDBOX_SCOPE = 'thread'


def _build_backend_factory(assistant_id: str):
    """Return a backend factory that builds the composite per invocation."""
    def _factory(ctx):  # noqa: ARG001
        from langgraph.config import get_config

        if SANDBOX_SCOPE == "assistant":
            cache_key = f"assistant:{assistant_id}"
        else:
            thread_id = get_config().get("configurable", {}).get("thread_id", "local")
            cache_key = f"thread:{thread_id}"
        sandbox_backend = _get_or_create_sandbox(cache_key)

        routes = {
            MEMORIES_PREFIX: StoreBackend(
                namespace=_make_namespace_factory(assistant_id),
            ),
            SKILLS_PREFIX: StoreBackend(
                namespace=_make_namespace_factory(assistant_id),
            ),
        }

        if HAS_USER_MEMORIES:
            routes[USER_PREFIX] = StoreBackend(
                namespace=_make_user_namespace_factory(assistant_id),
            )

        # Add subagent store routes for seeded sync subagents.
        seed = _load_seed()
        for sa_name in seed.get("subagents", {}):
            sa_prefix = f"{MEMORIES_PREFIX}subagents/{sa_name}/"
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
    configurable = (config or {}).get("configurable", {}) or {}
    assistant_id = str(configurable.get("assistant_id") or 'deepagents-deploy-parker')

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
    pass  # no MCP servers configured

    seed = _load_seed()
    all_subagents: list = []
    pass  # no sync subagents

    backend_factory = _build_backend_factory(assistant_id)

    # Preload AGENTS.md + user memory into the agent's context.
    memory_sources = [f"{MEMORIES_PREFIX}AGENTS.md"]
    if HAS_USER_MEMORIES and user_id:
        memory_sources.append(f"{USER_PREFIX}AGENTS.md")

    # AGENTS.md and skills are read-only; user memories are writable.
    permissions = [
        FilesystemPermission(
            operations=["write"],
            paths=[f"{MEMORIES_PREFIX}AGENTS.md", f"{SKILLS_PREFIX}**"],
            mode="deny",
        ),
    ]

    return create_deep_agent(
        model='anthropic:claude-sonnet-4-6',
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
