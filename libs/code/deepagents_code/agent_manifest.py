"""Agent manifest parsing and materialization.

The manifest is a Fleet-compatible file tree:

- `AGENTS.md` for the main system prompt
- `tools.json` for MCP server configuration
- `subagents/<name>/AGENTS.md` for delegated agents
- `skills/<name>/...` for local skills
- `manifest.json` for structured runtime metadata such as `runtime.model`
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote, urljoin

logger = logging.getLogger(__name__)

MANIFEST_JSON = "manifest.json"
SYSTEM_PROMPT_FILE = "AGENTS.md"
TOOLS_FILE = "tools.json"
SUBAGENTS_DIR = "subagents"
SKILLS_DIR = "skills"
LOCAL_BACKEND = "local"
SUPPORTED_BACKENDS = frozenset(
    {"local", "agentcore", "daytona", "langsmith", "modal", "runloop"}
)
DEFAULT_FLEET_API_BASE = "https://api.smith.langchain.com"
FLEET_AGENT_PATH_TEMPLATE = "/api/v1/fleet/agents/{agent_id}?include_files=true"
UNSUPPORTED_TOP_LEVEL_FIELDS = frozenset({"permissions", "shared_users", "sharedUsers"})
UNSUPPORTED_BACKEND_FIELDS = frozenset(
    {"policy_ids", "policyIds", "PolicyIDs", "sandbox_policy_ids"}
)


class AgentManifestError(ValueError):
    """Raised when an agent manifest is malformed."""


@dataclass(frozen=True, slots=True)
class AgentManifest:
    """A local agent definition loaded from a Fleet-style manifest tree.

    Args:
        name: Optional display name from the manifest.
        runtime_model: Optional model spec from `runtime.model`.
        system_prompt: Main agent prompt from `AGENTS.md`.
        tools: MCP configuration from `tools.json`, if present.
        backend_type: Local backend selection. Fleet backend policy metadata is
            intentionally dropped on import.
        files: Relative manifest file paths mapped to UTF-8 text content.
        memory_paths: Optional memory path metadata.
        metadata: Supported structured manifest metadata.
    """

    name: str | None = None
    runtime_model: str | None = None
    system_prompt: str = ""
    tools: dict[str, Any] | None = None
    backend_type: str = LOCAL_BACKEND
    files: dict[str, str] = field(default_factory=dict)
    memory_paths: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AgentManifestImportResult:
    """Result from materializing a manifest into an agent directory.

    Args:
        agent_name: Local agent identifier.
        path: Directory that now contains the local agent manifest.
        dropped_fields: Unsupported Fleet fields removed during import.
        backend_type: Locally selected backend type.
    """

    agent_name: str
    path: Path
    dropped_fields: tuple[str, ...]
    backend_type: str


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from `path`.

    Args:
        path: JSON file path.

    Returns:
        Parsed object.

    Raises:
        AgentManifestError: If the file is not a JSON object.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"Could not read agent manifest JSON from {path}: {exc}"
        raise AgentManifestError(msg) from exc
    if not isinstance(raw, dict):
        msg = f"Agent manifest JSON must be an object: {path}"
        raise AgentManifestError(msg)
    return raw


def _write_json(path: Path, value: dict[str, Any]) -> None:
    """Write `value` as stable formatted JSON.

    Args:
        path: Destination JSON path.
        value: JSON object to write.
    """
    content = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.write_text(content, encoding="utf-8")


def _validate_relative_path(path: str) -> str:
    """Validate a manifest file path is relative and non-traversing.

    Args:
        path: Manifest-relative path.

    Returns:
        Normalized POSIX path.

    Raises:
        AgentManifestError: If `path` is absolute or escapes the manifest root.
    """
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        msg = f"Manifest file path must stay inside the manifest tree: {path!r}"
        raise AgentManifestError(msg)
    return candidate.as_posix()


def _normalise_files(raw: Any) -> dict[str, str]:  # noqa: ANN401
    """Convert Fleet `files` payloads into a relative-path mapping.

    Args:
        raw: Either `{path: content}` or a list of objects with `path` and
            `content` fields.

    Returns:
        Relative file path to text content.

    Raises:
        AgentManifestError: If the file payload shape is unsupported.
    """
    if raw is None:
        return {}

    files: dict[str, str] = {}
    if isinstance(raw, dict):
        for path, content in raw.items():
            if not isinstance(path, str) or not isinstance(content, str):
                msg = "Manifest files mapping must contain string paths and content"
                raise AgentManifestError(msg)
            files[_validate_relative_path(path)] = content
        return files

    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                msg = "Manifest files list must contain objects"
                raise AgentManifestError(msg)
            path = item.get("path")
            content = item.get("content")
            if not isinstance(path, str) or not isinstance(content, str):
                msg = "Manifest file entries require string `path` and `content`"
                raise AgentManifestError(msg)
            files[_validate_relative_path(path)] = content
        return files

    msg = "Manifest `files` must be an object or list"
    raise AgentManifestError(msg)


def _extract_runtime_model(data: dict[str, Any]) -> str | None:
    """Extract `runtime.model` from a manifest object.

    Args:
        data: Structured manifest metadata.

    Returns:
        Model spec, or `None` when absent.
    """
    runtime = data.get("runtime")
    if not isinstance(runtime, dict):
        return None
    model = runtime.get("model")
    if isinstance(model, str):
        return model
    if isinstance(model, dict):
        for key in ("model_id", "modelId", "id", "name"):
            value = model.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _extract_memory_paths(data: dict[str, Any]) -> tuple[str, ...]:
    """Extract supported memory path metadata.

    Args:
        data: Structured manifest metadata.

    Returns:
        Tuple of memory path strings.
    """
    memory = data.get("memory")
    raw = memory.get("paths") if isinstance(memory, dict) else data.get("memory_paths")
    if not isinstance(raw, list):
        return ()
    return tuple(path for path in raw if isinstance(path, str))


def _extract_backend_type(data: dict[str, Any], fallback: str = LOCAL_BACKEND) -> str:
    """Extract a supported local backend type from metadata.

    Args:
        data: Structured manifest metadata.
        fallback: Backend to use when metadata is absent or unsupported.

    Returns:
        Supported backend type.
    """
    backend = data.get("backend")
    if isinstance(backend, str):
        backend_type = backend
    elif isinstance(backend, dict):
        raw = backend.get("type") or backend.get("provider")
        backend_type = raw if isinstance(raw, str) else fallback
    else:
        backend_type = fallback
    return backend_type if backend_type in SUPPORTED_BACKENDS else fallback


def _drop_unsupported_fields(
    data: dict[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Remove hosted-only Fleet fields from structured metadata.

    Args:
        data: Structured manifest metadata.

    Returns:
        Tuple of cleaned metadata and dotted field names that were dropped.
    """
    cleaned = dict(data)
    dropped: list[str] = []

    for key in UNSUPPORTED_TOP_LEVEL_FIELDS:
        if key in cleaned:
            cleaned.pop(key, None)
            dropped.append(key)

    backend = cleaned.get("backend")
    if isinstance(backend, dict):
        backend_copy = dict(backend)
        for key in UNSUPPORTED_BACKEND_FIELDS:
            if key in backend_copy:
                backend_copy.pop(key, None)
                dropped.append(f"backend.{key}")
        cleaned["backend"] = backend_copy

    return cleaned, tuple(sorted(dropped))


def _is_oauth_mcp_server(config: Any) -> bool:  # noqa: ANN401
    """Return whether an MCP server uses hosted OAuth-only configuration."""
    if not isinstance(config, dict):
        return False
    for key, value in config.items():
        lowered = str(key).lower()
        if "oauth" in lowered:
            return True
        if lowered == "auth":
            if isinstance(value, str) and value.lower() == "oauth":
                return True
            if isinstance(value, dict):
                auth_type = value.get("type") or value.get("provider")
                if isinstance(auth_type, str) and auth_type.lower() == "oauth":
                    return True
    return False


def _drop_unsupported_tools(
    tools: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, tuple[str, ...]]:
    """Drop MCP servers that require hosted OAuth configuration.

    Args:
        tools: MCP configuration from `tools.json`.

    Returns:
        Cleaned tools plus dropped field notes.
    """
    if tools is None:
        return None, ()

    servers = tools.get("mcpServers")
    if not isinstance(servers, dict):
        return tools, ()

    cleaned_servers: dict[str, Any] = {}
    dropped: list[str] = []
    for name, server in servers.items():
        if _is_oauth_mcp_server(server):
            dropped.append(f"tools.mcpServers.{name}.oauth")
            continue
        cleaned_servers[name] = server

    cleaned = dict(tools)
    cleaned["mcpServers"] = cleaned_servers
    return cleaned, tuple(sorted(dropped))


def load_agent_manifest(path: Path) -> AgentManifest:
    """Load an agent manifest from a directory or JSON payload file.

    Args:
        path: Manifest directory, or JSON file containing a Fleet payload.

    Returns:
        Parsed agent manifest.
    """
    if path.is_dir():
        return _load_manifest_dir(path)
    data = _read_json(path)
    return parse_fleet_agent_payload(data)


def fetch_fleet_agent_manifest(
    agent_id: str,
    *,
    api_base: str | None = None,
    api_key: str | None = None,
    timeout: float = 30.0,
) -> AgentManifest:
    """Fetch and parse a Fleet agent manifest by id.

    Args:
        agent_id: Fleet agent id.
        api_base: LangSmith API base URL. Defaults to `LANGSMITH_ENDPOINT`, then
            the public LangSmith API.
        api_key: LangSmith API key. Defaults to `LANGSMITH_API_KEY` or
            `LANGCHAIN_API_KEY`.
        timeout: HTTP timeout in seconds.

    Returns:
        Parsed manifest.

    Raises:
        AgentManifestError: If credentials are missing or the request fails.
    """
    resolved_key = api_key or os.environ.get("LANGSMITH_API_KEY")
    if resolved_key is None:
        resolved_key = os.environ.get("LANGCHAIN_API_KEY")
    if not resolved_key:
        msg = (
            "Fleet import by id requires LANGSMITH_API_KEY or LANGCHAIN_API_KEY. "
            "Pass a local manifest path to import offline."
        )
        raise AgentManifestError(msg)

    base = api_base or os.environ.get("LANGSMITH_ENDPOINT") or DEFAULT_FLEET_API_BASE
    path = FLEET_AGENT_PATH_TEMPLATE.format(agent_id=quote(agent_id, safe=""))
    url = urljoin(base.rstrip("/") + "/", path.lstrip("/"))
    try:
        import httpx

        response = httpx.get(
            url,
            headers={"x-api-key": resolved_key},
            timeout=timeout,
        )
        response.raise_for_status()
    except Exception as exc:
        msg = f"Could not fetch Fleet agent {agent_id!r}: {exc}"
        raise AgentManifestError(msg) from exc

    try:
        payload = response.json()
    except ValueError as exc:
        msg = f"Fleet response for {agent_id!r} was not valid JSON"
        raise AgentManifestError(msg) from exc
    if not isinstance(payload, dict):
        msg = f"Fleet response for {agent_id!r} must be a JSON object"
        raise AgentManifestError(msg)
    return parse_fleet_agent_payload(payload)


def _load_manifest_dir(root: Path) -> AgentManifest:
    """Load a manifest from a local file tree.

    Args:
        root: Manifest root directory.

    Returns:
        Parsed manifest.
    """
    manifest_path = root / MANIFEST_JSON
    metadata = _read_json(manifest_path) if manifest_path.is_file() else {}
    files: dict[str, str] = {}

    for relative in (SYSTEM_PROMPT_FILE, TOOLS_FILE):
        path = root / relative
        if path.is_file():
            files[relative] = path.read_text(encoding="utf-8")

    for tree_name in (SUBAGENTS_DIR, SKILLS_DIR):
        tree = root / tree_name
        if not tree.is_dir():
            continue
        for file_path in tree.rglob("*"):
            if file_path.is_file():
                relative = file_path.relative_to(root).as_posix()
                files[relative] = file_path.read_text(encoding="utf-8")

    return _manifest_from_parts(metadata, files)


def parse_fleet_agent_payload(payload: dict[str, Any]) -> AgentManifest:
    """Parse a Fleet agent payload into a local manifest.

    Args:
        payload: Fleet response body. Supports either a top-level `files`
            collection or a nested `manifest.files` collection.

    Returns:
        Parsed manifest with unsupported hosted-only fields removed.
    """
    manifest = payload.get("manifest")
    metadata = dict(manifest) if isinstance(manifest, dict) else dict(payload)
    raw_files = metadata.pop("files", None)
    if raw_files is None:
        raw_files = payload.get("files")
    return _manifest_from_parts(metadata, _normalise_files(raw_files))


def _manifest_from_parts(
    metadata: dict[str, Any],
    files: dict[str, str],
) -> AgentManifest:
    """Build an `AgentManifest` from structured metadata and file content.

    Args:
        metadata: Structured manifest metadata.
        files: Manifest file contents keyed by relative path.

    Returns:
        Parsed manifest.

    Raises:
        AgentManifestError: If `tools.json` is malformed.
    """
    cleaned_metadata, metadata_dropped = _drop_unsupported_fields(metadata)
    system_prompt = files.get(SYSTEM_PROMPT_FILE, "")
    raw_tools = files.get(TOOLS_FILE)
    tools = None
    if raw_tools is not None:
        try:
            loaded_tools = json.loads(raw_tools)
        except json.JSONDecodeError as exc:
            msg = f"{TOOLS_FILE} must contain valid JSON: {exc}"
            raise AgentManifestError(msg) from exc
        if not isinstance(loaded_tools, dict):
            msg = f"{TOOLS_FILE} must contain a JSON object"
            raise AgentManifestError(msg)
        tools = loaded_tools

    cleaned_tools, tool_dropped = _drop_unsupported_tools(tools)
    if cleaned_tools is not None:
        files = dict(files)
        files[TOOLS_FILE] = json.dumps(cleaned_tools, indent=2, sort_keys=True) + "\n"

    name = cleaned_metadata.get("name")
    return AgentManifest(
        name=name if isinstance(name, str) else None,
        runtime_model=_extract_runtime_model(cleaned_metadata),
        system_prompt=system_prompt,
        tools=cleaned_tools,
        backend_type=_extract_backend_type(cleaned_metadata),
        files=files,
        memory_paths=_extract_memory_paths(cleaned_metadata),
        metadata={
            **cleaned_metadata,
            "_dropped_fields": [*metadata_dropped, *tool_dropped],
        },
    )


def materialize_agent_manifest(
    manifest: AgentManifest,
    target_dir: Path,
    *,
    agent_name: str,
    backend_type: str = LOCAL_BACKEND,
    force: bool = False,
) -> AgentManifestImportResult:
    """Write a manifest into a local agent directory.

    Args:
        manifest: Parsed manifest.
        target_dir: Destination directory.
        agent_name: Local agent identifier.
        backend_type: Operator-selected local backend type.
        force: Replace `target_dir` if it already exists.

    Returns:
        Import result with dropped-field notes.

    Raises:
        AgentManifestError: If the backend is unsupported or the destination
            already exists without `force`.
    """
    if backend_type not in SUPPORTED_BACKENDS:
        msg = f"Unsupported backend type: {backend_type!r}"
        raise AgentManifestError(msg)
    if target_dir.exists():
        if not force:
            msg = f"Agent directory already exists: {target_dir}"
            raise AgentManifestError(msg)
        shutil.rmtree(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)
    for relative, content in manifest.files.items():
        safe_relative = _validate_relative_path(relative)
        path = target_dir / safe_relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    if SYSTEM_PROMPT_FILE not in manifest.files:
        (target_dir / SYSTEM_PROMPT_FILE).write_text(
            manifest.system_prompt,
            encoding="utf-8",
        )
    if manifest.tools is not None and TOOLS_FILE not in manifest.files:
        _write_json(target_dir / TOOLS_FILE, manifest.tools)

    metadata = dict(manifest.metadata)
    metadata.pop("_dropped_fields", None)
    metadata["name"] = manifest.name or agent_name
    runtime = metadata.get("runtime")
    runtime_metadata = (
        cast("dict[str, Any]", runtime) if isinstance(runtime, dict) else {}
    )
    metadata["runtime"] = {
        **runtime_metadata,
        "model": {"model_id": manifest.runtime_model} if manifest.runtime_model else {},
    }
    metadata["backend"] = {"type": backend_type}
    if manifest.memory_paths:
        metadata["memory"] = {"paths": list(manifest.memory_paths)}
    _write_json(target_dir / MANIFEST_JSON, metadata)

    dropped = tuple(manifest.metadata.get("_dropped_fields", ()))
    return AgentManifestImportResult(
        agent_name=agent_name,
        path=target_dir,
        dropped_fields=dropped,
        backend_type=backend_type,
    )


def get_agent_manifest_paths(agent_dir: Path) -> tuple[Path | None, Path | None]:
    """Return model/MCP config paths for a local agent manifest.

    Args:
        agent_dir: Local agent directory.

    Returns:
        Tuple of `(manifest_json_path, tools_json_path)`, with absent paths as
        `None`.
    """
    manifest_path = agent_dir / MANIFEST_JSON
    tools_path = agent_dir / TOOLS_FILE
    return (
        manifest_path if manifest_path.is_file() else None,
        tools_path if tools_path.is_file() else None,
    )


def load_manifest_model(agent_dir: Path) -> str | None:
    """Load the model spec from a local agent manifest, if present.

    Args:
        agent_dir: Local agent directory.

    Returns:
        Model spec, or `None` when absent.
    """
    manifest_path, _tools_path = get_agent_manifest_paths(agent_dir)
    if manifest_path is None:
        return None
    try:
        return _extract_runtime_model(_read_json(manifest_path))
    except AgentManifestError:
        logger.warning("Could not load agent manifest model from %s", agent_dir)
        return None


def load_manifest_backend(agent_dir: Path) -> str | None:
    """Load the local backend type from a local agent manifest, if present.

    Args:
        agent_dir: Local agent directory.

    Returns:
        Backend type, or `None` when absent.
    """
    manifest_path, _tools_path = get_agent_manifest_paths(agent_dir)
    if manifest_path is None:
        return None
    try:
        backend = _extract_backend_type(_read_json(manifest_path), fallback="")
    except AgentManifestError:
        logger.warning("Could not load agent manifest backend from %s", agent_dir)
        return None
    return backend or None
