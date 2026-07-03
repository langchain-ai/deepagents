"""Fleet run manifest schema and storage helpers.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Literal, cast

from deepagents_talon.fleet_export import fleet_tool_entries

if TYPE_CHECKING:
    from deepagents_talon.config import TalonConfig
    from deepagents_talon.fleet import FleetAgentComponents

MANIFEST_FILENAME = "fleet-run-manifest.json"
MANIFEST_SCHEMA_VERSION = 1

logger = logging.getLogger(__name__)

ModelSource = Literal["environment", "fleet"]


class FleetRunManifestValidationError(ValueError):
    """Raised when a Fleet run manifest file does not match the expected schema."""


@dataclass(frozen=True, slots=True)
class ChannelSelection:
    """Channel selected for the local Talon run.

    Args:
        provider: Selected channel provider, such as `whatsapp` or `telegram`.
        source: How the channel was selected.
        status: Selection readiness status.
        metadata: Non-secret selection metadata useful to follow-up setup agents.
    """

    provider: str
    source: str
    status: str = "ready"
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FleetToolRequirement:
    """MCP tool requirement declared by a Fleet export.

    Args:
        id: Stable requirement id derived from the requirement content.
        scope: Fleet scope declaring the requirement, such as `root` or
            `subagent:<name>`.
        tool_names: Requested MCP tool names.
        mcp_server_url: MCP server URL with query string and fragment removed.
        auth_path: Fleet authentication path for the server.
        loaded_tool_names: Requested tool names exposed by loaded Fleet components.
        needs_local_mcp: Whether this Fleet MCP server needs a local MCP replacement.
        server_display_name: Optional human-readable MCP server name from Fleet.
        server_registry_name: Optional registry MCP server name from Fleet.
    """

    id: str
    scope: str
    tool_names: tuple[str, ...]
    mcp_server_url: str
    auth_path: str
    loaded_tool_names: tuple[str, ...]
    needs_local_mcp: bool = True
    server_display_name: str | None = None
    server_registry_name: str | None = None

    @property
    def tool_name(self) -> str:
        """Return the first requested tool name for legacy callers."""
        return self.tool_names[0]

    @property
    def loaded(self) -> bool:
        """Return whether any requested tool is exposed by loaded Fleet components."""
        return bool(self.loaded_tool_names)


@dataclass(frozen=True, slots=True)
class SetupTask:
    """Follow-up local setup task for a Fleet MCP tool requirement.

    Args:
        id: Stable task id derived from the requirement id and target path.
        kind: Setup task kind.
        target_path: Assistant-local config file follow-up agents should edit.
        tool_requirement_ids: Fleet tool requirements covered by this task.
    """

    id: str
    kind: str
    target_path: str
    tool_requirement_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FleetRunManifest:
    """Assistant-scoped manifest describing a local Fleet-backed Talon run.

    Args:
        schema_version: Manifest schema version.
        assistant_id: Assistant id this manifest belongs to.
        fleet_dir: Operator-unzipped Fleet export directory.
        selected_channel: Channel selected for the run, when known.
        model_source: Whether the runtime model came from Fleet or the environment.
        model: Resolved model id used by the runtime.
        created_at: Creation time in UTC ISO 8601 form.
        local_mcp_config_path: Assistant-local MCP config target for setup tasks.
        tool_requirements: Fleet MCP tool requirements.
        setup_tasks: Follow-up setup tasks.
    """

    schema_version: int
    assistant_id: str
    fleet_dir: str
    selected_channel: ChannelSelection | None
    model_source: ModelSource
    model: str
    created_at: str
    local_mcp_config_path: str
    tool_requirements: tuple[FleetToolRequirement, ...]
    setup_tasks: tuple[SetupTask, ...]


def manifest_path(assistant_home: Path) -> Path:
    """Return the stable manifest path for an assistant home.

    Args:
        assistant_home: Per-assistant home directory.

    Returns:
        Path to the Fleet run manifest file.
    """
    return assistant_home / MANIFEST_FILENAME


def build_fleet_run_manifest(
    config: TalonConfig,
    components: FleetAgentComponents,
    *,
    selected_channel: ChannelSelection | None = None,
    created_at: datetime | str | None = None,
) -> FleetRunManifest:
    """Build a manifest for the current Fleet-backed runtime state.

    Args:
        config: Talon runtime configuration.
        components: Loaded Fleet components.
        selected_channel: Optional selected channel metadata.
        created_at: Creation time. Defaults to the current UTC time.

    Returns:
        Manifest ready to write under the assistant home.

    Raises:
        ValueError: If the configuration does not include a Fleet directory.
    """
    if config.fleet_dir is None:
        msg = "Fleet run manifests require a configured Fleet directory"
        raise ValueError(msg)

    target_path = str(config.manifest_dir / "tools.json")
    tool_requirements = _fleet_tool_requirements(
        config.fleet_dir,
        components,
    )
    return FleetRunManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        assistant_id=config.assistant_id,
        fleet_dir=str(config.fleet_dir),
        selected_channel=selected_channel,
        model_source="environment" if config.model else "fleet",
        model=config.model or components.model,
        created_at=_created_at(created_at),
        local_mcp_config_path=target_path,
        tool_requirements=tuple(tool_requirements),
        setup_tasks=tuple(_setup_tasks(tool_requirements, target_path=target_path)),
    )


def refresh_fleet_run_manifest(
    config: TalonConfig,
    components: FleetAgentComponents,
    *,
    selected_channel: ChannelSelection | None = None,
    now: datetime | None = None,
) -> FleetRunManifest:
    """Write a refreshed Fleet run manifest while preserving its creation time.

    Args:
        config: Talon runtime configuration.
        components: Loaded Fleet components.
        selected_channel: Optional selected channel metadata.
        now: Current time override for deterministic tests.

    Returns:
        Written manifest.
    """
    path = manifest_path(config.home)
    created_at: datetime | str | None = now
    try:
        previous = load_fleet_run_manifest(path)
    except FileNotFoundError:
        previous = None
    except FleetRunManifestValidationError:
        logger.warning("Replacing invalid Fleet run manifest at %s", path)
        previous = None

    if previous is not None and previous.assistant_id == config.assistant_id:
        created_at = previous.created_at

    manifest = build_fleet_run_manifest(
        config,
        components,
        selected_channel=selected_channel,
        created_at=created_at,
    )
    write_fleet_run_manifest(path, manifest)
    return manifest


def write_fleet_run_manifest(path: Path, manifest: FleetRunManifest) -> None:
    """Write a Fleet run manifest as deterministic JSON.

    Args:
        path: Destination manifest path.
        manifest: Manifest payload to write.
    """
    parent = path.parent
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    parent.chmod(0o700)

    payload = json.dumps(_manifest_to_dict(manifest), indent=2, sort_keys=True)
    payload = f"{payload}\n"
    with NamedTemporaryFile("w", dir=parent, encoding="utf-8", delete=False) as file:
        temp = Path(file.name)
        file.write(payload)
    temp.chmod(0o600)
    temp.replace(path)


def load_fleet_run_manifest(path: Path) -> FleetRunManifest:
    """Load and validate a Fleet run manifest.

    Args:
        path: Manifest path.

    Returns:
        Parsed manifest.

    Raises:
        FleetRunManifestValidationError: If the manifest is malformed.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        msg = f"Fleet run manifest {path} is not valid JSON"
        raise FleetRunManifestValidationError(msg) from error
    except OSError:
        raise

    if not isinstance(data, Mapping):
        msg = f"Fleet run manifest {path} must contain a JSON object"
        raise FleetRunManifestValidationError(msg)
    return _manifest_from_dict(cast("Mapping[str, object]", data), path=path)


def _fleet_tool_requirements(
    fleet_dir: Path,
    components: FleetAgentComponents,
) -> list[FleetToolRequirement]:
    loaded_tool_names = _component_tool_names(components)
    grouped = defaultdict(list)
    for entry in fleet_tool_entries(fleet_dir):
        grouped.setdefault((entry.scope, entry.mcp_server_url), []).append(entry)

    requirements: list[FleetToolRequirement] = []
    for key in sorted(grouped):
        group = grouped[key]
        first = group[0]
        tool_names = tuple(sorted({entry.tool_name for entry in group}))
        loaded_names = tuple(name for name in tool_names if name in loaded_tool_names)
        auth_paths = sorted({entry.auth_path for entry in group})
        auth_path = auth_paths[0] if len(auth_paths) == 1 else "mixed"
        requirement_id = _stable_id("tool", first.scope, first.mcp_server_url, *tool_names)
        requirements.append(
            FleetToolRequirement(
                id=requirement_id,
                scope=first.scope,
                tool_names=tool_names,
                mcp_server_url=first.mcp_server_url,
                auth_path=auth_path,
                loaded_tool_names=loaded_names,
                server_display_name=_first_present(entry.server_display_name for entry in group),
                server_registry_name=_first_present(entry.server_registry_name for entry in group),
            )
        )
    return sorted(requirements, key=lambda requirement: requirement.id)


def _setup_tasks(
    tool_requirements: Sequence[FleetToolRequirement],
    *,
    target_path: str,
) -> list[SetupTask]:
    return [
        SetupTask(
            id=_stable_id("setup", requirement.id, target_path),
            kind="local_mcp_config",
            target_path=target_path,
            tool_requirement_ids=(requirement.id,),
        )
        for requirement in tool_requirements
        if requirement.needs_local_mcp
    ]


def _first_present(values: Iterable[str | None]) -> str | None:
    return next((value for value in values if value is not None), None)


def _component_tool_names(components: FleetAgentComponents) -> set[str]:
    names: set[str] = set()
    names.update(_tool_names(components.tools))
    for subagent in components.subagents:
        names.update(_tool_names(_subagent_tools(subagent)))
    return names


def _subagent_tools(subagent: object) -> Sequence[object]:
    if isinstance(subagent, Mapping):
        tools = cast("Mapping[str, object]", subagent).get("tools")
    else:
        tools = getattr(subagent, "tools", None)
    if isinstance(tools, Sequence) and not isinstance(tools, (str, bytes)):
        return tools
    return ()


def _tool_names(tools: Sequence[object]) -> set[str]:
    names: set[str] = set()
    for tool in tools:
        name = getattr(tool, "name", None)
        if not isinstance(name, str):
            name = getattr(tool, "__name__", None)
        if isinstance(name, str) and name:
            names.add(name)
    return names


def _stable_id(*parts: str) -> str:
    raw = "::".join(parts)
    digest = hashlib.sha256(raw.encode()).hexdigest()[:12]
    slug = "-".join(_slug(part) for part in parts if part)
    return f"{slug[:120]}-{digest}"


def _slug(value: str) -> str:
    return "-".join("".join(char.lower() if char.isalnum() else "-" for char in value).split("-"))


def _created_at(value: datetime | str | None) -> str:
    if isinstance(value, str):
        return value
    current = datetime.now(UTC) if value is None else value
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    return current.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _manifest_to_dict(manifest: FleetRunManifest) -> dict[str, object]:
    return {
        "assistant_id": manifest.assistant_id,
        "created_at": manifest.created_at,
        "fleet_dir": manifest.fleet_dir,
        "local_mcp_config_path": manifest.local_mcp_config_path,
        "model": manifest.model,
        "model_source": manifest.model_source,
        "schema_version": manifest.schema_version,
        "selected_channel": _channel_to_dict(manifest.selected_channel),
        "setup_tasks": [_setup_task_to_dict(task) for task in manifest.setup_tasks],
        "tool_requirements": [
            _tool_requirement_to_dict(requirement) for requirement in manifest.tool_requirements
        ],
    }


def _channel_to_dict(channel: ChannelSelection | None) -> dict[str, object] | None:
    if channel is None:
        return None
    return {
        "metadata": dict(sorted(channel.metadata.items())),
        "provider": channel.provider,
        "source": channel.source,
        "status": channel.status,
    }


def _tool_requirement_to_dict(requirement: FleetToolRequirement) -> dict[str, object]:
    data: dict[str, object] = {
        "auth_path": requirement.auth_path,
        "id": requirement.id,
        "loaded_tool_names": list(requirement.loaded_tool_names),
        "mcp_server_url": requirement.mcp_server_url,
        "needs_local_mcp": requirement.needs_local_mcp,
        "scope": requirement.scope,
        "tool_names": list(requirement.tool_names),
    }
    if requirement.server_display_name is not None:
        data["server_display_name"] = requirement.server_display_name
    if requirement.server_registry_name is not None:
        data["server_registry_name"] = requirement.server_registry_name
    return data


def _setup_task_to_dict(task: SetupTask) -> dict[str, object]:
    return {
        "id": task.id,
        "kind": task.kind,
        "target_path": task.target_path,
        "tool_requirement_ids": list(task.tool_requirement_ids),
    }


def _manifest_from_dict(data: Mapping[str, object], *, path: Path) -> FleetRunManifest:
    schema_version = _required_int(data, "schema_version", path=path)
    if schema_version != MANIFEST_SCHEMA_VERSION:
        msg = f"Fleet run manifest {path} has unsupported schema_version {schema_version}"
        raise FleetRunManifestValidationError(msg)

    model_source = _required_str(data, "model_source", path=path)
    if model_source not in {"environment", "fleet"}:
        msg = f"Fleet run manifest {path} has invalid model_source"
        raise FleetRunManifestValidationError(msg)

    return FleetRunManifest(
        schema_version=schema_version,
        assistant_id=_required_str(data, "assistant_id", path=path),
        fleet_dir=_required_str(data, "fleet_dir", path=path),
        selected_channel=_optional_channel(data.get("selected_channel"), path=path),
        model_source=cast("ModelSource", model_source),
        model=_required_str(data, "model", path=path),
        created_at=_required_str(data, "created_at", path=path),
        local_mcp_config_path=_required_str(data, "local_mcp_config_path", path=path),
        tool_requirements=tuple(
            _tool_requirement_from_dict(item, path=path)
            for item in _required_list(data, "tool_requirements", path=path)
        ),
        setup_tasks=tuple(
            _setup_task_from_dict(item, path=path)
            for item in _required_list(data, "setup_tasks", path=path)
        ),
    )


def _optional_channel(value: object, *, path: Path) -> ChannelSelection | None:
    if value is None:
        return None
    data = _require_mapping(value, "selected_channel", path=path)
    metadata = data.get("metadata", {})
    if not isinstance(metadata, Mapping):
        msg = f"Fleet run manifest {path} selected_channel.metadata must be an object"
        raise FleetRunManifestValidationError(msg)
    return ChannelSelection(
        provider=_required_str(data, "provider", path=path),
        source=_required_str(data, "source", path=path),
        status=_optional_str(data.get("status"), path=path) or "ready",
        metadata={str(key): _string_value(item, path=path) for key, item in metadata.items()},
    )


def _tool_requirement_from_dict(value: object, *, path: Path) -> FleetToolRequirement:
    data = _require_mapping(value, "tool_requirements[]", path=path)
    tool_names = _required_list(data, "tool_names", path=path)
    loaded_tool_names = _required_list(data, "loaded_tool_names", path=path)
    return FleetToolRequirement(
        id=_required_str(data, "id", path=path),
        scope=_required_str(data, "scope", path=path),
        tool_names=tuple(_string_value(item, path=path) for item in tool_names),
        mcp_server_url=_required_str(data, "mcp_server_url", path=path),
        auth_path=_required_str(data, "auth_path", path=path),
        loaded_tool_names=tuple(_string_value(item, path=path) for item in loaded_tool_names),
        needs_local_mcp=_required_bool(data, "needs_local_mcp", path=path),
        server_display_name=_optional_str(data.get("server_display_name"), path=path),
        server_registry_name=_optional_str(data.get("server_registry_name"), path=path),
    )


def _setup_task_from_dict(value: object, *, path: Path) -> SetupTask:
    data = _require_mapping(value, "setup_tasks[]", path=path)
    return SetupTask(
        id=_required_str(data, "id", path=path),
        kind=_required_str(data, "kind", path=path),
        target_path=_required_str(data, "target_path", path=path),
        tool_requirement_ids=tuple(
            _string_value(item, path=path)
            for item in _required_list(data, "tool_requirement_ids", path=path)
        ),
    )


def _required_str(data: Mapping[str, object], key: str, *, path: Path) -> str:
    value = data.get(key)
    if isinstance(value, str) and value:
        return value
    msg = f"Fleet run manifest {path} field {key!r} must be a non-empty string"
    raise FleetRunManifestValidationError(msg)


def _required_int(data: Mapping[str, object], key: str, *, path: Path) -> int:
    value = data.get(key)
    if isinstance(value, int):
        return value
    msg = f"Fleet run manifest {path} field {key!r} must be an integer"
    raise FleetRunManifestValidationError(msg)


def _required_bool(data: Mapping[str, object], key: str, *, path: Path) -> bool:
    value = data.get(key)
    if isinstance(value, bool):
        return value
    msg = f"Fleet run manifest {path} field {key!r} must be a boolean"
    raise FleetRunManifestValidationError(msg)


def _required_list(data: Mapping[str, object], key: str, *, path: Path) -> Sequence[object]:
    value = data.get(key)
    if isinstance(value, list):
        return value
    msg = f"Fleet run manifest {path} field {key!r} must be a list"
    raise FleetRunManifestValidationError(msg)


def _require_mapping(value: object, name: str, *, path: Path) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return cast("Mapping[str, object]", value)
    msg = f"Fleet run manifest {path} field {name!r} must be an object"
    raise FleetRunManifestValidationError(msg)


def _string_value(value: object, *, path: Path) -> str:
    if isinstance(value, str) and value:
        return value
    msg = f"Fleet run manifest {path} expected a non-empty string value"
    raise FleetRunManifestValidationError(msg)


def _optional_str(value: object, *, path: Path) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value:
        return value
    msg = f"Fleet run manifest {path} expected an optional non-empty string value"
    raise FleetRunManifestValidationError(msg)
