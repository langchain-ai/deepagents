"""Fleet run manifest schema and storage helpers.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Literal, cast

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
        source: How the channel was selected, such as `cli` or `environment`.
        metadata: Non-secret selection metadata useful to follow-up setup agents.
    """

    provider: str
    source: str
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FleetToolRequirement:
    """MCP tool requirement declared by a Fleet export.

    Args:
        scope: Fleet scope declaring the requirement, such as `root` or
            `subagent:<name>`.
        tool_name: Requested MCP tool name.
        loaded: Whether the loaded Fleet components currently expose this tool.
    """

    scope: str
    tool_name: str
    loaded: bool


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
        local_mcp_config_path: Assistant-local MCP config target for follow-up setup.
        tool_requirements: Fleet MCP tool requirements.
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

    target_path = str(config.home / ".mcp.json")
    tool_requirements = _fleet_tool_requirements(config.fleet_dir, components)
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
    requirements: list[FleetToolRequirement] = []
    seen: set[tuple[str, str]] = set()
    for entry in _fleet_tool_entries(fleet_dir):
        key = (entry.scope, entry.tool_name)
        if key in seen:
            continue
        seen.add(key)
        requirements.append(
            FleetToolRequirement(
                scope=entry.scope,
                tool_name=entry.tool_name,
                loaded=entry.tool_name in loaded_tool_names,
            )
        )
    return sorted(requirements, key=lambda requirement: (requirement.scope, requirement.tool_name))


@dataclass(frozen=True, slots=True)
class _ToolEntry:
    scope: str
    tool_name: str


def _fleet_tool_entries(fleet_dir: Path) -> list[_ToolEntry]:
    entries: list[_ToolEntry] = []
    _extend_tool_entries(entries, fleet_dir / "tools.json", scope="root")

    subagents_dir = fleet_dir / "subagents"
    try:
        subagent_dirs = sorted(path for path in subagents_dir.iterdir() if path.is_dir())
    except OSError:
        return entries

    for subagent_dir in subagent_dirs:
        _extend_tool_entries(
            entries,
            subagent_dir / "tools.json",
            scope=f"subagent:{subagent_dir.name}",
        )
    return entries


def _extend_tool_entries(
    entries: list[_ToolEntry],
    path: Path,
    *,
    scope: str,
) -> None:
    if not path.is_file():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        msg = f"Could not read Fleet MCP tool entries from {path}"
        raise FleetRunManifestValidationError(msg) from exc
    if not isinstance(data, Mapping):
        return

    raw_tools = data.get("tools")
    if not isinstance(raw_tools, list):
        return

    for raw_tool in raw_tools:
        if not isinstance(raw_tool, Mapping):
            continue
        tool = cast("Mapping[str, object]", raw_tool)
        name = tool.get("name")
        server_url = tool.get("mcp_server_url")
        if not isinstance(name, str) or not name:
            continue
        if not isinstance(server_url, str) or not server_url:
            continue
        entries.append(_ToolEntry(scope=scope, tool_name=name))


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
    }


def _tool_requirement_to_dict(requirement: FleetToolRequirement) -> dict[str, object]:
    return {
        "loaded": requirement.loaded,
        "scope": requirement.scope,
        "tool_name": requirement.tool_name,
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
        metadata={str(key): _string_value(item, path=path) for key, item in metadata.items()},
    )


def _tool_requirement_from_dict(value: object, *, path: Path) -> FleetToolRequirement:
    data = _require_mapping(value, "tool_requirements[]", path=path)
    return FleetToolRequirement(
        scope=_required_str(data, "scope", path=path),
        tool_name=_required_str(data, "tool_name", path=path),
        loaded=_required_bool(data, "loaded", path=path),
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
