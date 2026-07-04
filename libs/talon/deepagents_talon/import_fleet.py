"""Fleet export import helpers for Talon.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast
from urllib.parse import urlsplit, urlunsplit

from deepagents_talon.config import TalonConfig


class FleetImportError(ValueError):
    """Raised when a Fleet export cannot be imported."""


@dataclass(frozen=True, slots=True)
class FleetReplacementTool:
    """Local MCP replacement needed for a Fleet-requested tool.

    Args:
        name: Fleet tool name.
        endpoint: Sanitized Fleet MCP server endpoint.
        auth_path: Fleet authentication path, when known.
        scope: Root agent or subagent scope that requested the tool.
    """

    name: str
    endpoint: str
    auth_path: str
    scope: str


@dataclass(frozen=True, slots=True)
class FleetSetupTask:
    """Operator task needed before local Talon runtime can replace Fleet MCP.

    Args:
        endpoint: Sanitized Fleet MCP server endpoint.
        auth_path: Fleet authentication path, when known.
        target: Local MCP config path the operator should edit.
        tool_names: Fleet tools requested from this endpoint.
    """

    endpoint: str
    auth_path: str
    target: str
    tool_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FleetImportSummary:
    """Result of importing a Fleet export into an assistant manifest.

    Args:
        fleet_dir: Validated Fleet export directory.
        assistant_id: Assistant id used to namespace Talon state.
        replacement_tool_count: Number of Fleet-requested tools requiring local replacement.
        setup_task_count: Number of local MCP replacement tasks.
        mcp_config_target: Assistant-scoped MCP config path written by the import.
        manifest_path: Manifest file written by the import.
        model_source: Whether the runtime model comes from Fleet or local env override.
    """

    fleet_dir: Path
    assistant_id: str
    replacement_tool_count: int
    setup_task_count: int
    mcp_config_target: Path
    manifest_path: Path
    model_source: Literal["fleet_config", "local_override"]


@dataclass(frozen=True, slots=True)
class FleetRunManifest:
    """Assistant-scoped Fleet run intent persisted by `import-fleet`.

    Args:
        fleet_dir: Validated Fleet export directory.
        assistant_id: Assistant id used to namespace Talon state.
        manifest_path: Manifest file that supplied the run intent.
        replacement_tool_count: Number of Fleet-requested tools requiring local replacement.
    """

    fleet_dir: Path
    assistant_id: str
    manifest_path: Path
    replacement_tool_count: int


@dataclass(frozen=True, slots=True)
class _FleetManifestContext:
    """Values shared by one manifest refresh."""

    target: Path
    fleet_dir: Path
    assistant_id: str
    model: str
    model_source: Literal["fleet_config", "local_override"]


def import_fleet_manifest(
    fleet_dir: Path,
    *,
    assistant_id: str,
    env: Mapping[str, str] | None = None,
) -> FleetImportSummary:
    """Write or refresh an assistant-scoped Fleet run manifest.

    Args:
        fleet_dir: Operator-unzipped Fleet export directory.
        assistant_id: Talon assistant id whose home receives the manifest.
        env: Environment mapping used to match runtime configuration.

    Returns:
        Summary of the import result.

    Raises:
        FleetImportError: If required Fleet export files are missing or malformed.
    """
    values = dict(os.environ if env is None else env)
    values["DEEPAGENTS_TALON_ASSISTANT_ID"] = assistant_id
    values["DEEPAGENTS_TALON_FLEET_DIR"] = str(fleet_dir)
    config = TalonConfig.from_env(values)

    source = _validate_fleet_dir(fleet_dir)
    fleet_config = _read_json_object(source / "config.json", label="Fleet config.json")
    fleet_model = _fleet_model(fleet_config)
    model_source: Literal["fleet_config", "local_override"] = (
        "local_override" if config.model else "fleet_config"
    )
    model = config.model or fleet_model

    config.ensure_home()
    target = config.manifest_dir / "tools.json"
    replacements = _replacement_tools(source, env=config.env)
    tasks = _setup_tasks(replacements, target=target)
    manifest = _manifest_payload(
        _FleetManifestContext(
            target=target,
            fleet_dir=source,
            assistant_id=config.assistant_id,
            model=model,
            model_source=model_source,
        ),
        replacements=replacements,
        tasks=tasks,
    )
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    target.chmod(0o600)

    return FleetImportSummary(
        fleet_dir=source,
        assistant_id=config.assistant_id,
        replacement_tool_count=len(replacements),
        setup_task_count=len(tasks),
        mcp_config_target=target,
        manifest_path=target,
        model_source=model_source,
    )


def load_fleet_run_manifest(
    *,
    assistant_id: str,
    env: Mapping[str, str] | None = None,
) -> FleetRunManifest:
    """Load and validate the assistant's Fleet run manifest.

    Args:
        assistant_id: Talon assistant id whose manifest should be loaded.
        env: Environment mapping used to resolve the assistant home.

    Returns:
        Validated Fleet run manifest.

    Raises:
        FleetImportError: If the manifest is missing, malformed, or points at a
            stale Fleet export directory.
    """
    values = dict(os.environ if env is None else env)
    values["DEEPAGENTS_TALON_ASSISTANT_ID"] = assistant_id
    config = TalonConfig.from_env(values)
    path = config.manifest_dir / "tools.json"
    if not path.is_file():
        msg = (
            f"Fleet run manifest not found for assistant {config.assistant_id!r}. "
            "Run `deepagents-talon import-fleet` first."
        )
        raise FleetImportError(msg)

    data = _read_json_object(path, label="Fleet run manifest")
    raw = data.get("manifest")
    if not isinstance(raw, Mapping):
        msg = "Malformed Fleet run manifest: missing manifest object"
        raise FleetImportError(msg)
    manifest = cast("Mapping[str, object]", raw)
    if manifest.get("source") != "fleet":
        msg = "Manifest is not a Fleet run manifest"
        raise FleetImportError(msg)

    manifest_assistant_id = manifest.get("assistant_id")
    if manifest_assistant_id != config.assistant_id:
        msg = (
            "Fleet run manifest assistant id does not match requested assistant "
            f"{config.assistant_id!r}"
        )
        raise FleetImportError(msg)

    fleet_dir = manifest.get("fleet_dir")
    if not isinstance(fleet_dir, str) or not fleet_dir:
        msg = "Malformed Fleet run manifest: fleet_dir is required"
        raise FleetImportError(msg)

    replacements = manifest.get("replacement_tools", ())
    if not isinstance(replacements, Sequence) or isinstance(replacements, (str, bytes, bytearray)):
        msg = "Malformed Fleet run manifest: replacement_tools must be an array"
        raise FleetImportError(msg)

    return FleetRunManifest(
        fleet_dir=_validate_fleet_dir(Path(fleet_dir)),
        assistant_id=config.assistant_id,
        manifest_path=path,
        replacement_tool_count=len(replacements),
    )


def _validate_fleet_dir(fleet_dir: Path) -> Path:
    source = fleet_dir.expanduser()
    try:
        source = source.resolve(strict=True)
    except OSError as exc:
        msg = f"Fleet directory does not exist: {fleet_dir}"
        raise FleetImportError(msg) from exc
    if not source.is_dir():
        msg = f"Fleet path is not a directory: {fleet_dir}"
        raise FleetImportError(msg)
    agents_md = source / "AGENTS.md"
    if not agents_md.is_file():
        msg = "Fleet export is missing required AGENTS.md"
        raise FleetImportError(msg)
    config_json = source / "config.json"
    if not config_json.is_file():
        msg = "Fleet export is missing required config.json"
        raise FleetImportError(msg)
    return source


def _read_json_object(path: Path, *, label: str) -> Mapping[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        msg = f"Could not read {label}: {path}"
        raise FleetImportError(msg) from exc
    except json.JSONDecodeError as exc:
        msg = f"Malformed {label}: {path}"
        raise FleetImportError(msg) from exc
    if not isinstance(data, Mapping):
        msg = f"Malformed {label}: expected a JSON object"
        raise FleetImportError(msg)
    return cast("Mapping[str, object]", data)


def _fleet_model(config: Mapping[str, object]) -> str:
    raw = config.get("model")
    if isinstance(raw, str) and raw:
        return raw
    agent = config.get("agent")
    if isinstance(agent, Mapping):
        nested = cast("Mapping[str, object]", agent).get("model")
        if isinstance(nested, str) and nested:
            return nested
    runtime = config.get("config")
    if isinstance(runtime, Mapping):
        configurable = cast("Mapping[str, object]", runtime).get("configurable")
        if isinstance(configurable, Mapping):
            model_config = cast("Mapping[str, object]", configurable).get("llm_model_config")
            if isinstance(model_config, Mapping):
                model_id = cast("Mapping[str, object]", model_config).get("modelId")
                if isinstance(model_id, str) and model_id:
                    return model_id
    msg = "Malformed Fleet config.json: missing model"
    raise FleetImportError(msg)


def _replacement_tools(
    fleet_dir: Path,
    *,
    env: Mapping[str, str],
) -> tuple[FleetReplacementTool, ...]:
    tools: list[FleetReplacementTool] = []
    tools.extend(_replacement_tools_from_file(fleet_dir / "tools.json", scope="root", env=env))

    subagents = fleet_dir / "subagents"
    if not subagents.is_dir():
        return tuple(tools)
    for child in sorted(path for path in subagents.iterdir() if path.is_dir()):
        tools.extend(
            _replacement_tools_from_file(
                child / "tools.json",
                scope=f"subagent:{child.name}",
                env=env,
            )
        )
    return tuple(sorted(tools, key=lambda tool: (tool.endpoint, tool.scope, tool.name)))


def _replacement_tools_from_file(
    path: Path,
    *,
    scope: str,
    env: Mapping[str, str],
) -> list[FleetReplacementTool]:
    if not path.is_file():
        return []
    data = _read_json_object(path, label=f"Fleet {path.name}")
    raw = data.get("tools")
    if raw is None:
        return []
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        msg = f"Malformed Fleet {path.name}: tools must be an array"
        raise FleetImportError(msg)

    tools: list[FleetReplacementTool] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, Mapping):
            msg = f"Malformed Fleet {path.name}: tools[{index}] must be an object"
            raise FleetImportError(msg)
        tool = cast("Mapping[str, object]", entry)
        name = tool.get("name")
        endpoint = tool.get("mcp_server_url")
        if not isinstance(name, str) or not name:
            msg = f"Malformed Fleet {path.name}: tools[{index}].name is required"
            raise FleetImportError(msg)
        if not isinstance(endpoint, str) or not endpoint:
            continue
        auth_path = _fleet_auth_path(tool, endpoint, env=env)
        if auth_path == "builtin" and env.get("BUILTIN_MCP_URL"):
            continue
        tools.append(
            FleetReplacementTool(
                name=name,
                endpoint=_safe_endpoint(endpoint),
                auth_path=auth_path,
                scope=scope,
            )
        )
    return tools


def _fleet_auth_path(
    tool: Mapping[str, object],
    endpoint: str,
    *,
    env: Mapping[str, str],
) -> str:
    raw = tool.get("auth_type") or tool.get("auth")
    if isinstance(raw, str) and raw in {"builtin", "headers", "oauth"}:
        return raw
    if _same_host(endpoint, env.get("BUILTIN_MCP_URL")):
        return "builtin"
    if "headers" in tool:
        return "headers"
    return "unknown"


def _setup_tasks(
    replacements: Sequence[FleetReplacementTool],
    *,
    target: Path,
) -> tuple[FleetSetupTask, ...]:
    grouped: dict[tuple[str, str], set[str]] = {}
    for tool in replacements:
        grouped.setdefault((tool.endpoint, tool.auth_path), set()).add(tool.name)
    return tuple(
        FleetSetupTask(
            endpoint=endpoint,
            auth_path=auth_path,
            target=str(target),
            tool_names=tuple(sorted(names)),
        )
        for (endpoint, auth_path), names in sorted(grouped.items())
    )


def _manifest_payload(
    context: _FleetManifestContext,
    *,
    replacements: Sequence[FleetReplacementTool],
    tasks: Sequence[FleetSetupTask],
) -> dict[str, object]:
    existing = _read_existing_manifest(context.target)
    return {
        "mcpServers": existing.get("mcpServers", {}),
        "manifest": {
            "source": "fleet",
            "fleet_dir": str(context.fleet_dir),
            "assistant_id": context.assistant_id,
            "model": context.model,
            "model_source": context.model_source,
            "replacement_tools": [asdict(tool) for tool in replacements],
            "setup_tasks": [asdict(task) for task in tasks],
        },
    }


def _read_existing_manifest(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        return {}
    data = _read_json_object(path, label="existing Talon manifest")
    servers = data.get("mcpServers")
    if servers is None:
        return {}
    if not isinstance(servers, Mapping):
        msg = "Malformed existing Talon manifest: mcpServers must be an object"
        raise FleetImportError(msg)
    return {"mcpServers": dict(cast("Mapping[str, object]", servers))}


def _safe_endpoint(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme and parsed.netloc:
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    return value.split("?", maxsplit=1)[0].split("#", maxsplit=1)[0]


def _same_host(left: str, right: str | None) -> bool:
    if not right:
        return False
    return urlsplit(left).hostname == urlsplit(right).hostname
