"""Plugin manifest parsing for plugins."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path, PureWindowsPath

from deepagents_code.plugins._json import json_object
from deepagents_code.plugins.agent_plugins import (
    AGENT_PLUGIN_FORMAT,
    AGENT_PLUGIN_MANIFEST_SCHEMA,
    AgentPluginError,
    validate_agent_plugin_manifest,
)
from deepagents_code.plugins.models import (
    ComponentInventory,
    JsonObject,
    PluginManifest,
    UnsupportedComponent,
)

logger = logging.getLogger(__name__)

_AGENT_PLUGIN_MANIFEST_PATH = Path("plugin.json")
_MANIFEST_RELATIVE_PATHS = (
    _AGENT_PLUGIN_MANIFEST_PATH,
    Path(".claude-plugin") / "plugin.json",
    Path(".codex-plugin") / "plugin.json",
)
_PATH_COMPONENT_FIELDS = {"skills", "mcpServers", "hooks"}
_UNSUPPORTED_COMPONENT_DIRS: tuple[UnsupportedComponent, ...] = (
    "agents",
    "commands",
)
_NAME_RE = re.compile(r"^[^\s]+$")


class PluginManifestError(ValueError):
    """Raised when a plugin manifest is malformed enough to skip the plugin."""


def _reject_nonstandard_constant(value: str) -> None:
    """Reject non-standard JSON constants (NaN, Infinity) during strict decode.

    Raises:
        AgentPluginError: Always, naming the offending constant.
    """
    msg = f"invalid JSON constant {value!r}"
    raise AgentPluginError(msg)


def find_manifest_path(root: Path) -> Path | None:
    """Return the first supported manifest path under `root`, if present.

    Args:
        root: Plugin root directory.

    Returns:
        Manifest path or `None`.
    """
    for rel in _MANIFEST_RELATIVE_PATHS:
        path = root / rel
        try:
            if path.is_file():
                return path
        except OSError:
            logger.warning("Could not inspect plugin manifest path %s", path)
    return None


def _validate_name(
    name: object, *, fallback: str | None = None, allow_at: bool = True
) -> str:
    """Validate a nonempty plugin name with no whitespace.

    Names such as `code-review` and `review@team` are valid; `code review` and
    the empty string are not.

    Returns:
        The validated name or fallback.

    Raises:
        PluginManifestError: If neither value is a valid name.
    """
    if (
        isinstance(name, str)
        and name
        and _NAME_RE.fullmatch(name)
        and (allow_at or "@" not in name)
    ):
        return name
    if fallback and _NAME_RE.fullmatch(fallback) and (allow_at or "@" not in fallback):
        return fallback
    msg = f"Invalid plugin name: {name!r}"
    raise PluginManifestError(msg)


def _is_windows_absolute(path: str) -> bool:
    return bool(PureWindowsPath(path).drive or PureWindowsPath(path).root)


def _resolve_component_path(
    declaration: str,
    plugin_root: Path,
    field_name: str,
    warnings: list[str],
) -> Path | None:
    if not declaration.startswith("./"):
        warnings.append(
            f"ignoring {field_name}: path must start with './' relative to plugin root"
        )
        return None
    relative = declaration[2:]
    if not relative:
        warnings.append(f"ignoring {field_name}: path must not be './'")
        return None
    path = Path(relative)
    if any(part == ".." for part in path.parts):
        warnings.append(f"ignoring {field_name}: path must not contain '..'")
        return None
    if path.is_absolute() or _is_windows_absolute(relative):
        warnings.append(f"ignoring {field_name}: path must stay within the plugin root")
        return None
    try:
        root_resolved = plugin_root.resolve()
        resolved = (plugin_root / path).resolve()
    except OSError as exc:
        warnings.append(
            f"ignoring {field_name}: could not resolve {declaration!r}: {exc}"
        )
        return None
    if not resolved.is_relative_to(root_resolved):
        warnings.append(f"ignoring {field_name}: path escapes plugin root")
        return None
    return resolved


def _resolve_component_paths(
    declaration: object,
    plugin_root: Path,
    field_name: str,
    warnings: list[str],
) -> tuple[Path, ...]:
    """Resolve one or more plugin-relative component paths.

    For example, `"./skills"` and `["./skills", "./extra-skills"]` are
    accepted. Absolute paths and paths containing `..` are rejected.

    Returns:
        Validated paths contained by the plugin root.
    """
    raw_paths: list[str]
    if isinstance(declaration, str):
        raw_paths = [declaration]
    elif isinstance(declaration, list):
        raw_paths = [item for item in declaration if isinstance(item, str)]
        warnings.extend(
            f"ignoring {field_name}: expected path string, got {type(item).__name__}"
            for item in declaration
            if not isinstance(item, str)
        )
    else:
        warnings.append(
            f"ignoring {field_name}: expected path string or list of strings"
        )
        return ()
    paths: list[Path] = []
    for raw_path in raw_paths:
        resolved = _resolve_component_path(raw_path, plugin_root, field_name, warnings)
        if resolved is not None:
            paths.append(resolved)
    return tuple(paths)


def _inline_mcp(value: object) -> JsonObject:
    if isinstance(value, dict):
        return json_object(value)
    if isinstance(value, list):
        merged: JsonObject = {}
        for item in value:
            if isinstance(item, dict):
                merged.update(json_object(item))
        return merged
    return {}


def _inline_hooks(value: object) -> JsonObject:
    """Normalize inline hooks to `hooks.json` document form.

    Returns:
        A wrapped hooks document, or an empty object.
    """
    if not isinstance(value, dict):
        return {}
    normalized = json_object(value)
    if not normalized:
        return {}
    wrapped = normalized.get("hooks")
    if isinstance(wrapped, dict):
        return {"hooks": wrapped}
    return {"hooks": normalized}


def load_manifest(
    root: Path, *, fallback_name: str | None = None
) -> tuple[PluginManifest | None, Path | None, tuple[str, ...]]:
    """Load an Agent Plugins, Claude, or Codex plugin manifest.

    Args:
        root: Plugin root directory.
        fallback_name: Name to use only when deriving a manifest-less plugin.

    Returns:
        `(manifest, manifest_path, warnings)`.

    Raises:
        PluginManifestError: If the manifest exists but is invalid.
    """
    manifest_path = find_manifest_path(root)
    if manifest_path is None:
        return None, None, ()
    is_root_manifest = manifest_path == root / _AGENT_PLUGIN_MANIFEST_PATH
    if is_root_manifest:
        # A root `plugin.json` is the Agent Plugins v1 manifest slot, but a
        # schema-less root manifest predates the v1 claim (legacy/auto-update
        # plugins). Sniff `$schema` to route: only a matching v1 schema uses the
        # strict v1 parser; anything else falls through to the legacy parser.
        try:
            if not manifest_path.resolve().is_relative_to(root.resolve()):
                msg = f"Agent Plugins manifest escapes plugin root: {manifest_path}"
                raise PluginManifestError(msg)
            raw_text = manifest_path.read_text(encoding="utf-8")
            decoded_root = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            msg = f"Invalid JSON syntax in {manifest_path}: {exc}"
            raise PluginManifestError(msg) from exc
        except (OSError, RuntimeError, UnicodeDecodeError) as exc:
            msg = f"Could not resolve Agent Plugins manifest {manifest_path}: {exc}"
            raise PluginManifestError(msg) from exc
        if isinstance(decoded_root, dict) and (
            decoded_root.get("$schema") == AGENT_PLUGIN_MANIFEST_SCHEMA
        ):
            try:
                # Re-decode strictly so non-standard JSON constants (NaN,
                # Infinity) are rejected as they are in `load_agent_plugin_manifest`.
                strict_decoded = json.loads(
                    raw_text, parse_constant=_reject_nonstandard_constant
                )
                raw_manifest, agent_plugin_warnings = validate_agent_plugin_manifest(
                    strict_decoded
                )
            except AgentPluginError as exc:
                raise PluginManifestError(str(exc)) from exc
            version_value = raw_manifest.get("version")
            manifest = PluginManifest(
                name=_validate_name(raw_manifest.get("name")),
                version=version_value if isinstance(version_value, str) else None,
                component_paths={},
                inline_mcp={},
                plugin_format=AGENT_PLUGIN_FORMAT,
            )
            return manifest, manifest_path, agent_plugin_warnings
    try:
        decoded = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"Invalid JSON syntax in {manifest_path}: {exc}"
        raise PluginManifestError(msg) from exc
    except (OSError, UnicodeDecodeError) as exc:
        msg = f"Could not read plugin manifest {manifest_path}: {exc}"
        raise PluginManifestError(msg) from exc
    if not isinstance(decoded, dict):
        msg = f"Plugin manifest {manifest_path} must be a JSON object"
        raise PluginManifestError(msg)
    raw_manifest = json_object(decoded)

    warnings: list[str] = []
    name = _validate_name(raw_manifest.get("name"), fallback=fallback_name)
    component_paths: dict[str, tuple[Path, ...]] = {}
    for field_name in _PATH_COMPONENT_FIELDS:
        declaration = raw_manifest.get(field_name)
        if declaration is None:
            continue
        if field_name in {"mcpServers", "hooks"} and isinstance(declaration, dict):
            continue
        paths = _resolve_component_paths(declaration, root, field_name, warnings)
        if paths:
            component_paths[field_name] = paths

    version_value = raw_manifest.get("version")
    version = version_value if isinstance(version_value, str) else None
    display_name_value = raw_manifest.get("displayName")
    auto_update_settings = raw_manifest.get("extensions")
    if isinstance(auto_update_settings, dict):
        auto_update_settings = auto_update_settings.get("com.langchain.deepagents.code")
    manifest = PluginManifest(
        name=name,
        version=version,
        component_paths=component_paths,
        inline_mcp=_inline_mcp(raw_manifest.get("mcpServers")),
        inline_hooks=_inline_hooks(raw_manifest.get("hooks")),
        display_name=(
            display_name_value if isinstance(display_name_value, str) else None
        ),
        auto_update=(
            isinstance(auto_update_settings, dict)
            and auto_update_settings.get("autoUpdate") is True
        ),
    )
    return manifest, manifest_path, tuple(warnings)


def _existing_component_path(path: Path, plugin_root: Path) -> tuple[Path, ...]:
    try:
        if not path.exists():
            return ()
        resolved = path.resolve()
        if not resolved.is_relative_to(plugin_root.resolve()):
            logger.warning("Ignoring plugin component outside plugin root: %s", path)
            return ()
    except OSError:
        logger.warning("Could not inspect plugin component path %s", path)
        return ()
    else:
        return (resolved,)


def _hooks_document_paths(path: Path, plugin_root: Path) -> tuple[Path, ...]:
    """Resolve a declared hooks file or directory.

    Returns:
        Existing hook document paths inside the plugin root.
    """
    try:
        target = path / "hooks.json" if path.is_dir() else path
    except OSError:
        logger.warning("Could not inspect plugin hooks path %s", path)
        return ()
    return _existing_component_path(target, plugin_root)


def _unsupported_component_dirs(
    plugin_root: Path,
) -> tuple[UnsupportedComponent, ...]:
    """Return present component dirs that deepagents-code does not load."""
    found: list[UnsupportedComponent] = []
    for name in _UNSUPPORTED_COMPONENT_DIRS:
        path = plugin_root / name
        try:
            if path.is_dir():
                found.append(name)
        except OSError:
            logger.warning("Could not inspect plugin component path %s", path)
    return tuple(found)


def _agent_plugin_skills(plugin_root: Path, warnings: list[str]) -> tuple[Path, ...]:
    skills_root = plugin_root / "skills"
    try:
        if not skills_root.exists():
            return ()
        resolved_root = skills_root.resolve()
        if not resolved_root.is_relative_to(plugin_root) or not resolved_root.is_dir():
            warnings.append("ignoring invalid Agent Plugins skills component")
            return ()
        children = sorted(resolved_root.iterdir(), key=lambda path: path.name)
    except (OSError, RuntimeError) as exc:
        warnings.append(f"ignoring Agent Plugins skills component: {exc}")
        return ()

    skills: list[Path] = []
    for child in children:
        skill_file = child / "SKILL.md"
        try:
            if not child.is_dir() or not skill_file.is_file():
                continue
            resolved = skill_file.resolve()
            if not resolved.is_relative_to(plugin_root):
                warnings.append(
                    f"ignoring Agent Plugins skill outside root: {child.name}"
                )
                continue
        except (OSError, RuntimeError) as exc:
            warnings.append(f"ignoring Agent Plugins skill {child.name}: {exc}")
            continue
        skills.append(resolved)
    return tuple(skills)


def _agent_plugin_mcp(plugin_root: Path, warnings: list[str]) -> tuple[Path, ...]:
    path = plugin_root / "mcp.json"
    try:
        if not path.exists():
            return ()
        resolved = path.resolve()
        if not resolved.is_relative_to(plugin_root) or not resolved.is_file():
            warnings.append("ignoring invalid Agent Plugins MCP component")
            return ()
    except (OSError, RuntimeError) as exc:
        warnings.append(f"ignoring Agent Plugins MCP component: {exc}")
        return ()
    return (resolved,)


def build_inventory(
    plugin_root: Path,
    manifest: PluginManifest | None,
    manifest_warnings: tuple[str, ...] = (),
) -> ComponentInventory:
    """Build component inventory for a plugin.

    Args:
        plugin_root: Plugin root directory.
        manifest: Parsed manifest or `None`.
        manifest_warnings: Warnings emitted during manifest parsing.

    Returns:
        Component inventory.
    """
    plugin_root = plugin_root.resolve()
    warnings = list(manifest_warnings)
    if manifest is not None and manifest.plugin_format == AGENT_PLUGIN_FORMAT:
        return ComponentInventory(
            skills=_agent_plugin_skills(plugin_root, warnings),
            mcp_files=_agent_plugin_mcp(plugin_root, warnings),
            warnings=tuple(warnings),
        )

    metadata_paths = manifest.component_paths if manifest else {}
    default_skills = _existing_component_path(plugin_root / "skills", plugin_root)
    root_skill = (
        ()
        if default_skills or (manifest and "skills" in manifest.component_paths)
        else _existing_component_path(plugin_root / "SKILL.md", plugin_root)
    )
    skills = (*default_skills, *metadata_paths.get("skills", ()), *root_skill)

    mcp_files = (
        *_existing_component_path(plugin_root / ".mcp.json", plugin_root),
        *metadata_paths.get("mcpServers", ()),
    )

    hook_files = (
        *_hooks_document_paths(plugin_root / "hooks", plugin_root),
        *(
            document
            for path in metadata_paths.get("hooks", ())
            for document in _hooks_document_paths(path, plugin_root)
        ),
    )

    unsupported = _unsupported_component_dirs(plugin_root)

    return ComponentInventory(
        skills=tuple(dict.fromkeys(skills)),
        mcp_files=tuple(dict.fromkeys(mcp_files)),
        hook_files=tuple(dict.fromkeys(hook_files)),
        unsupported=unsupported,
        warnings=tuple(warnings),
    )
