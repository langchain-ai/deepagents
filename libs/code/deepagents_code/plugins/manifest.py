"""Plugin manifest parsing for plugins."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path, PureWindowsPath

from deepagents_code.plugins._json import json_object
from deepagents_code.plugins.models import (
    ComponentInventory,
    JsonObject,
    PluginManifest,
    UnsupportedComponent,
)

logger = logging.getLogger(__name__)

_MANIFEST_RELATIVE_PATHS = (
    Path(".claude-plugin") / "plugin.json",
    Path(".codex-plugin") / "plugin.json",
)
_PATH_COMPONENT_FIELDS = {"skills", "mcpServers"}
_UNSUPPORTED_COMPONENT_DIRS: tuple[UnsupportedComponent, ...] = (
    "agents",
    "commands",
    "hooks",
)
_NAME_RE = re.compile(r"^[^\s]+$")


class PluginManifestError(ValueError):
    """Raised when a plugin manifest is malformed enough to skip the plugin."""


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


def load_manifest(
    root: Path, *, fallback_name: str | None = None
) -> tuple[PluginManifest | None, Path | None, tuple[str, ...]]:
    """Load a Claude/Codex plugin manifest.

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
    try:
        decoded = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"Invalid JSON syntax in {manifest_path}: {exc}"
        raise PluginManifestError(msg) from exc
    except OSError as exc:
        msg = f"Could not read plugin manifest {manifest_path}: {exc}"
        raise PluginManifestError(msg) from exc
    if not isinstance(decoded, dict):
        msg = f"Plugin manifest {manifest_path} must be a JSON object"
        raise PluginManifestError(msg)
    raw = json_object(decoded)

    warnings: list[str] = []
    name = _validate_name(raw.get("name"), fallback=fallback_name)
    component_paths: dict[str, tuple[Path, ...]] = {}
    for field_name in _PATH_COMPONENT_FIELDS:
        declaration = raw.get(field_name)
        if declaration is None:
            continue
        if field_name == "mcpServers" and isinstance(declaration, dict):
            continue
        paths = _resolve_component_paths(declaration, root, field_name, warnings)
        if paths:
            component_paths[field_name] = paths

    version_value = raw.get("version")
    version = version_value if isinstance(version_value, str) else None
    manifest = PluginManifest(
        name=name,
        version=version,
        component_paths=component_paths,
        inline_mcp=_inline_mcp(raw.get("mcpServers")),
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

    unsupported = _unsupported_component_dirs(plugin_root)

    return ComponentInventory(
        skills=tuple(dict.fromkeys(skills)),
        mcp_files=tuple(dict.fromkeys(mcp_files)),
        unsupported=unsupported,
        warnings=tuple(warnings),
    )
