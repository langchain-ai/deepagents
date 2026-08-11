"""Schema-selected layouts for supported plugin dialects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deepagents_code.plugins.models import PluginDialectName

_AGENT_PLUGIN_SCHEMA_PREFIX = "https://agent-plugins.org/schemas/"
AGENT_PLUGIN_V1_SCHEMA = "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json"
AGENT_PLUGIN_V1_MCP_SCHEMA = "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json"


@dataclass(frozen=True, slots=True)
class PluginLayout:
    """Conventional component locations for a plugin dialect."""

    skill_paths: tuple[Path, ...]
    mcp_paths: tuple[Path, ...]
    hook_paths: tuple[Path, ...]
    root_skill_fallback: bool


@dataclass(frozen=True, slots=True)
class PluginDialect:
    """Manifest interpretation and component layout selected by schema."""

    name: PluginDialectName
    schema: str | None
    mcp_schema: str | None
    layout: PluginLayout
    supports_auto_update: bool


@dataclass(frozen=True, slots=True)
class ComponentDiscoveryPlan:
    """Ordered conventional and manifest-declared component candidates."""

    skill_paths: tuple[Path, ...]
    declared_skill_paths: tuple[Path, ...]
    mcp_paths: tuple[Path, ...]
    declared_mcp_paths: tuple[Path, ...]
    hook_paths: tuple[Path, ...]
    declared_hook_paths: tuple[Path, ...]
    root_skill_path: Path | None


CLAUDE_PLUGIN_DIALECT = PluginDialect(
    name="claude",
    schema=None,
    mcp_schema=None,
    layout=PluginLayout(
        skill_paths=(Path("skills"),),
        mcp_paths=(Path(".mcp.json"),),
        hook_paths=(Path("hooks") / "hooks.json",),
        root_skill_fallback=True,
    ),
    supports_auto_update=True,
)

AGENT_PLUGIN_V1_DIALECT = PluginDialect(
    name="agent-plugin-v1",
    schema=AGENT_PLUGIN_V1_SCHEMA,
    mcp_schema=AGENT_PLUGIN_V1_MCP_SCHEMA,
    layout=PluginLayout(
        skill_paths=(Path("skills"),),
        mcp_paths=(Path("mcp.json"),),
        hook_paths=(Path("hooks") / "hooks.json",),
        root_skill_fallback=False,
    ),
    supports_auto_update=False,
)

_DIALECTS_BY_NAME = {
    CLAUDE_PLUGIN_DIALECT.name: CLAUDE_PLUGIN_DIALECT,
    AGENT_PLUGIN_V1_DIALECT.name: AGENT_PLUGIN_V1_DIALECT,
}
_DIALECTS_BY_SCHEMA = {
    dialect.schema: dialect
    for dialect in _DIALECTS_BY_NAME.values()
    if dialect.schema is not None
}


class UnsupportedPluginSchemaError(ValueError):
    """Raised when a manifest targets an unsupported Agent Plugins schema."""


def dialect_for_schema(schema: object) -> PluginDialect:
    """Select a plugin dialect from a manifest schema value.

    Args:
        schema: Raw `$schema` value from the selected manifest.

    Returns:
        The registered Agent Plugin dialect for a supported schema, otherwise
        the permissive Claude-style dialect.

    Raises:
        UnsupportedPluginSchemaError: If an Agent Plugins schema is recognized
            but its version is unsupported.
    """
    if not isinstance(schema, str):
        return CLAUDE_PLUGIN_DIALECT
    dialect = _DIALECTS_BY_SCHEMA.get(schema)
    if dialect is not None:
        return dialect
    if schema.startswith(_AGENT_PLUGIN_SCHEMA_PREFIX):
        msg = f"Unsupported Agent Plugins schema: {schema}"
        raise UnsupportedPluginSchemaError(msg)
    return CLAUDE_PLUGIN_DIALECT


def accepts_mcp_schema(dialect: PluginDialect, schema: object) -> bool:
    """Return whether a component schema is compatible with its dialect.

    Missing and non-Agent schemas remain permissive. Explicit Agent Plugins
    schema versions must match the version selected by the package manifest.

    Args:
        dialect: Schema-selected plugin dialect.
        schema: Raw `$schema` value from an MCP document.

    Returns:
        Whether the MCP document can use the selected loading strategy.
    """
    if not isinstance(schema, str) or not schema.startswith(
        _AGENT_PLUGIN_SCHEMA_PREFIX
    ):
        return True
    return dialect.mcp_schema == schema


def dialect_by_name(name: PluginDialectName) -> PluginDialect:
    """Return the registered dialect for a normalized dialect name.

    Args:
        name: Normalized plugin dialect name.

    Returns:
        Registered plugin dialect.
    """
    return _DIALECTS_BY_NAME[name]
