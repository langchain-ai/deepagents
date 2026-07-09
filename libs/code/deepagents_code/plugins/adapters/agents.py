"""Adapter for plugin-provided subagents."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import yaml

from deepagents_code.plugins.substitution import substitute_string
from deepagents_code.subagents import SubagentMetadata

if TYPE_CHECKING:
    from pathlib import Path

    from deepagents_code.plugins.models import PluginInstance

logger = logging.getLogger(__name__)
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)
_IGNORED_AGENT_FIELDS = ("permissionMode", "hooks", "mcpServers")


def _agent_from_file(
    plugin: PluginInstance, path: Path, base: Path
) -> SubagentMetadata | None:
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Skipping plugin agent %s: %s", path, exc)
        return None
    match = _FRONTMATTER_RE.match(content)
    if match is None:
        logger.warning("Skipping plugin agent %s: missing frontmatter", path)
        return None
    raw = yaml.safe_load(match.group(1))
    if not isinstance(raw, dict):
        logger.warning("Skipping plugin agent %s: invalid frontmatter", path)
        return None
    description = raw.get("description") or raw.get("when-to-use")
    if not isinstance(description, str) or not description.strip():
        logger.warning("Skipping plugin agent %s: missing description", path)
        return None
    for field in _IGNORED_AGENT_FIELDS:
        if field in raw:
            logger.warning("Ignoring %s in plugin agent %s", field, path)
    declared_name = raw.get("name")
    base_name = (
        declared_name if isinstance(declared_name, str) and declared_name else path.stem
    )
    relative_parent = path.parent.relative_to(base)
    name_parts = (plugin.name, *relative_parent.parts, base_name)
    model = raw.get("model")
    prompt = substitute_string(
        match.group(2).strip(),
        plugin_root=plugin.root,
        plugin_data=plugin.data_dir,
        warning_key=plugin.plugin_id,
    )
    return SubagentMetadata(
        name=":".join(name_parts),
        description=description,
        system_prompt=prompt,
        model=(
            model if isinstance(model, str) and model and model != "inherit" else None
        ),
        source=f"plugin:{plugin.plugin_id}",
        path=str(path),
    )


def plugin_agents(
    plugins: tuple[PluginInstance, ...],
) -> tuple[SubagentMetadata, ...]:
    """Load namespaced subagents from active plugins.

    Args:
        plugins: Active plugin instances.

    Returns:
        Namespaced subagent metadata.
    """
    agents: dict[str, SubagentMetadata] = {}
    for plugin in plugins:
        files: list[tuple[Path, Path]] = []
        for agent_path in plugin.inventory.agents:
            if agent_path.is_file() and agent_path.suffix.lower() == ".md":
                files.append((agent_path, agent_path.parent))
            elif agent_path.is_dir():
                files.extend(
                    (path, agent_path)
                    for path in sorted(agent_path.rglob("*.md"))
                    if path.is_file()
                )
        for path, base in files:
            agent = _agent_from_file(plugin, path, base)
            if agent is not None:
                agents[agent["name"]] = agent
    return tuple(agents.values())
