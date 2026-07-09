"""Adapter for plugin-provided prompt commands."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import yaml

from deepagents_code.plugins.substitution import substitute_string

if TYPE_CHECKING:
    from pathlib import Path

    from deepagents_code.plugins.models import PluginInstance

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)
_POSITIONAL_RE = re.compile(r"\$([1-9])")


@dataclass(frozen=True, slots=True, kw_only=True)
class PluginCommand:
    """A namespaced prompt command contributed by a plugin."""

    name: str
    description: str
    content: str
    plugin: PluginInstance
    path: Path | None = None
    argument_hint: str = ""
    model: str | None = None
    allowed_tools: tuple[str, ...] = ()

    def render(self, args: str, *, session_id: str | None = None) -> str:
        """Render the command prompt with arguments and plugin variables.

        Args:
            args: Raw command arguments.
            session_id: Optional active session identifier.

        Returns:
            Rendered prompt.
        """
        values = args.split()
        had_placeholder = "$ARGUMENTS" in self.content or bool(
            _POSITIONAL_RE.search(self.content)
        )
        rendered = self.content.replace("$ARGUMENTS", args)
        rendered = _POSITIONAL_RE.sub(
            lambda match: (
                values[int(match.group(1)) - 1]
                if len(values) >= int(match.group(1))
                else ""
            ),
            rendered,
        )
        rendered = substitute_string(
            rendered,
            plugin_root=self.plugin.root,
            plugin_data=self.plugin.data_dir,
            session_id=session_id,
            warning_key=self.plugin.plugin_id,
        )
        if args and not had_placeholder:
            rendered = f"{rendered.rstrip()}\n\n{args}"
        return rendered


def _parse_markdown(content: str) -> tuple[dict[str, object], str]:
    match = _FRONTMATTER_RE.match(content)
    if match is None:
        return {}, content
    raw = yaml.safe_load(match.group(1))
    frontmatter = (
        {key: value for key, value in raw.items() if isinstance(key, str)}
        if isinstance(raw, dict)
        else {}
    )
    return frontmatter, match.group(2)


def _allowed_tools(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, list):
        return tuple(item for item in value if isinstance(item, str))
    return ()


def _command_from_content(
    *,
    plugin: PluginInstance,
    name: str,
    content: str,
    path: Path | None,
    overrides: dict[str, object] | None = None,
) -> PluginCommand:
    frontmatter, body = _parse_markdown(content)
    metadata = {**frontmatter, **(overrides or {})}
    description = metadata.get("description")
    argument_hint = metadata.get("argumentHint", metadata.get("argument-hint"))
    model = metadata.get("model")
    allowed = metadata.get("allowedTools", metadata.get("allowed-tools"))
    return PluginCommand(
        name=f"{plugin.name}:{name}",
        description=description if isinstance(description, str) else "Plugin command",
        content=body,
        plugin=plugin,
        path=path,
        argument_hint=argument_hint if isinstance(argument_hint, str) else "",
        model=model if isinstance(model, str) and model != "inherit" else None,
        allowed_tools=_allowed_tools(allowed),
    )


def plugin_commands(
    plugins: tuple[PluginInstance, ...],
) -> tuple[PluginCommand, ...]:
    """Load namespaced prompt commands from active plugins.

    Args:
        plugins: Active plugin instances.

    Returns:
        Namespaced prompt commands.
    """
    commands: dict[str, PluginCommand] = {}
    for plugin in plugins:
        metadata = plugin.manifest.inline_commands if plugin.manifest else {}
        source_names: dict[Path, tuple[str, dict[str, object]]] = {}
        for name, item in metadata.items():
            source = item.get("source")
            if isinstance(source, str):
                source_names[(plugin.root / source).resolve()] = (name, item)
            content = item.get("content")
            if isinstance(content, str):
                command = _command_from_content(
                    plugin=plugin,
                    name=name,
                    content=content,
                    path=None,
                    overrides=item,
                )
                commands[command.name] = command
        files: list[tuple[Path, Path]] = []
        for command_path in plugin.inventory.commands:
            if command_path.is_file() and command_path.suffix.lower() == ".md":
                files.append((command_path, command_path.parent))
            elif command_path.is_dir():
                files.extend(
                    (path, command_path)
                    for path in sorted(command_path.rglob("*.md"))
                    if path.is_file()
                )
        for path, base in files:
            try:
                content = path.read_text(encoding="utf-8")
            except OSError:
                continue
            override = source_names.get(path.resolve())
            if override is None:
                relative = path.relative_to(base).with_suffix("")
                name = ":".join(relative.parts)
                metadata_override = None
            else:
                name, metadata_override = override
            command = _command_from_content(
                plugin=plugin,
                name=name,
                content=content,
                path=path,
                overrides=metadata_override,
            )
            commands[command.name] = command
    return tuple(commands.values())
