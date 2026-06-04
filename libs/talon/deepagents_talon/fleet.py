"""Fleet export loading for Talon."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from fleet_deepagents_export import StaticSkillsLoader, load_agent_components

if TYPE_CHECKING:
    from deepagents import AsyncSubAgent, CompiledSubAgent, SubAgent
    from langchain.agents.middleware import InterruptOnConfig
    from langchain.agents.middleware.types import AgentMiddleware
    from langchain_core.tools import BaseTool


@dataclass(frozen=True, slots=True)
class FleetAgentComponents:
    """Components returned by a Fleet export loader.

    Args:
        model: Chat model id from Fleet `config.json`.
        system_prompt: System prompt from Fleet `AGENTS.md`.
        tools: Resolved Fleet MCP tools.
        subagents: Fleet subagent specs.
        interrupt_on: Fleet human-in-the-loop config passed to the Deep Agents
            graph so Talon can surface tool approval over its channels.
        skills: Skill source paths for `create_deep_agent`.
        middleware: Middleware required by Fleet-loaded components.
    """

    model: str
    system_prompt: str
    tools: tuple[BaseTool | Callable[..., object], ...]
    subagents: tuple[SubAgent | CompiledSubAgent | AsyncSubAgent, ...]
    interrupt_on: Mapping[str, bool | InterruptOnConfig] | None
    skills: tuple[str, ...] = ()
    middleware: tuple[AgentMiddleware[Any, Any, Any], ...] = ()


async def load_fleet_agent_components(
    fleet_dir: Path,
    *,
    env: Mapping[str, str] | None = None,
) -> FleetAgentComponents:
    """Load a Fleet export directory through `fleet-deepagents-export`.

    Args:
        fleet_dir: Operator-unzipped Fleet export directory.
        env: Optional environment values to expose while loading the Fleet
            components. This lets embedding hosts pass TalonConfig-derived
            LangSmith settings even when they are not already in `os.environ`.

    Returns:
        Validated Fleet components ready for Talon's runtime wiring.

    Raises:
        TypeError: If the library returns an unexpected component shape.
    """
    with _patched_environ(env):
        raw = await load_agent_components(fleet_dir)
    components = _coerce_components(raw)
    return _with_static_skills_loader(components, fleet_dir=fleet_dir, env=env)


def _coerce_components(raw: object) -> FleetAgentComponents:
    if not isinstance(raw, Mapping):
        msg = "Fleet loader returned a non-mapping component payload"
        raise TypeError(msg)

    data = cast("Mapping[str, object]", raw)
    model = _required_str(data, "model")
    system_prompt = _required_str(data, "system_prompt")
    tools = cast(
        "tuple[BaseTool | Callable[..., object], ...]",
        _optional_sequence(data.get("tools"), "tools"),
    )
    subagents = cast(
        "tuple[SubAgent | CompiledSubAgent | AsyncSubAgent, ...]",
        _optional_sequence(data.get("subagents"), "subagents"),
    )
    interrupt_on = cast(
        "Mapping[str, bool | InterruptOnConfig] | None",
        _optional_mapping(data.get("interrupt_on"), "interrupt_on"),
    )
    return FleetAgentComponents(
        model=model,
        system_prompt=system_prompt,
        tools=tools,
        subagents=subagents,
        interrupt_on=interrupt_on,
    )


def _with_static_skills_loader(
    components: FleetAgentComponents,
    *,
    fleet_dir: Path,
    env: Mapping[str, str] | None,
) -> FleetAgentComponents:
    loader = StaticSkillsLoader(_static_skill_sources(fleet_dir, env=env))
    if not loader.skill_paths:
        return components
    return replace(
        components,
        skills=tuple(loader.skill_paths),
        middleware=(loader,),
    )


def _static_skill_sources(
    fleet_dir: Path,
    *,
    env: Mapping[str, str] | None,
) -> list[tuple[Path, str]]:
    sources: list[tuple[Path, str]] = []
    seen: set[str] = set()
    _append_skill_source(sources, seen, fleet_dir / "skills")
    values = os.environ if env is None else env
    for raw in _split_path_env(
        values.get("DEEPAGENTS_TALON_SKILLS_DIRS") or values.get("SKILLS_DIRS"),
    ):
        _append_skill_source(sources, seen, Path(raw).expanduser())
    return sources


def _append_skill_source(
    sources: list[tuple[Path, str]],
    seen: set[str],
    path: Path,
) -> None:
    marker = str(path)
    if marker in seen:
        return
    seen.add(marker)
    sources.append((path, marker))


def _required_str(data: Mapping[str, object], key: str) -> str:
    value = data.get(key)
    if isinstance(value, str):
        return value
    msg = f"Fleet loader returned invalid {key!r}; expected string"
    raise TypeError(msg)


def _optional_sequence(value: object, key: str) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    msg = f"Fleet loader returned invalid {key!r}; expected sequence"
    raise TypeError(msg)


def _optional_mapping(value: object, key: str) -> Mapping[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        msg = f"Fleet loader returned invalid {key!r}; expected mapping"
        raise TypeError(msg)

    result: dict[str, object] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str):
            msg = f"Fleet loader returned invalid {key!r}; expected string keys"
            raise TypeError(msg)
        result[raw_key] = raw_value
    return result


def _split_path_env(raw: str | None) -> list[str]:
    if not raw:
        return []
    separator = ";" if ";" in raw else os.pathsep
    return [str(Path(part).expanduser()) for part in raw.split(separator) if part.strip()]


@contextmanager
def _patched_environ(env: Mapping[str, str] | None) -> Iterator[None]:
    if not env:
        yield
        return

    previous = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
