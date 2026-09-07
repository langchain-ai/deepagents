"""Opt-in, editable research defaults; tool attachment is not a sandbox."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from deepagents_talon.subagents import LocalSubAgent

PROFILE_NAME = "research.json"
DEFAULT_PROFILE = Path(__file__).with_name("defaults") / PROFILE_NAME
_ROLES = {"internal-research", "external-research"}


@dataclass(frozen=True)
class ResearchProfile:
    """Validated instructions and exact optional attachments."""

    main_prompt: str
    direct_tools: list[str]
    agents: list[LocalSubAgent]


def install_research_defaults(home: Path) -> None:
    """Install the packaged profile without replacing an existing file.

    Args:
        home: Newly created, private assistant directory.
    """
    try:
        fd = os.open(home / PROFILE_NAME, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return
    with os.fdopen(fd, "w", encoding="utf-8") as target:
        target.write(DEFAULT_PROFILE.read_text(encoding="utf-8"))


def load_research_profile(home: Path | None) -> ResearchProfile | None:
    """Load a profile, rejecting malformed settings without printing their contents.

    Args:
        home: Assistant directory, or none for an embedding without a profile.

    Returns:
        Validated profile, or none when absent.

    Raises:
        ValueError: The saved profile is invalid.
    """
    if home is None:
        return None
    try:
        raw = (home / PROFILE_NAME).read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        return _parse_profile(json.loads(raw))
    except (ValueError, TypeError, KeyError):
        msg = "Invalid research.json; previous configuration retained"
        raise ValueError(msg) from None


def _names(value: object) -> list[str]:
    if (
        not isinstance(value, list)
        or any(not isinstance(name, str) or not name.strip() for name in value)
        or len(set(value)) != len(value)
    ):
        msg = "Expected unique exact tool names"
        raise ValueError(msg)
    return cast("list[str]", value)


def _parse_profile(data: object) -> ResearchProfile:
    fields = _object(data, {"version", "main_prompt", "direct_tools", "agents"})
    if (
        type(fields["version"]) is not int
        or fields["version"] != 1
        or not isinstance(fields["agents"], list)
    ):
        msg = "Expected a version 1 research profile"
        raise ValueError(msg)
    agents = [_parse_agent(agent) for agent in fields["agents"]]
    if len(agents) != len(_ROLES) or {agent["name"] for agent in agents} != _ROLES:
        msg = "Expected both research roles exactly once"
        raise ValueError(msg)
    return ResearchProfile(_text(fields["main_prompt"]), _names(fields["direct_tools"]), agents)


def _parse_agent(data: object) -> LocalSubAgent:
    fields = _object(data, {"name", "description", "system_prompt", "tools"})
    return {
        "name": _text(fields["name"]),
        "description": _text(fields["description"]),
        "system_prompt": _text(fields["system_prompt"]),
        "tool_names": _names(fields["tools"]),
    }


def _object(value: object, keys: set[str]) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        msg = "Unexpected profile fields"
        raise ValueError(msg)
    return cast("dict[str, object]", value)


def _text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        msg = "Expected nonempty text"
        raise ValueError(msg)
    return value
