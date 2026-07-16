"""Shared skill-merge helper with override (name-collision) debug logging.

Both skill discovery paths — the CLI `skills list` loader
(`deepagents_code.skills.load`) and the runtime agent loader
(`deepagents_code.plugins.adapters.skills_middleware.PluginSkillsMiddleware`) —
merge skills from multiple sources by precedence, last-one-wins, keyed on skill
name. A higher-precedence skill replaces a lower-precedence skill with the same
name. That override behavior is intentional; this helper leaves it unchanged and
makes each replacement observable in debug logs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Mapping, MutableMapping

logger = logging.getLogger(__name__)

_SkillT = TypeVar("_SkillT", bound="Mapping[str, object]")


def merge_skill(
    merged: MutableMapping[str, _SkillT],
    source_labels: MutableMapping[str, str | None],
    skill: _SkillT,
    *,
    source_label: str | None = None,
) -> None:
    """Merge one skill into `merged` by name, last-one-wins.

    Emits one `DEBUG` log whenever a skill replaces an already-merged skill with
    the same name, recording the skill name plus the previous and replacement
    source paths and labels so the winning definition is unambiguous. Nothing is
    logged when there is no collision.

    Callers must iterate sources in ascending precedence order so the replacing
    skill is always the higher-precedence one.

    Args:
        merged: Accumulator mapping skill name to merged metadata; mutated in
            place.
        source_labels: Parallel accumulator mapping skill name to the label of
            the source that last supplied it; mutated in place so the previous
            label is available on the next collision.
        skill: Skill metadata to merge. Must expose `name`; `path`, when present,
            is included in the override log to identify the colliding files.
        source_label: Human-readable label for the source supplying `skill`,
            when known. A missing label renders as `"unknown"` in the log.
    """
    name = str(skill["name"])
    previous = merged.get(name)
    if previous is not None:
        logger.debug(
            "Skill %r override: %s (source: %s) replaced by %s (source: %s)",
            name,
            previous.get("path"),
            source_labels.get(name) or "unknown",
            skill.get("path"),
            source_label or "unknown",
        )
    merged[name] = skill
    source_labels[name] = source_label
