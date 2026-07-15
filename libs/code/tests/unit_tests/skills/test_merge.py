"""Unit tests for skill-name collision (override) debug logging."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from deepagents_code.skills.load import list_skills
from deepagents_code.skills.merge import merge_skill

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

_MERGE_LOGGER = "deepagents_code.skills.merge"


def _create_skill(skill_dir: Path, name: str, description: str) -> None:
    """Create a minimal skill directory with a valid `SKILL.md`."""
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(f"""---
name: {name}
description: {description}
---
Content
""")


def _override_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """Return only the DEBUG override records emitted by the merge helper."""
    return [
        record
        for record in caplog.records
        if record.name == _MERGE_LOGGER and record.levelno == logging.DEBUG
    ]


class TestMergeSkillHelper:
    """Directly exercise `merge_skill`."""

    def test_collision_logs_debug_with_useful_fields(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A same-name replacement emits one DEBUG log with name and paths."""
        merged: dict[str, dict[str, str]] = {}
        labels: dict[str, str | None] = {}

        with caplog.at_level(logging.DEBUG, logger=_MERGE_LOGGER):
            merge_skill(
                merged,
                labels,
                {"name": "shared", "path": "/user/shared/SKILL.md"},
                source_label="user",
            )
            merge_skill(
                merged,
                labels,
                {"name": "shared", "path": "/project/shared/SKILL.md"},
                source_label="project",
            )

        records = _override_records(caplog)
        assert len(records) == 1
        message = records[0].getMessage()
        assert "shared" in message
        assert "/user/shared/SKILL.md" in message
        assert "/project/shared/SKILL.md" in message
        assert "user" in message
        assert "project" in message

        # Override behavior is unchanged: the higher-precedence skill wins.
        assert merged["shared"]["path"] == "/project/shared/SKILL.md"
        assert labels["shared"] == "project"

    def test_no_collision_does_not_log(self, caplog: pytest.LogCaptureFixture) -> None:
        """Distinct skill names produce no override DEBUG log."""
        merged: dict[str, dict[str, str]] = {}
        labels: dict[str, str | None] = {}

        with caplog.at_level(logging.DEBUG, logger=_MERGE_LOGGER):
            merge_skill(
                merged,
                labels,
                {"name": "alpha", "path": "/user/alpha/SKILL.md"},
                source_label="user",
            )
            merge_skill(
                merged,
                labels,
                {"name": "beta", "path": "/project/beta/SKILL.md"},
                source_label="project",
            )

        assert _override_records(caplog) == []
        assert set(merged) == {"alpha", "beta"}

    def test_repeated_replacement_logs_once_per_event(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Each effective replacement is logged once, not the final winner only."""
        merged: dict[str, dict[str, str]] = {}
        labels: dict[str, str | None] = {}

        with caplog.at_level(logging.DEBUG, logger=_MERGE_LOGGER):
            for label in ("built-in", "user", "project"):
                merge_skill(
                    merged,
                    labels,
                    {"name": "shared", "path": f"/{label}/shared/SKILL.md"},
                    source_label=label,
                )

        # built-in -> user, then user -> project == two replacement events.
        assert len(_override_records(caplog)) == 2
        assert merged["shared"]["path"] == "/project/shared/SKILL.md"


class TestListSkillsCollisionLogging:
    """Exercise collision logging through the CLI `list_skills` discovery path."""

    def test_project_overrides_user_logs_debug(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A project skill overriding a user skill logs a DEBUG override."""
        user_dir = tmp_path / "user_skills"
        project_dir = tmp_path / "project_skills"
        _create_skill(user_dir / "shared-skill", "shared-skill", "User version")
        _create_skill(project_dir / "shared-skill", "shared-skill", "Project version")

        with caplog.at_level(logging.DEBUG, logger=_MERGE_LOGGER):
            skills = list_skills(
                user_skills_dir=user_dir, project_skills_dir=project_dir
            )

        # Override behavior preserved: project wins, single merged entry.
        assert len(skills) == 1
        assert skills[0]["description"] == "Project version"

        records = _override_records(caplog)
        assert len(records) == 1
        message = records[0].getMessage()
        assert "shared-skill" in message
        assert "user" in message
        assert "project" in message

    def test_distinct_skills_do_not_log(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Distinct skill names across sources produce no override DEBUG log."""
        user_dir = tmp_path / "user_skills"
        project_dir = tmp_path / "project_skills"
        _create_skill(user_dir / "user-skill", "user-skill", "User skill")
        _create_skill(project_dir / "project-skill", "project-skill", "Project skill")

        with caplog.at_level(logging.DEBUG, logger=_MERGE_LOGGER):
            skills = list_skills(
                user_skills_dir=user_dir, project_skills_dir=project_dir
            )

        assert len(skills) == 2
        assert _override_records(caplog) == []
