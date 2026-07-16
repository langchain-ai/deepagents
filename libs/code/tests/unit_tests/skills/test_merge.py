"""Unit tests for skill-name collision (override) debug logging."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from deepagents.backends.filesystem import FilesystemBackend

from deepagents_code.plugins.adapters.skills_middleware import PluginSkillsMiddleware
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


def _override_records(
    caplog: pytest.LogCaptureFixture, level: int = logging.DEBUG
) -> list[logging.LogRecord]:
    """Return the merge-helper override records emitted at exactly `level`."""
    return [
        record
        for record in caplog.records
        if record.name == _MERGE_LOGGER and record.levelno == level
    ]


def _args(record: logging.LogRecord) -> tuple[object, ...]:
    """Return a record's positional log args, asserting they form a tuple."""
    args = record.args
    assert isinstance(args, tuple)
    return args


class TestMergeSkillHelper:
    """Directly exercise `merge_skill`."""

    def test_collision_logs_debug_with_useful_fields(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A replacement emits one DEBUG log with all identifying fields."""
        merged: dict[str, dict[str, str]] = {}
        labels: dict[str, str | None] = {}

        with caplog.at_level(logging.DEBUG, logger=_MERGE_LOGGER):
            merge_skill(
                merged,
                labels,
                {"name": "shared", "path": "/a/shared/SKILL.md"},
                source_label="user",
            )
            merge_skill(
                merged,
                labels,
                {"name": "shared", "path": "/b/shared/SKILL.md"},
                source_label="project",
            )

        records = _override_records(caplog)
        assert len(records) == 1
        assert records[0].args == (
            "shared",
            "/a/shared/SKILL.md",
            "user",
            "/b/shared/SKILL.md",
            "project",
        )
        assert _override_records(caplog, logging.WARNING) == []
        assert merged["shared"]["path"] == "/b/shared/SKILL.md"
        assert labels["shared"] == "project"

    def test_same_label_collision_logs_debug(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Labels do not change the DEBUG level used for collision diagnostics."""
        merged: dict[str, dict[str, str]] = {}
        labels: dict[str, str | None] = {}

        with caplog.at_level(logging.DEBUG, logger=_MERGE_LOGGER):
            merge_skill(
                merged,
                labels,
                {"name": "shared", "path": "/deepagents/shared/SKILL.md"},
                source_label="user",
            )
            merge_skill(
                merged,
                labels,
                {"name": "shared", "path": "/agents/shared/SKILL.md"},
                source_label="user",
            )

        records = _override_records(caplog)
        assert len(records) == 1
        assert records[0].args == (
            "shared",
            "/deepagents/shared/SKILL.md",
            "user",
            "/agents/shared/SKILL.md",
            "user",
        )
        assert _override_records(caplog, logging.WARNING) == []

    def test_same_path_replacement_logs_debug(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Every replacement logs even when both metadata paths are equal."""
        merged: dict[str, dict[str, str]] = {}
        labels: dict[str, str | None] = {}

        with caplog.at_level(logging.DEBUG, logger=_MERGE_LOGGER):
            merge_skill(
                merged,
                labels,
                {"name": "shared", "path": "/shared/SKILL.md"},
                source_label="user",
            )
            merge_skill(
                merged,
                labels,
                {"name": "shared", "path": "/shared/SKILL.md"},
                source_label="project",
            )

        records = _override_records(caplog)
        assert len(records) == 1
        assert records[0].args == (
            "shared",
            "/shared/SKILL.md",
            "user",
            "/shared/SKILL.md",
            "project",
        )
        assert _override_records(caplog, logging.WARNING) == []

    def test_missing_path_logs_none(self, caplog: pytest.LogCaptureFixture) -> None:
        """A skill without a `path` key logs `None` rather than raising."""
        merged: dict[str, dict[str, str]] = {}
        labels: dict[str, str | None] = {}

        with caplog.at_level(logging.DEBUG, logger=_MERGE_LOGGER):
            merge_skill(merged, labels, {"name": "shared"}, source_label="user")
            merge_skill(merged, labels, {"name": "shared"}, source_label="project")

        records = _override_records(caplog)
        assert len(records) == 1
        name, prev_path, _prev_label, new_path, _new_label = _args(records[0])
        assert name == "shared"
        assert prev_path is None
        assert new_path is None

    def test_unknown_label_falls_back(self, caplog: pytest.LogCaptureFixture) -> None:
        """A `None` source label renders as the literal `"unknown"`."""
        merged: dict[str, dict[str, str]] = {}
        labels: dict[str, str | None] = {}

        with caplog.at_level(logging.DEBUG, logger=_MERGE_LOGGER):
            merge_skill(merged, labels, {"name": "shared", "path": "/a/SKILL.md"})
            merge_skill(merged, labels, {"name": "shared", "path": "/b/SKILL.md"})

        records = _override_records(caplog)
        assert len(records) == 1
        assert records[0].args == (
            "shared",
            "/a/SKILL.md",
            "unknown",
            "/b/SKILL.md",
            "unknown",
        )

    def test_no_collision_does_not_log(self, caplog: pytest.LogCaptureFixture) -> None:
        """Distinct skill names produce no override log."""
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

        assert _override_records(caplog, logging.DEBUG) == []
        assert _override_records(caplog, logging.WARNING) == []
        assert set(merged) == {"alpha", "beta"}

    def test_repeated_replacement_logs_once_per_event(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Each replacement is logged once, not only the final winner."""
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

        assert len(skills) == 1
        assert skills[0]["description"] == "Project version"

        records = _override_records(caplog)
        assert len(records) == 1
        name, prev_path, prev_label, new_path, new_label = _args(records[0])
        assert name == "shared-skill"
        assert prev_label == "user"
        assert new_label == "project"
        assert str(user_dir) in str(prev_path)
        assert str(project_dir) in str(new_path)

    def test_distinct_skills_do_not_log(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Distinct skill names across sources produce no override log."""
        user_dir = tmp_path / "user_skills"
        project_dir = tmp_path / "project_skills"
        _create_skill(user_dir / "user-skill", "user-skill", "User skill")
        _create_skill(project_dir / "project-skill", "project-skill", "Project skill")

        with caplog.at_level(logging.DEBUG, logger=_MERGE_LOGGER):
            skills = list_skills(
                user_skills_dir=user_dir, project_skills_dir=project_dir
            )

        assert len(skills) == 2
        assert _override_records(caplog, logging.DEBUG) == []
        assert _override_records(caplog, logging.WARNING) == []


class TestMiddlewareCollisionLogging:
    """Exercise collision logging through `PluginSkillsMiddleware` (sync + async).

    These lock in the new three-way `zip(self.sources, self.source_labels,
    self._namespaces, ...)` wiring: both entry points must merge through
    `merge_skill` and log overrides identically.
    """

    @staticmethod
    def _middleware(user_dir: Path, project_dir: Path) -> PluginSkillsMiddleware:
        """Build a middleware over two colliding, non-namespaced sources."""
        _create_skill(user_dir / "review", "review", "User review")
        _create_skill(project_dir / "review", "review", "Project review")
        return PluginSkillsMiddleware(
            backend=FilesystemBackend(virtual_mode=False),
            sources=[(str(user_dir), "User"), (str(project_dir), "Project")],
            system_prompt=None,
        )

    def test_before_agent_collision_logs_override(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The sync path merges through `merge_skill` and logs the override."""
        middleware = self._middleware(tmp_path / "user", tmp_path / "project")

        with caplog.at_level(logging.DEBUG, logger=_MERGE_LOGGER):
            update = middleware.before_agent(
                cast("Any", {"messages": []}), runtime=cast("Any", None), config={}
            )

        assert update is not None
        # Precedence preserved: the later (project) source wins the collision.
        metadata = update["skills_metadata"]
        assert [skill["name"] for skill in metadata] == ["review"]
        assert metadata[0]["description"] == "Project review"

        records = _override_records(caplog)
        assert len(records) == 1
        name, _prev_path, prev_label, _new_path, new_label = _args(records[0])
        assert name == "review"
        assert prev_label == "User"
        assert new_label == "Project"

    async def test_abefore_agent_collision_logs_override(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The async path logs the override identically to the sync path."""
        middleware = self._middleware(tmp_path / "user", tmp_path / "project")

        with caplog.at_level(logging.DEBUG, logger=_MERGE_LOGGER):
            update = await middleware.abefore_agent(
                cast("Any", {"messages": []}), runtime=cast("Any", None), config={}
            )

        assert update is not None
        metadata = update["skills_metadata"]
        assert [skill["name"] for skill in metadata] == ["review"]
        assert metadata[0]["description"] == "Project review"

        records = _override_records(caplog)
        assert len(records) == 1
        name, _prev_path, prev_label, _new_path, new_label = _args(records[0])
        assert name == "review"
        assert prev_label == "User"
        assert new_label == "Project"
