"""Unit tests for subagent loading functionality."""

import logging
from pathlib import Path

import pytest

from deepagents_code.subagents import (
    _load_subagents_from_dir,
    _parse_subagent_file,
    list_subagents,
)


def make_subagent_content(
    name: str,
    description: str,
    model: str | None = None,
    system_prompt: str | None = None,
) -> str:
    """Create subagent markdown content with YAML frontmatter."""
    model_line = f"model: {model}\n" if model else ""
    prompt = (
        system_prompt
        or f"You are a {name} assistant.\n\n## Instructions\nDo your job well."
    )
    return f"""---
name: {name}
description: {description}
{model_line}---

{prompt}
"""


class TestParseSubagentFile:
    """Test _parse_subagent_file function."""


class TestLoadSubagentsFromDir:
    """Test _load_subagents_from_dir function."""


class TestListSubagents:
    """Test list_subagents function."""


class TestDiagnostics:
    """Test that discovery surfaces warnings for misconfigured subagents."""

    def test_warns_on_missing_frontmatter(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A file without frontmatter logs an explanatory warning."""
        subagent_file = tmp_path / "AGENTS.md"
        subagent_file.write_text("# Just markdown\n\nNo frontmatter.")

        with caplog.at_level(logging.WARNING):
            assert _parse_subagent_file(subagent_file) is None

        assert "missing YAML frontmatter" in caplog.text

    def test_warns_on_unreadable_file(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A file that cannot be read (here, a directory) logs a warning."""
        # Reading a directory with read_text raises OSError deterministically,
        # without relying on chmod (which is a no-op for root in CI).
        with caplog.at_level(logging.WARNING):
            assert _parse_subagent_file(tmp_path) is None

        assert "could not read file" in caplog.text

    def test_warns_on_invalid_yaml(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Frontmatter that is not valid YAML logs a warning."""
        subagent_file = tmp_path / "AGENTS.md"
        subagent_file.write_text("---\nname: [unclosed\n---\n\nContent\n")

        with caplog.at_level(logging.WARNING):
            assert _parse_subagent_file(subagent_file) is None

        assert "invalid YAML frontmatter" in caplog.text

    def test_warns_on_non_dict_frontmatter(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Frontmatter that parses to a non-mapping (a list) logs a warning."""
        subagent_file = tmp_path / "AGENTS.md"
        subagent_file.write_text("---\n- just\n- a\n- list\n---\n\nContent\n")

        with caplog.at_level(logging.WARNING):
            assert _parse_subagent_file(subagent_file) is None

        assert "must be a mapping" in caplog.text

    def test_warns_on_missing_description_field(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A missing description names the description field in the warning."""
        subagent_file = tmp_path / "AGENTS.md"
        subagent_file.write_text("---\nname: helper\n---\n\nContent\n")

        with caplog.at_level(logging.WARNING):
            assert _parse_subagent_file(subagent_file) is None

        assert "description (non-empty string required)" in caplog.text

    def test_warns_on_missing_name_field(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A missing name names the name field in the warning."""
        subagent_file = tmp_path / "AGENTS.md"
        subagent_file.write_text("---\ndescription: A helper\n---\n\nContent\n")

        with caplog.at_level(logging.WARNING):
            assert _parse_subagent_file(subagent_file) is None

        assert "name (non-empty string required)" in caplog.text

    def test_warns_on_non_string_model_field(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A non-string model names the model field in the warning."""
        subagent_file = tmp_path / "AGENTS.md"
        subagent_file.write_text(
            "---\nname: helper\ndescription: A helper\nmodel: 42\n---\n\nContent\n"
        )

        with caplog.at_level(logging.WARNING):
            assert _parse_subagent_file(subagent_file) is None

        assert "model (string required when present)" in caplog.text

    def test_warns_on_stray_file_in_agents_dir(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A markdown file placed directly in agents/ is flagged."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "researcher.md").write_text(
            make_subagent_content("researcher", "Research assistant")
        )

        with caplog.at_level(logging.WARNING):
            result = _load_subagents_from_dir(agents_dir, "project")

        assert result == {}
        assert "researcher.md" in caplog.text
        assert "AGENTS.md" in caplog.text

    def test_warns_on_folder_without_agents_md(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A folder with a misnamed definition (agent.md, not AGENTS.md) is flagged."""
        agents_dir = tmp_path / "agents"
        folder = agents_dir / "researcher"
        folder.mkdir(parents=True)
        (folder / "agent.md").write_text(
            make_subagent_content("researcher", "Research assistant")
        )

        with caplog.at_level(logging.WARNING):
            result = _load_subagents_from_dir(agents_dir, "user")

        assert result == {}
        assert "agent.md" in caplog.text
        assert "AGENTS.md" in caplog.text

    def test_warns_on_name_collision(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Two folders declaring the same frontmatter name are flagged."""
        agents_dir = tmp_path / "agents"
        for folder_name in ("researcher", "web-researcher"):
            folder = agents_dir / folder_name
            folder.mkdir(parents=True)
            # Both folders declare the same frontmatter `name`, so one silently
            # shadows the other without this warning.
            (folder / "AGENTS.md").write_text(
                make_subagent_content("researcher", f"Defined in {folder_name}")
            )

        with caplog.at_level(logging.WARNING):
            result = _load_subagents_from_dir(agents_dir, "project")

        # One definition wins (collapsed to a single entry); the collision warns.
        assert len(result) == 1
        assert "name collision" in caplog.text
        assert "researcher" in caplog.text

    def test_no_warning_for_valid_or_unrelated_entries(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A valid subagent alongside an unrelated non-markdown file stays silent."""
        agents_dir = tmp_path / "agents"
        folder = agents_dir / "researcher"
        folder.mkdir(parents=True)
        (folder / "AGENTS.md").write_text(
            make_subagent_content("researcher", "Research assistant")
        )
        # An unrelated file (not .md) directly under agents/ must not be flagged.
        (agents_dir / "notes.txt").write_text("just some notes")

        with caplog.at_level(logging.WARNING):
            result = _load_subagents_from_dir(agents_dir, "project")

        assert len(result) == 1
        assert caplog.records == []
