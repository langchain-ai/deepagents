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

    def test_parse_valid_subagent_with_all_fields(self, tmp_path: Path) -> None:
        """Test parsing a valid subagent file with all fields."""
        subagent_file = tmp_path / "researcher.md"
        subagent_file.write_text(
            make_subagent_content(
                "researcher",
                "Research topics on the web",
                model="anthropic:claude-haiku-4-5-20251001",
            )
        )

        result = _parse_subagent_file(subagent_file)

        assert result is not None
        assert result["name"] == "researcher"
        assert result["description"] == "Research topics on the web"
        assert result["model"] == "anthropic:claude-haiku-4-5-20251001"
        assert "researcher assistant" in result["system_prompt"]
        assert "## Instructions" in result["system_prompt"]
        assert result["path"] == str(subagent_file)

    def test_parse_subagent_without_model(self, tmp_path: Path) -> None:
        """Test parsing subagent without optional model field."""
        subagent_file = tmp_path / "helper.md"
        subagent_file.write_text(make_subagent_content("helper", "A helpful assistant"))

        result = _parse_subagent_file(subagent_file)

        assert result is not None
        assert result["name"] == "helper"
        assert result["description"] == "A helpful assistant"
        assert result["model"] is None

    def test_parse_subagent_with_multiline_system_prompt(self, tmp_path: Path) -> None:
        """Test parsing subagent with complex multiline system prompt."""
        subagent_file = tmp_path / "writer.md"
        content = """---
name: writer
description: Write content
---

You are a skilled writer.

## Guidelines

1. Write clearly
2. Use proper grammar
3. Be concise

## Output Format

Always structure your response with headings.
"""
        subagent_file.write_text(content)

        result = _parse_subagent_file(subagent_file)

        assert result is not None
        assert "## Guidelines" in result["system_prompt"]
        assert "## Output Format" in result["system_prompt"]
        assert "1. Write clearly" in result["system_prompt"]

    def test_parse_subagent_missing_name(self, tmp_path: Path) -> None:
        """Test that subagent without name is rejected without a fallback."""
        subagent_file = tmp_path / "invalid.md"
        subagent_file.write_text("""---
description: Missing name field
---

Content
""")

        assert _parse_subagent_file(subagent_file) is None

    def test_parse_subagent_uses_fallback_name(self, tmp_path: Path) -> None:
        """Test that fallback name is used when frontmatter omits name."""
        subagent_file = tmp_path / "AGENTS.md"
        subagent_file.write_text("""---
description: Missing name field
---

Content
""")

        result = _parse_subagent_file(subagent_file, fallback_name="helper")

        assert result is not None
        assert result["name"] == "helper"
        assert result["description"] == "Missing name field"

    def test_parse_subagent_missing_description(self, tmp_path: Path) -> None:
        """Test that subagent without description is rejected."""
        subagent_file = tmp_path / "invalid.md"
        subagent_file.write_text("""---
name: invalid
---

Content
""")

        assert _parse_subagent_file(subagent_file) is None

    def test_parse_subagent_no_frontmatter(self, tmp_path: Path) -> None:
        """Test that file without frontmatter is rejected."""
        subagent_file = tmp_path / "invalid.md"
        subagent_file.write_text("# Just markdown\n\nNo frontmatter.")

        assert _parse_subagent_file(subagent_file) is None

    def test_parse_subagent_invalid_yaml(self, tmp_path: Path) -> None:
        """Test that invalid YAML is rejected."""
        subagent_file = tmp_path / "invalid.md"
        subagent_file.write_text("""---
name: [unclosed
description: test
---

Content
""")

        assert _parse_subagent_file(subagent_file) is None

    def test_parse_subagent_empty_name(self, tmp_path: Path) -> None:
        """Test that empty name is rejected."""
        subagent_file = tmp_path / "invalid.md"
        subagent_file.write_text("""---
name: ""
description: Has empty name
---

Content
""")

        assert _parse_subagent_file(subagent_file) is None

    def test_parse_subagent_non_string_name(self, tmp_path: Path) -> None:
        """Test that non-string name is rejected."""
        subagent_file = tmp_path / "invalid.md"
        subagent_file.write_text("""---
name: 123
description: Has numeric name
---

Content
""")

        assert _parse_subagent_file(subagent_file) is None

    def test_parse_subagent_non_string_model(self, tmp_path: Path) -> None:
        """Test that non-string model is rejected."""
        subagent_file = tmp_path / "invalid.md"
        subagent_file.write_text("""---
name: test
description: Test
model: 123
---

Content
""")

        assert _parse_subagent_file(subagent_file) is None

    def test_parse_subagent_nonexistent_file(self, tmp_path: Path) -> None:
        """Test that nonexistent file returns None."""
        subagent_file = tmp_path / "nonexistent.md"

        assert _parse_subagent_file(subagent_file) is None

    def test_parse_subagent_frontmatter_not_dict(self, tmp_path: Path) -> None:
        """Test that non-dict frontmatter is rejected."""
        subagent_file = tmp_path / "invalid.md"
        subagent_file.write_text("""---
- item1
- item2
---

Content
""")

        assert _parse_subagent_file(subagent_file) is None


class TestLoadSubagentsFromDir:
    """Test _load_subagents_from_dir function."""

    def test_load_from_empty_directory(self, tmp_path: Path) -> None:
        """Test loading from empty directory."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        result = _load_subagents_from_dir(agents_dir, "user")
        assert result == {}

    def test_load_single_subagent(self, tmp_path: Path) -> None:
        """Test loading a single subagent."""
        agents_dir = tmp_path / "agents"
        folder = agents_dir / "researcher"
        folder.mkdir(parents=True)
        (folder / "AGENTS.md").write_text(
            make_subagent_content("researcher", "Research assistant")
        )

        result = _load_subagents_from_dir(agents_dir, "user")

        assert len(result) == 1
        assert "researcher" in result
        assert result["researcher"]["source"] == "user"

    def test_load_uses_folder_name_when_frontmatter_omits_name(
        self, tmp_path: Path
    ) -> None:
        """Test loading a subagent whose frontmatter omits name."""
        agents_dir = tmp_path / "agents"
        folder = agents_dir / "helper"
        folder.mkdir(parents=True)
        (folder / "AGENTS.md").write_text("""---
description: Helpful assistant
---

Content
""")

        result = _load_subagents_from_dir(agents_dir, "user")

        assert len(result) == 1
        assert "helper" in result
        assert result["helper"]["name"] == "helper"
        assert result["helper"]["description"] == "Helpful assistant"

    def test_load_multiple_subagents(self, tmp_path: Path) -> None:
        """Test loading multiple subagents."""
        agents_dir = tmp_path / "agents"

        for name in ["researcher", "writer", "reviewer"]:
            folder = agents_dir / name
            folder.mkdir(parents=True)
            (folder / "AGENTS.md").write_text(
                make_subagent_content(name, f"{name.title()} assistant")
            )

        result = _load_subagents_from_dir(agents_dir, "project")

        assert len(result) == 3
        assert all(s["source"] == "project" for s in result.values())

    def test_load_skips_misnamed_files(self, tmp_path: Path) -> None:
        """Test that files not matching expected name are skipped."""
        agents_dir = tmp_path / "agents"
        folder = agents_dir / "researcher"
        folder.mkdir(parents=True)
        # Wrong filename - should be AGENTS.md
        (folder / "agent.md").write_text(
            make_subagent_content("researcher", "Research assistant")
        )

        result = _load_subagents_from_dir(agents_dir, "user")
        assert result == {}

    def test_load_skips_invalid_subagents(self, tmp_path: Path) -> None:
        """Test that invalid subagents are skipped."""
        agents_dir = tmp_path / "agents"

        # Valid subagent
        valid_folder = agents_dir / "valid"
        valid_folder.mkdir(parents=True)
        (valid_folder / "AGENTS.md").write_text(
            make_subagent_content("valid", "Valid assistant")
        )

        # Invalid subagent (missing description)
        invalid_folder = agents_dir / "invalid"
        invalid_folder.mkdir(parents=True)
        (invalid_folder / "AGENTS.md").write_text("""---
name: invalid
---
Content
""")

        result = _load_subagents_from_dir(agents_dir, "user")

        assert len(result) == 1
        assert "valid" in result

    def test_load_skips_files_in_root(self, tmp_path: Path) -> None:
        """Test that files directly in agents dir (not in subfolders) are skipped."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "stray.md").write_text(
            make_subagent_content("stray", "Stray file")
        )

        result = _load_subagents_from_dir(agents_dir, "user")
        assert result == {}

    def test_load_nonexistent_directory(self, tmp_path: Path) -> None:
        """Test loading from nonexistent directory."""
        result = _load_subagents_from_dir(tmp_path / "nonexistent", "user")
        assert result == {}


class TestListSubagents:
    """Test list_subagents function."""

    def test_list_no_directories(self) -> None:
        """Test listing with no directories specified."""
        result = list_subagents()
        assert result == []

    def test_list_user_only(self, tmp_path: Path) -> None:
        """Test listing from user directory only."""
        user_dir = tmp_path / "user_agents"
        folder = user_dir / "researcher"
        folder.mkdir(parents=True)
        (folder / "AGENTS.md").write_text(
            make_subagent_content("researcher", "Research assistant")
        )

        result = list_subagents(user_agents_dir=user_dir)

        assert len(result) == 1
        assert result[0]["name"] == "researcher"
        assert result[0]["source"] == "user"

    def test_list_project_only(self, tmp_path: Path) -> None:
        """Test listing from project directory only."""
        project_dir = tmp_path / "project_agents"
        folder = project_dir / "reviewer"
        folder.mkdir(parents=True)
        (folder / "AGENTS.md").write_text(
            make_subagent_content("reviewer", "Code reviewer")
        )

        result = list_subagents(project_agents_dir=project_dir)

        assert len(result) == 1
        assert result[0]["name"] == "reviewer"
        assert result[0]["source"] == "project"

    def test_list_both_sources(self, tmp_path: Path) -> None:
        """Test listing from both user and project directories."""
        user_dir = tmp_path / "user_agents"
        project_dir = tmp_path / "project_agents"

        # User subagent
        user_folder = user_dir / "researcher"
        user_folder.mkdir(parents=True)
        (user_folder / "AGENTS.md").write_text(
            make_subagent_content("researcher", "Research assistant")
        )

        # Project subagent
        project_folder = project_dir / "reviewer"
        project_folder.mkdir(parents=True)
        (project_folder / "AGENTS.md").write_text(
            make_subagent_content("reviewer", "Code reviewer")
        )

        result = list_subagents(
            user_agents_dir=user_dir,
            project_agents_dir=project_dir,
        )

        assert len(result) == 2
        names = {s["name"] for s in result}
        assert names == {"researcher", "reviewer"}

    def test_list_project_overrides_user(self, tmp_path: Path) -> None:
        """Test that project subagents override user subagents with same name."""
        user_dir = tmp_path / "user_agents"
        project_dir = tmp_path / "project_agents"

        # User version
        user_folder = user_dir / "shared"
        user_folder.mkdir(parents=True)
        (user_folder / "AGENTS.md").write_text(
            make_subagent_content("shared", "User version")
        )

        # Project version (same name)
        project_folder = project_dir / "shared"
        project_folder.mkdir(parents=True)
        (project_folder / "AGENTS.md").write_text(
            make_subagent_content("shared", "Project version")
        )

        result = list_subagents(
            user_agents_dir=user_dir,
            project_agents_dir=project_dir,
        )

        assert len(result) == 1
        assert result[0]["name"] == "shared"
        assert result[0]["description"] == "Project version"
        assert result[0]["source"] == "project"

    def test_list_empty_directories(self, tmp_path: Path) -> None:
        """Test listing from empty directories."""
        user_dir = tmp_path / "user_agents"
        project_dir = tmp_path / "project_agents"
        user_dir.mkdir()
        project_dir.mkdir()

        result = list_subagents(
            user_agents_dir=user_dir,
            project_agents_dir=project_dir,
        )
        assert result == []

    def test_list_nonexistent_directories(self, tmp_path: Path) -> None:
        """Test listing from nonexistent directories."""
        result = list_subagents(
            user_agents_dir=tmp_path / "nonexistent_user",
            project_agents_dir=tmp_path / "nonexistent_project",
        )
        assert result == []

    def test_list_with_model_field(self, tmp_path: Path) -> None:
        """Test that model field is correctly loaded."""
        user_dir = tmp_path / "agents"
        folder = user_dir / "fast-researcher"
        folder.mkdir(parents=True)
        (folder / "AGENTS.md").write_text(
            make_subagent_content(
                "fast-researcher",
                "Fast research using Haiku",
                model="anthropic:claude-haiku-4-5-20251001",
            )
        )

        result = list_subagents(user_agents_dir=user_dir)

        assert len(result) == 1
        assert result[0]["model"] == "anthropic:claude-haiku-4-5-20251001"


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
