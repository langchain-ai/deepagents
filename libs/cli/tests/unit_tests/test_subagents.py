"""Unit tests for subagent loading functionality."""

from pathlib import Path

from deepagents_cli.subagents import (
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
        """Test that subagent without name is rejected."""
        subagent_file = tmp_path / "invalid.md"
        subagent_file.write_text("""---
description: Missing name field
---

Content
""")

        assert _parse_subagent_file(subagent_file) is None

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
