"""Tests for config module including project discovery utilities."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from deepagents_cli.config import (
    _find_project_agent_md,
    _find_project_root,
    validate_model_capabilities,
)


class TestProjectRootDetection:
    """Test project root detection via .git directory."""

    def test_find_project_root_with_git(self, tmp_path: Path) -> None:
        """Test that project root is found when .git directory exists."""
        # Create a mock project structure
        project_root = tmp_path / "my-project"
        project_root.mkdir()
        git_dir = project_root / ".git"
        git_dir.mkdir()

        # Create a subdirectory to search from
        subdir = project_root / "src" / "components"
        subdir.mkdir(parents=True)

        # Should find project root from subdirectory
        result = _find_project_root(subdir)
        assert result == project_root

    def test_find_project_root_no_git(self, tmp_path: Path) -> None:
        """Test that None is returned when no .git directory exists."""
        # Create directory without .git
        no_git_dir = tmp_path / "no-git"
        no_git_dir.mkdir()

        result = _find_project_root(no_git_dir)
        assert result is None

    def test_find_project_root_nested_git(self, tmp_path: Path) -> None:
        """Test that nearest .git directory is found (not parent repos)."""
        # Create nested git repos
        outer_repo = tmp_path / "outer"
        outer_repo.mkdir()
        (outer_repo / ".git").mkdir()

        inner_repo = outer_repo / "inner"
        inner_repo.mkdir()
        (inner_repo / ".git").mkdir()

        # Should find inner repo, not outer
        result = _find_project_root(inner_repo)
        assert result == inner_repo


class TestProjectAgentMdFinding:
    """Test finding project-specific AGENTS.md files."""

    def test_find_agent_md_in_deepagents_dir(self, tmp_path: Path) -> None:
        """Test finding AGENTS.md in .deepagents/ directory."""
        project_root = tmp_path / "project"
        project_root.mkdir()

        # Create .deepagents/AGENTS.md
        deepagents_dir = project_root / ".deepagents"
        deepagents_dir.mkdir()
        agent_md = deepagents_dir / "AGENTS.md"
        agent_md.write_text("Project instructions")

        result = _find_project_agent_md(project_root)
        assert len(result) == 1
        assert result[0] == agent_md

    def test_find_agent_md_in_root(self, tmp_path: Path) -> None:
        """Test finding AGENTS.md in project root (fallback)."""
        project_root = tmp_path / "project"
        project_root.mkdir()

        # Create root-level AGENTS.md (no .deepagents/)
        agent_md = project_root / "AGENTS.md"
        agent_md.write_text("Project instructions")

        result = _find_project_agent_md(project_root)
        assert len(result) == 1
        assert result[0] == agent_md

    def test_both_agent_md_files_combined(self, tmp_path: Path) -> None:
        """Test that both AGENTS.md files are returned when both exist."""
        project_root = tmp_path / "project"
        project_root.mkdir()

        # Create both locations
        deepagents_dir = project_root / ".deepagents"
        deepagents_dir.mkdir()
        deepagents_md = deepagents_dir / "AGENTS.md"
        deepagents_md.write_text("In .deepagents/")

        root_md = project_root / "AGENTS.md"
        root_md.write_text("In root")

        # Should return both, with .deepagents/ first
        result = _find_project_agent_md(project_root)
        assert len(result) == 2
        assert result[0] == deepagents_md
        assert result[1] == root_md

    def test_find_agent_md_not_found(self, tmp_path: Path) -> None:
        """Test that empty list is returned when no AGENTS.md exists."""
        project_root = tmp_path / "project"
        project_root.mkdir()

        result = _find_project_agent_md(project_root)
        assert result == []


class TestValidateModelCapabilities:
    """Tests for model capability validation."""

    @patch("deepagents_cli.config.console")
    def test_model_without_profile_attribute_warns(self, mock_console: Mock) -> None:
        """Test that models without profile attribute trigger a warning."""
        model = Mock(spec=[])  # No profile attribute
        validate_model_capabilities(model, "test-model")

        mock_console.print.assert_called_once()
        call_args = mock_console.print.call_args[0][0]
        assert "No capability profile" in call_args
        assert "test-model" in call_args

    @patch("deepagents_cli.config.console")
    def test_model_with_none_profile_warns(self, mock_console: Mock) -> None:
        """Test that models with `profile=None` trigger a warning."""
        model = Mock()
        model.profile = None

        validate_model_capabilities(model, "test-model")

        mock_console.print.assert_called_once()
        call_args = mock_console.print.call_args[0][0]
        assert "No capability profile" in call_args

    @patch("deepagents_cli.config.console")
    def test_model_with_tool_calling_false_exits(self, mock_console: Mock) -> None:
        """Test that models with `tool_calling=False` cause `sys.exit(1)`."""
        model = Mock()
        model.profile = {"tool_calling": False}

        with pytest.raises(SystemExit) as exc_info:
            validate_model_capabilities(model, "no-tools-model")

        assert exc_info.value.code == 1
        # Verify error messages were printed
        assert mock_console.print.call_count == 3
        error_call = mock_console.print.call_args_list[0][0][0]
        assert "does not support tool calling" in error_call
        assert "no-tools-model" in error_call

    @patch("deepagents_cli.config.console")
    def test_model_with_tool_calling_true_passes(self, mock_console: Mock) -> None:
        """Test that models with `tool_calling=True` pass without messages."""
        model = Mock()
        model.profile = {"tool_calling": True}

        validate_model_capabilities(model, "tools-model")

        mock_console.print.assert_not_called()

    @patch("deepagents_cli.config.console")
    def test_model_with_tool_calling_none_passes(self, mock_console: Mock) -> None:
        """Test that models with `tool_calling=None` (missing) pass."""
        model = Mock()
        model.profile = {"other_capability": True}

        validate_model_capabilities(model, "model-without-tool-key")

        mock_console.print.assert_not_called()

    @patch("deepagents_cli.config.console")
    def test_model_with_limited_context_warns(self, mock_console: Mock) -> None:
        """Test that models with <8000 token context trigger a warning."""
        model = Mock()
        model.profile = {"tool_calling": True, "max_input_tokens": 4096}

        validate_model_capabilities(model, "small-context-model")

        mock_console.print.assert_called_once()
        call_args = mock_console.print.call_args[0][0]
        assert "limited context" in call_args
        assert "4,096" in call_args
        assert "small-context-model" in call_args

    @patch("deepagents_cli.config.console")
    def test_model_with_adequate_context_passes(self, mock_console: Mock) -> None:
        """Confirm that models with >=8000 token context pass silently."""
        model = Mock()
        model.profile = {"tool_calling": True, "max_input_tokens": 128000}

        validate_model_capabilities(model, "large-context-model")

        mock_console.print.assert_not_called()

    @patch("deepagents_cli.config.console")
    def test_model_without_max_input_tokens_passes(self, mock_console: Mock) -> None:
        """Test that models without `max_input_tokens` key pass silently."""
        model = Mock()
        model.profile = {"tool_calling": True}

        validate_model_capabilities(model, "no-context-info-model")

        mock_console.print.assert_not_called()

    @patch("deepagents_cli.config.console")
    def test_model_with_zero_max_input_tokens_passes(self, mock_console: Mock) -> None:
        """Test that models with `max_input_tokens=0` pass (falsy value check)."""
        model = Mock()
        model.profile = {"tool_calling": True, "max_input_tokens": 0}

        validate_model_capabilities(model, "zero-context-model")

        # Should pass because 0 is falsy, so the condition `if max_input_tokens` fails
        mock_console.print.assert_not_called()

    @patch("deepagents_cli.config.console")
    def test_model_with_empty_profile_passes(self, mock_console: Mock) -> None:
        """Test that models with empty profile dict pass silently."""
        model = Mock()
        model.profile = {}

        validate_model_capabilities(model, "empty-profile-model")

        mock_console.print.assert_not_called()
