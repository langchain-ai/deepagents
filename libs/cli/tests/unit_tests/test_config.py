"""Tests for config module including project discovery utilities."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from deepagents_cli.config import (
    _find_project_agent_md,
    _find_project_root,
    create_model,
    settings,
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


class TestAgentsAliasDirectories:
    """Tests for .agents directory alias methods."""

    def test_user_agents_dir(self) -> None:
        """Test user_agents_dir returns ~/.agents."""
        from deepagents_cli.config import Settings

        settings = Settings.from_environment()
        expected = Path.home() / ".agents"
        assert settings.user_agents_dir == expected

    def test_get_user_agent_skills_dir(self) -> None:
        """Test get_user_agent_skills_dir returns ~/.agents/skills."""
        from deepagents_cli.config import Settings

        settings = Settings.from_environment()
        expected = Path.home() / ".agents" / "skills"
        assert settings.get_user_agent_skills_dir() == expected

    def test_get_project_agent_skills_dir_with_project(self, tmp_path: Path) -> None:
        """Test get_project_agent_skills_dir returns .agents/skills in project."""
        from deepagents_cli.config import Settings

        # Create a mock project with .git
        project_root = tmp_path / "my-project"
        project_root.mkdir()
        (project_root / ".git").mkdir()

        settings = Settings.from_environment(start_path=project_root)
        expected = project_root / ".agents" / "skills"
        assert settings.get_project_agent_skills_dir() == expected

    def test_get_project_agent_skills_dir_without_project(self, tmp_path: Path) -> None:
        """Test get_project_agent_skills_dir returns None when not in a project."""
        from deepagents_cli.config import Settings

        # Create a directory without .git
        no_project = tmp_path / "no-project"
        no_project.mkdir()

        settings = Settings.from_environment(start_path=no_project)
        assert settings.get_project_agent_skills_dir() is None


class TestCreateModelProfileExtraction:
    """Tests for profile extraction in create_model.

    These tests verify that create_model correctly extracts the model_context_limit
    from the model's profile attribute. We mock init_chat_model since create_model
    now uses it internally.
    """

    def setup_method(self) -> None:
        """Reset settings before each test."""
        settings.model_context_limit = None
        settings.model_name = None
        settings.model_provider = None

    @patch("deepagents_cli.config.init_chat_model")
    def test_extracts_context_limit_from_profile(
        self, mock_init_chat_model: Mock
    ) -> None:
        """Test that model_context_limit is extracted from model profile."""
        mock_model = Mock()
        mock_model.profile = {"max_input_tokens": 200000, "tool_calling": True}
        mock_init_chat_model.return_value = mock_model

        create_model("anthropic:claude-sonnet-4-5")
        assert settings.model_context_limit == 200000

    @patch("deepagents_cli.config.init_chat_model")
    def test_handles_missing_profile_gracefully(
        self, mock_init_chat_model: Mock
    ) -> None:
        """Test that missing profile attribute leaves context_limit as None."""
        mock_model = Mock(spec=["invoke"])  # No profile attribute
        mock_init_chat_model.return_value = mock_model

        create_model("anthropic:claude-sonnet-4-5")
        assert settings.model_context_limit is None

    @patch("deepagents_cli.config.init_chat_model")
    def test_handles_none_profile(self, mock_init_chat_model: Mock) -> None:
        """Test that profile=None leaves context_limit as None."""
        mock_model = Mock()
        mock_model.profile = None
        mock_init_chat_model.return_value = mock_model

        create_model("anthropic:claude-sonnet-4-5")
        assert settings.model_context_limit is None

    @patch("deepagents_cli.config.init_chat_model")
    def test_handles_non_dict_profile(self, mock_init_chat_model: Mock) -> None:
        """Test that non-dict profile is handled safely."""
        mock_model = Mock()
        mock_model.profile = "not a dict"
        mock_init_chat_model.return_value = mock_model

        create_model("anthropic:claude-sonnet-4-5")
        assert settings.model_context_limit is None

    @patch("deepagents_cli.config.init_chat_model")
    def test_handles_non_int_max_input_tokens(self, mock_init_chat_model: Mock) -> None:
        """Test that string max_input_tokens is ignored."""
        mock_model = Mock()
        mock_model.profile = {"max_input_tokens": "200000"}  # String, not int
        mock_init_chat_model.return_value = mock_model

        create_model("anthropic:claude-sonnet-4-5")
        assert settings.model_context_limit is None

    @patch("deepagents_cli.config.init_chat_model")
    def test_handles_missing_max_input_tokens_key(
        self, mock_init_chat_model: Mock
    ) -> None:
        """Test that profile without max_input_tokens key is handled."""
        mock_model = Mock()
        mock_model.profile = {"tool_calling": True}  # No max_input_tokens
        mock_init_chat_model.return_value = mock_model

        create_model("anthropic:claude-sonnet-4-5")
        assert settings.model_context_limit is None
