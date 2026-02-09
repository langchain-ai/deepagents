"""Tests for config module including project discovery utilities."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from deepagents_cli.config import (
    RECOMMENDED_SAFE_SHELL_COMMANDS,
    Settings,
    _find_project_agent_md,
    _find_project_root,
    create_model,
    fetch_langsmith_project_url,
    get_langsmith_project_name,
    parse_shell_allow_list,
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
        settings = Settings.from_environment()
        expected = Path.home() / ".agents"
        assert settings.user_agents_dir == expected

    def test_get_user_agent_skills_dir(self) -> None:
        """Test get_user_agent_skills_dir returns ~/.agents/skills."""
        settings = Settings.from_environment()
        expected = Path.home() / ".agents" / "skills"
        assert settings.get_user_agent_skills_dir() == expected

    def test_get_project_agent_skills_dir_with_project(self, tmp_path: Path) -> None:
        """Test get_project_agent_skills_dir returns .agents/skills in project."""
        # Create a mock project with .git
        project_root = tmp_path / "my-project"
        project_root.mkdir()
        (project_root / ".git").mkdir()

        settings = Settings.from_environment(start_path=project_root)
        expected = project_root / ".agents" / "skills"
        assert settings.get_project_agent_skills_dir() == expected

    def test_get_project_agent_skills_dir_without_project(self, tmp_path: Path) -> None:
        """Test get_project_agent_skills_dir returns None when not in a project."""
        # Create a directory without .git
        no_project = tmp_path / "no-project"
        no_project.mkdir()

        settings = Settings.from_environment(start_path=no_project)
        assert settings.get_project_agent_skills_dir() is None


class TestCreateModelProfileExtraction:
    """Tests for profile extraction in create_model."""

    def setup_method(self) -> None:
        """Reset settings before each test."""
        settings.model_context_limit = None
        settings.model_name = None
        settings.model_provider = None

    def _patch_settings_for_anthropic(self) -> dict[str, str | None]:
        """Return original settings and set up for Anthropic-only."""
        original = {
            "anthropic_api_key": settings.anthropic_api_key,
            "openai_api_key": settings.openai_api_key,
            "google_api_key": settings.google_api_key,
            "google_cloud_project": settings.google_cloud_project,
        }
        settings.anthropic_api_key = "test-key"
        settings.openai_api_key = None
        settings.google_api_key = None
        settings.google_cloud_project = None
        return original

    def _restore_settings(self, original: dict[str, str | None]) -> None:
        """Restore original settings."""
        settings.anthropic_api_key = original["anthropic_api_key"]
        settings.openai_api_key = original["openai_api_key"]
        settings.google_api_key = original["google_api_key"]
        settings.google_cloud_project = original["google_cloud_project"]

    @patch("langchain_anthropic.ChatAnthropic")
    def test_extracts_context_limit_from_profile(self, mock_chat_class: Mock) -> None:
        """Test that model_context_limit is extracted from model profile."""
        mock_model = Mock()
        mock_model.profile = {"max_input_tokens": 200000, "tool_calling": True}
        mock_chat_class.return_value = mock_model

        original = self._patch_settings_for_anthropic()
        try:
            create_model("claude-sonnet-4-5-20250929")
            assert settings.model_context_limit == 200000
        finally:
            self._restore_settings(original)

    @patch("langchain_anthropic.ChatAnthropic")
    def test_handles_missing_profile_gracefully(self, mock_chat_class: Mock) -> None:
        """Test that missing profile attribute leaves context_limit as None."""
        mock_model = Mock(spec=["invoke"])  # No profile attribute
        mock_chat_class.return_value = mock_model

        original = self._patch_settings_for_anthropic()
        try:
            create_model("claude-sonnet-4-5-20250929")
            assert settings.model_context_limit is None
        finally:
            self._restore_settings(original)

    @patch("langchain_anthropic.ChatAnthropic")
    def test_handles_none_profile(self, mock_chat_class: Mock) -> None:
        """Test that profile=None leaves context_limit as None."""
        mock_model = Mock()
        mock_model.profile = None
        mock_chat_class.return_value = mock_model

        original = self._patch_settings_for_anthropic()
        try:
            create_model("claude-sonnet-4-5-20250929")
            assert settings.model_context_limit is None
        finally:
            self._restore_settings(original)

    @patch("langchain_anthropic.ChatAnthropic")
    def test_handles_non_dict_profile(self, mock_chat_class: Mock) -> None:
        """Test that non-dict profile is handled safely."""
        mock_model = Mock()
        mock_model.profile = "not a dict"
        mock_chat_class.return_value = mock_model

        original = self._patch_settings_for_anthropic()
        try:
            create_model("claude-sonnet-4-5-20250929")
            assert settings.model_context_limit is None
        finally:
            self._restore_settings(original)

    @patch("langchain_anthropic.ChatAnthropic")
    def test_handles_non_int_max_input_tokens(self, mock_chat_class: Mock) -> None:
        """Test that string max_input_tokens is ignored."""
        mock_model = Mock()
        mock_model.profile = {"max_input_tokens": "200000"}  # String, not int
        mock_chat_class.return_value = mock_model

        original = self._patch_settings_for_anthropic()
        try:
            create_model("claude-sonnet-4-5-20250929")
            assert settings.model_context_limit is None
        finally:
            self._restore_settings(original)

    @patch("langchain_anthropic.ChatAnthropic")
    def test_handles_missing_max_input_tokens_key(self, mock_chat_class: Mock) -> None:
        """Test that profile without max_input_tokens key is handled."""
        mock_model = Mock()
        mock_model.profile = {"tool_calling": True}  # No max_input_tokens
        mock_chat_class.return_value = mock_model

        original = self._patch_settings_for_anthropic()
        try:
            create_model("claude-sonnet-4-5-20250929")
            assert settings.model_context_limit is None
        finally:
            self._restore_settings(original)


class TestParseShellAllowList:
    """Test parsing shell allow-list strings."""

    def test_none_input_returns_none(self) -> None:
        """Test that None input returns None."""
        result = parse_shell_allow_list(None)
        assert result is None

    def test_empty_string_returns_none(self) -> None:
        """Test that empty string returns None."""
        result = parse_shell_allow_list("")
        assert result is None

    def test_recommended_only(self) -> None:
        """Test that 'recommended' returns the full recommended list."""
        result = parse_shell_allow_list("recommended")
        assert result == list(RECOMMENDED_SAFE_SHELL_COMMANDS)

    def test_recommended_case_insensitive(self) -> None:
        """Test that 'RECOMMENDED', 'Recommended', etc. all work."""
        for variant in ["RECOMMENDED", "Recommended", "ReCoMmEnDeD", "  recommended  "]:
            result = parse_shell_allow_list(variant)
            assert result == list(RECOMMENDED_SAFE_SHELL_COMMANDS)

    def test_custom_commands_only(self) -> None:
        """Test parsing custom commands without 'recommended'."""
        result = parse_shell_allow_list("ls,cat,grep")
        assert result == ["ls", "cat", "grep"]

    def test_custom_commands_with_whitespace(self) -> None:
        """Test parsing custom commands with whitespace."""
        result = parse_shell_allow_list("ls , cat , grep")
        assert result == ["ls", "cat", "grep"]

    def test_recommended_merged_with_custom_commands(self) -> None:
        """Test that 'recommended' in list merges with custom commands."""
        result = parse_shell_allow_list("recommended,mycmd,myothercmd")
        expected = [*list(RECOMMENDED_SAFE_SHELL_COMMANDS), "mycmd", "myothercmd"]
        assert result == expected

    def test_custom_commands_before_recommended(self) -> None:
        """Test custom commands before 'recommended' keyword."""
        result = parse_shell_allow_list("mycmd,recommended,myothercmd")
        # mycmd first, then all recommended, then myothercmd
        expected = ["mycmd", *list(RECOMMENDED_SAFE_SHELL_COMMANDS), "myothercmd"]
        assert result == expected

    def test_duplicate_removal(self) -> None:
        """Test that duplicates are removed while preserving order."""
        result = parse_shell_allow_list("ls,cat,ls,grep,cat")
        assert result == ["ls", "cat", "grep"]

    def test_duplicate_removal_with_recommended(self) -> None:
        """Test that duplicates from recommended are removed."""
        # 'ls' is in RECOMMENDED_SAFE_SHELL_COMMANDS
        result = parse_shell_allow_list("ls,recommended,mycmd")
        # Should have ls once (first occurrence), then all recommended commands
        # except ls (since it's already in), then mycmd
        assert result is not None
        assert result[0] == "ls"
        # ls should not appear again
        assert result.count("ls") == 1
        # mycmd should appear once at the end
        assert result[-1] == "mycmd"
        # Total should be: 1 (ls) + len(recommended) - 1 (duplicate ls) + 1 (mycmd)
        # Which simplifies to: len(recommended) + 1
        assert len(result) == len(RECOMMENDED_SAFE_SHELL_COMMANDS) + 1

    def test_empty_commands_ignored(self) -> None:
        """Test that empty strings from split are ignored."""
        result = parse_shell_allow_list("ls,,cat,,,grep,")
        assert result == ["ls", "cat", "grep"]


class TestGetLangsmithProjectName:
    """Tests for get_langsmith_project_name()."""

    def test_returns_none_without_api_key(self) -> None:
        """Should return None when no LangSmith API key is set."""
        env = {
            "LANGSMITH_API_KEY": "",
            "LANGCHAIN_API_KEY": "",
            "LANGSMITH_TRACING": "true",
        }
        with patch.dict("os.environ", env, clear=False):
            assert get_langsmith_project_name() is None

    def test_returns_none_without_tracing(self) -> None:
        """Should return None when tracing is not enabled."""
        env = {
            "LANGSMITH_API_KEY": "lsv2_test",
            "LANGSMITH_TRACING": "",
            "LANGCHAIN_TRACING_V2": "",
        }
        with patch.dict("os.environ", env, clear=False):
            assert get_langsmith_project_name() is None

    def test_returns_project_from_settings(self) -> None:
        """Should prefer settings.deepagents_langchain_project."""
        env = {
            "LANGSMITH_API_KEY": "lsv2_test",
            "LANGSMITH_TRACING": "true",
            "LANGSMITH_PROJECT": "env-project",
        }
        with (
            patch.dict("os.environ", env, clear=False),
            patch("deepagents_cli.config.settings") as mock_settings,
        ):
            mock_settings.deepagents_langchain_project = "settings-project"
            assert get_langsmith_project_name() == "settings-project"

    def test_falls_back_to_env_project(self) -> None:
        """Should fall back to LANGSMITH_PROJECT env var."""
        env = {
            "LANGSMITH_API_KEY": "lsv2_test",
            "LANGSMITH_TRACING": "true",
            "LANGSMITH_PROJECT": "env-project",
        }
        with (
            patch.dict("os.environ", env, clear=False),
            patch("deepagents_cli.config.settings") as mock_settings,
        ):
            mock_settings.deepagents_langchain_project = None
            assert get_langsmith_project_name() == "env-project"

    def test_falls_back_to_default(self) -> None:
        """Should fall back to 'default' when no project name configured."""
        env = {
            "LANGSMITH_API_KEY": "lsv2_test",
            "LANGSMITH_TRACING": "true",
        }
        with (
            patch.dict("os.environ", env, clear=False),
            patch("deepagents_cli.config.settings") as mock_settings,
        ):
            mock_settings.deepagents_langchain_project = None
            assert get_langsmith_project_name() == "default"

    def test_accepts_langchain_api_key(self) -> None:
        """Should accept LANGCHAIN_API_KEY as alternative to LANGSMITH_API_KEY."""
        env = {
            "LANGSMITH_API_KEY": "",
            "LANGCHAIN_API_KEY": "lsv2_test",
            "LANGSMITH_TRACING": "true",
        }
        with (
            patch.dict("os.environ", env, clear=False),
            patch("deepagents_cli.config.settings") as mock_settings,
        ):
            mock_settings.deepagents_langchain_project = None
            assert get_langsmith_project_name() == "default"


class TestFetchLangsmithProjectUrl:
    """Tests for fetch_langsmith_project_url()."""

    def test_returns_url_on_success(self) -> None:
        """Should return the project URL from the LangSmith client."""

        class FakeProject:
            url = "https://smith.langchain.com/o/org/projects/p/proj"

        with patch("langsmith.Client") as mock_client_cls:
            mock_client_cls.return_value.read_project.return_value = FakeProject()
            result = fetch_langsmith_project_url("my-project")

        assert result == "https://smith.langchain.com/o/org/projects/p/proj"

    def test_returns_none_on_error(self) -> None:
        """Should return None when the LangSmith client raises."""
        with patch("langsmith.Client") as mock_client_cls:
            mock_client_cls.return_value.read_project.side_effect = OSError("timeout")
            result = fetch_langsmith_project_url("my-project")

        assert result is None

    def test_returns_none_when_url_is_none(self) -> None:
        """Should return None when the project has no URL."""

        class FakeProject:
            url = None

        with patch("langsmith.Client") as mock_client_cls:
            mock_client_cls.return_value.read_project.return_value = FakeProject()
            result = fetch_langsmith_project_url("my-project")

        assert result is None
