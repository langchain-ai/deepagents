"""Tests for config module including project discovery utilities."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from deepagents_cli import model_config
from deepagents_cli.config import (
    RECOMMENDED_SAFE_SHELL_COMMANDS,
    ModelResult,
    Settings,
    _create_model_from_class,
    _find_project_agent_md,
    _find_project_root,
    _get_provider_kwargs,
    build_langsmith_thread_url,
    create_model,
    detect_provider,
    fetch_langsmith_project_url,
    get_langsmith_project_name,
    parse_shell_allow_list,
    reset_langsmith_url_cache,
    settings,
    validate_model_capabilities,
)
from deepagents_cli.model_config import ModelConfigError, clear_caches


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

    def test_skips_paths_with_permission_errors(self, tmp_path: Path) -> None:
        """Test that OSError from Path.exists() is caught gracefully."""
        project_root = tmp_path / "project"
        project_root.mkdir()

        real_md = project_root / "AGENTS.md"
        real_md.write_text("root instructions")

        original_exists = Path.exists

        def patched_exists(self: Path) -> bool:
            if self.name == "AGENTS.md" and ".deepagents" in str(self):
                msg = "Permission denied"
                raise PermissionError(msg)
            return original_exists(self)

        with patch.object(Path, "exists", patched_exists):
            result = _find_project_agent_md(project_root)

        assert len(result) == 1
        assert result[0] == real_md


class TestSettingsGetProjectAgentMdPath:
    """Test Settings.get_project_agent_md_path() integration."""

    def test_returns_empty_list_when_no_project_root(self) -> None:
        """Should return [] when project_root is None."""
        s = Settings.__new__(Settings)
        s.project_root = None
        assert s.get_project_agent_md_path() == []

    def test_returns_existing_paths(self, tmp_path: Path) -> None:
        """Should return existing AGENTS.md paths from project root."""
        deepagents_dir = tmp_path / ".deepagents"
        deepagents_dir.mkdir()
        deepagents_md = deepagents_dir / "AGENTS.md"
        deepagents_md.write_text("inner")

        root_md = tmp_path / "AGENTS.md"
        root_md.write_text("root")

        s = Settings.__new__(Settings)
        s.project_root = tmp_path

        result = s.get_project_agent_md_path()
        assert result == [deepagents_md, root_md]

    def test_returns_empty_when_no_agents_md_files(self, tmp_path: Path) -> None:
        """Should return [] when project exists but has no AGENTS.md."""
        s = Settings.__new__(Settings)
        s.project_root = tmp_path
        assert s.get_project_agent_md_path() == []


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
    """Tests for profile extraction in create_model.

    These tests verify that create_model correctly extracts the context_limit
    from the model's profile attribute. We mock init_chat_model since create_model
    now uses it internally.
    """

    @patch("langchain.chat_models.init_chat_model")
    def test_extracts_context_limit_from_profile(
        self, mock_init_chat_model: Mock
    ) -> None:
        """Test that context_limit is extracted from model profile."""
        mock_model = Mock()
        mock_model.profile = {"max_input_tokens": 200000, "tool_calling": True}
        mock_init_chat_model.return_value = mock_model

        result = create_model("anthropic:claude-sonnet-4-5")
        assert result.context_limit == 200000

    @patch("langchain.chat_models.init_chat_model")
    def test_handles_missing_profile_gracefully(
        self, mock_init_chat_model: Mock
    ) -> None:
        """Test that missing profile attribute leaves context_limit as None."""
        mock_model = Mock(spec=["invoke"])  # No profile attribute
        mock_init_chat_model.return_value = mock_model

        result = create_model("anthropic:claude-sonnet-4-5")
        assert result.context_limit is None

    @patch("langchain.chat_models.init_chat_model")
    def test_handles_none_profile(self, mock_init_chat_model: Mock) -> None:
        """Test that profile=None leaves context_limit as None."""
        mock_model = Mock()
        mock_model.profile = None
        mock_init_chat_model.return_value = mock_model

        result = create_model("anthropic:claude-sonnet-4-5")
        assert result.context_limit is None

    @patch("langchain.chat_models.init_chat_model")
    def test_handles_non_dict_profile(self, mock_init_chat_model: Mock) -> None:
        """Test that non-dict profile is handled safely."""
        mock_model = Mock()
        mock_model.profile = "not a dict"
        mock_init_chat_model.return_value = mock_model

        result = create_model("anthropic:claude-sonnet-4-5")
        assert result.context_limit is None

    @patch("langchain.chat_models.init_chat_model")
    def test_handles_non_int_max_input_tokens(self, mock_init_chat_model: Mock) -> None:
        """Test that string max_input_tokens is ignored."""
        mock_model = Mock()
        mock_model.profile = {"max_input_tokens": "200000"}  # String, not int
        mock_init_chat_model.return_value = mock_model

        result = create_model("anthropic:claude-sonnet-4-5")
        assert result.context_limit is None

    @patch("langchain.chat_models.init_chat_model")
    def test_handles_missing_max_input_tokens_key(
        self, mock_init_chat_model: Mock
    ) -> None:
        """Test that profile without max_input_tokens key is handled."""
        mock_model = Mock()
        mock_model.profile = {"tool_calling": True}  # No max_input_tokens
        mock_init_chat_model.return_value = mock_model

        result = create_model("anthropic:claude-sonnet-4-5")
        assert result.context_limit is None


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

    def setup_method(self) -> None:
        """Clear LangSmith URL cache before each test."""
        reset_langsmith_url_cache()

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

    def test_caches_result_after_first_call(self) -> None:
        """Should only call the LangSmith client once for repeated invocations."""

        class FakeProject:
            url = "https://smith.langchain.com/o/org/projects/p/proj"

        with patch("langsmith.Client") as mock_client_cls:
            mock_client_cls.return_value.read_project.return_value = FakeProject()
            first = fetch_langsmith_project_url("my-project")
            second = fetch_langsmith_project_url("my-project")

        assert first == "https://smith.langchain.com/o/org/projects/p/proj"
        assert second == first
        mock_client_cls.assert_called_once()

    def test_caches_none_on_failure(self) -> None:
        """Should cache None on failure so retries don't make network calls."""
        with patch("langsmith.Client") as mock_client_cls:
            mock_client_cls.return_value.read_project.side_effect = OSError("timeout")
            first = fetch_langsmith_project_url("my-project")
            second = fetch_langsmith_project_url("my-project")

        assert first is None
        assert second is None
        mock_client_cls.assert_called_once()

    def test_different_project_name_fetches_again(self) -> None:
        """Should fetch again when called with a different project name."""

        class FakeProjectA:
            url = "https://smith.langchain.com/o/org/projects/p/a"

        class FakeProjectB:
            url = "https://smith.langchain.com/o/org/projects/p/b"

        with patch("langsmith.Client") as mock_client_cls:
            mock_client_cls.return_value.read_project.side_effect = [
                FakeProjectA(),
                FakeProjectB(),
            ]
            first = fetch_langsmith_project_url("project-a")
            second = fetch_langsmith_project_url("project-b")

        assert first == "https://smith.langchain.com/o/org/projects/p/a"
        assert second == "https://smith.langchain.com/o/org/projects/p/b"
        assert mock_client_cls.return_value.read_project.call_count == 2


class TestBuildLangsmithThreadUrl:
    """Tests for build_langsmith_thread_url()."""

    def setup_method(self) -> None:
        """Clear LangSmith URL cache before each test."""
        reset_langsmith_url_cache()

    def test_returns_url_when_configured(self) -> None:
        """Should return a full thread URL when LangSmith is configured."""

        class FakeProject:
            url = "https://smith.langchain.com/o/org/projects/p/proj"

        with (
            patch(
                "deepagents_cli.config.get_langsmith_project_name",
                return_value="my-project",
            ),
            patch("langsmith.Client") as mock_client_cls,
        ):
            mock_client_cls.return_value.read_project.return_value = FakeProject()
            result = build_langsmith_thread_url("thread-123")

        assert (
            result
            == "https://smith.langchain.com/o/org/projects/p/proj/t/thread-123?utm_source=deepagents-cli"
        )

    def test_strips_trailing_slash(self) -> None:
        """Should not produce double slashes when project URL has trailing slash."""

        class FakeProject:
            url = "https://smith.langchain.com/o/org/projects/p/proj/"

        with (
            patch(
                "deepagents_cli.config.get_langsmith_project_name",
                return_value="my-project",
            ),
            patch("langsmith.Client") as mock_client_cls,
        ):
            mock_client_cls.return_value.read_project.return_value = FakeProject()
            result = build_langsmith_thread_url("thread-123")

        assert (
            result
            == "https://smith.langchain.com/o/org/projects/p/proj/t/thread-123?utm_source=deepagents-cli"
        )

    def test_returns_none_when_no_project_name(self) -> None:
        """Should return None when LangSmith project name is not configured."""
        with patch(
            "deepagents_cli.config.get_langsmith_project_name",
            return_value=None,
        ):
            result = build_langsmith_thread_url("thread-123")

        assert result is None

    def test_returns_none_when_fetch_fails(self) -> None:
        """Should return None when the project URL cannot be resolved."""
        with (
            patch(
                "deepagents_cli.config.get_langsmith_project_name",
                return_value="my-project",
            ),
            patch("langsmith.Client") as mock_client_cls,
        ):
            mock_client_cls.return_value.read_project.side_effect = OSError("timeout")
            result = build_langsmith_thread_url("thread-123")

        assert result is None


class TestGetProviderKwargsConfigFallback:
    """Tests for _get_provider_kwargs() config-file fallback."""

    def setup_method(self) -> None:
        """Clear model config cache before each test."""
        clear_caches()

    def test_returns_base_url_from_config(self, tmp_path: Path) -> None:
        """Returns base_url from config for non-hardcoded provider."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.fireworks]
models = ["llama-v3p1-70b"]
base_url = "https://api.fireworks.ai/inference/v1"
api_key_env = "FIREWORKS_API_KEY"
""")
        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch.dict("os.environ", {"FIREWORKS_API_KEY": "test-key"}, clear=False),
        ):
            kwargs = _get_provider_kwargs("fireworks")

        assert kwargs["base_url"] == "https://api.fireworks.ai/inference/v1"
        assert kwargs["api_key"] == "test-key"

    def test_returns_api_key_from_config(self, tmp_path: Path) -> None:
        """Returns resolved api_key from config-file api_key_env."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.together]
models = ["meta-llama/Llama-3-70b"]
api_key_env = "TOGETHER_API_KEY"
""")
        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch.dict("os.environ", {"TOGETHER_API_KEY": "together-key"}, clear=False),
        ):
            kwargs = _get_provider_kwargs("together")

        assert kwargs["api_key"] == "together-key"
        assert "base_url" not in kwargs

    def test_omits_api_key_when_env_not_set(self, tmp_path: Path) -> None:
        """Omits api_key when the env var is not set."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.fireworks]
models = ["llama-v3p1-70b"]
api_key_env = "FIREWORKS_API_KEY"
""")
        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch.dict("os.environ", {}, clear=True),
        ):
            kwargs = _get_provider_kwargs("fireworks")

        assert "api_key" not in kwargs

    def test_returns_empty_for_unknown_config_provider(self) -> None:
        """Returns empty dict for provider not in hardcoded map or config."""
        kwargs = _get_provider_kwargs("nonexistent_provider_xyz")
        assert kwargs == {}

    def test_unconfigured_providers_return_empty(self) -> None:
        """Providers without config return empty kwargs."""
        kwargs = _get_provider_kwargs("anthropic")
        assert kwargs == {}

        kwargs = _get_provider_kwargs("google_genai")
        assert kwargs == {}

    def test_merges_config_params(self, tmp_path: Path) -> None:
        """Merges params from config with base_url and api_key."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.custom]
models = ["my-model"]
base_url = "https://my-endpoint.example.com"
api_key_env = "CUSTOM_KEY"

[models.providers.custom.params]
temperature = 0
max_tokens = 4096
""")
        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch.dict("os.environ", {"CUSTOM_KEY": "secret"}, clear=False),
        ):
            kwargs = _get_provider_kwargs("custom")

        assert kwargs["temperature"] == 0
        assert kwargs["max_tokens"] == 4096
        assert kwargs["base_url"] == "https://my-endpoint.example.com"
        assert kwargs["api_key"] == "secret"

    def test_passes_model_name_for_per_model_params(self, tmp_path: Path) -> None:
        """Per-model params are merged when model_name is provided."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.ollama]
models = ["qwen3:4b", "llama3"]

[models.providers.ollama.params]
temperature = 0
num_ctx = 8192

[models.providers.ollama.params."qwen3:4b"]
temperature = 0.5
num_ctx = 4000
""")
        with patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path):
            kwargs = _get_provider_kwargs("ollama", model_name="qwen3:4b")

        assert kwargs["temperature"] == 0.5
        assert kwargs["num_ctx"] == 4000

    def test_model_name_none_uses_provider_params(self, tmp_path: Path) -> None:
        """model_name=None returns provider params without per-model merge."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.ollama]
models = ["qwen3:4b"]

[models.providers.ollama.params]
temperature = 0

[models.providers.ollama.params."qwen3:4b"]
temperature = 0.5
""")
        with patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path):
            kwargs = _get_provider_kwargs("ollama")

        assert kwargs["temperature"] == 0

    def test_base_url_and_api_key_override_config_params(self, tmp_path: Path) -> None:
        """base_url/api_key from config fields override same keys in params."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.custom]
models = ["my-model"]
base_url = "https://correct-url.com"
api_key_env = "CUSTOM_KEY"

[models.providers.custom.params]
base_url = "https://wrong-url.com"
""")
        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch.dict("os.environ", {"CUSTOM_KEY": "secret"}, clear=False),
        ):
            kwargs = _get_provider_kwargs("custom")

        # Explicit base_url field should win over kwargs.base_url
        assert kwargs["base_url"] == "https://correct-url.com"


class TestOpenRouterHeaders:
    """Tests for OpenRouter default attribution headers."""

    def setup_method(self) -> None:
        """Clear model config cache before each test."""
        clear_caches()

    def test_injects_default_headers(self) -> None:
        """Injects HTTP-Referer and X-Title for openrouter provider."""
        kwargs = _get_provider_kwargs("openrouter")

        assert "default_headers" in kwargs
        assert kwargs["default_headers"]["HTTP-Referer"] == (
            "https://github.com/langchain-ai/deepagents"
        )
        assert kwargs["default_headers"]["X-Title"] == "Deep Agents CLI"

    def test_per_model_headers_override_defaults(self, tmp_path: Path) -> None:
        """Per-model default_headers override built-in defaults."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.openrouter]
models = ["deepseek/deepseek-chat"]

[models.providers.openrouter.params."deepseek/deepseek-chat"]
default_headers = {X-Title = "My Custom App"}
""")
        with patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path):
            kwargs = _get_provider_kwargs(
                "openrouter", model_name="deepseek/deepseek-chat"
            )

        assert kwargs["default_headers"]["X-Title"] == "My Custom App"
        # Built-in HTTP-Referer should still be present
        assert kwargs["default_headers"]["HTTP-Referer"] == (
            "https://github.com/langchain-ai/deepagents"
        )

    def test_no_headers_for_other_providers(self) -> None:
        """Other providers do not get OpenRouter attribution headers."""
        kwargs = _get_provider_kwargs("openai")
        assert "default_headers" not in kwargs


class TestCreateModelFromClass:
    """Tests for _create_model_from_class() custom class factory."""

    def test_raises_on_invalid_class_path_format(self) -> None:
        """Raises ModelConfigError when class_path lacks colon."""
        from deepagents_cli.model_config import ModelConfigError

        with pytest.raises(ModelConfigError, match="Invalid class_path"):
            _create_model_from_class("my_package.MyChatModel", "model", "provider", {})

    def test_raises_on_import_error(self) -> None:
        """Raises ModelConfigError when module cannot be imported."""
        from deepagents_cli.model_config import ModelConfigError

        with pytest.raises(ModelConfigError, match="Could not import module"):
            _create_model_from_class(
                "nonexistent_package_xyz.models:MyModel", "model", "provider", {}
            )

    def test_raises_when_class_not_found_in_module(self) -> None:
        """Raises ModelConfigError when class doesn't exist in module."""
        from deepagents_cli.model_config import ModelConfigError

        with pytest.raises(ModelConfigError, match="not found in module"):
            _create_model_from_class("os.path:NonExistentClass", "m", "p", {})

    def test_raises_when_not_base_chat_model_subclass(self) -> None:
        """Raises ModelConfigError when class is not a BaseChatModel."""
        from deepagents_cli.model_config import ModelConfigError

        # os.path:join is a function, not a BaseChatModel subclass
        with pytest.raises(ModelConfigError, match="not a BaseChatModel subclass"):
            _create_model_from_class("os.path:sep", "m", "p", {})

    def test_instantiates_valid_subclass(self) -> None:
        """Successfully instantiates a valid BaseChatModel subclass."""
        from unittest.mock import MagicMock

        from langchain_core.callbacks import CallbackManagerForLLMRun
        from langchain_core.language_models import BaseChatModel
        from langchain_core.messages import BaseMessage
        from langchain_core.outputs import ChatResult

        # Track what args the constructor receives
        captured: dict[str, object] = {}

        class FakeChatModel(BaseChatModel):
            """Minimal BaseChatModel subclass for testing."""

            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

            def _generate(
                self,
                messages: list[BaseMessage],
                stop: list[str] | None = None,
                run_manager: CallbackManagerForLLMRun | None = None,
                **kwargs: object,
            ) -> ChatResult:
                msg = "not implemented"
                raise NotImplementedError(msg)

            @property
            def _llm_type(self) -> str:
                return "fake"

        with patch("importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.MyChatModel = FakeChatModel
            mock_import.return_value = mock_module

            result = _create_model_from_class(
                "my_pkg:MyChatModel", "my-model", "custom", {"temp": 0}
            )

        assert isinstance(result, FakeChatModel)
        assert captured["model"] == "my-model"
        assert captured["temp"] == 0

    def test_raises_on_instantiation_error(self) -> None:
        """Raises ModelConfigError when constructor fails."""
        from unittest.mock import MagicMock

        from langchain_core.language_models import BaseChatModel

        from deepagents_cli.model_config import ModelConfigError

        class BadModel(BaseChatModel):
            def __init__(self, **kwargs: object) -> None:
                pass

        with (
            patch("importlib.import_module") as mock_import,
            patch.object(BadModel, "__init__", side_effect=TypeError("bad args")),
        ):
            mock_module = MagicMock()
            mock_module.BadModel = BadModel
            mock_import.return_value = mock_module

            with pytest.raises(ModelConfigError, match="Failed to instantiate"):
                _create_model_from_class("my_pkg:BadModel", "model", "custom", {})


class TestCreateModelWithCustomClass:
    """Tests for create_model() using custom class_path from config."""

    def setup_method(self) -> None:
        """Clear model config cache before each test."""
        clear_caches()

    def test_create_model_uses_class_path(self, tmp_path: Path) -> None:
        """create_model dispatches to custom class when class_path is set."""
        from unittest.mock import MagicMock

        from langchain_core.language_models import BaseChatModel

        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.custom]
class_path = "my_pkg.models:MyChatModel"
models = ["my-model"]

[models.providers.custom.params]
temperature = 0
""")
        mock_instance = MagicMock(spec=BaseChatModel)
        mock_instance.profile = None

        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch(
                "deepagents_cli.config._create_model_from_class",
                return_value=mock_instance,
            ) as mock_factory,
        ):
            result = create_model("custom:my-model")

        mock_factory.assert_called_once()
        call_args = mock_factory.call_args
        assert call_args[0][0] == "my_pkg.models:MyChatModel"
        assert call_args[0][1] == "my-model"
        assert call_args[0][2] == "custom"
        assert isinstance(result, ModelResult)
        assert result.model is mock_instance
        assert result.model_name == "my-model"
        assert result.provider == "custom"

    def test_create_model_falls_through_without_class_path(
        self, tmp_path: Path
    ) -> None:
        """create_model uses init_chat_model when no class_path is set."""
        from unittest.mock import MagicMock

        from langchain_core.language_models import BaseChatModel

        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.fireworks]
models = ["llama"]
api_key_env = "FIREWORKS_API_KEY"
""")
        mock_instance = MagicMock(spec=BaseChatModel)
        mock_instance.profile = None

        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch.dict("os.environ", {"FIREWORKS_API_KEY": "key"}, clear=False),
            patch(
                "deepagents_cli.config._create_model_via_init",
                return_value=mock_instance,
            ) as mock_init,
        ):
            result = create_model("fireworks:llama")

        mock_init.assert_called_once()
        assert result.model is mock_instance


class TestCreateModelExtraKwargs:
    """Tests for create_model() with extra_kwargs from --model-params."""

    @patch("langchain.chat_models.init_chat_model")
    def test_extra_kwargs_passed_to_model(self, mock_init_chat_model: Mock) -> None:
        """extra_kwargs are forwarded to init_chat_model."""
        mock_model = Mock()
        mock_model.profile = None
        mock_init_chat_model.return_value = mock_model

        create_model("anthropic:claude-sonnet-4-5", extra_kwargs={"temperature": 0.7})

        _, call_kwargs = mock_init_chat_model.call_args
        assert call_kwargs["temperature"] == 0.7

    @patch("langchain.chat_models.init_chat_model")
    def test_extra_kwargs_override_config(
        self, mock_init_chat_model: Mock, tmp_path: Path
    ) -> None:
        """extra_kwargs override values from config file."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.anthropic]
models = ["claude-sonnet-4-5"]

[models.providers.anthropic.params]
temperature = 0
max_tokens = 1024
""")
        mock_model = Mock()
        mock_model.profile = None
        mock_init_chat_model.return_value = mock_model

        clear_caches()
        with patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path):
            create_model(
                "anthropic:claude-sonnet-4-5",
                extra_kwargs={"temperature": 0.9},
            )

        _, call_kwargs = mock_init_chat_model.call_args
        # CLI kwarg wins over config
        assert call_kwargs["temperature"] == 0.9
        # Config kwarg preserved when not overridden
        assert call_kwargs["max_tokens"] == 1024

    @patch("langchain.chat_models.init_chat_model")
    def test_none_extra_kwargs_is_noop(self, mock_init_chat_model: Mock) -> None:
        """extra_kwargs=None does not affect behavior."""
        mock_model = Mock()
        mock_model.profile = None
        mock_init_chat_model.return_value = mock_model

        create_model("anthropic:claude-sonnet-4-5", extra_kwargs=None)
        mock_init_chat_model.assert_called_once()

    @patch("langchain.chat_models.init_chat_model")
    def test_empty_extra_kwargs_is_noop(self, mock_init_chat_model: Mock) -> None:
        """extra_kwargs={} does not affect behavior."""
        mock_model = Mock()
        mock_model.profile = None
        mock_init_chat_model.return_value = mock_model

        create_model("anthropic:claude-sonnet-4-5", extra_kwargs={})
        mock_init_chat_model.assert_called_once()


class TestCreateModelEdgeCaseParsing:
    """Tests for create_model() edge-case spec parsing."""

    @patch("langchain.chat_models.init_chat_model")
    def test_leading_colon_treated_as_bare_model(
        self, mock_init_chat_model: Mock
    ) -> None:
        """Leading colon (e.g., ':claude-opus-4-6') is treated as bare model name."""
        mock_model = Mock()
        mock_model.profile = None
        mock_init_chat_model.return_value = mock_model

        settings.anthropic_api_key = "test"
        try:
            result = create_model(":claude-opus-4-6")
        finally:
            settings.anthropic_api_key = None

        # Should have detected 'anthropic' provider and used 'claude-opus-4-6'
        assert result.model_name == "claude-opus-4-6"

    def test_trailing_colon_raises_error(self) -> None:
        """Trailing colon (e.g., 'anthropic:') raises ModelConfigError."""
        with pytest.raises(ModelConfigError, match="model name is required"):
            create_model("anthropic:")

    @patch("deepagents_cli.config._get_default_model_spec")
    @patch("langchain.chat_models.init_chat_model")
    def test_empty_string_uses_default(
        self, mock_init_chat_model: Mock, mock_default: Mock
    ) -> None:
        """Empty string falls through to _get_default_model_spec."""
        mock_default.return_value = "openai:gpt-4o"
        mock_model = Mock()
        mock_model.profile = None
        mock_init_chat_model.return_value = mock_model

        create_model("")
        mock_default.assert_called_once()


class TestDetectProvider:
    """Tests for detect_provider() auto-detection from model names."""

    @pytest.mark.parametrize(
        ("model_name", "expected"),
        [
            ("gpt-4o", "openai"),
            ("gpt-5.2", "openai"),
            ("o1-preview", "openai"),
            ("o3-mini", "openai"),
            ("o4-mini", "openai"),
            ("claude-sonnet-4-5", "anthropic"),
            ("claude-opus-4-5", "anthropic"),
            ("gemini-3.1-pro-preview", "google_genai"),
            ("llama3", None),
            ("mistral-large", None),
            ("some-unknown-model", None),
        ],
    )
    def test_detect_known_patterns(self, model_name: str, expected: str | None) -> None:
        """detect_provider returns the correct provider for known patterns."""
        # Ensure both Anthropic and Google credentials are "available" so the
        # default paths are taken (not the Vertex AI fallbacks).
        settings.anthropic_api_key = "test"
        settings.google_api_key = "test"
        try:
            assert detect_provider(model_name) == expected
        finally:
            settings.anthropic_api_key = None
            settings.google_api_key = None

    def test_claude_falls_back_to_vertex_when_no_anthropic(self) -> None:
        """Claude models route to google_vertexai when only Vertex AI is configured."""
        settings.anthropic_api_key = None
        settings.google_cloud_project = "my-project"
        settings.google_api_key = None
        try:
            assert detect_provider("claude-sonnet-4-5") == "google_vertexai"
        finally:
            settings.google_cloud_project = None

    def test_gemini_falls_back_to_vertex_when_no_google(self) -> None:
        """Gemini models route to google_vertexai when only Vertex AI is configured."""
        settings.google_api_key = None
        settings.google_cloud_project = "my-project"
        try:
            assert detect_provider("gemini-3-pro") == "google_vertexai"
        finally:
            settings.google_cloud_project = None

    def test_gemini_prefers_google_genai_when_both_available(self) -> None:
        """Gemini prefers google_genai when both Google and Vertex AI are configured."""
        settings.google_api_key = "test"
        settings.google_cloud_project = "my-project"
        try:
            # has_vertex_ai is False when google_api_key is set, so this
            # tests the google_genai path which is preferred.
            assert detect_provider("gemini-3-pro") == "google_genai"
        finally:
            settings.google_api_key = None
            settings.google_cloud_project = None

    def test_case_insensitive(self) -> None:
        """detect_provider is case-insensitive."""
        settings.anthropic_api_key = "test"
        try:
            assert detect_provider("Claude-Sonnet-4-5") == "anthropic"
            assert detect_provider("GPT-4o") == "openai"
        finally:
            settings.anthropic_api_key = None
