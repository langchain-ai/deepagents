"""Tests for config module including project discovery utilities."""

import os
from pathlib import Path
from unittest import mock

from deepagents_cli.config import (
    Settings,
    _detect_provider,
    _find_project_agent_md,
    _find_project_root,
    _infer_provider_from_config,
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
    """Test finding project-specific agent.md files."""

    def test_find_agent_md_in_deepagents_dir(self, tmp_path: Path) -> None:
        """Test finding agent.md in .deepagents/ directory."""
        project_root = tmp_path / "project"
        project_root.mkdir()

        # Create .deepagents/agent.md
        deepagents_dir = project_root / ".deepagents"
        deepagents_dir.mkdir()
        agent_md = deepagents_dir / "agent.md"
        agent_md.write_text("Project instructions")

        result = _find_project_agent_md(project_root)
        assert len(result) == 1
        assert result[0] == agent_md

    def test_find_agent_md_in_root(self, tmp_path: Path) -> None:
        """Test finding agent.md in project root (fallback)."""
        project_root = tmp_path / "project"
        project_root.mkdir()

        # Create root-level agent.md (no .deepagents/)
        agent_md = project_root / "agent.md"
        agent_md.write_text("Project instructions")

        result = _find_project_agent_md(project_root)
        assert len(result) == 1
        assert result[0] == agent_md

    def test_both_agent_md_files_combined(self, tmp_path: Path) -> None:
        """Test that both agent.md files are returned when both exist."""
        project_root = tmp_path / "project"
        project_root.mkdir()

        # Create both locations
        deepagents_dir = project_root / ".deepagents"
        deepagents_dir.mkdir()
        deepagents_md = deepagents_dir / "agent.md"
        deepagents_md.write_text("In .deepagents/")

        root_md = project_root / "agent.md"
        root_md.write_text("In root")

        # Should return both, with .deepagents/ first
        result = _find_project_agent_md(project_root)
        assert len(result) == 2
        assert result[0] == deepagents_md
        assert result[1] == root_md

    def test_find_agent_md_not_found(self, tmp_path: Path) -> None:
        """Test that empty list is returned when no agent.md exists."""
        project_root = tmp_path / "project"
        project_root.mkdir()

        result = _find_project_agent_md(project_root)
        assert result == []


class TestDetectProvider:
    """Test provider detection from model names."""

    def test_detect_openai_gpt(self) -> None:
        """Test detection of OpenAI GPT models."""
        assert _detect_provider("gpt-4o") == "openai"
        assert _detect_provider("gpt-4o-mini") == "openai"
        assert _detect_provider("GPT-4") == "openai"

    def test_detect_openai_o_series(self) -> None:
        """Test detection of OpenAI o1/o3 models."""
        assert _detect_provider("o1-preview") == "openai"
        assert _detect_provider("o3-mini") == "openai"

    def test_detect_anthropic_claude(self) -> None:
        """Test detection of Anthropic Claude models."""
        assert _detect_provider("claude-sonnet-4-5-20250929") == "anthropic"
        assert _detect_provider("claude-3-opus-20240229") == "anthropic"
        assert _detect_provider("CLAUDE-3-haiku") == "anthropic"

    def test_detect_google_gemini(self) -> None:
        """Test detection of Google Gemini models."""
        assert _detect_provider("gemini-2.0-flash") == "google"
        assert _detect_provider("gemini-1.5-pro") == "google"
        assert _detect_provider("GEMINI-pro") == "google"

    def test_detect_unknown(self) -> None:
        """Test that unknown models return None."""
        assert _detect_provider("deepseek-chat") is None
        assert _detect_provider("llama-3.1-70b") is None
        assert _detect_provider("qwen-72b") is None
        assert _detect_provider("glm-4") is None
        assert _detect_provider("custom-model") is None


class TestSettingsBaseUrl:
    """Test Settings class base_url configuration."""

    def test_settings_reads_base_url_env_vars(self, tmp_path: Path) -> None:
        """Test that Settings reads base URL environment variables."""
        env = {
            "OPENAI_API_KEY": "sk-test",
            "OPENAI_BASE_URL": "https://api.deepseek.com/v1",
            "ANTHROPIC_API_KEY": "test-key",
            "ANTHROPIC_BASE_URL": "https://open.bigmodel.cn/api/anthropic",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            settings = Settings.from_environment(start_path=tmp_path)

            assert settings.openai_base_url == "https://api.deepseek.com/v1"
            assert settings.anthropic_base_url == "https://open.bigmodel.cn/api/anthropic"
            assert settings.google_base_url is None

    def test_settings_has_custom_base_url_properties(self, tmp_path: Path) -> None:
        """Test has_custom_*_base_url properties."""
        # Create a Settings instance directly with controlled values
        settings = Settings(
            openai_api_key="sk-test",
            anthropic_api_key=None,
            google_api_key=None,
            tavily_api_key=None,
            openai_base_url="https://custom.api.com/v1",
            anthropic_base_url=None,
            google_base_url=None,
            deepagents_langchain_project=None,
            user_langchain_project=None,
            project_root=tmp_path,
        )

        assert settings.has_custom_openai_base_url is True
        assert settings.has_custom_anthropic_base_url is False
        assert settings.has_custom_google_base_url is False

    def test_settings_no_base_url_configured(self, tmp_path: Path) -> None:
        """Test Settings when no base URL is configured."""
        env = {
            "OPENAI_API_KEY": "sk-test",
        }
        # Clear any existing base URL env vars
        clear_env = {
            "OPENAI_BASE_URL": "",
            "ANTHROPIC_BASE_URL": "",
            "GOOGLE_BASE_URL": "",
        }
        with mock.patch.dict(os.environ, {**env, **clear_env}, clear=False):
            # Manually remove the empty strings to simulate unset
            for key in clear_env:
                os.environ.pop(key, None)

            settings = Settings.from_environment(start_path=tmp_path)

            assert settings.openai_base_url is None
            assert settings.anthropic_base_url is None
            assert settings.google_base_url is None


class TestInferProviderFromConfig:
    """Test provider inference from available configuration."""

    def test_infer_openai_when_only_openai_configured(self) -> None:
        """Test that OpenAI is inferred when only OpenAI key is available."""
        with mock.patch("deepagents_cli.config.settings") as mock_settings:
            mock_settings.has_openai = True
            mock_settings.has_anthropic = False
            mock_settings.has_google = False
            assert _infer_provider_from_config() == "openai"

    def test_infer_anthropic_when_only_anthropic_configured(self) -> None:
        """Test that Anthropic is inferred when only Anthropic key is available."""
        with mock.patch("deepagents_cli.config.settings") as mock_settings:
            mock_settings.has_openai = False
            mock_settings.has_anthropic = True
            mock_settings.has_google = False
            assert _infer_provider_from_config() == "anthropic"

    def test_infer_google_when_only_google_configured(self) -> None:
        """Test that Google is inferred when only Google key is available."""
        with mock.patch("deepagents_cli.config.settings") as mock_settings:
            mock_settings.has_openai = False
            mock_settings.has_anthropic = False
            mock_settings.has_google = True
            assert _infer_provider_from_config() == "google"

    def test_infer_openai_priority_when_multiple_configured(self) -> None:
        """Test that OpenAI has highest priority when multiple keys available."""
        with mock.patch("deepagents_cli.config.settings") as mock_settings:
            mock_settings.has_openai = True
            mock_settings.has_anthropic = True
            mock_settings.has_google = True
            assert _infer_provider_from_config() == "openai"

    def test_infer_none_when_no_provider_configured(self) -> None:
        """Test that None is returned when no provider is configured."""
        with mock.patch("deepagents_cli.config.settings") as mock_settings:
            mock_settings.has_openai = False
            mock_settings.has_anthropic = False
            mock_settings.has_google = False
            assert _infer_provider_from_config() is None
