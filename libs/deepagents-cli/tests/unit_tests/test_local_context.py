"""Tests for local context middleware."""

from unittest.mock import Mock, patch

from deepagents_cli.local_context import LocalContextMiddleware


class TestLocalContextMiddleware:
    """Test git context middleware functionality."""

    @patch("deepagents_cli.local_context.subprocess.run")
    def test_get_git_info_in_git_repo(self, mock_run) -> None:
        """Test git info gathering when in a git repository."""
        # Mock git rev-parse response (current branch)
        mock_branch_result = Mock()
        mock_branch_result.returncode = 0
        mock_branch_result.stdout = "feature/my-branch\n"

        # Mock git branch response (list of branches)
        mock_branches_result = Mock()
        mock_branches_result.returncode = 0
        mock_branches_result.stdout = "  feature/my-branch\n* main\n  master\n"

        # Configure subprocess.run to return different results for different calls
        mock_run.side_effect = [mock_branch_result, mock_branches_result]

        middleware = LocalContextMiddleware()
        git_info = middleware._get_git_info()

        assert git_info["branch"] == "feature/my-branch"
        assert "main" in git_info["main_branches"]
        assert "master" in git_info["main_branches"]
        assert len(git_info["main_branches"]) == 2

    @patch("deepagents_cli.local_context.subprocess.run")
    def test_get_git_info_not_in_git_repo(self, mock_run) -> None:
        """Test git info returns empty dict when not in a git repository."""
        # Mock git rev-parse failure (not a git repo)
        mock_result = Mock()
        mock_result.returncode = 128  # git error code for "not a git repository"
        mock_run.return_value = mock_result

        middleware = LocalContextMiddleware()
        git_info = middleware._get_git_info()

        assert git_info == {}

    @patch("deepagents_cli.local_context.subprocess.run")
    def test_get_git_info_only_main_branch(self, mock_run) -> None:
        """Test git info when only main branch exists."""
        mock_branch_result = Mock()
        mock_branch_result.returncode = 0
        mock_branch_result.stdout = "main\n"

        mock_branches_result = Mock()
        mock_branches_result.returncode = 0
        mock_branches_result.stdout = "* main\n"

        mock_run.side_effect = [mock_branch_result, mock_branches_result]

        middleware = LocalContextMiddleware()
        git_info = middleware._get_git_info()

        assert git_info["branch"] == "main"
        assert git_info["main_branches"] == ["main"]

    @patch("deepagents_cli.local_context.subprocess.run")
    def test_get_git_info_only_master_branch(self, mock_run) -> None:
        """Test git info when only master branch exists."""
        mock_branch_result = Mock()
        mock_branch_result.returncode = 0
        mock_branch_result.stdout = "master\n"

        mock_branches_result = Mock()
        mock_branches_result.returncode = 0
        mock_branches_result.stdout = "* master\n  feature/test\n"

        mock_run.side_effect = [mock_branch_result, mock_branches_result]

        middleware = LocalContextMiddleware()
        git_info = middleware._get_git_info()

        assert git_info["branch"] == "master"
        assert git_info["main_branches"] == ["master"]

    @patch("deepagents_cli.local_context.subprocess.run")
    def test_get_git_info_no_main_branches(self, mock_run) -> None:
        """Test git info when neither main nor master exist."""
        mock_branch_result = Mock()
        mock_branch_result.returncode = 0
        mock_branch_result.stdout = "develop\n"

        mock_branches_result = Mock()
        mock_branches_result.returncode = 0
        mock_branches_result.stdout = "* develop\n  feature/test\n"

        mock_run.side_effect = [mock_branch_result, mock_branches_result]

        middleware = LocalContextMiddleware()
        git_info = middleware._get_git_info()

        assert git_info["branch"] == "develop"
        assert git_info["main_branches"] == []

    @patch("deepagents_cli.local_context.subprocess.run")
    def test_before_agent_with_git_repo(self, mock_run) -> None:
        """Test before_agent returns git context when in git repo."""
        mock_branch_result = Mock()
        mock_branch_result.returncode = 0
        mock_branch_result.stdout = "main\n"

        mock_branches_result = Mock()
        mock_branches_result.returncode = 0
        mock_branches_result.stdout = "* main\n  master\n"

        mock_run.side_effect = [mock_branch_result, mock_branches_result]

        middleware = LocalContextMiddleware()
        state = {}
        runtime = Mock()

        result = middleware.before_agent(state, runtime)

        assert result is not None
        assert "local_context" in result
        assert "**Git**: Current branch `main`" in result["local_context"]
        assert "main branch available:" in result["local_context"]
        assert "`main`" in result["local_context"]
        assert "`master`" in result["local_context"]

    @patch("deepagents_cli.local_context.Path")
    @patch("deepagents_cli.local_context.subprocess.run")
    def test_before_agent_not_in_git_repo(self, mock_run, mock_path) -> None:
        """Test before_agent returns local context without git info when not in git repo."""
        # Mock git command failure (not in repo)
        mock_result = Mock()
        mock_result.returncode = 128
        mock_run.return_value = mock_result

        # Mock Path.cwd() and iterdir() to return empty
        mock_cwd = Mock()
        mock_cwd.name = "test-dir"
        mock_cwd.iterdir.return_value = []
        mock_path.cwd.return_value = mock_cwd

        middleware = LocalContextMiddleware()
        state = {}
        runtime = Mock()

        result = middleware.before_agent(state, runtime)

        # Should still return local context with CWD, just no git info
        assert result is not None
        assert "local_context" in result
        assert "Current Directory" in result["local_context"]
        assert "Git:" not in result["local_context"]

    def test_wrap_model_call_with_local_context(self) -> None:
        """Test that wrap_model_call appends local context to system prompt."""
        middleware = LocalContextMiddleware()

        # Create mock request with local context in state
        request = Mock()
        request.system_prompt = "Base system prompt"
        request.state = {
            "local_context": "## Local Context\n\nCurrent branch: `main`\nMain branch available: `main`"
        }

        # Mock the override method to return a new request
        overridden_request = Mock()
        request.override.return_value = overridden_request

        # Mock handler
        handler = Mock()
        handler.return_value = "response"

        result = middleware.wrap_model_call(request, handler)

        # Verify override was called with appended git context
        request.override.assert_called_once()
        call_args = request.override.call_args[1]
        assert "system_prompt" in call_args
        assert "Base system prompt" in call_args["system_prompt"]
        assert "Current branch: `main`" in call_args["system_prompt"]

        # Verify handler was called with overridden request
        handler.assert_called_once_with(overridden_request)
        assert result == "response"

    def test_wrap_model_call_without_local_context(self) -> None:
        """Test that wrap_model_call passes through when no local context."""
        middleware = LocalContextMiddleware()

        # Create mock request without local context
        request = Mock()
        request.system_prompt = "Base system prompt"
        request.state = {}

        # Mock handler
        handler = Mock()
        handler.return_value = "response"

        result = middleware.wrap_model_call(request, handler)

        # Verify override was NOT called
        request.override.assert_not_called()

        # Verify handler was called with original request
        handler.assert_called_once_with(request)
        assert result == "response"
