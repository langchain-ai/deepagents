"""Tests for local context middleware."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, Mock

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from deepagents_cli.local_context import (
    DETECT_CONTEXT_SCRIPT,
    LocalContextMiddleware,
    LocalContextState,
    _ExecutableBackend,
    _section_files,
    _section_git,
    _section_header,
    _section_makefile,
    _section_package_managers,
    _section_project,
    _section_runtimes,
    _section_test_command,
    _section_tree,
    build_detect_script,
)


def _make_backend(output: str = "", exit_code: int = 0) -> Mock:
    """Create a mock backend with execute() returning the given output."""
    backend = Mock()
    result = Mock()
    result.output = output
    result.exit_code = exit_code
    backend.execute.return_value = result
    return backend


def _make_summarization_event(cutoff: int) -> dict[str, Any]:
    """Create a minimal summarization event dict for testing.

    Only `cutoff_index` is used by the refresh logic; other fields
    are set to `None` for simplicity.
    """
    return {
        "cutoff_index": cutoff,
        "summary_message": None,
        "file_path": None,
    }


# Sample script output for testing
SAMPLE_CONTEXT = (
    "## Local Context\n\n"
    "**Current Directory**: `/home/user/project`\n\n"
    "**Git**: Current branch `main`, main branch available: `main`, `master`,"
    " 1 uncommitted change\n\n"
    "**Runtimes**: Python 3.12.4, Node 20.11.0\n"
)

SAMPLE_CONTEXT_NO_GIT = (
    "## Local Context\n\n"
    "**Current Directory**: `/home/user/project`\n\n"
    "**Runtimes**: Python 3.12.4\n"
)


class TestLocalContextMiddleware:
    """Test local context middleware functionality."""

    def test_before_agent_stores_context(self) -> None:
        """Test before_agent runs script and stores output in state."""
        backend = _make_backend(output=SAMPLE_CONTEXT)
        middleware = LocalContextMiddleware(backend=backend)
        state: LocalContextState = {"messages": []}
        runtime: Any = Mock()

        result = middleware.before_agent(state, runtime)

        assert result is not None
        assert "local_context" in result
        assert "## Local Context" in result["local_context"]
        assert "Current Directory" in result["local_context"]
        backend.execute.assert_called_once()

    def test_before_agent_skips_when_already_set(self) -> None:
        """Test before_agent returns None when local_context already exists."""
        backend = _make_backend(output=SAMPLE_CONTEXT)
        middleware = LocalContextMiddleware(backend=backend)
        state: LocalContextState = {
            "messages": [],
            "local_context": "already set",
        }
        runtime: Any = Mock()

        result = middleware.before_agent(state, runtime)

        assert result is None
        backend.execute.assert_not_called()

    def test_before_agent_handles_script_failure(self) -> None:
        """Test before_agent returns None when script exits non-zero."""
        backend = _make_backend(output="", exit_code=1)
        middleware = LocalContextMiddleware(backend=backend)
        state: LocalContextState = {"messages": []}
        runtime: Any = Mock()

        result = middleware.before_agent(state, runtime)

        assert result is None

    def test_before_agent_handles_empty_output(self) -> None:
        """Test before_agent returns None when script produces no output."""
        backend = _make_backend(output="   \n  ", exit_code=0)
        middleware = LocalContextMiddleware(backend=backend)
        state: LocalContextState = {"messages": []}
        runtime: Any = Mock()

        result = middleware.before_agent(state, runtime)

        assert result is None

    def test_before_agent_handles_execute_exception(self) -> None:
        """Test before_agent returns None when backend.execute() raises."""
        backend = Mock()
        backend.execute.side_effect = RuntimeError("connection failed")
        middleware = LocalContextMiddleware(backend=backend)
        state: LocalContextState = {"messages": []}
        runtime: Any = Mock()

        result = middleware.before_agent(state, runtime)

        assert result is None

    def test_before_agent_handles_none_output(self) -> None:
        """Test before_agent returns None when result.output is None."""
        backend = Mock()
        result_mock = Mock()
        result_mock.output = None
        result_mock.exit_code = 0
        backend.execute.return_value = result_mock
        middleware = LocalContextMiddleware(backend=backend)
        state: LocalContextState = {"messages": []}
        runtime: Any = Mock()

        result = middleware.before_agent(state, runtime)

        assert result is None

    def test_before_agent_git_context(self) -> None:
        """Test that git info is preserved from script output."""
        backend = _make_backend(output=SAMPLE_CONTEXT)
        middleware = LocalContextMiddleware(backend=backend)
        state: LocalContextState = {"messages": []}
        runtime: Any = Mock()

        result = middleware.before_agent(state, runtime)

        assert result is not None
        ctx = result["local_context"]
        assert "**Git**: Current branch `main`" in ctx
        assert "main branch available:" in ctx
        assert "`main`" in ctx
        assert "`master`" in ctx
        assert "1 uncommitted change" in ctx

    def test_before_agent_no_git(self) -> None:
        """Test output without git info."""
        backend = _make_backend(output=SAMPLE_CONTEXT_NO_GIT)
        middleware = LocalContextMiddleware(backend=backend)
        state: LocalContextState = {"messages": []}
        runtime: Any = Mock()

        result = middleware.before_agent(state, runtime)

        assert result is not None
        ctx = result["local_context"]
        assert "Current Directory" in ctx
        assert "**Git**:" not in ctx

    def test_wrap_model_call_with_local_context(self) -> None:
        """Test that wrap_model_call appends local context to system prompt."""
        backend = _make_backend()
        middleware = LocalContextMiddleware(backend=backend)

        request = Mock()
        request.system_prompt = "Base system prompt"
        request.state = {"local_context": SAMPLE_CONTEXT}

        overridden_request = Mock()
        request.override.return_value = overridden_request

        handler = Mock(return_value="response")

        result = middleware.wrap_model_call(request, handler)

        request.override.assert_called_once()
        call_args = request.override.call_args[1]
        assert "system_prompt" in call_args
        assert "Base system prompt" in call_args["system_prompt"]
        assert "Current branch `main`" in call_args["system_prompt"]

        handler.assert_called_once_with(overridden_request)
        assert result == "response"

    def test_wrap_model_call_without_local_context(self) -> None:
        """Test that wrap_model_call passes through when no local context."""
        backend = _make_backend()
        middleware = LocalContextMiddleware(backend=backend)

        request = Mock()
        request.system_prompt = "Base system prompt"
        request.state = {}

        handler = Mock(return_value="response")

        result = middleware.wrap_model_call(request, handler)

        request.override.assert_not_called()
        handler.assert_called_once_with(request)
        assert result == "response"

    @pytest.mark.asyncio
    async def test_awrap_model_call_with_local_context(self) -> None:
        """Test that awrap_model_call appends local context to system prompt."""
        backend = _make_backend()
        middleware = LocalContextMiddleware(backend=backend)

        request = Mock()
        request.system_prompt = "Base system prompt"
        request.state = {"local_context": SAMPLE_CONTEXT}

        overridden_request = Mock()
        request.override.return_value = overridden_request

        handler = AsyncMock(return_value="async response")

        result = await middleware.awrap_model_call(request, handler)

        request.override.assert_called_once()
        call_args = request.override.call_args[1]
        assert "system_prompt" in call_args
        assert "Base system prompt" in call_args["system_prompt"]
        assert "Current branch `main`" in call_args["system_prompt"]

        handler.assert_called_once_with(overridden_request)
        assert result == "async response"

    @pytest.mark.asyncio
    async def test_awrap_model_call_without_local_context(self) -> None:
        """Test that awrap_model_call passes through when no local context."""
        backend = _make_backend()
        middleware = LocalContextMiddleware(backend=backend)

        request = Mock()
        request.system_prompt = "Base system prompt"
        request.state = {}

        handler = AsyncMock(return_value="async response")

        result = await middleware.awrap_model_call(request, handler)

        request.override.assert_not_called()
        handler.assert_called_once_with(request)
        assert result == "async response"

    def test_before_agent_refreshes_on_summarization(self) -> None:
        """Test that a new summarization event triggers a context refresh."""
        ctx = "## Local Context\n\n**Current Directory**: `/new/path`\n"
        backend = _make_backend(output=ctx)
        middleware = LocalContextMiddleware(backend=backend)
        event = _make_summarization_event(5)
        state: dict[str, Any] = {
            "messages": [],
            "local_context": "stale context",
            "_summarization_event": event,
        }
        runtime: Any = Mock()

        result = middleware.before_agent(state, runtime)  # type: ignore[invalid-argument-type]

        assert result is not None
        assert result["local_context"] == ctx.strip()
        assert result["_local_context_refreshed_at_cutoff"] == 5
        backend.execute.assert_called_once()

    def test_before_agent_no_rerun_same_cutoff(self) -> None:
        """Test no re-run when cutoff matches last refreshed cutoff."""
        backend = _make_backend(output="anything")
        middleware = LocalContextMiddleware(backend=backend)
        event = _make_summarization_event(5)
        state: dict[str, Any] = {
            "messages": [],
            "local_context": "existing context",
            "_summarization_event": event,
            "_local_context_refreshed_at_cutoff": 5,
        }
        runtime: Any = Mock()

        result = middleware.before_agent(state, runtime)  # type: ignore[invalid-argument-type]

        # Falls through to initial-detection guard; local_context set.
        assert result is None
        backend.execute.assert_not_called()

    def test_before_agent_refresh_failure_records_cutoff(self) -> None:
        """Test failed refresh records cutoff but keeps existing context."""
        backend = _make_backend(output="", exit_code=1)
        middleware = LocalContextMiddleware(backend=backend)
        event = _make_summarization_event(10)
        state: dict[str, Any] = {
            "messages": [],
            "local_context": "keep this",
            "_summarization_event": event,
        }
        runtime: Any = Mock()

        result = middleware.before_agent(state, runtime)  # type: ignore[invalid-argument-type]

        assert result is not None
        # Cutoff recorded to prevent retry loop.
        assert result["_local_context_refreshed_at_cutoff"] == 10
        # local_context NOT overwritten.
        assert "local_context" not in result
        backend.execute.assert_called_once()

    def test_before_agent_second_summarization_refreshes(self) -> None:
        """Test a second summarization with different cutoff triggers re-run."""
        backend = _make_backend(output="refreshed again")
        middleware = LocalContextMiddleware(backend=backend)
        event = _make_summarization_event(20)
        state: dict[str, Any] = {
            "messages": [],
            "local_context": "first refresh",
            "_summarization_event": event,
            "_local_context_refreshed_at_cutoff": 10,
        }
        runtime: Any = Mock()

        result = middleware.before_agent(state, runtime)  # type: ignore[invalid-argument-type]

        assert result is not None
        assert result["local_context"] == "refreshed again"
        assert result["_local_context_refreshed_at_cutoff"] == 20

    def test_before_agent_cross_thread_isolation(self) -> None:
        """Test shared middleware produces independent results per thread."""
        backend = _make_backend(output="thread output")
        middleware = LocalContextMiddleware(backend=backend)
        runtime: Any = Mock()

        # Thread A: summarization at cutoff 5, not yet refreshed.
        state_a: dict[str, Any] = {
            "messages": [],
            "local_context": "old A",
            "_summarization_event": _make_summarization_event(5),
        }
        result_a = middleware.before_agent(state_a, runtime)  # type: ignore[invalid-argument-type]
        assert result_a is not None
        assert result_a["_local_context_refreshed_at_cutoff"] == 5

        backend.reset_mock()

        # Thread B: already refreshed at cutoff 5 — no re-run.
        state_b: dict[str, Any] = {
            "messages": [],
            "local_context": "old B",
            "_summarization_event": _make_summarization_event(5),
            "_local_context_refreshed_at_cutoff": 5,
        }
        result_b = middleware.before_agent(state_b, runtime)  # type: ignore[invalid-argument-type]
        assert result_b is None
        backend.execute.assert_not_called()

        # Thread C: no summarization event, context already set.
        state_c: dict[str, Any] = {
            "messages": [],
            "local_context": "existing C",
        }
        result_c = middleware.before_agent(state_c, runtime)  # type: ignore[invalid-argument-type]
        assert result_c is None
        backend.execute.assert_not_called()

    def test_before_agent_refresh_exception_records_cutoff(self) -> None:
        """Test exception during refresh records cutoff and keeps context."""
        backend = Mock()
        backend.execute.side_effect = RuntimeError("sandbox unreachable")
        middleware = LocalContextMiddleware(backend=backend)
        event = _make_summarization_event(7)
        state: dict[str, Any] = {
            "messages": [],
            "local_context": "keep this",
            "_summarization_event": event,
        }
        runtime: Any = Mock()

        result = middleware.before_agent(state, runtime)  # type: ignore[invalid-argument-type]

        assert result is not None
        assert result["_local_context_refreshed_at_cutoff"] == 7
        assert "local_context" not in result
        backend.execute.assert_called_once()

    def test_before_agent_missing_cutoff_index_skips_refresh(self) -> None:
        """Test that a summarization event missing cutoff_index skips refresh."""
        backend = _make_backend(output="anything")
        middleware = LocalContextMiddleware(backend=backend)
        state: dict[str, Any] = {
            "messages": [],
            "local_context": "existing",
            "_summarization_event": {"summary_message": None, "file_path": None},
        }
        runtime: Any = Mock()

        result = middleware.before_agent(state, runtime)  # type: ignore[invalid-argument-type]

        # Both cutoff and refreshed_cutoff are None, so cutoff != refreshed_cutoff
        # is False. Falls through to initial-detection guard; local_context set.
        assert result is None
        backend.execute.assert_not_called()


# ---------------------------------------------------------------------------
# Section-level bash tests
# ---------------------------------------------------------------------------


def _run_section(section_bash: str, cwd: Path, *, with_header: bool = False) -> str:
    """Run a bash section snippet and return stdout.

    Note: bash scripts may return exit code 1 when their last conditional
    evaluates to false (e.g., `[ -n "" ] && echo ...`). This is normal bash
    behavior, not an error. We check stderr for real failures instead.
    """
    script = (_section_header() + "\n" + section_bash) if with_header else section_bash
    result = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        cwd=cwd,
        check=False,
    )
    # Fail on genuine bash errors (syntax errors, etc.) indicated by stderr
    assert not result.stderr, (
        f"Bash section produced stderr (exit code {result.returncode}).\n"
        f"stderr: {result.stderr}\nstdout: {result.stdout}"
    )
    return result.stdout


class TestBuildDetectScript:
    """Smoke tests for the script assembly."""

    def test_build_detect_script_returns_string(self) -> None:
        script = build_detect_script()
        assert isinstance(script, str)
        assert script.startswith("bash <<'__DETECT_CONTEXT_EOF__'")
        assert script.rstrip().endswith("__DETECT_CONTEXT_EOF__")

    def test_module_constant_matches_builder(self) -> None:
        assert build_detect_script() == DETECT_CONTEXT_SCRIPT


class TestSectionHeader:
    """Tests for _section_header."""

    def test_prints_cwd(self, tmp_path: Path) -> None:
        out = _run_section(_section_header(), tmp_path)
        assert "## Local Context" in out
        assert f"**Current Directory**: `{tmp_path}`" in out

    def test_in_git_false_outside_repo(self, tmp_path: Path) -> None:
        # Append a check so we can see the value
        script = _section_header() + '\necho "IN_GIT=$IN_GIT"'
        result = subprocess.run(
            ["bash", "-c", script],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            check=False,
        )
        assert "IN_GIT=false" in result.stdout

    def test_in_git_true_inside_repo(self, tmp_path: Path) -> None:
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=False)
        script = _section_header() + '\necho "IN_GIT=$IN_GIT"'
        result = subprocess.run(
            ["bash", "-c", script],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            check=False,
        )
        assert "IN_GIT=true" in result.stdout


class TestSectionProject:
    """Tests for _section_project."""

    def test_python_project(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("")
        out = _run_section(_section_project(), tmp_path, with_header=True)
        assert "**Project**:" in out
        assert "Language: python" in out

    def test_javascript_project(self, tmp_path: Path) -> None:
        (tmp_path / "package.json").write_text("{}")
        out = _run_section(_section_project(), tmp_path, with_header=True)
        assert "Language: javascript/typescript" in out

    def test_rust_project(self, tmp_path: Path) -> None:
        (tmp_path / "Cargo.toml").write_text("")
        out = _run_section(_section_project(), tmp_path, with_header=True)
        assert "Language: rust" in out

    def test_monorepo_libs_apps(self, tmp_path: Path) -> None:
        (tmp_path / "libs").mkdir()
        (tmp_path / "apps").mkdir()
        out = _run_section(_section_project(), tmp_path, with_header=True)
        assert "Monorepo: yes" in out

    def test_envs_detected(self, tmp_path: Path) -> None:
        (tmp_path / ".venv").mkdir()
        out = _run_section(_section_project(), tmp_path, with_header=True)
        assert "Environments: .venv" in out

    def test_no_project_files_no_output(self, tmp_path: Path) -> None:
        out = _run_section(_section_project(), tmp_path, with_header=True)
        assert "**Project**:" not in out


class TestSectionPackageManagers:
    """Tests for _section_package_managers."""

    def test_uv_lock(self, tmp_path: Path) -> None:
        (tmp_path / "uv.lock").write_text("")
        out = _run_section(_section_package_managers(), tmp_path)
        assert "Python: uv" in out

    def test_poetry_lock(self, tmp_path: Path) -> None:
        (tmp_path / "poetry.lock").write_text("")
        out = _run_section(_section_package_managers(), tmp_path)
        assert "Python: poetry" in out

    def test_pyproject_with_uv_tool(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("[tool.uv]\n")
        out = _run_section(_section_package_managers(), tmp_path)
        assert "Python: uv" in out

    def test_requirements_txt(self, tmp_path: Path) -> None:
        (tmp_path / "requirements.txt").write_text("flask\n")
        out = _run_section(_section_package_managers(), tmp_path)
        assert "Python: pip" in out

    def test_bun_lockb(self, tmp_path: Path) -> None:
        (tmp_path / "bun.lockb").write_text("")
        out = _run_section(_section_package_managers(), tmp_path)
        assert "Node: bun" in out

    def test_yarn_lock(self, tmp_path: Path) -> None:
        (tmp_path / "yarn.lock").write_text("")
        out = _run_section(_section_package_managers(), tmp_path)
        assert "Node: yarn" in out

    def test_combined_python_and_node(self, tmp_path: Path) -> None:
        (tmp_path / "uv.lock").write_text("")
        (tmp_path / "yarn.lock").write_text("")
        out = _run_section(_section_package_managers(), tmp_path)
        assert "Python: uv" in out
        assert "Node: yarn" in out

    def test_no_package_manager(self, tmp_path: Path) -> None:
        out = _run_section(_section_package_managers(), tmp_path)
        assert "**Package Manager**" not in out


class TestSectionRuntimes:
    """Tests for _section_runtimes."""

    def test_runs_and_detects_python(self, tmp_path: Path) -> None:
        out = _run_section(_section_runtimes(), tmp_path)
        # python3 is available in CI and dev; just check format
        assert "**Runtimes**:" in out
        assert "Python " in out


def _git_env(tmp_path: Path) -> dict[str, str]:
    """Minimal env for `git commit` in an isolated temp dir."""
    return {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
        "HOME": str(tmp_path),
    }


def _git_init_commit(tmp_path: Path, *, branch: str | None = None) -> None:
    """`git init` (optionally with *branch*) + empty commit."""
    cmd = ["git", "init"]
    if branch:
        cmd += ["-b", branch]
    subprocess.run(cmd, cwd=tmp_path, capture_output=True, check=False)
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "init"],
        cwd=tmp_path,
        capture_output=True,
        env=_git_env(tmp_path),
        check=False,
    )


class TestSectionGit:
    """Tests for _section_git."""

    def test_branch_name(self, tmp_path: Path) -> None:
        _git_init_commit(tmp_path, branch="feat-x")
        out = _run_section(_section_git(), tmp_path, with_header=True)
        assert "Current branch `feat-x`" in out

    def test_main_branch_listed(self, tmp_path: Path) -> None:
        _git_init_commit(tmp_path, branch="main")
        out = _run_section(_section_git(), tmp_path, with_header=True)
        assert "main branch available: `main`" in out

    def test_uncommitted_changes_singular(self, tmp_path: Path) -> None:
        _git_init_commit(tmp_path)
        (tmp_path / "new.txt").write_text("hello")
        out = _run_section(_section_git(), tmp_path, with_header=True)
        assert "1 uncommitted change\n" in out or out.rstrip().endswith(
            "1 uncommitted change"
        )

    def test_uncommitted_changes_plural(self, tmp_path: Path) -> None:
        _git_init_commit(tmp_path)
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        out = _run_section(_section_git(), tmp_path, with_header=True)
        assert "2 uncommitted changes" in out

    def test_no_output_outside_git(self, tmp_path: Path) -> None:
        out = _run_section(_section_git(), tmp_path, with_header=True)
        assert "**Git**" not in out


class TestSectionTestCommand:
    """Tests for _section_test_command."""

    def test_makefile_test_target(self, tmp_path: Path) -> None:
        (tmp_path / "Makefile").write_text("test:\n\tpytest\n")
        out = _run_section(_section_test_command(), tmp_path)
        assert "`make test`" in out

    def test_pytest_via_pyproject(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
        out = _run_section(_section_test_command(), tmp_path)
        assert "`pytest`" in out

    def test_pytest_via_tests_dir(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("")
        (tmp_path / "tests").mkdir()
        out = _run_section(_section_test_command(), tmp_path)
        assert "`pytest`" in out

    def test_npm_test(self, tmp_path: Path) -> None:
        (tmp_path / "package.json").write_text('{"scripts": {"test": "jest"}}\n')
        out = _run_section(_section_test_command(), tmp_path)
        assert "`npm test`" in out

    def test_no_test_command(self, tmp_path: Path) -> None:
        out = _run_section(_section_test_command(), tmp_path)
        assert "**Run Tests**" not in out


class TestSectionFiles:
    """Tests for _section_files."""

    def test_lists_files_and_dirs(self, tmp_path: Path) -> None:
        (tmp_path / "README.md").write_text("hi")
        (tmp_path / "src").mkdir()
        out = _run_section(_section_files(), tmp_path)
        assert "- README.md" in out
        assert "- src/" in out

    def test_caps_at_20(self, tmp_path: Path) -> None:
        for i in range(25):
            (tmp_path / f"file{i:02d}.txt").write_text("")
        out = _run_section(_section_files(), tmp_path)
        assert "(20 shown)" in out
        assert "5 more files" in out

    def test_excludes_pycache(self, tmp_path: Path) -> None:
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "keep.py").write_text("")
        out = _run_section(_section_files(), tmp_path)
        assert "__pycache__" not in out
        assert "keep.py" in out

    def test_includes_deepagents(self, tmp_path: Path) -> None:
        (tmp_path / ".deepagents").mkdir()
        out = _run_section(_section_files(), tmp_path)
        assert ".deepagents" in out


class TestSectionTree:
    """Tests for _section_tree."""

    def test_tree_output_format(self, tmp_path: Path) -> None:
        import shutil

        if shutil.which("tree") is None:
            pytest.skip("tree not installed")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("")
        out = _run_section(_section_tree(), tmp_path)
        assert "**Tree** (3 levels):" in out
        assert "```text" in out
        assert "```" in out

    def test_skips_when_tree_missing(self, tmp_path: Path) -> None:
        # Use absolute bash path; bogus PATH so `command -v tree` fails
        script = _section_tree()
        result = subprocess.run(
            ["/bin/bash", "-c", script],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            env={"PATH": "/nonexistent"},
            check=False,
        )
        assert "**Tree**" not in result.stdout


class TestSectionMakefile:
    """Tests for _section_makefile."""

    def test_shows_makefile_contents(self, tmp_path: Path) -> None:
        """Makefile in CWD is shown with path, header, and contents."""
        (tmp_path / "Makefile").write_text("all:\n\techo hello\n")
        out = _run_section(_section_makefile(), tmp_path, with_header=True)
        assert "**Makefile** (`Makefile`, first 20 lines):" in out
        assert "```makefile" in out
        assert "echo hello" in out

    def test_truncation_note_for_long_makefile(self, tmp_path: Path) -> None:
        """Makefiles longer than 20 lines show a truncation notice."""
        lines = [f"target{i}:\n\techo {i}\n" for i in range(30)]
        (tmp_path / "Makefile").write_text("".join(lines))
        out = _run_section(_section_makefile(), tmp_path, with_header=True)
        assert "... (truncated)" in out

    def test_no_output_without_makefile(self, tmp_path: Path) -> None:
        """No Makefile section is emitted when no Makefile exists."""
        out = _run_section(_section_makefile(), tmp_path, with_header=True)
        assert "**Makefile**" not in out

    def test_fallback_to_git_root_makefile(self, tmp_path: Path) -> None:
        """Falls back to the git root Makefile when CWD is a subdirectory.

        In a monorepo the user may be working in a nested package directory
        that has no Makefile of its own. The script should discover the
        Makefile at the git root and display it with its full path.

        Example layout:

            repo/           <- git root, contains Makefile
            └── packages/
                └── foo/    <- CWD (no Makefile here)
        """
        _git_init_commit(tmp_path, branch="main")
        (tmp_path / "Makefile").write_text("test:\n\tpytest\n")
        subdir = tmp_path / "packages" / "foo"
        subdir.mkdir(parents=True)
        # Need _section_project() to set ROOT before _section_makefile()
        script = _section_project() + "\n" + _section_makefile()
        out = _run_section(script, subdir, with_header=True)
        assert f"`{tmp_path}/Makefile`" in out
        assert "pytest" in out


# ---------------------------------------------------------------------------
# Protocol tests
# ---------------------------------------------------------------------------


class TestExecutableBackend:
    """Tests for _ExecutableBackend runtime-checkable protocol."""

    def test_object_with_execute_satisfies_protocol(self) -> None:
        class HasExecute:
            def execute(self, command: str) -> None: ...

        assert isinstance(HasExecute(), _ExecutableBackend)

    def test_object_without_execute_does_not_satisfy(self) -> None:
        class NoExecute:
            pass

        assert not isinstance(NoExecute(), _ExecutableBackend)


# ---------------------------------------------------------------------------
# End-to-end script test
# ---------------------------------------------------------------------------


class TestFullScript:
    """End-to-end tests for the assembled DETECT_CONTEXT_SCRIPT."""

    def test_full_script_executes_successfully(self, tmp_path: Path) -> None:
        """Full assembled script runs without errors."""
        (tmp_path / "pyproject.toml").write_text("[tool.uv]\n")
        (tmp_path / "uv.lock").write_text("")
        result = subprocess.run(
            ["bash", "-c", DETECT_CONTEXT_SCRIPT],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            check=False,
        )
        assert result.returncode == 0
        assert "## Local Context" in result.stdout
        assert "Python: uv" in result.stdout

    def test_full_script_in_git_repo(self, tmp_path: Path) -> None:
        """Full script with git repo produces git section."""
        _git_init_commit(tmp_path, branch="main")
        result = subprocess.run(
            ["bash", "-c", DETECT_CONTEXT_SCRIPT],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            check=False,
        )
        assert result.returncode == 0
        assert "Current branch `main`" in result.stdout


# ---------------------------------------------------------------------------
# Additional coverage tests
# ---------------------------------------------------------------------------


class TestSectionProjectExtended:
    """Extended tests for _section_project."""

    def test_go_project(self, tmp_path: Path) -> None:
        (tmp_path / "go.mod").write_text("")
        out = _run_section(_section_project(), tmp_path, with_header=True)
        assert "Language: go" in out

    def test_java_project_pom(self, tmp_path: Path) -> None:
        (tmp_path / "pom.xml").write_text("")
        out = _run_section(_section_project(), tmp_path, with_header=True)
        assert "Language: java" in out

    def test_java_project_gradle(self, tmp_path: Path) -> None:
        (tmp_path / "build.gradle").write_text("")
        out = _run_section(_section_project(), tmp_path, with_header=True)
        assert "Language: java" in out

    def test_node_modules_env(self, tmp_path: Path) -> None:
        (tmp_path / "node_modules").mkdir()
        out = _run_section(_section_project(), tmp_path, with_header=True)
        assert "Environments: node_modules" in out

    def test_project_root_shown_in_subdirectory(self, tmp_path: Path) -> None:
        _git_init_commit(tmp_path, branch="main")
        subdir = tmp_path / "packages" / "foo"
        subdir.mkdir(parents=True)
        out = _run_section(_section_project(), subdir, with_header=True)
        assert f"Project root: `{tmp_path}`" in out


class TestSectionPackageManagersExtended:
    """Extended tests for _section_package_managers."""

    def test_pipenv_via_pipfile(self, tmp_path: Path) -> None:
        (tmp_path / "Pipfile").write_text("")
        out = _run_section(_section_package_managers(), tmp_path)
        assert "Python: pipenv" in out

    def test_pipenv_via_pipfile_lock(self, tmp_path: Path) -> None:
        (tmp_path / "Pipfile.lock").write_text("")
        out = _run_section(_section_package_managers(), tmp_path)
        assert "Python: pipenv" in out

    def test_pnpm_lock(self, tmp_path: Path) -> None:
        (tmp_path / "pnpm-lock.yaml").write_text("")
        out = _run_section(_section_package_managers(), tmp_path)
        assert "Node: pnpm" in out

    def test_poetry_via_pyproject(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("[tool.poetry]\n")
        out = _run_section(_section_package_managers(), tmp_path)
        assert "Python: poetry" in out


class TestSectionGitExtended:
    """Extended tests for _section_git."""

    def test_both_main_and_master_listed(self, tmp_path: Path) -> None:
        _git_init_commit(tmp_path, branch="main")
        subprocess.run(
            ["git", "branch", "master"],
            cwd=tmp_path,
            capture_output=True,
            check=False,
        )
        out = _run_section(_section_git(), tmp_path, with_header=True)
        assert "`main`" in out
        assert "`master`" in out
