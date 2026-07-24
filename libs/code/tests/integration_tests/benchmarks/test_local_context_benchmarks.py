"""Wall-time benchmarks for `LocalContextMiddleware`.

Run locally:  `make benchmark`
Run with CodSpeed:  `make bench`

The shell benchmarks exercise the environment-detection work that delays the first
interaction and each post-summarization refresh. The middleware-bookkeeping benchmark
isolates `before_agent`'s per-invocation overhead from shell execution (using a static
backend). The request benchmark isolates the prompt composition performed before every
model call.
"""

from __future__ import annotations

import os
import subprocess
from typing import TYPE_CHECKING, Any, cast

import pytest
from deepagents.backends import LocalShellBackend
from deepagents.backends.protocol import ExecuteResponse
from langchain.agents.middleware.types import AgentState, ModelRequest

from deepagents_code.local_context import (
    _DETECT_SCRIPT_TIMEOUT,
    DETECT_CONTEXT_SCRIPT,
    LocalContextMiddleware,
    LocalContextState,
    _section_files,
    _section_makefile,
)
from deepagents_code.mcp_tools import MCPServerInfo, MCPToolInfo

if TYPE_CHECKING:
    from pathlib import Path

    from langchain_core.language_models import BaseChatModel
    from langgraph.runtime import Runtime
    from pytest_benchmark.fixture import BenchmarkFixture

pytestmark = pytest.mark.benchmark


class _StaticBackend:
    """Return prebuilt detection output without shell or mock overhead."""

    def __init__(self, output: str) -> None:
        self._result = ExecuteResponse(output=output, exit_code=0)

    def execute(
        self,
        command: str,  # noqa: ARG002
        *,
        timeout: int | None = None,  # noqa: ARG002
    ) -> ExecuteResponse:
        """Return the prebuilt result."""
        return self._result


def _git_env(home: Path) -> dict[str, str]:
    """Build a deterministic environment for Git fixture setup."""
    return {
        **os.environ,
        "GIT_AUTHOR_NAME": "benchmark",
        "GIT_AUTHOR_EMAIL": "benchmark@example.com",
        "GIT_COMMITTER_NAME": "benchmark",
        "GIT_COMMITTER_EMAIL": "benchmark@example.com",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "HOME": str(home),
        "LC_ALL": "C",
        "LANG": "C",
    }


def _shell_env(home: Path, tools: Path) -> dict[str, str]:
    """Build an isolated environment with controlled optional executables."""
    return {
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "HOME": str(home),
        "LC_ALL": "C",
        "LANG": "C",
        "PATH": f"{tools}{os.pathsep}{os.environ['PATH']}",
        "TMPDIR": str(home),
    }


def _install_benchmark_tools(directory: Path) -> None:
    """Install deterministic stand-ins for optional detection commands."""
    directory.mkdir()
    scripts = {
        "python3": "#!/bin/sh\necho 'Python 3.12.0'\n",
        "node": "#!/bin/sh\necho 'v22.0.0'\n",
        "gh": ("#!/bin/sh\ncat <<'EOF'\nJSON FIELDS\n  number, title, url\n\nEOF\n"),
        "tree": ("#!/bin/sh\nprintf '.\\n|-- apps\\n|-- libs\\n`-- tests\\n'\n"),
    }
    for name, script in scripts.items():
        executable = directory / name
        executable.write_text(script)
        executable.chmod(0o755)


@pytest.fixture(scope="module")
def realistic_backend(tmp_path_factory: pytest.TempPathFactory) -> LocalShellBackend:
    """Create a dirty mixed-language monorepo for full-script measurements."""
    root = tmp_path_factory.mktemp("local-context-repo")
    home = tmp_path_factory.mktemp("local-context-home")
    tools = home / "bin"
    _install_benchmark_tools(tools)
    (root / "libs" / "python_pkg").mkdir(parents=True)
    (root / "apps" / "web" / "src").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / ".deepagents").mkdir()
    (root / ".venv").mkdir()
    (root / "node_modules").mkdir()
    (root / "pyproject.toml").write_text(
        "[tool.uv]\n[tool.pytest.ini_options]\naddopts = '-q'\n"
    )
    (root / "uv.lock").write_text("")
    (root / "package.json").write_text('{"scripts": {"test": "vitest"}}\n')
    (root / "pnpm-lock.yaml").write_text("lockfileVersion: '9.0'\n")
    (root / "Makefile").write_text(
        "test:\n\tpytest\n" + "".join(f"target-{i}:\n\t@true\n" for i in range(24))
    )
    for index in range(120):
        parent = root / ("libs/python_pkg" if index % 2 else "apps/web/src")
        (parent / f"module_{index:03d}.py").write_text(f"VALUE = {index}\n")

    env = _git_env(home)
    subprocess.run(
        ["git", "init", "-b", "main"],
        cwd=root,
        env=env,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "add", "Makefile", "apps", "libs", "package.json", "pyproject.toml"],
        cwd=root,
        env=env,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "benchmark fixture"],
        cwd=root,
        env=env,
        capture_output=True,
        check=True,
    )
    (root / "apps" / "web" / "src" / "module_000.py").write_text("VALUE = -1\n")
    (root / "untracked.txt").write_text("dirty\n")

    return LocalShellBackend(
        root_dir=root,
        virtual_mode=False,
        inherit_env=False,
        env=_shell_env(home, tools),
    )


@pytest.fixture(scope="module")
def large_makefile_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a large Makefile whose preview should require only 21 lines."""
    root = tmp_path_factory.mktemp("local-context-large-makefile")
    (root / "Makefile").write_text("test:\n\t@true\n" + "# filler\n" * 500_000)
    return root


@pytest.fixture(scope="module")
def large_directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create enough top-level entries to expose non-linear shell processing."""
    root = tmp_path_factory.mktemp("local-context-large-directory")
    for index in range(10_000):
        (root / f"file-{index:05d}").touch()
    return root


def test_detect_context_script_realistic_monorepo(
    benchmark: BenchmarkFixture,
    realistic_backend: LocalShellBackend,
) -> None:
    """Measure production detection through `LocalShellBackend.execute`."""
    result = benchmark.pedantic(
        realistic_backend.execute,
        args=(DETECT_CONTEXT_SCRIPT,),
        kwargs={"timeout": _DETECT_SCRIPT_TIMEOUT},
        rounds=10,
        warmup_rounds=2,
        iterations=1,
    )

    assert result.exit_code == 0
    assert "## Local Context" in result.output
    assert "Current branch `main`" in result.output
    assert "uncommitted changes" in result.output


def test_large_makefile_preview(
    benchmark: BenchmarkFixture,
    large_makefile_dir: Path,
) -> None:
    """Measure that Makefile preview work stays bounded by its 20-line output."""

    def run_section() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", "-c", _section_makefile()],
            cwd=large_makefile_dir,
            capture_output=True,
            text=True,
            check=False,
        )

    result = benchmark.pedantic(
        run_section,
        rounds=10,
        warmup_rounds=2,
        iterations=1,
    )

    assert result.returncode == 0
    assert "... (truncated)" in result.stdout


def test_large_directory_listing(
    benchmark: BenchmarkFixture,
    large_directory: Path,
) -> None:
    """Measure top-level filtering/counting without a per-entry Bash loop."""

    def run_section() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", "-c", _section_files()],
            cwd=large_directory,
            capture_output=True,
            text=True,
            check=False,
        )

    result = benchmark.pedantic(
        run_section,
        rounds=10,
        warmup_rounds=2,
        iterations=1,
    )

    assert result.returncode == 0
    assert "**Files** (showing 20 of 10000):" in result.stdout


def test_initial_detection_python_overhead(benchmark: BenchmarkFixture) -> None:
    """Measure middleware bookkeeping separately from shell execution."""
    middleware = LocalContextMiddleware(_StaticBackend("local context\n" * 200))
    state: LocalContextState = {"messages": []}
    runtime = cast("Runtime[Any]", None)

    def run_batch() -> dict[str, Any]:
        result = None
        for _ in range(10_000):
            result = middleware.before_agent(state, runtime)
        assert result is not None
        return result

    result = benchmark(run_batch)

    assert result["_local_context"].startswith("local context")


def test_compose_model_request(benchmark: BenchmarkFixture) -> None:
    """Measure the prompt composition performed before every model call."""
    servers = [
        MCPServerInfo(
            name=f"server-{server_index}",
            transport="http",
            tools=tuple(
                MCPToolInfo(name=f"tool_{server_index}_{tool_index}", description="")
                for tool_index in range(12)
            ),
        )
        for server_index in range(3)
    ]
    middleware = LocalContextMiddleware(
        _StaticBackend(""),
        mcp_server_info=servers,
        tracing_project="shared-deepagents-code",
        user_tracing_project="user-project",
    )
    local_context = "## Local Context\n\n" + "context detail\n" * 600
    state = cast(
        "AgentState[Any]",
        {"messages": [], "_local_context": local_context},
    )
    request = ModelRequest(
        model=cast("BaseChatModel", None),
        messages=[],
        system_prompt="system instruction\n" * 800,
        state=state,
    )

    def run_batch() -> ModelRequest:
        result = None
        for _ in range(1_000):
            result = middleware._get_modified_request(request)
        assert result is not None
        return result

    result = benchmark(run_batch)

    assert result.system_prompt is not None
    assert "## Local Context" in result.system_prompt
    assert "**LangSmith Tracing**:" in result.system_prompt
    assert "**MCP Servers**" in result.system_prompt
