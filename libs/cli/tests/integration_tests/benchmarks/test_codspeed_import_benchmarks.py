"""CodSpeed-compatible import benchmarks for CLI startup performance.

These tests use the pytest-benchmark fixture so CodSpeed can track import times
across commits and flag regressions. Module caches are evicted between
iterations to simulate a cold-start import.

Run locally:

```bash
make benchmark
uv run --group test pytest ./tests -m benchmark -v
```

Run with CodSpeed:

```bash
uv run --group test pytest ./tests -m benchmark --codspeed
```
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pytest_benchmark.fixture import BenchmarkFixture

# Module-name prefixes to evict between benchmark iterations so each
# run simulates a cold-start import.
_EVICT_PREFIXES = (
    "deepagents_cli",
    "deepagents",
    "langchain",
    "langchain_core",
    "langchain_anthropic",
    "langchain_openai",
    "langchain_google_genai",
    "langgraph",
)


def _evict_modules() -> None:
    """Remove CLI and heavy-dependency modules from `sys.modules`."""
    for key in list(sys.modules):
        if any(key == p or key.startswith(f"{p}.") for p in _EVICT_PREFIXES):
            del sys.modules[key]


pytestmark = pytest.mark.benchmark


@pytest.fixture(autouse=True)
def _clean_module_cache() -> Iterator[None]:
    """Evict cached modules before *and* after every test."""
    _evict_modules()
    yield
    _evict_modules()


# ---------------------------------------------------------------------------
# Lightweight startup-path modules
#
# These are imported during CLI startup and must stay fast.  Regressions
# here directly impact `deepagents --help` and time-to-first-prompt.
# ---------------------------------------------------------------------------


class TestStartupPathBenchmarks:
    """Cold-start import benchmarks for modules on the CLI startup path."""

    def test_import_app(self, benchmark: BenchmarkFixture) -> None:
        """Full `app` module — the critical startup path."""

        def do_import() -> None:
            _evict_modules()
            import deepagents_cli.app

        benchmark(do_import)

    def test_import_main(self, benchmark: BenchmarkFixture) -> None:
        """`main` module — CLI entry point with argparse."""

        def do_import() -> None:
            _evict_modules()
            import deepagents_cli.main

        benchmark(do_import)

    def test_import_cli_context(self, benchmark: BenchmarkFixture) -> None:
        """`_cli_context` — lightweight TypedDict for runtime overrides."""

        def do_import() -> None:
            _evict_modules()
            from deepagents_cli._cli_context import CLIContext

        benchmark(do_import)

    def test_import_ask_user_types(self, benchmark: BenchmarkFixture) -> None:
        """`_ask_user_types` — lightweight TypedDicts for ask-user protocol."""

        def do_import() -> None:
            _evict_modules()
            import deepagents_cli._ask_user_types

        benchmark(do_import)

    def test_import_textual_adapter(self, benchmark: BenchmarkFixture) -> None:
        """`textual_adapter` — heavy langchain deps are deferred."""

        def do_import() -> None:
            _evict_modules()
            import deepagents_cli.textual_adapter

        benchmark(do_import)

    def test_import_tool_display(self, benchmark: BenchmarkFixture) -> None:
        """`tool_display` — SDK backends are deferred."""

        def do_import() -> None:
            _evict_modules()
            import deepagents_cli.tool_display

        benchmark(do_import)

    def test_import_config(self, benchmark: BenchmarkFixture) -> None:
        """`config` — settings module, should be lightweight."""

        def do_import() -> None:
            _evict_modules()
            import deepagents_cli.config

        benchmark(do_import)

    def test_import_ui(self, benchmark: BenchmarkFixture) -> None:
        """`ui` — display helpers, should be lightweight."""

        def do_import() -> None:
            _evict_modules()
            import deepagents_cli.ui

        benchmark(do_import)

    def test_import_file_ops(self, benchmark: BenchmarkFixture) -> None:
        """`file_ops` — file operation tracking, SDK import deferred."""

        def do_import() -> None:
            _evict_modules()
            import deepagents_cli.file_ops

        benchmark(do_import)


# ---------------------------------------------------------------------------
# Heavy runtime modules
#
# These are NOT on the startup path (imported by agent.py at runtime) but
# are tracked to catch regressions in agent initialization time.
# ---------------------------------------------------------------------------


class TestRuntimePathBenchmarks:
    """Cold-start import benchmarks for heavy runtime modules."""

    def test_import_configurable_model(self, benchmark: BenchmarkFixture) -> None:
        """`configurable_model` — middleware + langchain deps."""

        def do_import() -> None:
            _evict_modules()
            import deepagents_cli.configurable_model

        benchmark(do_import)

    def test_import_ask_user(self, benchmark: BenchmarkFixture) -> None:
        """`ask_user` — middleware + langchain/langgraph deps."""

        def do_import() -> None:
            _evict_modules()
            import deepagents_cli.ask_user

        benchmark(do_import)
