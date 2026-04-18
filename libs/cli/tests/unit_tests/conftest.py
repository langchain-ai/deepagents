"""Shared fixtures for CLI unit tests."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


@pytest.fixture(autouse=True, scope="session")
def _warm_model_caches() -> None:
    """Pre-populate model-config caches once per xdist worker.

    Tests like the model-selector UI tests call `get_available_models()` and
    `get_model_profiles()` during widget init.  Without a warm cache the first
    invocation in each worker process pays ~800-1200 ms of disk I/O to discover
    provider profiles via `importlib.util`.  Paying that cost once per session
    instead of once per test shaves significant time off the overall run.

    Tests that explicitly need a clean cache (e.g. `test_model_config.py`) use
    their own function-scoped `clear_caches()` fixture which overrides this.
    """
    with contextlib.suppress(Exception):
        from deepagents_cli.model_config import get_available_models, get_model_profiles

        get_available_models()
        get_model_profiles()


@pytest.fixture(autouse=True)
def _clear_langsmith_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent LangSmith env vars loaded from .env from leaking into tests.

    `dotenv.load_dotenv()` runs at `deepagents_cli.config` import time and
    may inject `LANGSMITH_*` variables from a local `.env` file.  These
    cause spurious failures in unit tests that run with `--disable-socket`
    because the LangSmith client attempts real HTTP requests.

    Each test that *needs* LangSmith variables should set them explicitly via
    `monkeypatch.setenv` or `patch.dict("os.environ", ...)`.
    """
    for key in (
        "LANGSMITH_API_KEY",
        "LANGCHAIN_API_KEY",
        "LANGSMITH_TRACING",
        "LANGCHAIN_TRACING_V2",
        "LANGSMITH_PROJECT",
        "DEEPAGENTS_CLI_LANGSMITH_PROJECT",
        "DEEPAGENTS_CLI_LANGSMITH_API_KEY",
        "DEEPAGENTS_CLI_LANGCHAIN_API_KEY",
        "DEEPAGENTS_CLI_LANGSMITH_TRACING",
        "DEEPAGENTS_CLI_LANGCHAIN_TRACING_V2",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def _register_theme_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make app-specific CSS variables available to all test `App` instances.

    Production code defines these in `DeepAgentsApp.get_theme_variable_defaults`
    but many tests use lightweight `App[None]` subclasses that lack the override.
    Patching the base class ensures `$mode-bash`, `$mode-command` resolve
    everywhere without requiring each test app to opt in.
    """
    from textual.app import App

    from deepagents_cli.theme import get_css_variable_defaults

    original = App.get_theme_variable_defaults
    custom = get_css_variable_defaults(dark=True)

    def _with_custom_vars(self: App) -> dict[str, str]:
        base = original(self)
        base.update(custom)
        return base

    monkeypatch.setattr(App, "get_theme_variable_defaults", _with_custom_vars)


@pytest.fixture(autouse=True)
def _provide_app_context() -> Generator[None]:
    """Set Textual's `active_app` context var for sync widget tests.

    Many unit tests construct widgets and call `compose()` directly without a
    running Textual app. Widget code that calls `self.app` (e.g., for
    theme-aware color lookups) needs a valid app in the context. This fixture
    provides a minimal `App` instance with the default LangChain theme
    registered so that `get_theme_colors()` returns the LC brand palette
    (matching `DARK_COLORS`).
    """
    from textual._context import active_app
    from textual.app import App
    from textual.theme import Theme

    from deepagents_cli import theme

    app = App()
    c = theme.DARK_COLORS
    app.register_theme(
        Theme(
            name="langchain",
            primary=c.primary,
            secondary=c.secondary,
            accent=c.accent,
            foreground=c.foreground,
            background=c.background,
            surface=c.surface,
            panel=c.panel,
            warning=c.warning,
            error=c.error,
            success=c.success,
            dark=True,
        )
    )
    app.theme = "langchain"
    token = active_app.set(app)
    try:
        yield
    finally:
        active_app.reset(token)


@pytest.fixture(autouse=True)
def _isolate_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect ChatInput history to a temp file.

    Without this, every test that mounts a `ChatInput` widget writes to the
    real `~/.deepagents/history.jsonl`, causing duplicate/stale entries that
    persist across test runs and branch switches.
    """
    monkeypatch.setattr(
        "deepagents_cli.widgets.chat_input._default_history_path",
        lambda: tmp_path / "history.jsonl",
    )
