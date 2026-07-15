"""Shared fixtures for unit tests."""

from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


_UPDATE_CHECK_SELF_MANAGED_MARK = "self_managed_update_check"


def _self_manages_update_check(request: pytest.FixtureRequest) -> bool:
    """Return whether the test owns app-startup update-check setup."""
    return request.node.get_closest_marker(_UPDATE_CHECK_SELF_MANAGED_MARK) is not None


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
        from deepagents_code.model_config import (
            get_available_models,
            get_model_profiles,
        )

        get_available_models()
        get_model_profiles()


@pytest.fixture(autouse=True)
def _restore_os_environ() -> Generator[None, None, None]:
    """Snapshot and restore `os.environ` around every test.

    Production code under test (`_ensure_bootstrap`, `_load_dotenv`,
    `_apply_default_langsmith_project`) writes to `os.environ` directly. When a
    test clears a variable with `monkeypatch.delenv(name, raising=False)` that
    was already absent, monkeypatch records no undo entry — so a later direct
    write by that code survives teardown and leaks into subsequent tests (e.g.
    a dotenv-reload test leaking `DEEPAGENTS_CODE_OPENAI_API_KEY` into a gateway
    key-mismatch test). Defined before the other autouse fixtures so it tears
    down last, leaving `os.environ` pristine no matter how a key was set.

    Restores by diffing against the snapshot rather than a blanket
    `clear()`/`update()`, so a test that never touches `os.environ` (the vast
    majority) triggers zero `putenv` calls on teardown.
    """
    snapshot = dict(os.environ)
    try:
        yield
    finally:
        for key in [key for key in os.environ if key not in snapshot]:
            del os.environ[key]
        for key, value in snapshot.items():
            if os.environ.get(key) != value:
                os.environ[key] = value


@pytest.fixture(autouse=True)
def _clear_langsmith_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent LangSmith env vars loaded from .env from leaking into tests.

    `dotenv.load_dotenv()` runs at `deepagents_code.config` import time and
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
        "LANGSMITH_ENDPOINT",
        "LANGCHAIN_ENDPOINT",
        "LANGSMITH_PROJECT",
        "DEEPAGENTS_CODE_LANGSMITH_PROJECT",
        "DEEPAGENTS_CODE_LANGSMITH_REDACT",
        "DEEPAGENTS_CODE_LANGSMITH_API_KEY",
        "DEEPAGENTS_CODE_LANGCHAIN_API_KEY",
        "DEEPAGENTS_CODE_LANGSMITH_TRACING",
        "DEEPAGENTS_CODE_LANGCHAIN_TRACING_V2",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def _clear_tavily_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent a Tavily key loaded from .env from leaking into tests.

    Like `LANGSMITH_*`, `dotenv.load_dotenv()` at `deepagents_code.config`
    import time may inject `TAVILY_API_KEY` from a developer's local `.env`.
    A leaked key flips `settings.has_tavily` to `True`, which silently changes
    onboarding behavior: the launch sequence short-circuits the Tavily step on
    a dev machine but runs it on CI, so a test that reaches the step passes
    locally yet hangs (real screen push) or writes a credential on CI.

    Each test that *needs* a Tavily key should set it explicitly via
    `monkeypatch.setenv` or patch `settings.has_tavily`.
    """
    for key in ("TAVILY_API_KEY", "DEEPAGENTS_CODE_TAVILY_API_KEY"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def _clear_provider_base_url_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent provider base-URL env vars from leaking into tests.

    A developer machine provisioned with the LangSmith gateway exports
    `OPENAI_BASE_URL` / `ANTHROPIC_BASE_URL`, which `get_base_url` now reads as
    a fallback. Clear them (and the `DEEPAGENTS_CODE_` overrides) so base-URL
    tests are deterministic. Tests that need a value set it explicitly.
    """
    for key in (
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_API_URL",
        "BASETEN_BASE_URL",
        "BASETEN_API_BASE",
        "GOOGLE_GEMINI_BASE_URL",
        "DEEPAGENTS_CODE_OPENAI_BASE_URL",
        "DEEPAGENTS_CODE_ANTHROPIC_BASE_URL",
        "DEEPAGENTS_CODE_BASETEN_BASE_URL",
        "DEEPAGENTS_CODE_BASETEN_API_BASE",
        "DEEPAGENTS_CODE_GOOGLE_GEMINI_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def _clear_onboarding_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent local debug onboarding env vars from affecting tests."""
    monkeypatch.delenv("DEEPAGENTS_CODE_DEBUG_ONBOARDING", raising=False)


@pytest.fixture(autouse=True)
def _clear_update_env(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prevent update debug/loop-guard and toggle env vars from affecting tests.

    `DEEPAGENTS_CODE_DEBUG_UPDATE` short-circuits the install path, and the
    internal `DEEPAGENTS_CODE_RESTARTED_AFTER_UPDATE` sentinel suppresses
    auto-update to break a restart loop. The sentinel can leak in not just from a
    developer shell but from a prior test exercising the production code that sets
    it, so it is cleared unconditionally. `DEEPAGENTS_CODE_AUTO_UPDATE` is read
    directly from the environment by `is_auto_update_enabled`, so a developer who
    exports it would otherwise make auto-update tests fail or pass spuriously.

    Most unit tests should not run the app startup PyPI update check at all: it
    performs DNS in a background worker, which pytest-socket reports under
    `--disable-socket` even when the app swallows the failure. Set the production
    opt-out env var by default so subprocess tests inherit the same no-network
    behavior. Tests marked `self_managed_update_check` cover the update-check
    gate directly, so they opt out of this default below.
    """
    monkeypatch.delenv("DEEPAGENTS_CODE_DEBUG_UPDATE", raising=False)
    monkeypatch.delenv("DEEPAGENTS_CODE_RESTARTED_AFTER_UPDATE", raising=False)
    monkeypatch.delenv("DEEPAGENTS_CODE_AUTO_UPDATE", raising=False)

    if _self_manages_update_check(request):
        monkeypatch.delenv("DEEPAGENTS_CODE_NO_UPDATE_CHECK", raising=False)
    else:
        monkeypatch.setenv("DEEPAGENTS_CODE_NO_UPDATE_CHECK", "1")


@pytest.fixture(autouse=True)
def _disable_app_startup_update_checks(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep app startup tests from racing PyPI or user update config.

    Tests marked `self_managed_update_check` manage the gate themselves, so leave
    `is_update_check_enabled` untouched for them.
    """
    if _self_manages_update_check(request):
        return
    monkeypatch.setattr(
        "deepagents_code.update_check.is_update_check_enabled",
        lambda: False,
    )


@pytest.fixture(autouse=True)
def _clear_behavior_override_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent developer behavior overrides from changing default-path tests."""
    for key in (
        "DEEPAGENTS_CODE_CURSOR_STYLE",
        "DEEPAGENTS_CODE_EXPERIMENTAL",
        "DEEPAGENTS_CODE_MEMORY_AUTO_SAVE",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def _clear_external_event_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent local alpha event-listener env vars from affecting tests."""
    monkeypatch.delenv("DEEPAGENTS_CODE_EXTERNAL_EVENT_SOCKET", raising=False)
    monkeypatch.delenv("DEEPAGENTS_CODE_EXTERNAL_EVENT_SOCKET_PATH", raising=False)


@pytest.fixture(autouse=True)
def _disable_terminal_escape(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop tests from leaking terminal control sequences to the real terminal.

    Production code constructs `DeepAgentsApp` and exercises the spinner / theme
    paths, which emit `OSC 11` (background color) and `OSC 9;4` (taskbar
    progress) via `terminal_escape.write_terminal_escape`. That writer targets
    `/dev/tty`, which pytest does not capture, so running the suite from inside
    a real terminal (e.g. an editable install) visibly recolors the developer's
    session. Opting out keeps the run inert. `test_terminal_escape.py` clears
    this var in its own fixture so its assertions still exercise the real path.
    """
    monkeypatch.setenv("DEEPAGENTS_CODE_NO_TERMINAL_ESCAPE", "1")


@pytest.fixture(autouse=True)
def _register_theme_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make app-specific CSS variables available to all test `App` instances.

    Production code defines these in `DeepAgentsApp.get_theme_variable_defaults`
    but many tests use lightweight `App[None]` subclasses that lack the override.
    Patching the base class ensures custom mode variables resolve everywhere
    without requiring each test app to opt in.
    """
    from textual.app import App

    from deepagents_code.theme import get_css_variable_defaults

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

    from deepagents_code import theme

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
def _isolate_state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect app-managed state away from the developer's real data."""
    state_dir = tmp_path / ".state"
    monkeypatch.setattr("deepagents_code.model_config.DEFAULT_STATE_DIR", state_dir)

    from deepagents_code import sessions

    monkeypatch.setattr(sessions, "_db_path", None)
    sessions._message_count_cache.clear()
    sessions._initial_prompt_cache.clear()
    sessions._recent_threads_cache.clear()


@pytest.fixture(autouse=True)
def _isolate_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect ChatInput history to a temp file.

    Without this, every test that mounts a `ChatInput` widget writes to the
    real `~/.deepagents/.state/history.jsonl`, causing duplicate/stale
    entries that persist across test runs and branch switches.
    """
    monkeypatch.setattr(
        "deepagents_code.tui.widgets.chat_input._default_history_path",
        lambda: tmp_path / "history.jsonl",
    )


@pytest.fixture(autouse=True)
def _clear_kitty_kbd_probe_cache() -> None:
    """Reset the `functools.cache` on the kitty-keyboard-protocol probe.

    The probe is cached for the lifetime of the process in production,
    but stale state leaks across tests that patch the probe function or
    rely on platform-specific behaviour. Clearing on every test keeps
    results deterministic regardless of file order or `pytest-xdist`
    sharding.
    """
    from deepagents_code.terminal_capabilities import supports_kitty_keyboard_protocol

    supports_kitty_keyboard_protocol.cache_clear()
