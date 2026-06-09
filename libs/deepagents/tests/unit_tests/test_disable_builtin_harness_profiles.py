"""Tests for disabling the SDK's built-in harness profiles.

Covers both opt-out mechanisms — the `DEEPAGENTS_DISABLE_BUILTIN_HARNESS_PROFILES`
environment variable and the `disable_builtin_harness_profiles()` function — across
the pre-bootstrap and post-bootstrap cases.

These tests mutate process-global bootstrap state, so the `fresh_bootstrap`
fixture snapshots and faithfully restores it (see the note in
`tests/unit_tests/conftest.py` about the save-and-restore pattern).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from deepagents import (
    HarnessProfile,
    disable_builtin_harness_profiles,
    register_harness_profile,
)
from deepagents.profiles import _builtin_profiles as bp
from deepagents.profiles.harness.harness_profiles import (
    _HARNESS_PROFILES,
    _get_harness_profile,
)
from deepagents.profiles.provider.provider_profiles import _PROVIDER_PROFILES

if TYPE_CHECKING:
    from collections.abc import Iterator

_SONNET_KEY = "anthropic:claude-sonnet-4-6"
_HAIKU_KEY = "anthropic:claude-haiku-4-5"
_CODEX_KEY = "openai:gpt-5.3-codex"


@pytest.fixture
def fresh_bootstrap(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Hand the test a clean, un-bootstrapped registry and restore afterward.

    Snapshots the live registries and bootstrap flags, resets to a pristine
    un-bootstrapped slate so the test fully controls (re)registration, then
    restores the exact prior state so later tests in the session still observe
    populated registries.
    """
    saved_harness = dict(_HARNESS_PROFILES)
    saved_provider = dict(_PROVIDER_PROFILES)
    saved_builtin = dict(bp._BUILTIN_HARNESS_PROFILES)
    saved_loaded = bp._loaded
    saved_thread = bp._loading_thread_id
    saved_keys = bp._BOOTSTRAP_HARNESS_KEYS
    saved_disabled = bp._harness_profiles_disabled

    monkeypatch.delenv(bp._DISABLE_BUILTIN_HARNESS_PROFILES_ENV, raising=False)
    bp._harness_profiles_disabled = False
    bp._loaded = False
    bp._loading_thread_id = None
    bp._BUILTIN_HARNESS_PROFILES.clear()
    _HARNESS_PROFILES.clear()
    _PROVIDER_PROFILES.clear()
    try:
        yield
    finally:
        bp._harness_profiles_disabled = saved_disabled
        bp._loading_thread_id = saved_thread
        bp._BOOTSTRAP_HARNESS_KEYS = saved_keys
        bp._BUILTIN_HARNESS_PROFILES.clear()
        bp._BUILTIN_HARNESS_PROFILES.update(saved_builtin)
        bp._loaded = saved_loaded
        _HARNESS_PROFILES.clear()
        _HARNESS_PROFILES.update(saved_harness)
        _PROVIDER_PROFILES.clear()
        _PROVIDER_PROFILES.update(saved_provider)


@pytest.mark.usefixtures("fresh_bootstrap")
class TestDefault:
    """Without any opt-out, the built-in harness profiles load as before."""

    def test_builtins_present_by_default(self) -> None:
        bp._ensure_builtin_profiles_loaded()
        assert _get_harness_profile(_SONNET_KEY) is not None
        assert _get_harness_profile(_CODEX_KEY) is not None


@pytest.mark.usefixtures("fresh_bootstrap")
class TestEnvVar:
    """`DEEPAGENTS_DISABLE_BUILTIN_HARNESS_PROFILES` gates registration."""

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " on "])
    def test_truthy_values_suppress_builtins(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        monkeypatch.setenv(bp._DISABLE_BUILTIN_HARNESS_PROFILES_ENV, value)
        bp._ensure_builtin_profiles_loaded()
        assert _get_harness_profile(_SONNET_KEY) is None
        assert _get_harness_profile(_HAIKU_KEY) is None
        assert _get_harness_profile(_CODEX_KEY) is None

    @pytest.mark.parametrize("value", ["0", "false", "no", "", "maybe"])
    def test_falsey_values_keep_builtins(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        monkeypatch.setenv(bp._DISABLE_BUILTIN_HARNESS_PROFILES_ENV, value)
        bp._ensure_builtin_profiles_loaded()
        assert _get_harness_profile(_SONNET_KEY) is not None

    def test_provider_profiles_unaffected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(bp._DISABLE_BUILTIN_HARNESS_PROFILES_ENV, "1")
        bp._ensure_builtin_profiles_loaded()
        # Built-in harness profiles are gone, but model-construction provider
        # profiles still register.
        assert _PROVIDER_PROFILES
        assert not _HARNESS_PROFILES


@pytest.mark.usefixtures("fresh_bootstrap")
class TestFunction:
    """`disable_builtin_harness_profiles()` works before and after bootstrap."""

    def test_before_bootstrap_suppresses_registration(self) -> None:
        disable_builtin_harness_profiles()
        bp._ensure_builtin_profiles_loaded()
        assert _get_harness_profile(_SONNET_KEY) is None
        assert not _HARNESS_PROFILES

    def test_after_bootstrap_removes_untouched_builtins(self) -> None:
        bp._ensure_builtin_profiles_loaded()
        assert _get_harness_profile(_SONNET_KEY) is not None

        disable_builtin_harness_profiles()
        assert _get_harness_profile(_SONNET_KEY) is None
        assert _get_harness_profile(_CODEX_KEY) is None
        assert not _HARNESS_PROFILES

    def test_after_bootstrap_preserves_caller_layered_profile(self) -> None:
        bp._ensure_builtin_profiles_loaded()
        # Layer a caller registration on top of a built-in key; this merges into
        # a new profile object, so it must survive the disable.
        register_harness_profile(_SONNET_KEY, HarnessProfile(system_prompt_suffix="caller suffix"))

        disable_builtin_harness_profiles()

        preserved = _get_harness_profile(_SONNET_KEY)
        assert preserved is not None
        assert preserved.system_prompt_suffix == "caller suffix"
        # An untouched built-in under a different key is still removed.
        assert _get_harness_profile(_HAIKU_KEY) is None

    def test_sets_process_wide_flag(self) -> None:
        disable_builtin_harness_profiles()
        assert bp._builtin_harness_profiles_disabled() is True
