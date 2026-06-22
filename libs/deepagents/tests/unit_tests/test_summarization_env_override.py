"""Tests for env-var overrides of the summarization context budget."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from deepagents.middleware.summarization import (
    _positive_int_env,
    compute_summarization_defaults,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

_TRIGGER = "DEEPAGENTS_SUMMARIZE_TRIGGER_TOKENS"
_KEEP = "DEEPAGENTS_SUMMARIZE_KEEP_TOKENS"


class _NoProfileModel:
    """Minimal stand-in: compute_summarization_defaults only reads `.profile`."""

    profile = None


def _model() -> BaseChatModel:
    return cast("BaseChatModel", _NoProfileModel())


def test_positive_int_env_unset_or_invalid(monkeypatch) -> None:
    monkeypatch.delenv("X_BUDGET", raising=False)
    assert _positive_int_env("X_BUDGET") is None
    monkeypatch.setenv("X_BUDGET", "0")
    assert _positive_int_env("X_BUDGET") is None
    monkeypatch.setenv("X_BUDGET", "-5")
    assert _positive_int_env("X_BUDGET") is None
    monkeypatch.setenv("X_BUDGET", "abc")
    assert _positive_int_env("X_BUDGET") is None


def test_positive_int_env_valid(monkeypatch) -> None:
    monkeypatch.setenv("X_BUDGET", "75000")
    assert _positive_int_env("X_BUDGET") == 75000


def test_defaults_unchanged_without_env(monkeypatch) -> None:
    monkeypatch.delenv(_TRIGGER, raising=False)
    monkeypatch.delenv(_KEEP, raising=False)
    defaults = compute_summarization_defaults(_model())
    assert defaults["trigger"] == ("tokens", 170000)
    assert defaults["keep"] == ("messages", 6)


def test_env_overrides_trigger_and_keep(monkeypatch) -> None:
    monkeypatch.setenv(_TRIGGER, "75000")
    monkeypatch.setenv(_KEEP, "15000")
    defaults = compute_summarization_defaults(_model())
    assert defaults["trigger"] == ("tokens", 75000)
    assert defaults["keep"] == ("tokens", 15000)
    assert defaults["truncate_args_settings"]["trigger"] == ("tokens", 75000)
    assert defaults["truncate_args_settings"]["keep"] == ("tokens", 15000)
