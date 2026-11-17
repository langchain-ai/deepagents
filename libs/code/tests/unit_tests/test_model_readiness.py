"""Tests for the session readiness check shared by `auth login` and ACP."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from deepagents_code import auth_store, model_config
from deepagents_code.model_readiness import (
    InvalidModelConfig,
    MissingProviderKey,
    NoModelSelected,
    Ready,
    UnverifiedProviderAuth,
    check_readiness,
    readiness_problem,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture
def config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Isolate `config.toml`, `auth.json`, and provider key env vars."""
    path = tmp_path / "config.toml"
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", path)
    monkeypatch.setattr(model_config, "DEFAULT_STATE_DIR", tmp_path / ".state")
    for name in list(os.environ):
        if name.endswith("_API_KEY"):
            monkeypatch.delenv(name)
    model_config.clear_caches()
    yield path
    model_config.clear_caches()


@pytest.mark.usefixtures("config_path")
def test_no_saved_model_names_no_model() -> None:
    problem = readiness_problem()

    assert isinstance(problem, NoModelSelected)
    assert problem.message == "no model selected"


def test_model_without_key_names_the_provider(config_path: Path) -> None:
    config_path.write_text(
        '[models]\ndefault = "anthropic:claude-test"\n', encoding="utf-8"
    )

    problem = readiness_problem()

    assert isinstance(problem, MissingProviderKey)
    assert problem.model_spec == "anthropic:claude-test"
    assert problem.provider == "anthropic"
    assert problem.message == "no key for anthropic"


def test_model_and_key_is_ready(config_path: Path) -> None:
    config_path.write_text(
        '[models]\ndefault = "anthropic:claude-test"\n', encoding="utf-8"
    )
    auth_store.set_stored_key("anthropic", "sk-ant-test")

    assert check_readiness() == Ready(
        model_spec="anthropic:claude-test", provider="anthropic"
    )
    assert readiness_problem() is None


def test_recent_model_counts_when_no_default(config_path: Path) -> None:
    config_path.write_text(
        '[models]\nrecent = "anthropic:claude-test"\n', encoding="utf-8"
    )
    auth_store.set_stored_key("anthropic", "sk-ant-test")

    assert readiness_problem() is None


def test_disallowed_default_is_not_ready(config_path: Path) -> None:
    config_path.write_text(
        '[models]\ndefault = "anthropic:claude-test"\nallowed = ["openai:*"]\n',
        encoding="utf-8",
    )
    auth_store.set_stored_key("anthropic", "sk-ant-test")

    assert isinstance(readiness_problem(), NoModelSelected)


def test_deny_all_policy_is_a_config_error(config_path: Path) -> None:
    """Signing in cannot fix a policy that allows nothing, so it is not auth."""
    config_path.write_text("[models]\nallowed = []\n", encoding="utf-8")

    problem = readiness_problem()

    assert isinstance(problem, InvalidModelConfig)
    assert "allows no models" in problem.message


def test_unknown_provider_auth_is_not_ready(config_path: Path) -> None:
    """An unknown provider is not proof of credentials, unlike a missing key."""
    config_path.write_text(
        '[models]\ndefault = "nonexistent_provider:made-up-model"\n',
        encoding="utf-8",
    )

    problem = readiness_problem()

    assert isinstance(problem, UnverifiedProviderAuth)
    assert problem.model_spec == "nonexistent_provider:made-up-model"
    assert problem.message == "cannot verify credentials for nonexistent_provider"


def test_model_override_outranks_saved_default(config_path: Path) -> None:
    """A launch `--model` is what the session uses, so its key is the one checked."""
    config_path.write_text(
        '[models]\ndefault = "anthropic:claude-test"\n', encoding="utf-8"
    )
    auth_store.set_stored_key("anthropic", "sk-ant-test")

    problem = readiness_problem(model_override="openai:gpt-test")

    assert problem == MissingProviderKey(
        model_spec="openai:gpt-test", provider="openai"
    )


@pytest.mark.usefixtures("config_path")
def test_model_override_without_saved_model_is_ready() -> None:
    auth_store.set_stored_key("openai", "sk-openai-test")

    assert check_readiness(model_override="openai:gpt-test") == Ready(
        model_spec="openai:gpt-test", provider="openai"
    )


def test_model_override_outside_allowlist_is_a_config_error(
    config_path: Path,
) -> None:
    config_path.write_text('[models]\nallowed = ["anthropic:*"]\n', encoding="utf-8")
    auth_store.set_stored_key("openai", "sk-openai-test")

    problem = readiness_problem(model_override="openai:gpt-test")

    assert isinstance(problem, InvalidModelConfig)
    assert "openai:gpt-test" in problem.message
