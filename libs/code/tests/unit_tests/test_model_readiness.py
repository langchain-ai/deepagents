"""Tests for the session readiness check shared by `auth login` and ACP."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from deepagents_code import auth_store, model_config
from deepagents_code.model_readiness import (
    InvalidModelConfig,
    LaunchOptions,
    MissingProviderKey,
    NoModelSelected,
    Ready,
    UnverifiedProviderAuth,
    check_readiness,
    construction_problem,
    keyed_providers,
    provider_problem,
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


def test_unverifiable_provider_auth_can_start(config_path: Path) -> None:
    """Only a known-missing key blocks a session; the SDK reports real failures."""
    config_path.write_text(
        '[models]\ndefault = "nonexistent_provider:made-up-model"\n',
        encoding="utf-8",
    )

    assert check_readiness() == UnverifiedProviderAuth(
        model_spec="nonexistent_provider:made-up-model",
        provider="nonexistent_provider",
    )
    assert readiness_problem() is None


@pytest.mark.usefixtures("config_path")
@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        pytest.param(
            "google_vertexai:claude-test",
            "google_anthropic_vertex:claude-test",
            id="provider-cannot-serve-model",
        ),
        pytest.param("nonexistent_provider:made-up-model", "", id="unknown-provider"),
    ],
)
def test_construction_problem_reports_what_session_start_would(
    spec: str, expected: str
) -> None:
    """Readiness passes these; constructing the model, as a session does, fails."""
    problem = construction_problem(spec, spec.partition(":")[0], LaunchOptions())

    assert isinstance(problem, InvalidModelConfig)
    assert expected in problem.message


@pytest.mark.usefixtures("config_path")
def test_construction_leaves_the_process_environment_alone() -> None:
    """Repro: a check left a stored key in the environment after its deletion."""
    auth_store.set_stored_key("anthropic", "sk-ant-test")

    problem = construction_problem(
        "anthropic:claude-test", "anthropic", LaunchOptions()
    )

    assert problem is None
    assert "ANTHROPIC_API_KEY" not in os.environ


@pytest.mark.usefixtures("config_path")
@pytest.mark.parametrize("error", [ValueError, ImportError])
def test_constructor_errors_from_provider_code_are_not_shown(
    error: type[Exception], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repro: `create_model` wrapped an SDK error that echoed the key it was given."""
    auth_store.set_stored_key("anthropic", "sk-ant-test")

    def init_chat_model(*_args: object, **_kwargs: object) -> None:
        msg = "invalid api_key sk-synthetic-secret"
        raise error(msg)

    monkeypatch.setattr("langchain.chat_models.init_chat_model", init_chat_model)

    problem = construction_problem("anthropic:claude-x", "anthropic", LaunchOptions())

    assert isinstance(problem, InvalidModelConfig)
    assert "sk-synthetic-secret" not in problem.message
    assert error.__name__ in problem.message


@pytest.mark.usefixtures("config_path")
def test_credential_failure_in_construction_asks_for_a_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An expired ChatGPT session passes readiness but fails when the model builds."""
    from deepagents_code.model_config import MissingCredentialsError

    def create_model(*_args: object, **_kwargs: object) -> None:
        msg = "sign in to ChatGPT again"
        raise MissingCredentialsError(msg, provider="openai_codex", env_var=None)

    monkeypatch.setattr("deepagents_code.config.create_model", create_model)

    problem = construction_problem(
        "openai_codex:gpt-x", "openai_codex", LaunchOptions()
    )

    assert problem == MissingProviderKey(
        model_spec="openai_codex:gpt-x", provider="openai_codex"
    )


def test_provider_problem_names_why_a_provider_is_unusable(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path.write_text(
        '[models]\nallowed = ["anthropic:*", "groq:*", "cohere:*"]\n'
        '[models.providers.groq]\nclass_path = "my_pkg:MyChat"\n'
        "[models.providers.openai]\nenabled = false\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "deepagents_code.config_manifest.is_provider_package_installed",
        lambda provider: provider not in {"cohere", "groq"},
    )

    assert provider_problem("anthropic") is None
    assert provider_problem("groq") is None, "a custom class_path needs no package"
    assert "dcode install cohere" in (provider_problem("cohere") or "")
    assert "disabled" in (provider_problem("openai") or "")
    assert "permits no google_genai" in (provider_problem("google_genai") or "")


def test_keyed_providers_skip_providers_policy_blocks(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repro: an OpenAI key under `allowed = ["anthropic:*"]` opened an empty picker."""
    config_path.write_text('[models]\nallowed = ["anthropic:*"]\n', encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    auth_store.set_stored_key("anthropic", "sk-ant-test")

    assert keyed_providers() == {"anthropic"}


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
