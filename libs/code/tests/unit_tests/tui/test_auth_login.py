"""Pilot tests for the standalone `dcode auth login` app."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from functools import partial
from typing import TYPE_CHECKING

import pytest
from textual.widgets import Input, OptionList, Static

from deepagents_code import auth_store, model_config
from deepagents_code.integrations import openai_codex
from deepagents_code.model_readiness import MissingProviderKey, Ready, check_readiness
from deepagents_code.tui.auth_login import (
    AuthLoginApp,
    LoginCancelled,
    LoginOutcome,
    selection_problem,
)
from deepagents_code.tui.widgets.auth import AuthManagerScreen, AuthPromptScreen
from deepagents_code.tui.widgets.model_selector import (
    ModelOption,
    ModelSelectorScreen,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from textual.pilot import Pilot


@pytest.fixture
def config_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> Iterator[Path]:
    """Isolate `config.toml`, `auth.json`, and provider key env vars.

    Parametrize indirectly with TOML text to start from a saved config.
    """
    path = tmp_path / "config.toml"
    text = getattr(request, "param", None)
    if text is not None:
        path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", path)
    monkeypatch.setattr(model_config, "DEFAULT_STATE_DIR", tmp_path / ".state")
    for name in list(os.environ):
        if name.endswith("_API_KEY"):
            monkeypatch.delenv(name)
    model_config.clear_caches()
    yield path
    model_config.clear_caches()


async def _wait_until(
    pilot: Pilot[LoginOutcome], condition: Callable[[], bool]
) -> None:
    for _ in range(200):
        if condition():
            return
        await pilot.pause(0.01)
    msg = "condition not met"
    raise AssertionError(msg)


async def _save_anthropic_key(pilot: Pilot[LoginOutcome]) -> None:
    """Drive the key step: open the highlighted provider and submit a key."""
    app = pilot.app
    await _wait_until(pilot, lambda: isinstance(app.screen, AuthManagerScreen))
    await pilot.press("enter")
    await _wait_until(pilot, lambda: isinstance(app.screen, AuthPromptScreen))
    app.screen.query_one("#auth-prompt-input", Input).value = "sk-ant-test"
    await pilot.press("enter")


async def _wait_for_models(pilot: Pilot[LoginOutcome]) -> list[ModelOption]:
    app = pilot.app
    await _wait_until(
        pilot,
        lambda: (
            isinstance(app.screen, ModelSelectorScreen)
            and bool(app.screen.query(ModelOption))
        ),
    )
    return list(app.screen.query(ModelOption))


@pytest.mark.usefixtures("config_path")
async def test_key_step_lists_only_installed_model_providers() -> None:
    app = AuthLoginApp(provider="anthropic")
    async with app.run_test() as pilot:
        await _wait_until(pilot, lambda: isinstance(app.screen, AuthManagerScreen))
        options = app.screen.query_one("#auth-manager-options", OptionList)
        ids = {options.get_option_at_index(i).id for i in range(options.option_count)}
        highlighted = options.get_option_at_index(options.highlighted or 0).id
        await pilot.press("escape")

    assert highlighted == "anthropic"
    assert "anthropic" in ids
    assert not ids & {"tavily", "langsmith"}


@pytest.mark.usefixtures("config_path")
async def test_save_key_then_choose_model_exits_ready() -> None:
    app = AuthLoginApp(provider="anthropic")
    async with app.run_test() as pilot:
        await _save_anthropic_key(pilot)
        rows = await _wait_for_models(pilot)
        await pilot.press("enter")

    assert {row.model_spec.partition(":")[0] for row in rows} == {"anthropic"}
    outcome = app.return_value
    assert isinstance(outcome, Ready)
    assert outcome.provider == "anthropic"
    assert model_config.ModelConfig.load().default_model == outcome.model_spec
    assert auth_store.get_stored_key("anthropic") == "sk-ant-test"


@pytest.mark.usefixtures("config_path")
async def test_cancel_at_key_step_exits_cancelled() -> None:
    app = AuthLoginApp(provider="anthropic")
    async with app.run_test() as pilot:
        await _wait_until(pilot, lambda: isinstance(app.screen, AuthManagerScreen))
        await pilot.press("escape")

    assert app.return_value == LoginCancelled("key")


@pytest.mark.usefixtures("config_path")
async def test_cancel_at_model_step_exits_cancelled() -> None:
    app = AuthLoginApp(provider="anthropic")
    async with app.run_test() as pilot:
        await _save_anthropic_key(pilot)
        await _wait_for_models(pilot)
        await pilot.press("escape")

    assert app.return_value == LoginCancelled("model")
    assert model_config.ModelConfig.load().default_model is None


@pytest.mark.parametrize(
    "config_path", ['[models]\ndefault = "anthropic:claude-saved"\n'], indirect=True
)
@pytest.mark.usefixtures("config_path")
async def test_model_step_skipped_when_permitted_default_exists() -> None:
    app = AuthLoginApp(provider="anthropic")
    async with app.run_test() as pilot:
        await _save_anthropic_key(pilot)
        await _wait_until(pilot, lambda: app.return_value is not None)
        screens = list(app.screen_stack)

    assert not any(isinstance(screen, ModelSelectorScreen) for screen in screens)
    assert app.return_value == Ready(
        model_spec="anthropic:claude-saved", provider="anthropic"
    )
    assert model_config.ModelConfig.load().default_model == "anthropic:claude-saved"


@pytest.mark.parametrize(
    "config_path",
    ['[models]\ndefault = "openai:blocked"\nallowed = ["anthropic:*"]\n'],
    indirect=True,
)
@pytest.mark.usefixtures("config_path")
async def test_blocked_default_is_kept_and_choice_saved_as_recent() -> None:
    """A default outside `models.allowed` is still never overwritten."""
    app = AuthLoginApp(provider="anthropic")
    async with app.run_test() as pilot:
        await _save_anthropic_key(pilot)
        await _wait_for_models(pilot)
        await pilot.press("enter")

    outcome = app.return_value
    assert isinstance(outcome, Ready)
    config = model_config.ModelConfig.load()
    assert config.default_model == "openai:blocked"
    assert config.recent_model == outcome.model_spec


@pytest.mark.usefixtures("config_path")
async def test_model_override_is_checked_instead_of_signed_in_provider() -> None:
    """A launch `--model` without its key is not ready after another sign-in.

    Repro: `--acp --model openai:gpt-test auth login anthropic`.
    """
    readiness = partial(check_readiness, model_override="openai:gpt-test")
    app = AuthLoginApp(provider="anthropic", readiness=readiness)
    async with app.run_test() as pilot:
        await _save_anthropic_key(pilot)
        await _wait_until(pilot, lambda: app.return_value is not None)
        screens = list(app.screen_stack)

    assert not any(isinstance(screen, ModelSelectorScreen) for screen in screens)
    assert app.return_value == MissingProviderKey(
        model_spec="openai:gpt-test", provider="openai"
    )


@pytest.mark.usefixtures("config_path")
async def test_typed_spec_from_another_provider_is_rejected() -> None:
    """Repro: the Anthropic-only selector accepted and saved `openai:other-model`."""
    app = AuthLoginApp(provider="anthropic")
    async with app.run_test() as pilot:
        await _save_anthropic_key(pilot)
        await _wait_for_models(pilot)
        app.screen.query_one("#model-filter", Input).value = "openai:other-model"
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        selector = app.screen
        footer = str(selector.query_one(".model-selector-help", Static).content)
        await pilot.press("escape")

    assert isinstance(selector, ModelSelectorScreen)
    assert "Choose a model from Anthropic" in footer
    assert app.return_value == LoginCancelled("model")
    assert model_config.ModelConfig.load().default_model is None


@pytest.mark.parametrize(
    "config_path", ['[models]\nallowed = ["anthropic:claude-ok"]\n'], indirect=True
)
@pytest.mark.usefixtures("config_path")
@pytest.mark.parametrize(
    ("spec", "accepted"),
    [
        ("anthropic:claude-ok", True),
        ("anthropic:claude-blocked", False),
        ("openai:gpt-test", False),
        ("not a spec", False),
    ],
)
def test_selection_problem_limits_to_provider_and_allowlist(
    spec: str, accepted: bool
) -> None:
    assert (selection_problem("anthropic", spec) is None) is accepted


@pytest.fixture
def chatgpt_login(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the ChatGPT OAuth flow succeed without a browser or network."""

    async def _signed_in(  # noqa: RUF029  # async signature dictated by protocol
        *_args: object, **_kwargs: object
    ) -> openai_codex.CodexAuthStatus:
        return openai_codex.CodexAuthStatus(
            logged_in=True,
            store_path=tmp_path / "codex.json",
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )

    monkeypatch.setattr(openai_codex, "run_browser_login", _signed_in)
    monkeypatch.setattr(
        openai_codex, "default_store_path", lambda: tmp_path / "codex.json"
    )


@pytest.mark.usefixtures("config_path", "chatgpt_login")
async def test_chatgpt_sign_in_advances_to_model_step_then_cancel() -> None:
    app = AuthLoginApp(provider="openai_codex")
    async with app.run_test() as pilot:
        await _wait_until(pilot, lambda: isinstance(app.screen, AuthManagerScreen))
        await pilot.press("enter")
        rows = await _wait_for_models(pilot)
        await pilot.press("escape")

    assert {row.model_spec.partition(":")[0] for row in rows} == {"openai_codex"}
    assert app.signed_in == "openai_codex"
    assert app.return_value == LoginCancelled("model")
