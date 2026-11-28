"""Pilot tests for the standalone `dcode auth login` app."""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime, timedelta
from functools import partial
from typing import TYPE_CHECKING

import pytest
from textual.widgets import Input, OptionList, Static

from deepagents_code import auth_store, model_config
from deepagents_code.integrations import openai_codex
from deepagents_code.model_readiness import (
    InvalidModelConfig,
    LaunchOptions,
    MissingProviderKey,
    NoModelSelected,
    Readiness,
    Ready,
    check_readiness,
)
from deepagents_code.tui.auth_login import (
    AuthLoginApp,
    KeyStep,
    LoginCancelled,
    LoginOutcome,
    ModelStep,
    Step,
    accept_model_choice,
    plan_step,
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


def _screen_text(app: AuthLoginApp) -> str:
    return " ".join(str(widget.content) for widget in app.screen.query(Static))


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
async def test_blocked_default_is_replaced_by_the_sign_in_choice() -> None:
    app = AuthLoginApp(provider="anthropic")
    async with app.run_test() as pilot:
        await _save_anthropic_key(pilot)
        await _wait_for_models(pilot)
        await pilot.press("enter")

    outcome = app.return_value
    assert isinstance(outcome, Ready)
    assert model_config.ModelConfig.load().default_model == outcome.model_spec


@pytest.mark.parametrize(
    "config_path", ['[models]\ndefault = "openai:gpt-test"\n'], indirect=True
)
@pytest.mark.usefixtures("config_path")
async def test_signing_in_to_another_provider_than_the_default_finishes() -> None:
    """Repro: an OpenAI default without its key made an Anthropic sign-in exit 1."""
    app = AuthLoginApp(provider="anthropic")
    async with app.run_test() as pilot:
        await _save_anthropic_key(pilot)
        await _wait_for_models(pilot)
        await pilot.press("enter")

    outcome = app.return_value
    assert isinstance(outcome, Ready)
    assert outcome.provider == "anthropic"
    assert model_config.ModelConfig.load().default_model == outcome.model_spec


@pytest.mark.usefixtures("config_path")
async def test_model_override_asks_for_its_own_providers_key() -> None:
    """Repro: `--model openai:gpt-test auth login anthropic` ended on the wrong key."""
    readiness = partial(check_readiness, model_override="openai:gpt-test")
    app = AuthLoginApp(provider="anthropic", readiness=readiness, fixed_model=True)
    async with app.run_test() as pilot:
        await _wait_until(pilot, lambda: isinstance(app.screen, AuthManagerScreen))
        options = app.screen.query_one("#auth-manager-options", OptionList)
        highlighted = options.get_option_at_index(options.highlighted or 0).id
        await pilot.press("enter")
        await _wait_until(pilot, lambda: isinstance(app.screen, AuthPromptScreen))
        app.screen.query_one("#auth-prompt-input", Input).value = "sk-openai-test"
        await pilot.press("enter")
        await _wait_until(pilot, lambda: app.return_value is not None)
        screens = list(app.screen_stack)

    assert highlighted == "openai"
    assert not any(isinstance(screen, ModelSelectorScreen) for screen in screens)
    assert app.return_value == Ready(model_spec="openai:gpt-test", provider="openai")


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
    ("spec", "saved"),
    [
        ("anthropic:claude-ok", "anthropic:claude-ok"),
        pytest.param(
            "claude-ok", "anthropic:claude-ok", id="bare-name-saved-canonical"
        ),
        ("anthropic:claude-blocked", None),
        ("openai:gpt-test", None),
        ("not a spec", None),
    ],
)
async def test_accept_model_choice_saves_only_a_permitted_canonical_spec(
    spec: str, saved: str | None
) -> None:
    auth_store.set_stored_key("anthropic", "sk-ant-test")

    problem = await accept_model_choice(frozenset({"anthropic"}), LaunchOptions(), spec)

    assert (problem is None) is (saved is not None)
    assert model_config.ModelConfig.load().default_model == saved


@pytest.fixture
def chatgpt_login(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the ChatGPT OAuth flow succeed without a browser or network.

    Status reads report signed in only after the login runs, as with a real
    token store.
    """
    signed_in: list[bool] = []

    def _status(*_args: object, **_kwargs: object) -> openai_codex.CodexAuthStatus:
        if not signed_in:
            return openai_codex.CodexAuthStatus(
                logged_in=False, store_path=tmp_path / "codex.json"
            )
        return openai_codex.CodexAuthStatus(
            logged_in=True,
            store_path=tmp_path / "codex.json",
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )

    async def _signed_in(  # noqa: RUF029  # async signature dictated by protocol
        *_args: object, **_kwargs: object
    ) -> openai_codex.CodexAuthStatus:
        signed_in.append(True)
        return _status()

    monkeypatch.setattr(openai_codex, "run_browser_login", _signed_in)
    monkeypatch.setattr(openai_codex, "get_status", _status)
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
    assert app.return_value == LoginCancelled("model")


@pytest.mark.usefixtures("config_path")
async def test_existing_env_key_goes_straight_to_model_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The common ACP case: a key exists but `session/new` found no model."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
    app = AuthLoginApp(provider="anthropic")
    async with app.run_test() as pilot:
        await _wait_for_models(pilot)
        screens = list(app.screen_stack)
        await pilot.press("enter")

    assert not any(isinstance(screen, AuthManagerScreen) for screen in screens)
    outcome = app.return_value
    assert isinstance(outcome, Ready)
    assert outcome.provider == "anthropic"
    assert model_config.ModelConfig.load().default_model == outcome.model_spec


@pytest.mark.parametrize(
    "config_path", ['[models]\ndefault = "anthropic:claude-saved"\n'], indirect=True
)
@pytest.mark.usefixtures("config_path")
async def test_already_ready_exits_without_a_screen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
    pushed: list[object] = []
    app = AuthLoginApp(provider="anthropic")
    monkeypatch.setattr(
        app, "push_screen", lambda screen, *_a, **_k: pushed.append(screen)
    )
    async with app.run_test() as pilot:
        await _wait_until(pilot, lambda: app.return_value is not None)

    assert pushed == []
    assert app.return_value == Ready(
        model_spec="anthropic:claude-saved", provider="anthropic"
    )


@pytest.mark.usefixtures("config_path")
async def test_escape_on_key_step_continues_when_a_key_appeared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A key found without a save (Ctrl+R reload, ChatGPT session) is not a cancel."""
    app = AuthLoginApp(provider="anthropic")
    async with app.run_test() as pilot:
        await _wait_until(pilot, lambda: isinstance(app.screen, AuthManagerScreen))
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
        await pilot.press("escape")
        await _wait_for_models(pilot)
        await pilot.press("enter")

    assert isinstance(app.return_value, Ready)


@pytest.mark.parametrize(
    "config_path", ['[models]\ndefault = "openai:gpt-test"\n'], indirect=True
)
@pytest.mark.usefixtures("config_path")
async def test_escape_on_key_step_exits_ready_when_already_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Naming a provider without a key opens the key step, but dcode is ready."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-env")
    app = AuthLoginApp(provider="anthropic")
    async with app.run_test() as pilot:
        await _wait_until(pilot, lambda: isinstance(app.screen, AuthManagerScreen))
        await pilot.press("escape")

    assert app.return_value == Ready(model_spec="openai:gpt-test", provider="openai")


@pytest.mark.usefixtures("config_path")
async def test_login_without_a_provider_reuses_a_saved_key() -> None:
    """Repro: a key saved by an earlier, cancelled login forced a re-entry of it."""
    auth_store.set_stored_key("anthropic", "sk-ant-test")
    app = AuthLoginApp()
    async with app.run_test() as pilot:
        rows = await _wait_for_models(pilot)
        screens = list(app.screen_stack)
        await pilot.press("enter")

    assert not any(isinstance(screen, AuthManagerScreen) for screen in screens)
    assert {row.model_spec.partition(":")[0] for row in rows} == {"anthropic"}
    assert isinstance(app.return_value, Ready)


@pytest.mark.usefixtures("config_path")
async def test_quit_on_the_model_step_reports_that_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`run_login` names the step on screen when the app quits without a result."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
    app = AuthLoginApp(provider="anthropic")
    async with app.run_test() as pilot:
        await _wait_for_models(pilot)
        app.exit()

    assert app.return_value is None
    assert app.active_step == "model"


@pytest.mark.usefixtures("config_path")
async def test_another_providers_key_keeps_an_explicit_sign_in_open() -> None:
    """Repro: `auth login anthropic` finished on a saved OpenAI key instead."""
    app = AuthLoginApp(provider="anthropic")
    async with app.run_test() as pilot:
        await _wait_until(pilot, lambda: isinstance(app.screen, AuthManagerScreen))
        auth_store.set_stored_key("openai", "sk-openai-test")
        app.screen.post_message(AuthManagerScreen.CredentialSaved("openai"))
        await pilot.pause(0.1)
        still_on_key_step = isinstance(app.screen, AuthManagerScreen)
        await pilot.press("escape")

    assert still_on_key_step
    assert app.return_value == LoginCancelled("key")


@pytest.mark.usefixtures("config_path")
async def test_choice_is_not_saved_when_its_key_vanished(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repro: a key removed while the picker was open still replaced the default.

    The failed credential sends the user back to sign in to that provider.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
    app = AuthLoginApp(provider="anthropic")
    async with app.run_test() as pilot:
        await _wait_for_models(pilot)
        monkeypatch.delenv("ANTHROPIC_API_KEY")
        await pilot.press("enter")
        await _wait_until(pilot, lambda: isinstance(app.screen, AuthManagerScreen))
        await pilot.press("escape")
        await _wait_until(pilot, lambda: app.return_value is not None)

    assert app.return_value == LoginCancelled("key")
    assert model_config.ModelConfig.load().default_model is None


@pytest.mark.parametrize(
    "config_path", ['[models]\ndefault = "anthropic:claude-saved"\n'], indirect=True
)
@pytest.mark.usefixtures("config_path")
async def test_credential_failure_when_the_model_builds_returns_to_the_key_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repro: an expired ChatGPT session exited as a config error, not a sign-in."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
    monkeypatch.setattr(
        "deepagents_code.tui.auth_login.construction_problem",
        lambda spec, provider, _options: MissingProviderKey(
            model_spec=spec, provider=provider
        ),
    )
    app = AuthLoginApp(provider="anthropic")
    async with app.run_test() as pilot:
        await _wait_until(pilot, lambda: isinstance(app.screen, AuthManagerScreen))
        options = app.screen.query_one("#auth-manager-options", OptionList)
        highlighted = options.get_option_at_index(options.highlighted or 0).id
        await pilot.press("escape")
        await _wait_until(pilot, lambda: app.return_value is not None)

    assert highlighted == "anthropic"
    assert app.return_value == LoginCancelled("key"), "Escape must not rebuild"


@pytest.mark.parametrize(
    "config_path",
    [
        '[models]\nallowed = ["anthropic:claude-private"]\n',
        pytest.param(
            '[models]\nallowed = ["anthropic:claude-private"]\n'
            "[models.providers.ollama]\nbase_url_env = true\n",
            id="unrelated-provider-discovery-error",
        ),
    ],
    indirect=True,
)
@pytest.mark.usefixtures("config_path")
async def test_an_allowlisted_model_discovery_does_not_list_can_be_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repro: an empty model list stopped login before the ID could be entered."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
    app = AuthLoginApp(provider="anthropic")
    async with app.run_test() as pilot:
        await _wait_until(pilot, lambda: isinstance(app.screen, ModelSelectorScreen))
        text = _screen_text(app)
        app.screen.query_one("#model-filter", Input).value = "anthropic:claude-private"
        await pilot.pause()
        await pilot.press("enter")
        await _wait_until(pilot, lambda: app.return_value is not None)

    assert "type one as anthropic:<model>" in text
    assert app.return_value == Ready(
        model_spec="anthropic:claude-private", provider="anthropic"
    )
    assert model_config.ModelConfig.load().default_model == "anthropic:claude-private"


@pytest.mark.parametrize(
    "config_path", ["[models.providers.openai]\nenabled = false\n"], indirect=True
)
@pytest.mark.usefixtures("config_path", "chatgpt_login")
async def test_chatgpt_with_openai_disabled_reaches_the_model_step() -> None:
    """Discovery lists ChatGPT models through openai; typing one still works."""
    app = AuthLoginApp(provider="openai_codex")
    async with app.run_test() as pilot:
        await _wait_until(pilot, lambda: isinstance(app.screen, AuthManagerScreen))
        await pilot.press("enter")
        await _wait_until(pilot, lambda: isinstance(app.screen, ModelSelectorScreen))
        text = _screen_text(app)
        await pilot.press("escape")

    assert "openai_codex:<model>" in text
    assert app.return_value == LoginCancelled("model")


@pytest.mark.parametrize(
    "config_path",
    [
        (
            '[models]\ndefault = "openai_codex:gpt-test"\n'
            "[models.providers.openai]\nenabled = false\n"
        )
    ],
    indirect=True,
)
@pytest.mark.usefixtures("config_path")
async def test_chatgpt_sign_in_for_a_saved_model_works_with_openai_disabled() -> None:
    """Repro: a saved ChatGPT model needs no picker, so openai is not required."""
    app = AuthLoginApp(provider="openai_codex")
    async with app.run_test() as pilot:
        await _wait_until(pilot, lambda: isinstance(app.screen, AuthManagerScreen))
        options = app.screen.query_one("#auth-manager-options", OptionList)
        ids = {options.get_option_at_index(i).id for i in range(options.option_count)}
        highlighted = options.get_option_at_index(options.highlighted or 0).id
        await pilot.press("escape")

    assert "openai_codex" in ids
    assert highlighted == "openai_codex"


@pytest.mark.usefixtures("config_path")
async def test_a_second_enter_does_not_start_a_second_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each check may construct a model or refresh a ChatGPT token."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
    calls: list[str] = []
    release = asyncio.Event()

    async def accept(_providers: object, _options: object, spec: str) -> str | None:
        calls.append(spec)
        await release.wait()
        return "rejected"

    monkeypatch.setattr("deepagents_code.tui.auth_login.accept_model_choice", accept)
    app = AuthLoginApp(provider="anthropic")
    async with app.run_test() as pilot:
        await _wait_for_models(pilot)
        await pilot.press("enter")
        await pilot.press("enter")
        await pilot.pause(0.1)
        release.set()
        await pilot.pause(0.1)
        await pilot.press("escape")

    assert len(calls) == 1
    assert app.return_value == LoginCancelled("model")


_READY = Ready(model_spec="anthropic:x", provider="anthropic")
_NO_KEY_OPENAI = MissingProviderKey(model_spec="openai:x", provider="openai")
_BAD = InvalidModelConfig("blocked")
_ANTHROPIC = frozenset({"anthropic"})
_NONE: frozenset[str] = frozenset()


@pytest.mark.parametrize(
    ("readiness", "provider", "fixed_model", "has_key", "keyed", "expected"),
    [
        pytest.param(_READY, None, False, False, _NONE, _READY, id="ready"),
        pytest.param(_READY, "anthropic", False, True, _NONE, _READY, id="ready-same"),
        pytest.param(
            _READY, "openai", False, False, _NONE, KeyStep("openai"), id="switch"
        ),
        pytest.param(
            NoModelSelected(),
            "anthropic",
            False,
            True,
            _NONE,
            ModelStep(_ANTHROPIC),
            id="key-no-model",
        ),
        pytest.param(
            NoModelSelected(),
            "anthropic",
            False,
            False,
            _NONE,
            KeyStep("anthropic"),
            id="no-key",
        ),
        pytest.param(
            NoModelSelected(),
            None,
            False,
            False,
            _ANTHROPIC,
            ModelStep(_ANTHROPIC),
            id="bare-with-saved-key",
        ),
        pytest.param(
            NoModelSelected(), None, False, False, _NONE, KeyStep(None), id="bare"
        ),
        pytest.param(
            _NO_KEY_OPENAI,
            "anthropic",
            False,
            True,
            _NONE,
            ModelStep(_ANTHROPIC),
            id="other-provider-than-default",
        ),
        pytest.param(
            _NO_KEY_OPENAI,
            None,
            False,
            False,
            _NONE,
            KeyStep("openai"),
            id="default-provider-key",
        ),
        pytest.param(
            _NO_KEY_OPENAI,
            None,
            False,
            False,
            _ANTHROPIC,
            ModelStep(_ANTHROPIC),
            id="bare-prefers-a-provider-with-a-key",
        ),
        pytest.param(
            _NO_KEY_OPENAI,
            "anthropic",
            True,
            True,
            _NONE,
            KeyStep("openai"),
            id="fixed-model-needs-its-key",
        ),
        pytest.param(_READY, "openai", True, False, _NONE, _READY, id="fixed-ready"),
        pytest.param(_BAD, "anthropic", False, True, _NONE, _BAD, id="config-error"),
    ],
)
def test_plan_step(
    readiness: Readiness,
    provider: str | None,
    fixed_model: bool,
    has_key: bool,
    keyed: frozenset[str],
    expected: Step,
) -> None:
    step = plan_step(
        readiness,
        provider=provider,
        fixed_model=fixed_model,
        has_key=lambda _p: has_key,
        unavailable=lambda _p: None,
        keyed=lambda: keyed,
    )

    assert step == expected


def test_plan_step_reports_an_unusable_explicit_provider() -> None:
    """Repro: an uninstalled provider opened an empty picker or a manager without it."""
    step = plan_step(
        NoModelSelected(),
        provider="cohere",
        fixed_model=False,
        has_key=lambda _p: True,
        unavailable=lambda p: f"{p} is not installed",
        keyed=lambda: _NONE,
    )

    assert step == InvalidModelConfig("cohere is not installed")
