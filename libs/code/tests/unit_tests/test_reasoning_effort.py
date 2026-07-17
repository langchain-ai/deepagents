"""Tests for `/effort` reasoning effort handling.

Support data comes from LangChain model profiles, so most tests mock
`get_model_profiles()` instead of relying on installed provider packages.
"""

import logging
from collections.abc import Coroutine, Iterator
from contextlib import AbstractContextManager
from pathlib import Path
from typing import get_args
from unittest.mock import AsyncMock, Mock, patch

import pytest
from textual.app import App
from textual.widgets import OptionList

from deepagents_code import model_config, reasoning_effort
from deepagents_code.app import DeepAgentsApp
from deepagents_code.command_registry import COMMANDS
from deepagents_code.config import settings
from deepagents_code.model_config import ModelProfileEntry
from deepagents_code.reasoning_effort import (
    EffortLabel,
    current_effort_from_model_params,
    default_effort_for_model,
    is_effort_supported_for_model,
    merge_effort_model_params,
    supported_efforts_for_model,
    without_effort_model_params,
)
from deepagents_code.tui.widgets.effort_selector import EffortSelectorScreen
from deepagents_code.tui.widgets.messages import ErrorMessage


@pytest.fixture(autouse=True)
def _restore_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    original_name = settings.model_name
    original_provider = settings.model_provider
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", tmp_path / "config.toml")
    model_config.clear_caches()
    yield
    settings.model_name = original_name
    settings.model_provider = original_provider
    model_config.clear_caches()


def _profile_entry(**profile: object) -> ModelProfileEntry:
    return ModelProfileEntry(profile=dict(profile), overridden_keys=frozenset())


def _mock_profiles(
    mapping: dict[str, ModelProfileEntry],
) -> AbstractContextManager[Mock]:
    """Patch `get_model_profiles` to return a fixed, hermetic mapping."""
    return patch.object(reasoning_effort, "get_model_profiles", return_value=mapping)


# Reading logic (mocked profiles, provider-agnostic)


def test_supported_efforts_for_model_reads_profile_field() -> None:
    with _mock_profiles(
        {"acme:foo": _profile_entry(reasoning_effort_levels=["low", "high"])}
    ):
        assert supported_efforts_for_model("acme:foo") == ("low", "high")


def test_supported_efforts_for_model_missing_spec_is_empty() -> None:
    with _mock_profiles({}):
        assert supported_efforts_for_model("acme:unknown") == ()
    assert supported_efforts_for_model(None) == ()
    assert supported_efforts_for_model("") == ()


def test_supported_efforts_for_model_missing_field_is_empty() -> None:
    with _mock_profiles({"acme:foo": _profile_entry(max_input_tokens=1000)}):
        assert supported_efforts_for_model("acme:foo") == ()


def test_supported_efforts_for_model_malformed_field_is_empty() -> None:
    """A non-list `reasoning_effort_levels` (e.g. bad hand-edited data) is discarded."""
    with _mock_profiles(
        {"acme:foo": _profile_entry(reasoning_effort_levels="not-a-list")}
    ):
        assert supported_efforts_for_model("acme:foo") == ()


def test_default_effort_for_model_reads_profile_field() -> None:
    with _mock_profiles({"acme:foo": _profile_entry(reasoning_effort_default="high")}):
        assert default_effort_for_model("acme:foo") == "high"


def test_default_effort_for_model_missing_spec_is_none() -> None:
    with _mock_profiles({}):
        assert default_effort_for_model("acme:unknown") is None
    assert default_effort_for_model(None) is None


def test_default_effort_for_model_malformed_field_is_none() -> None:
    with _mock_profiles({"acme:foo": _profile_entry(reasoning_effort_default=5)}):
        assert default_effort_for_model("acme:foo") is None


def test_is_effort_supported_for_model() -> None:
    with _mock_profiles(
        {"acme:foo": _profile_entry(reasoning_effort_levels=["low", "high"])}
    ):
        assert is_effort_supported_for_model("acme:foo", "high")
        assert not is_effort_supported_for_model("acme:foo", "medium")
        assert not is_effort_supported_for_model("acme:unknown", "high")


def test_supported_efforts_for_model_forwards_cli_override() -> None:
    with _mock_profiles({}) as mock_profiles:
        supported_efforts_for_model("acme:foo", cli_override={"x": 1})
    mock_profiles.assert_called_once_with(cli_override={"x": 1})


def test_default_effort_for_model_forwards_cli_override() -> None:
    with _mock_profiles({}) as mock_profiles:
        default_effort_for_model("acme:foo", cli_override={"x": 1})
    mock_profiles.assert_called_once_with(cli_override={"x": 1})


# --- Reading logic against real, installed profile data ---------------------
# openai/anthropic/google-genai are direct dependencies, so always installed;
# fireworks/xai are optional extras and not assumed present here.


@pytest.mark.parametrize(
    ("model_spec", "efforts"),
    [
        ("openai:gpt-5.5", ("none", "low", "medium", "high", "xhigh")),
        ("openai:gpt-5.6-sol", ("none", "low", "medium", "high", "xhigh", "max")),
        # `openai_codex` mirrors the curated `CODEX_MODELS` subset of `openai`
        # profiles (handled inside `get_model_profiles` itself), so reasoning
        # effort support resolves identically under either provider name.
        ("openai_codex:gpt-5.5", ("none", "low", "medium", "high", "xhigh")),
        ("anthropic:claude-opus-4-8", ("low", "medium", "high", "xhigh", "max")),
        ("anthropic:claude-opus-4-5", ("low", "medium", "high")),
        # Sonnet 4.5 predates reasoning effort: no profile entry for it.
        ("anthropic:claude-sonnet-4-5", ()),
        ("google_genai:gemini-3-pro-preview", ("minimal", "low", "medium", "high")),
        # Gemini 2.5 only supports the numeric `thinking_budget`, not named
        # levels, so it's intentionally excluded from `reasoning_effort_levels`.
        ("google_genai:gemini-2.5-pro", ()),
        # A model with no reasoning capability at all.
        ("openai:gpt-4o", ()),
        # An unrecognized spec.
        ("ollama:llama3.1", ()),
    ],
)
def test_supported_efforts_for_model_real_profiles(
    model_spec: str, efforts: tuple[str, ...]
) -> None:
    assert supported_efforts_for_model(model_spec) == efforts


@pytest.mark.parametrize(
    ("model_spec", "default"),
    [
        ("openai:gpt-5.5", "medium"),
        ("openai:gpt-5.4", None),
        ("anthropic:claude-opus-4-8", "high"),
        ("anthropic:claude-sonnet-4-5", None),
        ("google_genai:gemini-3-pro-preview", "high"),
    ],
)
def test_default_effort_for_model_real_profiles(
    model_spec: str, default: str | None
) -> None:
    assert default_effort_for_model(model_spec) == default


# current_effort_from_model_params
# `/effort` writes only the standard `reasoning_effort` key going forward, but
# a provider-shaped key can still arrive via a raw `--model-params` value that
# bypassed `/effort` -- it must still be recognized as an explicit effort
# setting, since `_restore_effort_override` uses this function to decide
# whether a saved preference should be merged on top of it.


def test_current_effort_reads_flat_sentinel() -> None:
    assert (
        current_effort_from_model_params(
            "anthropic:claude-opus-4-8", {"reasoning_effort": "high"}
        )
        == "high"
    )


@pytest.mark.parametrize(
    "model_params",
    [
        {"reasoning": {"effort": "low"}},
        {"output_config": {"effort": "low"}},
        {"model_kwargs": {"reasoning_effort": "low"}},
        {"extra_body": {"reasoning_effort": "low"}},
        {"thinking_level": "low"},
    ],
)
def test_current_effort_recognizes_provider_shaped_values(
    model_params: dict[str, object],
) -> None:
    """A raw `--model-params` value in any provider's shape is still detected.

    Regression test: an explicit provider-shaped effort must not be treated
    as "no effort set", or `_restore_effort_override` would incorrectly merge
    a saved preference on top of it.
    """
    assert (
        current_effort_from_model_params("anthropic:claude-opus-4-8", model_params)
        == "low"
    )


def test_current_effort_flat_sentinel_takes_priority_over_provider_shape() -> None:
    """When both are present, the flat sentinel (`/effort`'s own write path) wins."""
    assert (
        current_effort_from_model_params(
            "anthropic:claude-opus-4-8",
            {"reasoning_effort": "high", "output_config": {"effort": "low"}},
        )
        == "high"
    )


@pytest.mark.parametrize(
    "model_params",
    [
        {"reasoning": {"effort": 5}},
        {"output_config": {"effort": 5}},
        {"model_kwargs": {"reasoning_effort": 5}},
        {"extra_body": {"reasoning_effort": 5}},
        {"thinking_level": 5},
    ],
)
def test_current_effort_warns_on_malformed_provider_shaped_value(
    model_params: dict[str, object],
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        assert (
            current_effort_from_model_params("anthropic:claude-opus-4-8", model_params)
            is None
        )
    assert any(record.levelno == logging.WARNING for record in caplog.records)


@pytest.mark.parametrize(
    "model_params",
    [
        {"reasoning": "raw"},
        {"output_config": "raw"},
        {"model_kwargs": "raw"},
        {"extra_body": "raw"},
    ],
)
def test_current_effort_non_dict_container_is_silent(
    model_params: dict[str, object],
) -> None:
    """A non-dict container is a legitimate shape and must not warn.

    E.g. preserved verbatim by `without_effort_model_params`.
    """
    assert (
        current_effort_from_model_params("anthropic:claude-opus-4-8", model_params)
        is None
    )


def test_current_effort_requires_spec_and_params() -> None:
    assert current_effort_from_model_params(None, {"reasoning_effort": "high"}) is None
    assert current_effort_from_model_params("anthropic:claude-opus-4-8", None) is None
    assert current_effort_from_model_params("anthropic:claude-opus-4-8", {}) is None


def test_merge_and_clear_effort_model_params_preserves_unrelated_params() -> None:
    merged = merge_effort_model_params(
        {"temperature": 0.2, "model_kwargs": {"top_p": 0.9}},
        {"model_kwargs": {"reasoning_effort": "high"}},
    )

    assert merged == {
        "temperature": 0.2,
        "model_kwargs": {"top_p": 0.9, "reasoning_effort": "high"},
    }
    assert without_effort_model_params(merged) == {
        "temperature": 0.2,
        "model_kwargs": {"top_p": 0.9},
    }


def test_merge_and_clear_xai_effort_preserves_extra_body_params() -> None:
    merged = merge_effort_model_params(
        {"extra_body": {"prompt_cache_key": "thread-1"}},
        {"extra_body": {"reasoning_effort": "high"}},
    )

    assert merged == {
        "extra_body": {"prompt_cache_key": "thread-1", "reasoning_effort": "high"}
    }
    assert without_effort_model_params(merged) == {
        "extra_body": {"prompt_cache_key": "thread-1"}
    }


def test_without_effort_clears_anthropic_thinking_and_effort() -> None:
    """A raw `--model-params` value using the legacy nested shape still clears."""
    format_config = {"type": "json_schema", "schema": {"type": "object"}}
    params = {
        "temperature": 0.3,
        "output_config": {"format": format_config, "effort": "xhigh"},
        "thinking": {"type": "adaptive", "display": "summarized"},
    }
    assert without_effort_model_params(params) == {
        "temperature": 0.3,
        "output_config": {"format": format_config},
    }


def test_without_effort_clears_legacy_anthropic_top_level_effort() -> None:
    assert without_effort_model_params({"temperature": 0.3, "effort": "xhigh"}) == {
        "temperature": 0.3
    }


def test_without_effort_clears_google_thinking_level() -> None:
    assert without_effort_model_params({"thinking_level": "low"}) is None


def test_without_effort_clears_top_level_openai_reasoning_effort() -> None:
    cleaned = without_effort_model_params(
        {"reasoning_effort": "high", "temperature": 0.1}
    )
    assert cleaned == {"temperature": 0.1}


def test_without_effort_preserves_non_dict_model_kwargs() -> None:
    """A non-dict `model_kwargs` is preserved verbatim while effort keys drop."""
    cleaned = without_effort_model_params(
        {"model_kwargs": "raw", "temperature": 0.1, "effort": "high"}
    )
    assert cleaned == {"model_kwargs": "raw", "temperature": 0.1}


@pytest.mark.parametrize(
    "effort_params",
    [
        {"reasoning_effort": "high"},
        {"reasoning": {"effort": "none"}},
        {"thinking": {"type": "adaptive"}, "output_config": {"effort": "xhigh"}},
        {"thinking_level": "low"},
        {"model_kwargs": {"reasoning_effort": "max"}},
        {"extra_body": {"reasoning_effort": "medium"}},
    ],
)
def test_effort_params_round_trip_clears_to_none(
    effort_params: dict[str, object],
) -> None:
    """The clear-set must strip every shape effort params can arrive in."""
    merged = merge_effort_model_params(None, effort_params)
    assert without_effort_model_params(merged) is None


def test_effort_argument_hint_covers_effort_vocabulary() -> None:
    """The `/effort` argument hint must list every `EffortLabel` plus a reset.

    The label vocabulary is hand-duplicated into the command's `argument_hint`
    (and `COMMANDS.md`), none of which is type-checked against `EffortLabel`.
    This pins the hint so a new label can't silently drift out of the hint text.
    """
    effort_command = next(cmd for cmd in COMMANDS if cmd.name == "/effort")
    hint = effort_command.argument_hint
    assert hint is not None
    tokens = set(hint.strip("[]").split("|"))
    assert set(get_args(EffortLabel)) <= tokens
    # At least one reset token (handled by `_set_effort_override`) is offered.
    assert tokens & {"clear", "--clear", "reset"}


# app.py integration (uses real profile data for openai/anthropic)


async def test_effort_command_sets_current_model_params() -> None:
    app = DeepAgentsApp()
    app._mount_message = AsyncMock()  # ty: ignore
    settings.model_provider = "openai"
    settings.model_name = "gpt-5.5"

    await app._handle_effort_command("/effort high")

    # Support is only validated now; the actual provider-specific shape is
    # built natively inside the model from a plain `reasoning_effort` sentinel.
    assert app._model_params_override == {"reasoning_effort": "high"}
    assert model_config.load_effort_for_model("openai:gpt-5.5") == "high"
    assert app._mount_message.await_count == 2  # ty: ignore[unresolved-attribute]


async def test_restore_effort_override_applies_persisted_model_choice() -> None:
    model_config.save_effort_for_model("openai:gpt-5.6-luna", "max")
    app = DeepAgentsApp()
    app._model_params_override = {"temperature": 0.2}

    await app._restore_effort_override("openai:gpt-5.6-luna")

    assert app._model_params_override == {
        "temperature": 0.2,
        "reasoning_effort": "max",
    }


async def test_restore_effort_override_keeps_explicit_params() -> None:
    model_config.save_effort_for_model("openai:gpt-5.5", "high")
    app = DeepAgentsApp()
    # Explicit per-session params already specify an effort.
    app._model_params_override = {"reasoning_effort": "low"}

    await app._restore_effort_override("openai:gpt-5.5")

    # The explicit low effort wins; the saved high is not merged over it.
    assert app._model_params_override == {"reasoning_effort": "low"}


async def test_startup_model_params_precede_persisted_effort() -> None:
    model_config.save_effort_for_model("openai:gpt-5.5", "high")
    app = DeepAgentsApp(
        model_kwargs={
            "model_spec": "openai:gpt-5.5",
            "extra_kwargs": {"reasoning_effort": "low"},
        }
    )

    # `on_mount` restores effort before deferred model creation consumes the
    # startup kwargs. The explicit CLI value must already be active by then.
    await app._restore_effort_override("openai:gpt-5.5")

    assert app._model_params_override == {"reasoning_effort": "low"}


async def test_restore_effort_override_keeps_explicit_provider_shaped_params() -> None:
    """A raw `--model-params` value in a provider's own shape blocks the merge.

    Regression test for the exact scenario a review flagged: an explicit
    per-session effort supplied through `--model-params` in a provider's own
    shape (here, OpenAI's `reasoning.effort`) must be recognized so a saved
    preference isn't merged on top of -- and silently conflicting with -- it.
    """
    model_config.save_effort_for_model("openai:gpt-5.5", "high")
    app = DeepAgentsApp()
    app._model_params_override = {"reasoning": {"effort": "low"}}

    await app._restore_effort_override("openai:gpt-5.5")

    assert app._model_params_override == {"reasoning": {"effort": "low"}}


async def test_restore_effort_override_prunes_invalid_model_choice() -> None:
    # gpt-5.5 does not support `max`, so the saved label is invalid for it.
    model_config.save_effort_for_model("openai:gpt-5.5", "max")
    app = DeepAgentsApp()
    # No effort in the active params, so the invalid saved label is pruned and
    # unrelated params are preserved.
    app._model_params_override = {"temperature": 0.2}

    await app._restore_effort_override("openai:gpt-5.5")

    assert app._model_params_override == {"temperature": 0.2}
    assert model_config.load_effort_for_model("openai:gpt-5.5") is None


async def test_effort_command_without_args_opens_selector() -> None:
    app = DeepAgentsApp()
    app._mount_message = AsyncMock()  # ty: ignore
    app.push_screen = Mock()  # ty: ignore
    app._model_params_override = {"reasoning_effort": "medium"}
    settings.model_provider = "openai"
    settings.model_name = "gpt-5.5"

    await app._handle_effort_command("/effort")

    app.push_screen.assert_called_once()  # ty: ignore[unresolved-attribute]
    screen = app.push_screen.call_args.args[0]  # ty: ignore[unresolved-attribute]
    assert isinstance(screen, EffortSelectorScreen)
    assert screen._model_spec == "openai:gpt-5.5"
    assert screen._efforts == ("none", "low", "medium", "high", "xhigh")
    assert screen._current_effort == "medium"
    assert screen._default_effort == "medium"
    app._mount_message.assert_not_awaited()  # ty: ignore[unresolved-attribute]


async def test_effort_command_clear_removes_only_effort_params() -> None:
    app = DeepAgentsApp()
    app._mount_message = AsyncMock()  # ty: ignore
    app._model_params_override = {
        "temperature": 0.2,
        "reasoning_effort": "high",
    }
    settings.model_provider = "openai"
    settings.model_name = "gpt-5.5"
    model_config.save_effort_for_model("openai:gpt-5.5", "high")

    await app._handle_effort_command("/effort clear")

    assert app._model_params_override == {"temperature": 0.2}
    assert model_config.load_effort_for_model("openai:gpt-5.5") is None


async def test_effort_command_save_failure_reports_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = DeepAgentsApp()
    app._mount_message = AsyncMock()  # ty: ignore
    settings.model_provider = "openai"
    settings.model_name = "gpt-5.5"
    monkeypatch.setattr(
        model_config, "save_effort_for_model", lambda *_args, **_kwargs: False
    )

    await app._set_effort_override("high")

    # The effort still applies for the session, but the user is told it could
    # not be persisted, and the success message is suppressed by the early
    # return (so the only mounted message is the error).
    assert app._model_params_override == {"reasoning_effort": "high"}
    assert app._mount_message.await_count == 1  # ty: ignore[unresolved-attribute]
    message = app._mount_message.await_args.args[0]  # ty: ignore[unresolved-attribute]
    assert isinstance(message, ErrorMessage)
    assert "could not be saved" in message._content
    assert model_config.load_effort_for_model("openai:gpt-5.5") is None


async def test_effort_command_clear_failure_reports_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = DeepAgentsApp()
    app._mount_message = AsyncMock()  # ty: ignore
    app._model_params_override = {"reasoning_effort": "high"}
    settings.model_provider = "openai"
    settings.model_name = "gpt-5.5"
    monkeypatch.setattr(
        model_config, "clear_effort_for_model", lambda *_args, **_kwargs: False
    )

    await app._set_effort_override("clear")

    # The session override is dropped (no params remain, so it collapses to
    # None), but the user is warned the saved preference could not be removed,
    # and the success message is suppressed by the early return.
    assert app._model_params_override is None
    assert app._mount_message.await_count == 1  # ty: ignore[unresolved-attribute]
    message = app._mount_message.await_args.args[0]  # ty: ignore[unresolved-attribute]
    assert isinstance(message, ErrorMessage)
    assert "could not be removed" in message._content


async def test_effort_command_updates_status_bar_effort() -> None:
    app = DeepAgentsApp()
    app._mount_message = AsyncMock()  # ty: ignore
    app._status_bar = Mock()  # ty: ignore
    settings.model_provider = "openai"
    settings.model_name = "gpt-5.5"

    await app._handle_effort_command("/effort xhigh")

    app._status_bar.set_model.assert_called_once_with(  # ty: ignore[unresolved-attribute]
        provider="openai",
        model="gpt-5.5",
        effort="xhigh",
    )


async def test_effort_command_clear_refreshes_status_bar_to_default() -> None:
    """Clearing an override refreshes the status bar to the reverted effort."""
    app = DeepAgentsApp()
    app._mount_message = AsyncMock()  # ty: ignore
    app._status_bar = Mock()  # ty: ignore
    app._model_params_override = {"reasoning_effort": "high"}
    settings.model_provider = "openai"
    settings.model_name = "gpt-5.5"

    await app._handle_effort_command("/effort clear")

    # gpt-5.5's documented default is `medium`; the bar reverts to it once the
    # `high` override is gone. A dropped `_sync_status_model()` call in the
    # clear branch would leave the stale `high` suffix and fail this.
    app._status_bar.set_model.assert_called_once_with(  # ty: ignore[unresolved-attribute]
        provider="openai",
        model="gpt-5.5",
        effort="medium",
    )


async def test_effort_command_rejects_unsupported_effort() -> None:
    app = DeepAgentsApp()
    app._mount_message = AsyncMock()  # ty: ignore
    settings.model_provider = "anthropic"
    settings.model_name = "claude-opus-4-5"

    # Opus 4.5 supports up to `high`; `xhigh` postdates it (Opus 4.7+ only).
    await app._handle_effort_command("/effort xhigh")

    assert app._model_params_override is None
    assert app._mount_message.await_count == 2  # ty: ignore[unresolved-attribute]


@pytest.mark.parametrize("token", ["clear", "--clear", "reset"])
async def test_effort_command_clear_aliases(token: str) -> None:
    app = DeepAgentsApp()
    app._mount_message = AsyncMock()  # ty: ignore
    app._model_params_override = {
        "temperature": 0.2,
        "reasoning_effort": "high",
    }
    settings.model_provider = "openai"
    settings.model_name = "gpt-5.5"

    await app._handle_effort_command(f"/effort {token}")

    assert app._model_params_override == {"temperature": 0.2}


async def test_effort_selector_reports_no_model_configured() -> None:
    app = DeepAgentsApp()
    app._mount_message = AsyncMock()  # ty: ignore
    app.push_screen = Mock()  # ty: ignore
    settings.model_provider = None
    settings.model_name = None

    await app._handle_effort_command("/effort")

    app.push_screen.assert_not_called()  # ty: ignore[unresolved-attribute]
    assert app._mount_message.await_count == 2  # ty: ignore[unresolved-attribute]


async def test_effort_command_reports_not_configurable_model() -> None:
    app = DeepAgentsApp()
    app._mount_message = AsyncMock()  # ty: ignore
    settings.model_provider = "anthropic"
    settings.model_name = "claude-sonnet-4-5"

    await app._handle_effort_command("/effort high")

    assert app._model_params_override is None
    assert app._mount_message.await_count == 2  # ty: ignore[unresolved-attribute]


async def test_effort_selector_not_configurable_model_skips_screen() -> None:
    """Bare `/effort` on a non-configurable model reports instead of opening.

    The typed-arg path is covered separately; this guards the *selector* arm so
    a regression can't push the modal for a model that supports no efforts.
    """
    app = DeepAgentsApp()
    app._mount_message = AsyncMock()  # ty: ignore
    app.push_screen = Mock()  # ty: ignore
    settings.model_provider = "anthropic"
    settings.model_name = "claude-sonnet-4-5"

    await app._handle_effort_command("/effort")

    app.push_screen.assert_not_called()  # ty: ignore[unresolved-attribute]
    # Echoed UserMessage + the "not configurable" AppMessage.
    assert app._mount_message.await_count == 2  # ty: ignore[unresolved-attribute]


async def test_set_effort_override_guards_non_configurable_model() -> None:
    """`_set_effort_override` re-checks configurability before applying.

    The selector path applies effort in a worker scheduled after the model was
    resolved, so the sink re-resolves the context to guard against the model
    becoming non-configurable in between.
    """
    app = DeepAgentsApp()
    app._mount_message = AsyncMock()  # ty: ignore
    settings.model_provider = "anthropic"
    settings.model_name = "claude-sonnet-4-5"

    await app._set_effort_override("high")

    assert app._model_params_override is None
    # Single AppMessage — the direct sink does not echo a UserMessage.
    app._mount_message.assert_awaited_once()  # ty: ignore[unresolved-attribute]


async def test_effort_selector_result_applies_and_refocuses() -> None:
    """Choosing an effort schedules the apply worker and restores input focus."""
    app = DeepAgentsApp()
    app._mount_message = AsyncMock()  # ty: ignore
    app.push_screen = Mock()  # ty: ignore
    app._set_effort_override = AsyncMock()  # ty: ignore
    app._chat_input = Mock()  # ty: ignore
    scheduled: list[tuple[Coroutine[object, object, None], dict[str, object]]] = []
    app.run_worker = Mock(  # ty: ignore
        side_effect=lambda coro, **kwargs: scheduled.append((coro, kwargs))
    )
    settings.model_provider = "openai"
    settings.model_name = "gpt-5.5"

    await app._handle_effort_command("/effort")
    handle_result = app.push_screen.call_args.args[1]  # ty: ignore[unresolved-attribute]

    handle_result("high")

    assert scheduled[0][1]["group"] == "effort-selection"
    app._chat_input.focus_input.assert_called_once()  # ty: ignore[unresolved-attribute]

    # Running the scheduled worker coroutine applies the chosen effort.
    await scheduled[0][0]
    app._set_effort_override.assert_awaited_once_with(  # ty: ignore[unresolved-attribute]
        "high"
    )


async def test_effort_selector_cancel_refocuses_without_applying() -> None:
    """Dismissing the selector refocuses input and schedules no work."""
    app = DeepAgentsApp()
    app._mount_message = AsyncMock()  # ty: ignore
    app.push_screen = Mock()  # ty: ignore
    app._chat_input = Mock()  # ty: ignore
    app.run_worker = Mock()  # ty: ignore
    settings.model_provider = "openai"
    settings.model_name = "gpt-5.5"

    await app._handle_effort_command("/effort")
    handle_result = app.push_screen.call_args.args[1]  # ty: ignore[unresolved-attribute]

    handle_result(None)

    app.run_worker.assert_not_called()  # ty: ignore[unresolved-attribute]
    app._chat_input.focus_input.assert_called_once()  # ty: ignore[unresolved-attribute]


async def test_effort_selector_apply_failure_reports_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A failure applying the selected effort logs and surfaces an error.

    The worker running `apply_effort` is not covered by the app's worker-state
    error net, so the callback catches, logs, and mounts an `ErrorMessage`
    itself — otherwise the failure would die silently in the background.
    """
    app = DeepAgentsApp()
    app._mount_message = AsyncMock()  # ty: ignore
    app.push_screen = Mock()  # ty: ignore
    app._set_effort_override = AsyncMock(  # ty: ignore
        side_effect=RuntimeError("boom")
    )
    app._chat_input = Mock()  # ty: ignore
    scheduled: list[Coroutine[object, object, None]] = []
    app.run_worker = Mock(  # ty: ignore
        side_effect=lambda coro, **_kwargs: scheduled.append(coro)
    )
    settings.model_provider = "openai"
    settings.model_name = "gpt-5.5"

    await app._handle_effort_command("/effort")
    handle_result = app.push_screen.call_args.args[1]  # ty: ignore[unresolved-attribute]
    handle_result("high")

    with caplog.at_level(logging.ERROR):
        await scheduled[0]

    assert any(
        "Failed to apply reasoning effort" in record.message
        for record in caplog.records
    )
    mounted = app._mount_message.await_args.args[0]  # ty: ignore[unresolved-attribute]
    assert isinstance(mounted, ErrorMessage)


class _EffortSelectorHost(App[None]):
    """Minimal host app for mounting `EffortSelectorScreen` in tests."""


@pytest.mark.parametrize(
    ("current_effort", "default_effort", "expected_index"),
    [("medium", "low", 2), (None, "medium", 2), (None, None, 0), ("bogus", None, 0)],
)
async def test_effort_selector_highlights_current(
    current_effort: str | None, default_effort: str | None, expected_index: int
) -> None:
    app = _EffortSelectorHost()
    async with app.run_test() as pilot:
        await app.push_screen(
            EffortSelectorScreen(
                model_spec="openai:gpt-5.5",
                efforts=("none", "low", "medium", "high", "xhigh"),
                current_effort=current_effort,
                default_effort=default_effort,
            )
        )
        await pilot.pause()
        option_list = app.screen.query_one("#effort-options", OptionList)
        assert option_list.highlighted == expected_index


async def test_effort_selector_enter_selects_highlighted() -> None:
    app = _EffortSelectorHost()
    async with app.run_test() as pilot:
        results: list[str | None] = []
        await app.push_screen(
            EffortSelectorScreen(
                model_spec="openai:gpt-5.5",
                efforts=("low", "medium", "high"),
                current_effort="low",
            ),
            results.append,
        )
        await pilot.pause()
        app.screen.query_one("#effort-options", OptionList).focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert results == ["low"]


async def test_effort_selector_escape_cancels() -> None:
    app = _EffortSelectorHost()
    async with app.run_test() as pilot:
        results: list[str | None] = []
        await app.push_screen(
            EffortSelectorScreen(
                model_spec="openai:gpt-5.5",
                efforts=("low", "high"),
                current_effort=None,
            ),
            results.append,
        )
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert results == [None]


def test_effort_selector_format_label_marks_current_and_default() -> None:
    screen = EffortSelectorScreen(
        model_spec="openai:gpt-5.5",
        efforts=("low", "high"),
        current_effort="high",
        default_effort="low",
    )
    assert "(current)" in str(screen._format_label("high"))
    assert "(default)" in str(screen._format_label("low"))


def test_effort_selector_format_label_combines_current_default() -> None:
    screen = EffortSelectorScreen(
        model_spec="openai:gpt-5.5",
        efforts=("low", "high"),
        current_effort="high",
        default_effort="high",
    )
    assert "(current, default)" in str(screen._format_label("high"))
