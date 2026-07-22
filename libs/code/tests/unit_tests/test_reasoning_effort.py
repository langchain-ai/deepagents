"""Tests for `/effort` reasoning effort handling.

Support data comes from LangChain model profiles, so most tests mock
`get_model_profiles()` instead of relying on installed provider packages.
"""

import logging
from collections.abc import Coroutine, Iterator
from contextlib import AbstractContextManager
from pathlib import Path
from unittest.mock import AsyncMock, Mock, call, patch

import pytest
from textual.app import App
from textual.widgets import OptionList

from deepagents_code import model_config, reasoning_effort
from deepagents_code.app import DeepAgentsApp
from deepagents_code.command_registry import COMMANDS
from deepagents_code.config import settings
from deepagents_code.model_config import ModelProfileEntry
from deepagents_code.reasoning_effort import (
    current_effort_from_model_params,
    default_effort_for_model,
    has_explicit_effort_model_params,
    is_effort_supported_for_model,
    supported_efforts_for_model,
    with_effort_model_params,
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


def test_supported_efforts_for_model_reads_ordered_open_ended_levels() -> None:
    with _mock_profiles(
        {
            "acme:foo": _profile_entry(
                reasoning_output=True,
                reasoning_effort_levels=["minimal", "turbo-v2", "max"],
            )
        }
    ):
        assert supported_efforts_for_model("acme:foo") == (
            "minimal",
            "turbo-v2",
            "max",
        )


@pytest.mark.parametrize(
    "profile",
    [
        {},
        {"reasoning_output": False, "reasoning_effort_levels": ["high"]},
        {"reasoning_output": True},
        {"reasoning_output": True, "reasoning_effort_levels": []},
    ],
)
def test_supported_efforts_for_model_fails_closed(profile: dict[str, object]) -> None:
    with _mock_profiles({"acme:foo": _profile_entry(**profile)}):
        assert supported_efforts_for_model("acme:foo") == ()


def test_supported_efforts_for_model_missing_spec_is_empty() -> None:
    with _mock_profiles({}):
        assert supported_efforts_for_model("acme:unknown") == ()
    assert supported_efforts_for_model(None) == ()
    assert supported_efforts_for_model("") == ()


@pytest.mark.parametrize(
    ("profile", "bad_value"),
    [
        (
            {
                "reasoning_output": "enabled",
                "reasoning_effort_levels": ["high"],
            },
            "enabled",
        ),
        (
            {
                "reasoning_output": True,
                "reasoning_effort_levels": ("low", "high"),
            },
            "low",
        ),
        (
            {
                "reasoning_output": True,
                "reasoning_effort_levels": ["low", 7],
            },
            "7",
        ),
    ],
)
def test_supported_efforts_for_model_logs_only_malformed_types(
    profile: dict[str, object],
    bad_value: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with (
        _mock_profiles({"acme:foo": _profile_entry(**profile)}),
        caplog.at_level(logging.WARNING),
    ):
        assert supported_efforts_for_model("acme:foo") == ()
    assert caplog.records
    assert bad_value not in caplog.text


def test_default_effort_is_independent_of_selectable_levels() -> None:
    with _mock_profiles(
        {
            "acme:foo": _profile_entry(
                reasoning_output=True,
                reasoning_effort_levels=["low", "high"],
                reasoning_effort_default="automatic",
            )
        }
    ):
        assert default_effort_for_model("acme:foo") == "automatic"


def test_default_effort_can_exist_without_configurable_levels() -> None:
    with _mock_profiles(
        {
            "acme:foo": _profile_entry(
                reasoning_output=True,
                reasoning_effort_default="provider-default",
            )
        }
    ):
        assert supported_efforts_for_model("acme:foo") == ()
        assert default_effort_for_model("acme:foo") == "provider-default"


@pytest.mark.parametrize(
    "profile",
    [
        {},
        {"reasoning_output": False, "reasoning_effort_default": "high"},
        {"reasoning_output": True},
    ],
)
def test_default_effort_missing_or_disabled_is_none(profile: dict[str, object]) -> None:
    with _mock_profiles({"acme:foo": _profile_entry(**profile)}):
        assert default_effort_for_model("acme:foo") is None


def test_default_effort_malformed_value_logs_type_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with (
        _mock_profiles(
            {
                "acme:foo": _profile_entry(
                    reasoning_output=True,
                    reasoning_effort_default=42,
                )
            }
        ),
        caplog.at_level(logging.WARNING),
    ):
        assert default_effort_for_model("acme:foo") is None
    assert "int" in caplog.text
    assert "42" not in caplog.text


def test_is_effort_supported_for_model() -> None:
    with _mock_profiles(
        {
            "acme:foo": _profile_entry(
                reasoning_output=True,
                reasoning_effort_levels=["low", "high"],
            )
        }
    ):
        assert is_effort_supported_for_model("acme:foo", "high")
        assert not is_effort_supported_for_model("acme:foo", "medium")
        assert not is_effort_supported_for_model("acme:unknown", "high")


@pytest.mark.parametrize(
    "model_spec",
    [
        "openai:gpt-test",
        "openai_codex:gpt-test",
        "anthropic:claude-test",
        "google_genai:gemini-test",
        "fireworks:accounts/fireworks/models/test",
        "xai:grok-test",
        "custom:model",
    ],
)
def test_profile_support_is_not_limited_by_provider(model_spec: str) -> None:
    with _mock_profiles(
        {
            model_spec: _profile_entry(
                reasoning_output=True,
                reasoning_effort_levels=["provider-specific"],
                reasoning_effort_default="provider-default",
            )
        }
    ):
        assert supported_efforts_for_model(model_spec) == ("provider-specific",)
        assert default_effort_for_model(model_spec) == "provider-default"


def test_xai_released_profiles_replace_old_grok_45_matrix() -> None:
    with _mock_profiles(
        {
            "xai:grok-4.3": _profile_entry(
                reasoning_output=True,
                reasoning_effort_levels=["none", "low", "medium", "high"],
                reasoning_effort_default="low",
            ),
            "xai:grok-4.5": _profile_entry(reasoning_output=True),
        }
    ):
        assert supported_efforts_for_model("xai:grok-4.3") == (
            "none",
            "low",
            "medium",
            "high",
        )
        assert default_effort_for_model("xai:grok-4.3") == "low"
        assert supported_efforts_for_model("xai:grok-4.5") == ()


def test_profile_helpers_forward_cli_override() -> None:
    override = {"reasoning_effort_levels": ["custom"]}
    with _mock_profiles({}) as mock_profiles:
        supported_efforts_for_model("acme:foo", cli_override=override)
        default_effort_for_model("acme:foo", cli_override=override)
    assert mock_profiles.call_args_list == [
        call(cli_override=override),
        call(cli_override=override),
    ]


def test_cli_profile_override_supports_unregistered_model() -> None:
    override = {
        "reasoning_output": True,
        "reasoning_effort_levels": ["custom"],
        "reasoning_effort_default": "provider-default",
    }
    with _mock_profiles({}):
        assert supported_efforts_for_model(
            "custom:unregistered", cli_override=override
        ) == ("custom",)
        assert (
            default_effort_for_model("custom:unregistered", cli_override=override)
            == "provider-default"
        )


def test_config_and_cli_profile_override_precedence(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("""
[models.providers.acme]
models = ["foo"]
[models.providers.acme.profile]
reasoning_output = true
reasoning_effort_levels = ["config-low", "config-high"]
reasoning_effort_default = "config-default"
""")
    upstream = {
        "foo": {
            "reasoning_output": True,
            "reasoning_effort_levels": ["upstream"],
            "reasoning_effort_default": "upstream-default",
        }
    }
    with (
        patch.object(
            model_config,
            "_get_provider_profile_modules",
            return_value=[("acme", "acme.data._profiles")],
        ),
        patch.object(model_config, "_load_provider_profiles", return_value=upstream),
        patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
    ):
        model_config.clear_caches()
        assert supported_efforts_for_model("acme:foo") == (
            "config-low",
            "config-high",
        )
        assert default_effort_for_model("acme:foo") == "config-default"

        cli_override = {
            "reasoning_effort_levels": ["cli"],
            "reasoning_effort_default": "cli-default",
        }
        assert supported_efforts_for_model("acme:foo", cli_override=cli_override) == (
            "cli",
        )
        assert (
            default_effort_for_model("acme:foo", cli_override=cli_override)
            == "cli-default"
        )
        assert (
            supported_efforts_for_model(
                "acme:foo", cli_override={"reasoning_effort_levels": []}
            )
            == ()
        )


# Contract checks against required minimum integrations.


def test_gemini_36_profile_contract() -> None:
    assert supported_efforts_for_model("google_genai:gemini-3.6-flash") == (
        "minimal",
        "low",
        "medium",
        "high",
    )
    assert default_effort_for_model("google_genai:gemini-3.6-flash") == "medium"


def test_openai_and_codex_use_mirrored_profile_contract() -> None:
    expected = ("none", "low", "medium", "high", "xhigh")
    assert supported_efforts_for_model("openai:gpt-5.5") == expected
    assert supported_efforts_for_model("openai_codex:gpt-5.5") == expected
    assert default_effort_for_model("openai:gpt-5.5") == "medium"
    assert default_effort_for_model("openai_codex:gpt-5.5") == "medium"


def test_anthropic_profile_contract() -> None:
    assert supported_efforts_for_model("anthropic:claude-opus-4-5") == (
        "low",
        "medium",
        "high",
    )
    assert default_effort_for_model("anthropic:claude-opus-4-5") == "high"


def test_openai_integration_translates_standard_effort_without_summary() -> None:
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI

    with (
        patch("langchain_openai.chat_models.base.openai.OpenAI"),
        patch("langchain_openai.chat_models.base.openai.AsyncOpenAI"),
    ):
        model = ChatOpenAI(
            model="gpt-5.5",
            api_key="test",
            reasoning_effort="high",
            use_responses_api=True,
        )
    payload = model._get_request_payload([HumanMessage("hello")])

    assert payload["reasoning"] == {"effort": "high"}
    assert "reasoning_effort" not in payload


def test_anthropic_integration_translates_standard_effort() -> None:
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage

    model = ChatAnthropic(
        model="claude-opus-4-5",
        api_key="test",
        reasoning_effort="high",
        output_config={"format": {"type": "json_schema", "schema": {}}},
    )
    payload = model._get_request_payload([HumanMessage("hello")])

    assert payload["output_config"] == {
        "format": {"type": "json_schema", "schema": {}},
        "effort": "high",
    }
    assert "reasoning_effort" not in payload


def test_google_integration_translates_standard_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from langchain_google_genai import ChatGoogleGenerativeAI

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    model = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key="test",
        reasoning_effort="minimal",
        thinking_config={"include_thoughts": True},
    )
    config = model._build_thinking_config()

    assert config is not None
    assert config.thinking_level is not None
    assert config.thinking_level.value == "MINIMAL"
    assert config.include_thoughts is True


# Compatibility reader for canonical and legacy/native model params.


@pytest.mark.parametrize(
    ("model_spec", "model_params", "expected"),
    [
        ("openai:gpt-5.5", {"reasoning": {"effort": "low"}}, "low"),
        (
            "openai_codex:gpt-5.5",
            {"reasoning": {"effort": "high"}},
            "high",
        ),
        ("anthropic:claude-opus-4-5", {"effort": "max"}, "max"),
        (
            "anthropic:claude-opus-4-5",
            {"output_config": {"effort": "low"}},
            "low",
        ),
        ("google_genai:gemini-3.6-flash", {"thinking_level": "minimal"}, "minimal"),
        (
            "google_genai:gemini-3.6-flash",
            {"thinking_config": {"thinking_level": "medium"}},
            "medium",
        ),
        (
            "fireworks:accounts/fireworks/models/deepseek-v4-pro",
            {"model_kwargs": {"reasoning_effort": "xhigh"}},
            "xhigh",
        ),
        ("xai:grok-4.3", {"extra_body": {"reasoning_effort": "low"}}, "low"),
        ("custom:model", {"reasoning_effort": "custom"}, "custom"),
    ],
)
def test_current_effort_recognizes_canonical_and_native_settings(
    model_spec: str,
    model_params: dict[str, object],
    expected: str,
) -> None:
    assert current_effort_from_model_params(model_spec, model_params) == expected


@pytest.mark.parametrize(
    ("model_spec", "model_params", "expected"),
    [
        (
            "openai:gpt-5.5",
            {"reasoning_effort": "high", "reasoning": {"effort": "low"}},
            "low",
        ),
        (
            "anthropic:claude-opus-4-5",
            {
                "effort": "max",
                "reasoning_effort": "high",
                "output_config": {"effort": "low"},
            },
            "max",
        ),
        (
            "google_genai:gemini-3.6-flash",
            {
                "thinking_level": "minimal",
                "reasoning_effort": "high",
                "thinking_config": {"thinking_level": "medium"},
            },
            "minimal",
        ),
        (
            "anthropic:claude-opus-4-5",
            {
                "effort": None,
                "reasoning_effort": "high",
                "output_config": {"effort": "low"},
            },
            "low",
        ),
        (
            "google_genai:gemini-3.6-flash",
            {
                "thinking_level": None,
                "reasoning_effort": "high",
                "thinking_config": {"thinking_level": "low"},
            },
            "low",
        ),
        (
            "xai:grok-4.3",
            {
                "reasoning_effort": "high",
                "extra_body": {"reasoning_effort": "low"},
            },
            "high",
        ),
        (
            "xai:grok-4.3",
            {
                "reasoning_effort": None,
                "extra_body": {"reasoning_effort": "low"},
            },
            "low",
        ),
    ],
)
def test_current_effort_matches_integration_precedence(
    model_spec: str,
    model_params: dict[str, object],
    expected: str,
) -> None:
    assert current_effort_from_model_params(model_spec, model_params) == expected


def test_openai_native_null_suppresses_flat_effort() -> None:
    model_spec = "openai:gpt-5.5"
    model_params = {
        "reasoning": {"effort": None},
        "reasoning_effort": "high",
    }

    assert current_effort_from_model_params(model_spec, model_params) is None
    assert has_explicit_effort_model_params(model_spec, model_params)


def test_fireworks_duplicate_forms_fail_closed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model_spec = "fireworks:accounts/fireworks/models/deepseek-v4-pro"
    model_params = {
        "reasoning_effort": "high",
        "model_kwargs": {"reasoning_effort": "low"},
    }

    with caplog.at_level(logging.WARNING):
        assert current_effort_from_model_params(model_spec, model_params) is None
    assert has_explicit_effort_model_params(model_spec, model_params)
    assert "conflicting Fireworks" in caplog.text


@pytest.mark.parametrize(
    ("model_spec", "model_params"),
    [
        (
            "openai:gpt-5.5",
            {"reasoning": {"effort": 5}, "reasoning_effort": "high"},
        ),
        ("anthropic:claude-opus-4-5", {"effort": 5}),
        (
            "google_genai:gemini-3.6-flash",
            {"thinking_config": {"thinking_level": 5}},
        ),
        (
            "fireworks:accounts/fireworks/models/deepseek-v4-pro",
            {"model_kwargs": {"reasoning_effort": 5}},
        ),
        ("xai:grok-4.3", {"extra_body": {"reasoning_effort": 5}}),
        ("custom:model", {"reasoning_effort": 5}),
    ],
)
def test_current_effort_warns_on_malformed_values(
    model_spec: str,
    model_params: dict[str, object],
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        assert current_effort_from_model_params(model_spec, model_params) is None
    assert "int" in caplog.text
    assert "5" not in caplog.text


def test_current_effort_non_dict_container_is_silent() -> None:
    assert (
        current_effort_from_model_params("openai:gpt-5.5", {"reasoning": "raw"}) is None
    )


def test_current_effort_requires_spec_and_params() -> None:
    assert current_effort_from_model_params(None, {"reasoning_effort": "high"}) is None
    assert current_effort_from_model_params("anthropic:claude-opus-4-5", None) is None
    assert current_effort_from_model_params("anthropic:claude-opus-4-5", {}) is None


@pytest.mark.parametrize(
    ("model_spec", "existing", "cleaned"),
    [
        (
            "openai:gpt-5.5",
            {
                "temperature": 0.1,
                "reasoning_effort": "high",
                "reasoning": {"effort": "low", "summary": "auto"},
            },
            {"temperature": 0.1, "reasoning": {"summary": "auto"}},
        ),
        (
            "openai_codex:gpt-5.5",
            {"reasoning": {"effort": "high", "summary": "concise"}},
            {"reasoning": {"summary": "concise"}},
        ),
        (
            "anthropic:claude-opus-4-5",
            {
                "effort": "high",
                "output_config": {"effort": "low", "format": "json"},
                "thinking": {"type": "adaptive", "display": "summarized"},
            },
            {"output_config": {"format": "json"}},
        ),
        (
            "google_genai:gemini-3.6-flash",
            {
                "thinking_level": "high",
                "thinking_config": {
                    "thinking_level": "low",
                    "include_thoughts": True,
                },
            },
            {"thinking_config": {"include_thoughts": True}},
        ),
        (
            "fireworks:accounts/fireworks/models/deepseek-v4-pro",
            {"model_kwargs": {"reasoning_effort": "max", "top_p": 0.9}},
            {"model_kwargs": {"top_p": 0.9}},
        ),
        (
            "xai:grok-4.3",
            {
                "extra_body": {
                    "reasoning_effort": "high",
                    "prompt_cache_key": "thread-1",
                }
            },
            {"extra_body": {"prompt_cache_key": "thread-1"}},
        ),
        (
            "custom:model",
            {"reasoning_effort": "custom", "temperature": 0.2},
            {"temperature": 0.2},
        ),
    ],
)
def test_set_and_clear_preserve_unrelated_native_settings(
    model_spec: str,
    existing: dict[str, object],
    cleaned: dict[str, object],
) -> None:
    original = dict(existing)

    assert without_effort_model_params(model_spec, existing) == cleaned
    assert with_effort_model_params(model_spec, existing, "replacement") == {
        **cleaned,
        "reasoning_effort": "replacement",
    }
    assert existing == original


def test_anthropic_clear_preserves_arbitrary_thinking_config() -> None:
    params = {
        "reasoning_effort": "high",
        "thinking": {"type": "enabled", "budget_tokens": 4096},
    }

    assert without_effort_model_params("anthropic:claude-opus-4-5", params) == {
        "thinking": {"type": "enabled", "budget_tokens": 4096}
    }


def test_clear_preserves_non_dict_nested_values() -> None:
    assert without_effort_model_params(
        "fireworks:accounts/fireworks/models/deepseek-v4-pro",
        {"reasoning_effort": "high", "model_kwargs": "raw"},
    ) == {"model_kwargs": "raw"}


def test_effort_argument_hint_is_profile_agnostic() -> None:
    effort_command = next(cmd for cmd in COMMANDS if cmd.name == "/effort")
    assert effort_command.argument_hint == "[<level>|clear]"


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


async def test_effort_command_replaces_native_effort_and_preserves_summary() -> None:
    app = DeepAgentsApp()
    app._mount_message = AsyncMock()  # ty: ignore
    app._model_params_override = {"reasoning": {"effort": "low", "summary": "auto"}}
    settings.model_provider = "openai"
    settings.model_name = "gpt-5.5"

    await app._handle_effort_command("/effort high")

    assert app._model_params_override == {
        "reasoning": {"summary": "auto"},
        "reasoning_effort": "high",
    }


async def test_profile_override_controls_selector_and_validation() -> None:
    override = {
        "reasoning_output": True,
        "reasoning_effort_levels": ["Ultra"],
        "reasoning_effort_default": "provider-default",
    }
    app = DeepAgentsApp(profile_override=override)
    app._mount_message = AsyncMock()  # ty: ignore
    app.push_screen = Mock()  # ty: ignore
    settings.model_provider = "openai"
    settings.model_name = "gpt-5.5"

    await app._handle_effort_command("/effort")

    screen = app.push_screen.call_args.args[0]  # ty: ignore[unresolved-attribute]
    assert screen._efforts == ("Ultra",)
    assert screen._default_effort == "provider-default"

    await app._handle_effort_command("/effort Ultra")
    assert app._model_params_override == {"reasoning_effort": "Ultra"}


async def test_profile_override_controls_persisted_restoration() -> None:
    model_config.save_effort_for_model("openai:gpt-5.5", "custom")
    app = DeepAgentsApp(
        profile_override={
            "reasoning_output": True,
            "reasoning_effort_levels": ["custom"],
        }
    )

    await app._restore_effort_override("openai:gpt-5.5")

    assert app._model_params_override == {"reasoning_effort": "custom"}


def test_profile_override_controls_status_default() -> None:
    app = DeepAgentsApp(
        profile_override={
            "reasoning_output": True,
            "reasoning_effort_levels": ["custom"],
            "reasoning_effort_default": "outside-levels",
        }
    )
    app._status_bar = Mock()  # ty: ignore
    settings.model_provider = "openai"
    settings.model_name = "gpt-5.5"

    app._sync_status_model()

    app._status_bar.set_model.assert_called_once_with(  # ty: ignore[unresolved-attribute]
        provider="openai",
        model="gpt-5.5",
        effort="outside-levels",
    )


async def test_empty_profile_override_levels_disable_effort() -> None:
    app = DeepAgentsApp(
        profile_override={
            "reasoning_output": True,
            "reasoning_effort_levels": [],
        }
    )
    app._mount_message = AsyncMock()  # ty: ignore
    settings.model_provider = "openai"
    settings.model_name = "gpt-5.5"

    await app._handle_effort_command("/effort high")

    assert app._model_params_override is None
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


@pytest.mark.parametrize(
    ("model_spec", "model_params"),
    [
        ("openai:gpt-5.5", {"reasoning": {"effort": "low"}}),
        ("openai_codex:gpt-5.5", {"reasoning": {"effort": "low"}}),
        ("anthropic:claude-opus-4-5", {"effort": "low"}),
        (
            "anthropic:claude-opus-4-5",
            {"output_config": {"effort": "low"}},
        ),
        ("google_genai:gemini-3.6-flash", {"thinking_level": "low"}),
        (
            "google_genai:gemini-3.6-flash",
            {"thinking_config": {"thinking_level": "low"}},
        ),
        (
            "fireworks:accounts/fireworks/models/deepseek-v4-pro",
            {"model_kwargs": {"reasoning_effort": "low"}},
        ),
        (
            "fireworks:accounts/fireworks/models/deepseek-v4-pro",
            {
                "reasoning_effort": "high",
                "model_kwargs": {"reasoning_effort": "low"},
            },
        ),
        ("xai:grok-4.3", {"extra_body": {"reasoning_effort": "low"}}),
        ("custom:model", {"reasoning_effort": "low"}),
    ],
)
async def test_restore_keeps_explicit_canonical_and_native_params(
    model_spec: str,
    model_params: dict[str, object],
) -> None:
    model_config.save_effort_for_model(model_spec, "high")
    app = DeepAgentsApp()
    app._model_params_override = model_params

    await app._restore_effort_override(model_spec)

    assert app._model_params_override == model_params


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


async def test_gemini_36_selector_offers_minimal_with_medium_default() -> None:
    app = DeepAgentsApp()
    app._mount_message = AsyncMock()  # ty: ignore
    app.push_screen = Mock()  # ty: ignore
    settings.model_provider = "google_genai"
    settings.model_name = "gemini-3.6-flash"

    await app._handle_effort_command("/effort")

    screen = app.push_screen.call_args.args[0]  # ty: ignore[unresolved-attribute]
    assert screen._efforts == ("minimal", "low", "medium", "high")
    assert screen._default_effort == "medium"


async def test_effort_command_clear_removes_only_effort_params() -> None:
    app = DeepAgentsApp()
    app._mount_message = AsyncMock()  # ty: ignore
    app._model_params_override = {
        "temperature": 0.2,
        "reasoning_effort": "high",
        "reasoning": {"effort": "low", "summary": "auto"},
    }
    settings.model_provider = "openai"
    settings.model_name = "gpt-5.5"
    model_config.save_effort_for_model("openai:gpt-5.5", "high")

    await app._handle_effort_command("/effort clear")

    assert app._model_params_override == {
        "temperature": 0.2,
        "reasoning": {"summary": "auto"},
    }
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


async def test_effort_clear_works_when_profile_is_not_configurable() -> None:
    app = DeepAgentsApp()
    app._mount_message = AsyncMock()  # ty: ignore
    app._model_params_override = {"output_config": {"effort": "low", "format": "json"}}
    settings.model_provider = "anthropic"
    settings.model_name = "claude-sonnet-4-5"

    await app._handle_effort_command("/effort clear")

    assert app._model_params_override == {"output_config": {"format": "json"}}


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
