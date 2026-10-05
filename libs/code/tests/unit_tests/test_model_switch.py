"""Tests for model switching functionality."""

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal
from unittest.mock import AsyncMock, Mock, patch

import pytest
from textual.app import App, ComposeResult

from deepagents_code import model_config
from deepagents_code._cli_context import INHERIT_SUMMARIZATION_MODEL
from deepagents_code._paths import PATHS
from deepagents_code.app import DeepAgentsApp, _extract_model_params_flag
from deepagents_code.client.remote_client import RemoteAgent
from deepagents_code.config import runtime_state
from deepagents_code.model_config import (
    ModelSpec,
    ProviderAuthSource,
    ProviderAuthState,
    ProviderAuthStatus,
)
from deepagents_code.tui.widgets.messages import AppMessage, ErrorMessage
from deepagents_code.tui.widgets.status import StatusBar

_CONFIGURED_AUTH_STATUS = ProviderAuthStatus(
    state=ProviderAuthState.CONFIGURED,
    provider="anthropic",
    env_var="ANTHROPIC_API_KEY",
    source=ProviderAuthSource.ENV,
)
"""Generic non-blocking auth status for tests that bypass the credential check."""


def _make_remote_agent() -> RemoteAgent:
    """Create a RemoteAgent pointing at a dummy URL for test scaffolding."""
    return RemoteAgent("http://test:0")


class _FakeModelResult:
    """Minimal model result for `_switch_model` tests."""

    def __init__(
        self,
        *,
        model_name: str,
        provider: str,
        context_limit: int,
        unsupported_modalities: frozenset[str] = frozenset(),
    ) -> None:
        self.model_name = model_name
        self.provider = provider
        self.context_limit = context_limit
        self.unsupported_modalities = unsupported_modalities

    def apply_to_runtime_state(self) -> None:
        """Mirror `ModelResult.apply_to_runtime_state()` for test isolation."""
        runtime_state.model_name = self.model_name
        runtime_state.model_provider = self.provider
        runtime_state.model_context_limit = self.context_limit
        runtime_state.model_unsupported_modalities = self.unsupported_modalities


@pytest.fixture(autouse=True)
def _restore_runtime_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Save and restore global runtime state mutated by tests."""
    original_name = runtime_state.model_name
    original_provider = runtime_state.model_provider
    original_context_limit = runtime_state.model_context_limit
    original_modalities = runtime_state.model_unsupported_modalities
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", tmp_path / "config.toml")
    yield
    runtime_state.model_name = original_name
    runtime_state.model_provider = original_provider
    runtime_state.model_context_limit = original_context_limit
    runtime_state.model_unsupported_modalities = original_modalities


@pytest.fixture(autouse=True)
def mock_create_model() -> Iterator[Mock]:
    """Avoid provider package imports while preserving metadata updates."""
    context_limits = {
        "anthropic:claude-opus-4-5": 200_000,
        "anthropic:claude-sonnet-4-5": 200_000,
        "fireworks:llama-v3p1-70b": 131_072,
        "ollama:llama3": 8_192,
        "openai:gpt-5.5": 128_000,
    }

    def fake_create_model(
        _config: object,
        model_spec: str,
        *,
        extra_kwargs: dict[str, object] | None = None,
        profile_overrides: dict[str, object] | None = None,
        cli_max_retries: int | None = None,
    ) -> _FakeModelResult:
        del extra_kwargs, profile_overrides, cli_max_retries
        parsed = ModelSpec.try_parse(model_spec)
        if parsed is None:
            provider = "openai"
            model_name = model_spec
        else:
            provider = parsed.provider
            model_name = parsed.model

        context_limit = context_limits.get(f"{provider}:{model_name}", 65_536)
        return _FakeModelResult(
            model_name=model_name,
            provider=provider,
            context_limit=context_limit,
        )

    with patch(
        "deepagents_code.client.remote_client.RemoteAgent.aresolve_model",
        side_effect=fake_create_model,
    ) as mock:
        yield mock


class TestFormatModelParams:
    """Tests for the `_format_model_params` rendering helper."""


class TestModelSwitchWarning:
    """Tests for the large-context confirmation gate."""

    @pytest.mark.parametrize("result", [False, None])
    async def test_cancel_or_dismissal_fails_closed(self, result: bool | None) -> None:
        app = DeepAgentsApp()
        app._context_tokens = 100_001
        app._model_switch_warning_threshold = 100_000
        app._push_screen_wait = AsyncMock(return_value=result)  # ty: ignore
        app._switch_model = AsyncMock()  # ty: ignore
        runtime_state.model_provider = "anthropic"
        runtime_state.model_name = "claude-opus-4-5"

        await app._confirm_and_switch_model("openai:gpt-5.5")

        app._switch_model.assert_not_awaited()


class TestModelSwitchNoOp:
    """Tests for no-op when switching to the same model."""

    async def test_real_switch_resets_unchanged_toast_suppression(self) -> None:
        """A real switch clears suppression so the next no-op toasts again.

        Without the reset, re-selecting A after an A -> B -> A round trip
        would be swallowed by the stale suppression entry left by the first
        no-op.
        """
        app = DeepAgentsApp()
        notify_mock = Mock()
        app._mount_message = AsyncMock()  # ty: ignore
        app.notify = notify_mock  # ty: ignore
        app._agent = _make_remote_agent()

        runtime_state.model_name = "claude-opus-4-5"
        runtime_state.model_provider = "anthropic"

        # Hold the clock still so a second toast is attributable to the reset
        # rather than to the toast lifetime quietly expiring.
        with (
            patch(
                "deepagents_code.model_config.get_provider_auth_status",
                return_value=_CONFIGURED_AUTH_STATUS,
            ),
            patch("deepagents_code.model_config.save_recent_model", return_value=True),
            patch("deepagents_code.app._monotonic", return_value=100.0),
        ):
            # No-op records the suppression entry.
            await app._switch_model("anthropic:claude-opus-4-5")
            # Real switches away and back must clear it.
            await app._switch_model("anthropic:claude-sonnet-4-5")
            await app._switch_model("anthropic:claude-opus-4-5")
            # Identical message, same instant on the clock: only the reset can
            # let this through.
            await app._switch_model("anthropic:claude-opus-4-5")

        unchanged_toasts = [
            call.args[0]
            for call in notify_mock.call_args_list
            if call.args[0].startswith("Already using")
        ]
        assert unchanged_toasts == [
            "Already using anthropic:claude-opus-4-5",
            "Already using anthropic:claude-opus-4-5",
        ]

    async def test_same_model_with_new_params_refreshes_status_effort(self) -> None:
        """Same-model param updates should refresh the status bar effort."""
        app = DeepAgentsApp()
        app._mount_message = AsyncMock()  # ty: ignore
        app._status_bar = Mock()  # ty: ignore
        app._agent = _make_remote_agent()

        runtime_state.model_name = "gpt-5.5"
        runtime_state.model_provider = "openai"

        with patch(
            "deepagents_code.model_config.get_provider_auth_status",
            return_value=ProviderAuthStatus(
                state=ProviderAuthState.CONFIGURED,
                provider="openai",
                env_var="OPENAI_API_KEY",
                source=ProviderAuthSource.ENV,
            ),
        ):
            await app._switch_model(
                "openai:gpt-5.5",
                extra_kwargs={"reasoning_effort": "low"},
            )

        app._status_bar.set_model.assert_called_once_with(  # ty: ignore[unresolved-attribute]
            provider="openai",
            model="gpt-5.5",
            effort="low",
        )

    async def test_same_model_without_params_clears_prior_override(self) -> None:
        """Re-selecting the same model with no params must clear stale params.

        Regression test for the asymmetry where the regular-switch path always
        wrote `_model_params_override = extra_kwargs` (clearing on `None`) but
        the already-active branch left previously-set params in place.
        """
        app = DeepAgentsApp()
        app._mount_message = AsyncMock()  # ty: ignore
        app._agent = _make_remote_agent()

        runtime_state.model_name = "claude-opus-4-5"
        runtime_state.model_provider = "anthropic"

        # Simulate a prior `/model <current> --model-params {...}` call.
        app._model_override = "anthropic:claude-opus-4-5"
        app._model_params_override = {"num_ctx": 16384}

        with patch(
            "deepagents_code.model_config.get_provider_auth_status",
            return_value=_CONFIGURED_AUTH_STATUS,
        ):
            await app._switch_model("anthropic:claude-opus-4-5")

        assert app._model_override == "anthropic:claude-opus-4-5"
        assert app._model_params_override is None

    async def test_switch_restores_persisted_effort_for_model(self) -> None:
        app = DeepAgentsApp()
        app._mount_message = AsyncMock()  # ty: ignore
        app._agent = _make_remote_agent()
        runtime_state.model_name = "gpt-5.5"
        runtime_state.model_provider = "openai"
        model_config.save_effort_for_model(
            "anthropic:claude-opus-4-5",
            "high",
        )

        with patch(
            "deepagents_code.model_config.get_provider_auth_status",
            return_value=_CONFIGURED_AUTH_STATUS,
        ):
            await app._switch_model("anthropic:claude-opus-4-5")

        assert app._model_params_override == {"reasoning_effort": "high"}

    async def test_switch_model_params_effort_overrides_saved(self) -> None:
        app = DeepAgentsApp()
        app._mount_message = AsyncMock()  # ty: ignore
        app._agent = _make_remote_agent()
        runtime_state.model_name = "gpt-5.5"
        runtime_state.model_provider = "openai"
        model_config.save_effort_for_model("openai:gpt-5.5", "high")

        with patch(
            "deepagents_code.model_config.get_provider_auth_status",
            return_value=ProviderAuthStatus(
                state=ProviderAuthState.CONFIGURED,
                provider="openai",
                env_var="OPENAI_API_KEY",
                source=ProviderAuthSource.ENV,
            ),
        ):
            await app._switch_model(
                "openai:gpt-5.5",
                extra_kwargs={"reasoning_effort": "low"},
            )

        # Explicit --model-params effort wins over the saved preference.
        assert app._model_params_override == {"reasoning_effort": "low"}


class TestModelSwitchErrorHandling:
    """Tests for error handling in _switch_model."""

    async def test_missing_credentials_shows_error(self) -> None:
        """_switch_model shows error when provider credentials are missing."""
        app = DeepAgentsApp()
        app._mount_message = AsyncMock()  # ty: ignore
        app._agent = _make_remote_agent()

        # Set a different current model
        runtime_state.model_name = "gpt-5.5"
        runtime_state.model_provider = "openai"

        captured_errors: list[str] = []
        original_init = ErrorMessage.__init__

        def capture_init(self: ErrorMessage, message: str, **kwargs: Any) -> None:
            captured_errors.append(message)
            original_init(self, message, **kwargs)

        with (
            patch.object(
                RemoteAgent,
                "aresolve_model",
                AsyncMock(
                    side_effect=RuntimeError("Missing credentials: ANTHROPIC_API_KEY")
                ),
            ),
            patch(
                "deepagents_code.model_config.get_provider_auth_status",
                return_value=ProviderAuthStatus(
                    state=ProviderAuthState.MISSING,
                    provider="anthropic",
                    env_var="ANTHROPIC_API_KEY",
                ),
            ),
            patch.object(ErrorMessage, "__init__", capture_init),
        ):
            await app._switch_model("anthropic:claude-sonnet-4-5")

        app._mount_message.assert_called_once()  # ty: ignore
        assert len(captured_errors) == 1
        assert "Missing credentials" in captured_errors[0]
        assert "ANTHROPIC_API_KEY" in captured_errors[0]
        assert app._model_switching is False

    async def test_save_recent_model_failure_shows_warning(self) -> None:
        """Permission error saving recent model shows error, no success message."""
        app = DeepAgentsApp()
        app._mount_message = AsyncMock()  # ty: ignore
        app._agent = _make_remote_agent()

        runtime_state.model_name = "gpt-5.5"
        runtime_state.model_provider = "openai"

        captured_errors: list[str] = []
        original_err_init = ErrorMessage.__init__

        def capture_err(self: ErrorMessage, message: str, **kwargs: Any) -> None:
            captured_errors.append(message)
            original_err_init(self, message, **kwargs)

        captured_messages: list[str] = []
        original_app_init = AppMessage.__init__

        def capture_app(self: AppMessage, message: str, **kwargs: Any) -> None:
            captured_messages.append(message)
            original_app_init(self, message, **kwargs)

        with (
            patch(
                "deepagents_code.model_config.get_provider_auth_status",
                return_value=_CONFIGURED_AUTH_STATUS,
            ),
            patch("deepagents_code.model_config.save_recent_model", return_value=False),
            patch.object(ErrorMessage, "__init__", capture_err),
            patch.object(AppMessage, "__init__", capture_app),
        ):
            await app._switch_model("anthropic:claude-sonnet-4-5")

        # Should warn about save failure
        assert len(captured_errors) == 1
        assert "could not save" in captured_errors[0].lower()
        assert PATHS.display(PATHS.profile.root) in captured_errors[0]

        # Should NOT show success message when save fails
        assert not any("Switched to" in m for m in captured_messages)
        assert app._model_override == "anthropic:claude-sonnet-4-5"

    async def test_remote_agent_sets_model_override(self) -> None:
        """With remote agent, sets model override for ConfigurableModelMiddleware."""
        app = DeepAgentsApp()
        app._mount_message = AsyncMock()  # ty: ignore
        app._agent = _make_remote_agent()

        runtime_state.model_name = "gpt-5.5"
        runtime_state.model_provider = "openai"

        captured_messages: list[str] = []
        original_init = AppMessage.__init__

        def capture_init(self: AppMessage, message: str, **kwargs: Any) -> None:
            captured_messages.append(message)
            original_init(self, message, **kwargs)

        with (
            patch(
                "deepagents_code.model_config.get_provider_auth_status",
                return_value=_CONFIGURED_AUTH_STATUS,
            ),
            patch(
                "deepagents_code.model_config.save_recent_model", return_value=True
            ) as mock_save,
            patch.object(AppMessage, "__init__", capture_init),
        ):
            await app._switch_model("anthropic:claude-sonnet-4-5")

        assert app._model_override == "anthropic:claude-sonnet-4-5"
        assert app._model_params_override is None
        mock_save.assert_called_once()
        assert runtime_state.model_name == "claude-sonnet-4-5"
        assert runtime_state.model_provider == "anthropic"
        assert any("Switched to" in m for m in captured_messages)

    async def test_remote_agent_refreshes_model_metadata(
        self, mock_create_model: Mock
    ) -> None:
        """Switching models should refresh derived settings like context size."""
        app = DeepAgentsApp()
        app._mount_message = AsyncMock()  # ty: ignore
        app._agent = _make_remote_agent()
        app._profile_override = {"max_input_tokens": 180_000}

        runtime_state.model_name = "gpt-5.5"
        runtime_state.model_provider = "openai"
        runtime_state.model_context_limit = 128_000

        with (
            patch(
                "deepagents_code.model_config.get_provider_auth_status",
                return_value=_CONFIGURED_AUTH_STATUS,
            ),
            patch("deepagents_code.model_config.save_recent_model", return_value=True),
        ):
            await app._switch_model(
                "anthropic:claude-sonnet-4-5",
                extra_kwargs={"temperature": 0.7},
            )

        assert runtime_state.model_name == "claude-sonnet-4-5"
        assert runtime_state.model_provider == "anthropic"
        assert runtime_state.model_context_limit == 200_000
        mock_create_model.assert_called_once_with(
            {"configurable": {"thread_id": app._lc_thread_id}},
            "anthropic:claude-sonnet-4-5",
            extra_kwargs={"temperature": 0.7},
        )

    async def test_remote_agent_sets_model_params_override(self) -> None:
        """With remote agent, extra_kwargs are stored as _model_params_override."""
        app = DeepAgentsApp()
        app._mount_message = AsyncMock()  # ty: ignore
        app._agent = _make_remote_agent()

        runtime_state.model_name = "gpt-5.5"
        runtime_state.model_provider = "openai"

        with (
            patch(
                "deepagents_code.model_config.get_provider_auth_status",
                return_value=_CONFIGURED_AUTH_STATUS,
            ),
            patch("deepagents_code.model_config.save_recent_model", return_value=True),
        ):
            await app._switch_model(
                "anthropic:claude-sonnet-4-5",
                extra_kwargs={"temperature": 0.7, "max_tokens": 1024},
            )

        assert app._model_override == "anthropic:claude-sonnet-4-5"
        assert app._model_params_override == {
            "temperature": 0.7,
            "max_tokens": 1024,
        }

    async def test_switched_to_message_echoes_params(self) -> None:
        """The 'Switched to' confirmation should echo `--model-params`."""
        app = DeepAgentsApp()
        app._mount_message = AsyncMock()  # ty: ignore
        app._agent = _make_remote_agent()

        runtime_state.model_name = "gpt-5.5"
        runtime_state.model_provider = "openai"

        captured_messages: list[str] = []
        original_init = AppMessage.__init__

        def capture_init(self: AppMessage, message: str, **kwargs: Any) -> None:
            captured_messages.append(message)
            original_init(self, message, **kwargs)

        with (
            patch(
                "deepagents_code.model_config.get_provider_auth_status",
                return_value=_CONFIGURED_AUTH_STATUS,
            ),
            patch("deepagents_code.model_config.save_recent_model", return_value=True),
            patch.object(AppMessage, "__init__", capture_init),
        ):
            await app._switch_model(
                "anthropic:claude-sonnet-4-5",
                extra_kwargs={"temperature": 0.7, "num_ctx": 16384},
            )

        assert any(
            m == "Switched to anthropic:claude-sonnet-4-5 with model params "
            '{"num_ctx": 16384, "temperature": 0.7}'
            for m in captured_messages
        )


class TestModelSwitchConcurrencyGuard:
    """Tests for _model_switching concurrency guard."""

    async def test_concurrent_model_switch_blocked(self) -> None:
        """Second _switch_model call is rejected while first is in-flight."""
        app = DeepAgentsApp()
        app._mount_message = AsyncMock()  # ty: ignore
        app._model_switching = True

        captured_messages: list[str] = []
        original_init = AppMessage.__init__

        def capture_init(self: AppMessage, message: str, **kwargs: Any) -> None:
            captured_messages.append(message)
            original_init(self, message, **kwargs)

        with patch.object(AppMessage, "__init__", capture_init):
            await app._switch_model("anthropic:claude-sonnet-4-5")

        app._mount_message.assert_called_once()  # ty: ignore
        assert len(captured_messages) == 1
        assert "already in progress" in captured_messages[0]

    async def test_model_switching_flag_reset_on_success(self) -> None:
        """_model_switching resets to False after a successful switch."""
        app = DeepAgentsApp()
        app._mount_message = AsyncMock()  # ty: ignore
        app._agent = _make_remote_agent()

        runtime_state.model_name = "gpt-5.5"
        runtime_state.model_provider = "openai"

        with (
            patch(
                "deepagents_code.model_config.get_provider_auth_status",
                return_value=_CONFIGURED_AUTH_STATUS,
            ),
            patch("deepagents_code.model_config.save_recent_model", return_value=True),
        ):
            await app._switch_model("anthropic:claude-sonnet-4-5")

        assert app._model_switching is False


class TestModelSwitchSessionReadiness:
    """Tests for gating model switch on server-backed session readiness."""

    async def test_deferred_switch_completes_after_server_ready(self) -> None:
        """End-to-end: defer during connect, drain after ready, switch completes."""
        app = DeepAgentsApp()
        app._mount_message = AsyncMock()  # ty: ignore
        notify_mock = Mock()
        app.notify = notify_mock  # ty: ignore
        app._agent = None
        app._connecting = True

        runtime_state.model_name = "gpt-5.5"
        runtime_state.model_provider = "openai"

        with (
            patch(
                "deepagents_code.model_config.get_provider_auth_status",
                return_value=_CONFIGURED_AUTH_STATUS,
            ),
            patch("deepagents_code.model_config.save_recent_model", return_value=True),
        ):
            await app._switch_model("anthropic:claude-sonnet-4-5")

            assert len(app._deferred_actions) == 1

            # Simulate `ServerReady`: agent arrives, connecting flips off, drain runs.
            app._agent = _make_remote_agent()
            app._connecting = False
            await app._maybe_drain_deferred()

        assert app._deferred_actions == []
        assert app._model_override == "anthropic:claude-sonnet-4-5"
        assert runtime_state.model_name == "claude-sonnet-4-5"
        assert runtime_state.model_provider == "anthropic"
        assert app._model_switching is False


class TestModelSwitchFailedStartupRecovery:
    """Tests for `/model` recovery after a failed initial server startup."""

    async def test_retry_with_still_missing_credentials_errors(self) -> None:
        """Retrying with creds still missing surfaces the credentials error.

        Avoids looping right back into the same `ModelConfigError` by
        applying the standard tri-state credentials check before
        re-launching the startup worker.
        """
        app = DeepAgentsApp()
        app._mount_message = AsyncMock()  # ty: ignore
        app._agent = None
        app._connecting = False
        app._server_startup_error = "ModelConfigError: ANTHROPIC_API_KEY not set"
        app._server_kwargs = {
            "assistant_id": None,
            "model_name": "anthropic:claude-opus-4-5",
            "model_params": None,
            "interactive": True,
        }
        run_worker_mock = Mock()
        app.run_worker = run_worker_mock  # ty: ignore

        captured_errors: list[str] = []
        original_init = ErrorMessage.__init__

        def capture_init(self: ErrorMessage, message: str, **kwargs: Any) -> None:
            captured_errors.append(message)
            original_init(self, message, **kwargs)

        with (
            patch(
                "deepagents_code.model_config.get_provider_auth_status",
                return_value=ProviderAuthStatus(
                    state=ProviderAuthState.MISSING,
                    provider="anthropic",
                    env_var="ANTHROPIC_API_KEY",
                ),
            ),
            patch.object(ErrorMessage, "__init__", capture_init),
        ):
            await app._switch_model("anthropic:claude-sonnet-4-5")

        # No worker scheduled, failure state preserved so the user can retry.
        run_worker_mock.assert_not_called()
        assert app._server_startup_error == (
            "ModelConfigError: ANTHROPIC_API_KEY not set"
        )
        assert app._connecting is False
        assert any(
            "Missing credentials" in msg and "ANTHROPIC_API_KEY" in msg
            for msg in captured_errors
        )

    async def test_remote_server_mode_keeps_original_error(self) -> None:
        """In remote-server mode (no `_server_kwargs`), recovery is not possible.

        The CLI doesn't own the subprocess so it can't restart it; fall back
        to the existing "server-backed session" error rather than silently
        no-op'ing.
        """
        app = DeepAgentsApp()
        app._mount_message = AsyncMock()  # ty: ignore
        app._agent = None
        app._connecting = False
        app._server_startup_error = "RuntimeError: connection refused"
        app._server_kwargs = None
        run_worker_mock = Mock()
        app.run_worker = run_worker_mock  # ty: ignore

        captured_errors: list[str] = []
        original_init = ErrorMessage.__init__

        def capture_init(self: ErrorMessage, message: str, **kwargs: Any) -> None:
            captured_errors.append(message)
            original_init(self, message, **kwargs)

        with patch.object(ErrorMessage, "__init__", capture_init):
            await app._switch_model("anthropic:claude-sonnet-4-5")

        run_worker_mock.assert_not_called()
        assert any("server-backed session" in msg for msg in captured_errors)


class TestModelSwitchBareModelName:
    """Tests for _switch_model with bare model names (no provider prefix)."""

    async def test_bare_model_name_auto_detects_provider(self) -> None:
        """Bare model name like 'gpt-5.5' auto-detects provider and switches."""
        app = DeepAgentsApp()
        app._mount_message = AsyncMock()  # ty: ignore
        app._agent = _make_remote_agent()

        runtime_state.model_name = "claude-sonnet-4-5"
        runtime_state.model_provider = "anthropic"

        captured_messages: list[str] = []
        original_init = AppMessage.__init__

        def capture_init(self: AppMessage, message: str, **kwargs: Any) -> None:
            captured_messages.append(message)
            original_init(self, message, **kwargs)

        with (
            patch("deepagents_code.config.detect_provider", return_value="openai"),
            patch(
                "deepagents_code.model_config.get_provider_auth_status",
                return_value=_CONFIGURED_AUTH_STATUS,
            ),
            patch(
                "deepagents_code.model_config.save_recent_model", return_value=True
            ) as mock_save,
            patch.object(AppMessage, "__init__", capture_init),
        ):
            await app._switch_model("gpt-5.5")

        mock_save.assert_called_once_with("openai:gpt-5.5")
        assert app._model_override == "openai:gpt-5.5"
        assert runtime_state.model_name == "gpt-5.5"
        assert runtime_state.model_provider == "openai"
        assert any("Switched to openai:gpt-5.5" in m for m in captured_messages)

    async def test_fireworks_qualified_id_gets_provider_prefix(self) -> None:
        """A Fireworks `accounts/...` ID resolves to a `fireworks:` prefix.

        The server's resolved provider appears in the confirmation message,
        status bar, and subsequent inference override.
        """
        app = DeepAgentsApp()
        app._mount_message = AsyncMock()  # ty: ignore
        app._agent = _make_remote_agent()

        runtime_state.model_name = "claude-sonnet-4-5"
        runtime_state.model_provider = "anthropic"

        captured_messages: list[str] = []
        original_init = AppMessage.__init__

        def capture_init(self: AppMessage, message: str, **kwargs: Any) -> None:
            captured_messages.append(message)
            original_init(self, message, **kwargs)

        model_id = "accounts/fireworks/models/kimi-k2p7-code"
        with (
            patch.object(
                RemoteAgent,
                "aresolve_model",
                AsyncMock(
                    return_value=_FakeModelResult(
                        model_name=model_id, provider="fireworks", context_limit=131_072
                    )
                ),
            ),
            patch(
                "deepagents_code.model_config.get_provider_auth_status",
                return_value=_CONFIGURED_AUTH_STATUS,
            ),
            patch(
                "deepagents_code.model_config.save_recent_model", return_value=True
            ) as mock_save,
            patch.object(AppMessage, "__init__", capture_init),
        ):
            await app._switch_model(model_id)

        mock_save.assert_called_once_with(f"fireworks:{model_id}")
        assert app._model_override == f"fireworks:{model_id}"
        assert runtime_state.model_name == model_id
        assert runtime_state.model_provider == "fireworks"
        assert any(f"Switched to fireworks:{model_id}" in m for m in captured_messages)


class TestExtractModelParamsFlag:
    """Tests for _extract_model_params_flag helper."""

    def test_double_quoted_json_with_escaped_quotes(self) -> None:
        """Extracts JSON from double-quoted value with escaped inner quotes."""
        raw = '--model-params "{\\"temperature\\": 0.7}" anthropic:claude-sonnet-4-5'
        remaining, params = _extract_model_params_flag(raw)
        assert remaining == "anthropic:claude-sonnet-4-5"
        assert params == {"temperature": 0.7}

    def test_bare_braces(self) -> None:
        """Extracts JSON from unquoted braces with balanced matching."""
        raw = '--model-params {"temperature": 0.7, "max_tokens": 100}'
        remaining, params = _extract_model_params_flag(raw)
        assert remaining == ""
        assert params == {"temperature": 0.7, "max_tokens": 100}

    def test_bare_braces_with_model_after(self) -> None:
        """Model arg after bare-brace JSON is preserved."""
        raw = '--model-params {"temperature":0.7} anthropic:claude-sonnet-4-5'
        remaining, params = _extract_model_params_flag(raw)
        assert remaining == "anthropic:claude-sonnet-4-5"
        assert params == {"temperature": 0.7}

    def test_unbalanced_braces_raises(self) -> None:
        """Raises ValueError for unbalanced braces."""
        with pytest.raises(ValueError, match="Unbalanced"):
            _extract_model_params_flag('--model-params {"temperature": 0.7')


class TestModelCommandIntegration:
    """Tests for /model command handler integration."""


class TestSummarizationModelCommand:
    @staticmethod
    def _capture_errors() -> tuple[list[str], Any]:
        """Return a captured-error list and the `ErrorMessage.__init__` patch."""
        captured: list[str] = []
        original_init = ErrorMessage.__init__

        def capture_init(self: ErrorMessage, message: str, **kwargs: Any) -> None:
            captured.append(message)
            original_init(self, message, **kwargs)

        return captured, capture_init

    @pytest.mark.parametrize("word", ["clear", "--clear", "reset", "CLEAR"])
    async def test_every_clearing_spelling_effort_accepts_works_here(
        self, word: str
    ) -> None:
        """`/effort` takes all of these, so the habit has to transfer.

        A rejected spelling falls through to model resolution and surfaces a
        confusing "unknown model" error for what is really a valid request.
        """
        app = DeepAgentsApp(summarization_model="openai:gpt-5.4-mini")
        app._mount_message = AsyncMock()  # ty: ignore[invalid-assignment]

        with patch(
            "deepagents_code.app.DeepAgentsApp._resolve_auxiliary_model"
        ) as create_model:
            await app._handle_command(f"/offload model {word}")

        create_model.assert_not_called()
        assert app._summarization_model_override == INHERIT_SUMMARIZATION_MODEL

    @pytest.mark.parametrize(
        "command", ["/offload model", "/offload  model", "/offload\tMODEL"]
    )
    async def test_no_argument_opens_selector_without_changing_override(
        self, command: str
    ) -> None:
        app = DeepAgentsApp(summarization_model="openai:gpt-5.4-mini")
        app._mount_message = AsyncMock()  # ty: ignore[invalid-assignment]

        with patch.object(
            app,
            "_show_summarization_model_selector",
            new_callable=AsyncMock,
        ) as show_selector:
            await app._handle_command(command)

        show_selector.assert_awaited_once()
        assert app._summarization_model_override == "openai:gpt-5.4-mini"

    async def test_selector_preserves_in_flight_provider_setup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A second selection must not cancel an installation already running."""
        app = DeepAgentsApp()
        host = App()
        started = asyncio.Event()
        release = asyncio.Event()
        completed: list[str] = []

        async def apply_selection(spec: str, _extra: str | None) -> None:
            if spec == "custom:first":
                started.set()
                await release.wait()
            completed.append(spec)

        monkeypatch.setattr(app, "run_worker", host.run_worker)
        monkeypatch.setattr(
            app, "_apply_summarization_model_selection", apply_selection
        )
        with (
            patch.object(app, "push_screen") as push,
            patch.object(
                app,
                "call_after_refresh",
                side_effect=lambda callback: callback(),
            ),
        ):
            async with host.run_test() as pilot:
                await app._show_summarization_model_selector()
                handle_result = push.call_args.args[1]
                handle_result(("custom:first", "custom"))
                await asyncio.wait_for(started.wait(), timeout=2)
                handle_result(("custom:second", "custom"))
                await pilot.pause()
                release.set()
                await host.workers.wait_for_complete()

        assert set(completed) == {"custom:first", "custom:second"}

    async def test_external_remote_selector_skips_local_provider_requirements(
        self,
    ) -> None:
        app = DeepAgentsApp(summarization_model="server_provider:remote-model")
        app._agent = _make_remote_agent()  # ty: ignore[invalid-assignment]

        with patch.object(app, "push_screen") as push:
            await app._show_summarization_model_selector()

        screen = push.call_args.args[0]
        assert screen._check_provider_requirements is False

    async def test_selector_falls_back_to_main_model_when_unset(self) -> None:
        app = DeepAgentsApp(summarization_model="")

        with (
            patch.object(
                app,
                "_effective_model_spec",
                return_value="anthropic:claude-sonnet-4-5",
            ),
            patch.object(app, "push_screen") as push,
        ):
            await app._show_summarization_model_selector()

        screen = push.call_args.args[0]
        assert screen._current_provider == "anthropic"
        assert screen._current_model == "claude-sonnet-4-5"

    async def test_cancelled_post_install_auth_keeps_summary_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dismissing auth leaves the installed provider unapplied."""
        app = DeepAgentsApp(summarization_model="openai:gpt-5.4-mini")
        install = AsyncMock(return_value=True)
        authenticate = AsyncMock(return_value=False)
        mount_message = AsyncMock()
        monkeypatch.setattr(app, "_install_extra", install)
        monkeypatch.setattr(app, "_prompt_model_auth_if_needed", authenticate)
        monkeypatch.setattr(app, "_mount_message", mount_message)

        await app._apply_summarization_model_selection(
            "baseten:moonshotai/Kimi-K3", "baseten"
        )

        assert app._summarization_model_override == "openai:gpt-5.4-mini"
        assert mount_message.await_args is not None
        mounted = str(mount_message.await_args.args[0]._content)
        assert "Installed 'baseten'" in mounted
        assert "/auth" in mounted

    async def test_multi_word_argument_is_rejected_without_resolving(self) -> None:
        """The grammar is a single bare spec -- no params, unlike `/model`."""
        app = DeepAgentsApp()
        app._mount_message = AsyncMock()  # ty: ignore[invalid-assignment]
        captured, capture_init = self._capture_errors()

        with (
            patch(
                "deepagents_code.app.DeepAgentsApp._resolve_auxiliary_model"
            ) as create_model,
            patch.object(ErrorMessage, "__init__", capture_init),
        ):
            await app._handle_command("/offload model openai:gpt-5.4-mini extra")

        create_model.assert_not_called()
        assert len(captured) == 1
        assert "Usage:" in captured[0]
        assert app._summarization_model_override is None


class _StatusBarHarness(App[None]):
    """Minimal app that mounts a `StatusBar` so its child widgets exist.

    `_switch_model`'s success path calls `set_model`, which queries the
    `#model-display` child, so the bar must be mounted to be driven end-to-end.
    """

    def compose(self) -> ComposeResult:
        """Yield a single status bar."""
        yield StatusBar(id="status-bar")


@pytest.mark.parametrize(
    ("role", "authenticated"),
    [
        ("main", True),
        ("summarization", True),
        ("auto", True),
        ("goal", True),
        ("rubric", True),
        ("main", False),
        ("auto", False),
    ],
)
async def test_every_picker_defers_install_and_requires_authentication(
    role: Literal["main", "summarization", "auto", "goal", "rubric"],
    authenticated: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepagents_code.model_metadata import ModelMetadata

    app = DeepAgentsApp()
    remote = _make_remote_agent()
    app._agent = remote
    app._server_kwargs = {}
    app._lc_thread_id = "thread"
    app._agent_running = True
    app._model_override = "custom:main"
    app._summarization_model_override = "custom:summary"
    app._auto_classifier_model = "custom:classifier"
    app._rubric_model = "custom:grader"
    app._rubric_model_recorded = True
    install = AsyncMock(return_value=True)
    authenticate = AsyncMock(return_value=authenticated)
    monkeypatch.setattr(app, "_install_extra", install)
    monkeypatch.setattr(app, "_prompt_model_auth_if_needed", authenticate)
    monkeypatch.setattr(app, "_mount_message", AsyncMock())
    monkeypatch.setattr(app, "notify", Mock())
    monkeypatch.setattr(app, "_restore_effort_override", AsyncMock())
    monkeypatch.setattr(app, "_persist_goal_rubric_state", AsyncMock(return_value=True))
    monkeypatch.setattr(
        remote, "aresolve_model", AsyncMock(return_value=ModelMetadata("new", "custom"))
    )
    monkeypatch.setattr(model_config, "save_recent_model", Mock(return_value=True))
    monkeypatch.setattr(model_config, "touch_recent_model", Mock())
    previous = (
        app._model_override,
        app._summarization_model_override,
        app._auto_classifier_model,
        app._rubric_model,
    )

    if role == "main":
        await app._install_extra_then_switch(
            "test-extra", "custom:new", interactive=False
        )
    else:
        await app._apply_auxiliary_model_selection(
            "custom:new", "test-extra", role=role
        )
    install.assert_not_awaited()
    assert len(app._deferred_actions) == 1
    app._agent_running = False
    await app._deferred_actions.pop().execute()

    install.assert_awaited_once_with("test-extra", auto_restart=True)
    authenticate.assert_awaited_once_with("custom:new")
    expected = list(previous)
    if authenticated:
        index = {"main": 0, "summarization": 1, "auto": 2, "goal": 3, "rubric": 3}[role]
        expected[index] = "custom:new"
    assert [
        app._model_override,
        app._summarization_model_override,
        app._auto_classifier_model,
        app._rubric_model,
    ] == expected


@pytest.mark.parametrize(
    "command", ["/offload model", "/auto model", "/goal model", "/rubric model"]
)
async def test_clearing_auxiliary_choices_waits_for_active_turn(
    command: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = DeepAgentsApp()
    remote = _make_remote_agent()
    app._agent = remote
    app._lc_thread_id = "thread"
    app._agent_running = True
    app._model_override = "custom:main"
    app._summarization_model_override = "custom:old"
    app._auto_classifier_model = "custom:old"
    app._rubric_model = "custom:old"
    app._rubric_model_recorded = True
    monkeypatch.setattr(app, "notify", Mock())
    monkeypatch.setattr(app, "_mount_message", AsyncMock())
    monkeypatch.setattr(app, "_persist_goal_rubric_state", AsyncMock(return_value=True))
    resolve = AsyncMock()
    monkeypatch.setattr(remote, "aresolve_model", resolve)

    await app._handle_command(f"{command} clear")
    assert app._summarization_model_override == "custom:old"
    assert app._auto_classifier_model == "custom:old"
    assert app._rubric_model == "custom:old"
    app._agent_running = False
    await app._deferred_actions.pop().execute()
    if command == "/offload model":
        assert app._summarization_model_override == INHERIT_SUMMARIZATION_MODEL
    elif command == "/auto model":
        assert app._auto_classifier_model is None
        assert app._auto_classifier_model_cleared
    else:
        assert app._rubric_model is None
        assert app._rubric_model_recorded
    assert app._model_override == "custom:main"
    resolve.assert_not_awaited()
