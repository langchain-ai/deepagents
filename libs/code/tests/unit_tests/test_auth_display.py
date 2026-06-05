"""Tests for shared auth status rendering."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

from deepagents_code import model_config
from deepagents_code.auth_display import format_auth_status
from deepagents_code.config import get_glyphs
from deepagents_code.model_config import (
    ProviderAuthSource,
    ProviderAuthState,
    ProviderAuthStatus,
    get_provider_auth_status,
)


@pytest.fixture(autouse=True)
def _clear_model_caches() -> Iterator[None]:
    """Clear module-level model config caches around each test."""
    model_config.clear_caches()
    yield
    model_config.clear_caches()


@pytest.fixture
def isolated_model_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point config and credential state at temporary paths."""
    config_path = tmp_path / "config.toml"
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", config_path)
    monkeypatch.setattr(model_config, "DEFAULT_STATE_DIR", tmp_path / ".state")


_AUTH_STATUS_CASES = [
    (
        ProviderAuthStatus(
            state=ProviderAuthState.CONFIGURED,
            provider="anthropic",
            env_var="ANTHROPIC_API_KEY",
            source=ProviderAuthSource.ENV,
        ),
        "[env: ANTHROPIC_API_KEY]",
        "",
    ),
    (
        ProviderAuthStatus(
            state=ProviderAuthState.MISSING,
            provider="anthropic",
            env_var="ANTHROPIC_API_KEY",
        ),
        "[missing]",
        f"{get_glyphs().warning} missing ANTHROPIC_API_KEY",
    ),
    (
        ProviderAuthStatus(
            state=ProviderAuthState.NOT_REQUIRED,
            provider="ollama",
            detail="local provider",
        ),
        "[local provider]",
        "local provider",
    ),
    (
        ProviderAuthStatus(
            state=ProviderAuthState.IMPLICIT,
            provider="google_vertexai",
            env_var="GOOGLE_CLOUD_PROJECT",
            detail="implicit auth",
        ),
        "[implicit auth]",
        "implicit auth",
    ),
    (
        ProviderAuthStatus(
            state=ProviderAuthState.MANAGED,
            provider="custom",
            detail="custom auth",
        ),
        "[custom auth]",
        "custom auth",
    ),
    (
        ProviderAuthStatus(
            state=ProviderAuthState.UNKNOWN,
            provider="unknown",
            detail="credentials unknown",
        ),
        "[? credentials unknown]",
        f"{get_glyphs().question} credentials unknown",
    ),
]


@pytest.mark.parametrize(("status", "auth_label", "model_label"), _AUTH_STATUS_CASES)
def test_format_auth_status_covers_all_states(
    status: ProviderAuthStatus, auth_label: str, model_label: str
) -> None:
    """Both UI styles render every provider auth state from the shared helper."""
    assert str(format_auth_status(status, style="auth")) == auth_label
    assert format_auth_status(status, style="model", glyphs=get_glyphs()) == model_label


def test_auth_style_formats_stored_credentials() -> None:
    """The auth manager keeps its stored-credential badge."""
    status = ProviderAuthStatus(
        state=ProviderAuthState.CONFIGURED,
        provider="openai",
        env_var="OPENAI_API_KEY",
        source=ProviderAuthSource.STORED,
    )

    assert str(format_auth_status(status, style="auth")) == "[stored]"


def test_auth_style_uses_resolved_env_var_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """The auth manager names the env var that wins resolution."""
    monkeypatch.setenv("OPENAI_API_KEY", "canonical")
    monkeypatch.setenv("DEEPAGENTS_CODE_OPENAI_API_KEY", "prefixed")
    status = ProviderAuthStatus(
        state=ProviderAuthState.CONFIGURED,
        provider="openai",
        env_var="OPENAI_API_KEY",
        source=ProviderAuthSource.ENV,
    )

    label = str(format_auth_status(status, style="auth"))

    assert label == "[env: DEEPAGENTS_CODE_OPENAI_API_KEY]"


@pytest.mark.usefixtures("isolated_model_config")
def test_vertex_adc_path_renders_implicit_not_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vertex AI without env credentials uses implicit ADC auth labels."""
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("DEEPAGENTS_CODE_GOOGLE_CLOUD_PROJECT", raising=False)
    status = get_provider_auth_status("google_vertexai")

    assert status.state is ProviderAuthState.IMPLICIT
    assert status.env_var == "GOOGLE_CLOUD_PROJECT"
    assert str(format_auth_status(status, style="auth")) == "[implicit auth]"
    assert (
        format_auth_status(status, style="model", glyphs=get_glyphs())
        == "implicit auth"
    )
