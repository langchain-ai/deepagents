"""Tests for shared auth status rendering."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

from deepagents_code import model_config
from deepagents_code.auth_display import format_auth_badge, format_auth_indicator
from deepagents_code.config import get_glyphs
from deepagents_code.model_config import (
    CODEX_PROVIDER,
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
        "[env set: ANTHROPIC_API_KEY]",
        "",
    ),
    (
        ProviderAuthStatus(
            state=ProviderAuthState.MISSING,
            provider="anthropic",
            env_var="ANTHROPIC_API_KEY",
        ),
        "[missing]",
        f"{get_glyphs().warning} missing credentials",
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
def test_format_auth_covers_all_states(
    status: ProviderAuthStatus,
    auth_label: str,
    model_label: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both UI surfaces render every provider auth state."""
    if status.env_var:
        monkeypatch.delenv(f"DEEPAGENTS_CODE_{status.env_var}", raising=False)
    assert format_auth_badge(status).plain == auth_label
    assert format_auth_indicator(status, get_glyphs()) == model_label


def test_auth_badge_formats_stored_credentials() -> None:
    """The auth manager keeps its stored-credential badge."""
    status = ProviderAuthStatus(
        state=ProviderAuthState.CONFIGURED,
        provider="openai",
        env_var="OPENAI_API_KEY",
        source=ProviderAuthSource.STORED,
    )

    assert format_auth_badge(status).plain == "[stored]"


def test_auth_badge_env_source_without_var() -> None:
    """An ENV-source status with no env var falls back to a bare `[env]` badge."""
    status = ProviderAuthStatus(
        state=ProviderAuthState.CONFIGURED,
        provider="openai",
        env_var=None,
        source=ProviderAuthSource.ENV,
    )

    assert format_auth_badge(status).plain == "[env]"


def test_auth_badge_uses_resolved_env_var_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """The auth manager names the env var that wins resolution."""
    monkeypatch.setenv("OPENAI_API_KEY", "canonical")
    monkeypatch.setenv("DEEPAGENTS_CODE_OPENAI_API_KEY", "prefixed")
    status = ProviderAuthStatus(
        state=ProviderAuthState.CONFIGURED,
        provider="openai",
        env_var="OPENAI_API_KEY",
        source=ProviderAuthSource.ENV,
    )

    assert (
        format_auth_badge(status).plain == "[env set: DEEPAGENTS_CODE_OPENAI_API_KEY]"
    )


def _badge_styles(status: ProviderAuthStatus) -> str:
    """Return the concatenated span styles of a provider's auth badge."""
    return " ".join(str(span.style or "") for span in format_auth_badge(status).spans)


def test_missing_badge_carries_warning_style() -> None:
    """A missing-credential badge is styled as a warning, not muted text."""
    status = ProviderAuthStatus(
        state=ProviderAuthState.MISSING,
        provider="anthropic",
        env_var="ANTHROPIC_API_KEY",
    )

    assert "$warning" in _badge_styles(status)


def test_stored_badge_carries_success_style() -> None:
    """A stored-credential badge is styled as a success, not muted text."""
    status = ProviderAuthStatus(
        state=ProviderAuthState.CONFIGURED,
        provider="openai",
        env_var="OPENAI_API_KEY",
        source=ProviderAuthSource.STORED,
    )

    assert "$success" in _badge_styles(status)


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
    assert format_auth_badge(status).plain == "[implicit auth]"
    assert format_auth_indicator(status, get_glyphs()) == "implicit auth"


def test_codex_badge_configured_with_plan() -> None:
    """A signed-in codex status renders the plan parsed from the detail."""
    status = ProviderAuthStatus(
        state=ProviderAuthState.CONFIGURED,
        provider=CODEX_PROVIDER,
        source=ProviderAuthSource.STORED,
        detail="signed in to ChatGPT (pro)",
    )
    assert format_auth_badge(status).plain == "[chatgpt: pro]"


def test_codex_badge_configured_with_expired_token_refresh_detail() -> None:
    """The codex badge plan parser ignores refresh details after the plan."""
    status = ProviderAuthStatus(
        state=ProviderAuthState.CONFIGURED,
        provider=CODEX_PROVIDER,
        source=ProviderAuthSource.STORED,
        detail="signed in to ChatGPT (pro); access token will refresh on use",
    )
    assert format_auth_badge(status).plain == "[chatgpt: pro]"


def test_codex_badge_configured_without_plan() -> None:
    """A signed-in codex status with no plan in the detail renders bare."""
    status = ProviderAuthStatus(
        state=ProviderAuthState.CONFIGURED,
        provider=CODEX_PROVIDER,
        source=ProviderAuthSource.STORED,
        detail="signed in to ChatGPT",
    )
    assert format_auth_badge(status).plain == "[chatgpt]"


def test_codex_badge_missing_prompts_sign_in() -> None:
    """A missing codex credential renders the sign-in prompt, not `[missing]`."""
    status = ProviderAuthStatus(
        state=ProviderAuthState.MISSING,
        provider=CODEX_PROVIDER,
        detail="not signed in to ChatGPT",
    )
    assert format_auth_badge(status).plain == "[sign in to chatgpt]"
