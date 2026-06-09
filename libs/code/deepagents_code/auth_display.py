"""Shared provider auth status formatting."""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_never

from textual.content import Content

from deepagents_code.model_config import (
    ProviderAuthSource,
    ProviderAuthState,
    ProviderAuthStatus,
    resolved_env_var_name,
)

if TYPE_CHECKING:
    from deepagents_code.config import Glyphs


def format_auth_badge(status: ProviderAuthStatus) -> Content:
    """Format an auth manager badge for a provider.

    Used by the `/auth` manager, where each provider renders a bracketed,
    styled badge (e.g. `[stored]`, `[env: ANTHROPIC_API_KEY]`, `[missing]`).

    Args:
        status: Provider auth/readiness status.

    Returns:
        A styled badge `Content` for the auth manager surface.
    """
    state = status.state
    match state:
        case ProviderAuthState.CONFIGURED:
            return _format_configured_badge(status)
        case ProviderAuthState.MISSING:
            return Content.styled("[missing]", "bold $warning")
        case ProviderAuthState.NOT_REQUIRED:
            return _auth_badge(status.detail or "no API key required")
        case ProviderAuthState.IMPLICIT:
            return _auth_badge(status.detail or "implicit auth")
        case ProviderAuthState.MANAGED:
            return _auth_badge(status.detail or "custom auth")
        case ProviderAuthState.UNKNOWN:
            return _auth_badge(status.detail or "credentials unknown", prefix="? ")
        case _:
            assert_never(state)


def format_auth_indicator(status: ProviderAuthStatus, glyphs: Glyphs) -> str:
    """Format a model selector provider-header indicator.

    Used by `/model`, where the indicator is plain text shown next to the
    provider name. Returns an empty string for `CONFIGURED` providers, which
    need no indicator.

    Args:
        status: Provider auth/readiness status.
        glyphs: Glyph table for the active terminal mode.

    Returns:
        Text shown next to the provider name, or an empty string when no
            indicator should be rendered (e.g., `CONFIGURED`).
    """
    state = status.state
    match state:
        case ProviderAuthState.CONFIGURED:
            return ""
        case ProviderAuthState.MISSING:
            if status.env_var:
                return f"{glyphs.warning} missing {status.env_var}"
            return f"{glyphs.warning} missing credentials"
        case ProviderAuthState.NOT_REQUIRED:
            return status.detail or "no API key required"
        case ProviderAuthState.IMPLICIT:
            return status.detail or "implicit auth"
        case ProviderAuthState.MANAGED:
            return status.detail or "custom auth"
        case ProviderAuthState.UNKNOWN:
            detail = status.detail or "credentials unknown"
            return f"{glyphs.question} {detail}"
        case _:
            assert_never(state)


def _auth_badge(detail: str, *, prefix: str = "") -> Content:
    """Format a muted auth manager badge.

    Args:
        detail: Badge text inside the brackets.
        prefix: Text prepended verbatim before `detail` inside the brackets.
            Callers include any separator themselves (e.g. `"? "`); this
            helper does not insert one.

    Returns:
        Formatted auth manager badge content.
    """
    return Content.assemble(
        ("[", "$text-muted"),
        (prefix, "$text-muted"),
        Content.styled(detail, "$text-muted"),
        ("]", "$text-muted"),
    )


def _format_configured_badge(status: ProviderAuthStatus) -> Content:
    """Format the auth manager badge for a `CONFIGURED` provider.

    Args:
        status: A `CONFIGURED` provider auth status.

    Returns:
        A styled badge naming the credential source (`[stored]` or `[env: …]`).

    Raises:
        ValueError: If the status carries no source. `ProviderAuthStatus`
            guarantees `CONFIGURED` implies a source, so this guards against
            that invariant being violated rather than a normal input.
    """
    match status.source:
        case ProviderAuthSource.STORED:
            return Content.styled("[stored]", "bold $success")
        case ProviderAuthSource.ENV:
            if status.env_var:
                return Content.assemble(
                    ("[env: ", "$text-muted"),
                    Content.styled(
                        resolved_env_var_name(status.env_var), "$text-muted"
                    ),
                    ("]", "$text-muted"),
                )
            return Content.styled("[env]", "$text-muted")
        case None:
            msg = f"CONFIGURED auth status has no source: {status!r}"
            raise ValueError(msg)
        case _:
            assert_never(status.source)
