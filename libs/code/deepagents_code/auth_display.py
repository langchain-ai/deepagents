"""Shared provider auth status formatting."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, assert_never, overload

from textual.content import Content

from deepagents_code.model_config import (
    ProviderAuthSource,
    ProviderAuthState,
    ProviderAuthStatus,
    resolved_env_var_name,
)

if TYPE_CHECKING:
    from deepagents_code.config import Glyphs

AuthStatusStyle = Literal["auth", "model"]


@overload
def format_auth_status(
    status: ProviderAuthStatus,
    *,
    style: Literal["auth"],
    glyphs: Glyphs | None = None,
) -> Content: ...


@overload
def format_auth_status(
    status: ProviderAuthStatus,
    *,
    style: Literal["model"],
    glyphs: Glyphs,
) -> str: ...


def format_auth_status(
    status: ProviderAuthStatus,
    *,
    style: AuthStatusStyle,
    glyphs: Glyphs | None = None,
) -> Content | str:
    """Format provider auth status for a UI surface.

    Args:
        status: Provider auth/readiness status.
        style: UI surface vocabulary to render.
        glyphs: Glyph table required for the model selector style.

    Returns:
        A status label in the selected surface style.

    Raises:
        ValueError: If model selector style is requested without glyphs.
    """
    if style == "auth":
        return _format_auth_manager_status(status)
    if glyphs is None:
        msg = "glyphs are required for model auth status formatting"
        raise ValueError(msg)
    return _format_model_selector_status(status, glyphs)


def _auth_badge(detail: str, *, prefix: str = "") -> Content:
    """Format a muted auth manager badge.

    Args:
        detail: Badge text inside the brackets.
        prefix: Text to prepend inside the brackets.

    Returns:
        Formatted auth manager badge content.
    """
    return Content.assemble(
        ("[", "$text-muted"),
        (prefix, "$text-muted"),
        Content.styled(detail, "$text-muted"),
        ("]", "$text-muted"),
    )


def _format_auth_manager_status(status: ProviderAuthStatus) -> Content:
    """Format an auth manager badge.

    Args:
        status: Provider auth/readiness status.

    Returns:
        Formatted auth manager badge content.

    Raises:
        ValueError: If a configured status has an unsupported source.
    """
    state = status.state
    match state:
        case ProviderAuthState.CONFIGURED:
            if status.source is ProviderAuthSource.STORED:
                return Content.styled("[stored]", "bold $success")
            if status.source is ProviderAuthSource.ENV:
                if status.env_var:
                    return Content.assemble(
                        ("[env: ", "$text-muted"),
                        Content.styled(
                            resolved_env_var_name(status.env_var), "$text-muted"
                        ),
                        ("]", "$text-muted"),
                    )
                return Content.styled("[env]", "$text-muted")
            msg = f"Unsupported configured auth source: {status.source!r}"
            raise ValueError(msg)
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


def _format_model_selector_status(status: ProviderAuthStatus, glyphs: Glyphs) -> str:
    """Format a model selector provider-header indicator.

    Args:
        status: Provider auth/readiness status.
        glyphs: Glyph table for the active terminal mode.

    Returns:
        Text shown next to the provider name.
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
