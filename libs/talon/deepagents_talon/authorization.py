"""Host-mediated authorization events for Talon integrations.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

import contextvars
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True, slots=True)
class AuthorizationBinding:
    """Identity and lifetime of one authorization attempt."""

    server_name: str
    invocation_id: str
    expires_at: float


@dataclass(frozen=True, slots=True)
class AuthorizationURL:
    """Authorization URL that the host must deliver outside model context."""

    binding: AuthorizationBinding
    url: str = field(repr=False)
    type: Literal["authorization_url"] = "authorization_url"


@dataclass(frozen=True, slots=True)
class CallbackURLRequested:
    """Request for a pasted OAuth callback URL from the bound operator."""

    binding: AuthorizationBinding
    type: Literal["callback_url_requested"] = "callback_url_requested"


@dataclass(frozen=True, slots=True)
class DeviceCode:
    """Device authorization instructions that bypass model context."""

    binding: AuthorizationBinding
    verification_uri: str = field(repr=False)
    user_code: str = field(repr=False)
    type: Literal["device_code"] = "device_code"


@dataclass(frozen=True, slots=True)
class AuthorizationCompleted:
    """Bounded notification that authorization completed."""

    binding: AuthorizationBinding
    type: Literal["completed"] = "completed"


AuthorizationFailureReason = Literal[
    "cancelled",
    "expired",
    "invalid_callback",
    "unavailable",
    "error",
]


@dataclass(frozen=True, slots=True)
class AuthorizationFailed:
    """Bounded notification that authorization failed."""

    binding: AuthorizationBinding
    reason: AuthorizationFailureReason
    type: Literal["failed"] = "failed"


type AuthorizationEvent = (
    AuthorizationURL
    | CallbackURLRequested
    | DeviceCode
    | AuthorizationCompleted
    | AuthorizationFailed
)
type AuthorizationHandler = Callable[[AuthorizationEvent], Awaitable[str | None]]


@dataclass(slots=True)
class AuthorizationAttempt:
    """Task-local marker populated only when an OAuth flow starts."""

    binding: AuthorizationBinding | None = None
    completed: bool = False


_AUTHORIZATION_HANDLER: contextvars.ContextVar[AuthorizationHandler | None] = (
    contextvars.ContextVar("talon_authorization_handler", default=None)
)
_AUTHORIZATION_INVOCATION: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "talon_authorization_invocation",
    default=None,
)
_AUTHORIZATION_ATTEMPT: contextvars.ContextVar[AuthorizationAttempt | None] = (
    contextvars.ContextVar("talon_authorization_attempt", default=None)
)


def current_authorization_handler() -> AuthorizationHandler | None:
    """Return the handler bound to the active Talon request."""
    return _AUTHORIZATION_HANDLER.get()


def current_authorization_invocation() -> str | None:
    """Return the exact active tool-call identifier."""
    return _AUTHORIZATION_INVOCATION.get()


def current_authorization_attempt() -> AuthorizationAttempt | None:
    """Return the task-local authorization attempt marker."""
    return _AUTHORIZATION_ATTEMPT.get()


def set_authorization_handler(
    handler: AuthorizationHandler | None,
) -> contextvars.Token[AuthorizationHandler | None]:
    """Bind a host handler to the active request."""
    return _AUTHORIZATION_HANDLER.set(handler)


def reset_authorization_handler(token: contextvars.Token[AuthorizationHandler | None]) -> None:
    """Restore the previous request handler."""
    _AUTHORIZATION_HANDLER.reset(token)


def set_authorization_invocation(
    invocation_id: str | None,
) -> contextvars.Token[str | None]:
    """Bind the exact active tool-call identifier."""
    return _AUTHORIZATION_INVOCATION.set(invocation_id)


def reset_authorization_invocation(token: contextvars.Token[str | None]) -> None:
    """Restore the previous tool-call identifier."""
    _AUTHORIZATION_INVOCATION.reset(token)


def set_authorization_attempt(
    attempt: AuthorizationAttempt,
) -> contextvars.Token[AuthorizationAttempt | None]:
    """Bind a mutable flow marker to one tool invocation."""
    return _AUTHORIZATION_ATTEMPT.set(attempt)


def reset_authorization_attempt(token: contextvars.Token[AuthorizationAttempt | None]) -> None:
    """Restore the previous flow marker."""
    _AUTHORIZATION_ATTEMPT.reset(token)


__all__ = [
    "AuthorizationAttempt",
    "AuthorizationBinding",
    "AuthorizationCompleted",
    "AuthorizationEvent",
    "AuthorizationFailed",
    "AuthorizationFailureReason",
    "AuthorizationHandler",
    "AuthorizationURL",
    "CallbackURLRequested",
    "DeviceCode",
    "current_authorization_attempt",
    "current_authorization_handler",
    "current_authorization_invocation",
    "reset_authorization_attempt",
    "reset_authorization_handler",
    "reset_authorization_invocation",
    "set_authorization_attempt",
    "set_authorization_handler",
    "set_authorization_invocation",
]
