"""Stable structured errors exposed by the browser capability."""

from __future__ import annotations

from enum import StrEnum


class BrowserErrorCode(StrEnum):
    """Machine-readable browser failure categories."""

    ACCESS_DISABLED = "access_disabled"
    ASYNC_REQUIRED = "async_required"
    INVALID_URL = "invalid_url"
    BLOCKED_URL = "blocked_url"
    STALE_PAGE_REFERENCE = "stale_page_reference"
    STALE_ELEMENT_REFERENCE = "stale_element_reference"
    ELEMENT_IDENTITY_UNAVAILABLE = "element_identity_unavailable"
    ELEMENT_CHANGED = "element_changed"
    PAGE_NAVIGATED = "page_navigated"
    NAVIGATION_INVALIDATED_REFERENCE = "navigation_invalidated_reference"
    SENSITIVE_CONTROL_BLOCKED = "sensitive_control_blocked"
    TAB_LIMIT_REACHED = "tab_limit_reached"
    CONTEXT_LIMIT_REACHED = "context_limit_reached"
    SCREENSHOT_TOO_LARGE = "screenshot_too_large"
    STARTUP_TIMEOUT = "startup_timeout"
    BROWSER_NOT_PROVISIONED = "browser_not_provisioned"
    STARTUP_FAILED = "startup_failed"
    INVALID_RUNTIME_FACTORY = "invalid_runtime_factory"
    ELEMENT_DISABLED = "element_disabled"
    ELEMENT_NOT_EDITABLE = "element_not_editable"
    ACTION_TARGET_MISMATCH = "action_target_mismatch"
    ACTION_TIMEOUT = "action_timeout"
    SCROLL_FAILED = "scroll_failed"
    ACTION_FAILED = "action_failed"
    CLEANUP_FAILED = "cleanup_failed"
    RUNTIME_CLOSED = "runtime_closed"


class BrowserError(Exception):
    """Base browser failure with stable machine-readable details."""

    def __init__(self, message: str, *, code: BrowserErrorCode) -> None:
        """Initialize an error with human- and machine-readable details."""
        super().__init__(message)
        self.code = code.value

    def as_dict(self) -> dict[str, str]:
        """Return a serialization-safe representation."""
        return {"code": self.code, "message": str(self)}

    def __repr__(self) -> str:
        """Return a structured debug representation."""
        return f"{type(self).__name__}(code={self.code!r}, message={str(self)!r})"


class BrowserAccessError(BrowserError, PermissionError):
    """Raised when browser access has not been explicitly activated."""


class BrowserRuntimeError(BrowserError, RuntimeError):
    """Raised when a browser operation cannot safely be completed."""

    def __init__(
        self,
        message: str,
        *,
        code: BrowserErrorCode = BrowserErrorCode.ACTION_FAILED,
    ) -> None:
        """Initialize a runtime failure."""
        super().__init__(message, code=code)


class NetworkPolicyError(BrowserError, ValueError):
    """Raised when a URL or resolved address violates browser network policy."""

    def __init__(
        self,
        message: str,
        *,
        code: BrowserErrorCode = BrowserErrorCode.INVALID_URL,
    ) -> None:
        """Initialize a network policy failure."""
        super().__init__(message, code=code)
