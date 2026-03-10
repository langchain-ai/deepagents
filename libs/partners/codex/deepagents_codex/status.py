"""Rich auth status types for Codex OAuth."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CodexAuthStatus(Enum):
    """Authentication status for Codex OAuth credentials."""

    PACKAGE_MISSING = "package_missing"
    NOT_AUTHENTICATED = "not_authenticated"
    AUTHENTICATED = "authenticated"
    EXPIRED = "expired"
    REFRESH_FAILED = "refresh_failed"
    CORRUPT = "corrupt"


@dataclass(frozen=True)
class CodexAuthInfo:
    """Structured authentication status information."""

    status: CodexAuthStatus
    user_email: str | None = None
    expires_at: float | None = None
    message: str | None = None

    @property
    def is_usable(self) -> bool:
        """Whether credentials can be used for API calls (may need refresh)."""
        return self.status in (CodexAuthStatus.AUTHENTICATED, CodexAuthStatus.EXPIRED)
