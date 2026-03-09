"""Codex-specific error types."""


class CodexAuthError(Exception):
    """Raised when authentication with the Codex backend fails."""


class CodexTokenExpiredError(CodexAuthError):
    """Raised when the access token has expired and refresh failed."""


class CodexAPIError(Exception):
    """Raised when the Codex API returns an error response."""

    def __init__(self, status_code: int, message: str) -> None:
        """Initialize CodexAPIError.

        Args:
            status_code: HTTP status code from the API.
            message: Error message from the API response.
        """
        self.status_code = status_code
        self.message = message
        super().__init__(f"Codex API error {status_code}: {message}")
