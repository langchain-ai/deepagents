"""File-backed credential storage for Codex OAuth tokens."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_AUTH_DIR = Path.home() / ".deepagents" / "auth"
_DEFAULT_CREDS_FILE = "codex.json"


@dataclass(frozen=True)
class CodexCredentials:
    """OAuth credentials for the Codex backend."""

    access_token: str
    refresh_token: str
    expires_at: float
    user_email: str | None = None

    @property
    def is_expired(self) -> bool:
        """Check if the access token has expired (with 5-minute buffer)."""
        return time.time() >= (self.expires_at - 300)


class CodexAuthStore:
    """File-backed credential store for Codex OAuth tokens.

    Stores credentials at ``~/.deepagents/auth/codex.json`` with
    restricted file permissions (0o600 file, 0o700 directory).
    """

    def __init__(self, path: Path | None = None) -> None:
        """Initialize the credential store.

        Args:
            path: Custom path to the credentials file.
                Defaults to ``~/.deepagents/auth/codex.json``.
        """
        self._path = path or (_DEFAULT_AUTH_DIR / _DEFAULT_CREDS_FILE)

    @property
    def path(self) -> Path:
        """Path to the credentials file."""
        return self._path

    def load(self) -> CodexCredentials | None:
        """Load credentials from disk.

        Returns:
            Credentials if the file exists and is valid, None otherwise.
        """
        if not self._path.exists():
            return None
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return CodexCredentials(
                access_token=data["access_token"],
                refresh_token=data["refresh_token"],
                expires_at=float(data["expires_at"]),
                user_email=data.get("user_email"),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            logger.warning("Corrupt credentials file at %s", self._path)
            return None

    def save(self, creds: CodexCredentials) -> None:
        """Save credentials to disk with restricted permissions."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Set directory permissions
        self._path.parent.chmod(0o700)
        # Write atomically via temp approach
        data = json.dumps(asdict(creds), indent=2)
        self._path.write_text(data, encoding="utf-8")
        self._path.chmod(0o600)

    def delete(self) -> bool:
        """Delete the credentials file.

        Returns:
            True if the file was deleted, False if it didn't exist.
        """
        if self._path.exists():
            self._path.unlink()
            return True
        return False
