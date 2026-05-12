"""User-level credential storage for model providers.

Persists API keys (and, in the future, OAuth tokens) under
`~/.deepagents/.state/auth.json` (file mode 0600, parent 0700) so users can
enter credentials directly in the TUI rather than exporting environment
variables before launch.

Security notes:

- The stored value (`ApiKeyCredential.key`) must never be logged, formatted
    via `%r`/`!r`, or interpolated into exception messages — every helper here
    reports only structural facts ("set credential for provider X").
- The file is written via `O_EXCL | 0o600` to a temp path, then atomically
    replaced. A second `chmod 0600` runs on the final path so filesystems that
    ignore the create-mode argument still end up with private perms. Permission
    failures are reported back to the caller in `WriteOutcome.warnings` so the
    UI can surface them to the user — `logger.warning` alone is invisible
    inside a Textual TUI session.
- On Windows, POSIX mode bits don't apply; the chmod calls are best-effort
    and skipped silently.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import stat
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, TypedDict

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

_STORAGE_VERSION = 1
"""Schema version stamped into `auth.json`; bump on incompatible shape changes."""


class ApiKeyCredential(TypedDict):
    """A persisted API key credential.

    The `type` field is the discriminator that lets `OAuthCredential` (added
    later) coexist in the same file without migration.
    """

    type: Literal["api_key"]
    """Credential kind discriminator."""

    key: str
    """The API key value as entered by the user. Never log this field."""

    added_at: str
    """ISO-8601 UTC timestamp recording when the credential was stored."""


class OAuthCredential(TypedDict):
    """A persisted OAuth subscription credential.

    Stub kept here so the `StoredCredential` discriminated union narrows
    correctly today and the OAuth implementation lands as a pure addition.
    No code path produces or consumes this shape yet.
    """

    type: Literal["oauth"]
    """Credential kind discriminator."""

    access_token: str
    """OAuth access token. Never log."""

    refresh_token: str
    """OAuth refresh token. Never log."""

    expires_at: str
    """ISO-8601 UTC expiry timestamp."""


StoredCredential = ApiKeyCredential | OAuthCredential
"""Tagged union of every persisted credential shape, narrowed by `type`."""


@dataclass(frozen=True, slots=True)
class WriteOutcome:
    """Result of a credential write that may have warnings to surface."""

    warnings: tuple[str, ...] = field(default_factory=tuple)
    """User-visible warning strings (e.g., chmod failures). Empty on success."""


def _auth_path() -> Path:
    """Return `~/.deepagents/.state/auth.json`.

    Resolved at call time (not import time) so tests can redirect storage by
    monkeypatching `deepagents_cli.model_config.DEFAULT_STATE_DIR` — same
    pattern `mcp_auth._tokens_dir` uses.
    """
    from deepagents_cli.model_config import DEFAULT_STATE_DIR

    return DEFAULT_STATE_DIR / "auth.json"


def _read_raw() -> dict | None:
    """Read and validate the on-disk auth file.

    Returns:
        The decoded JSON object, or `None` when the file is missing.

    Raises:
        RuntimeError: If the file exists but cannot be parsed or has an
            unsupported schema version.
    """
    path = _auth_path()
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except FileNotFoundError:
        return None
    except OSError as exc:
        msg = (
            f"Failed to read credential file {path}: {exc}. "
            "Check the file permissions on the parent directory."
        )
        raise RuntimeError(msg) from exc
    except json.JSONDecodeError as exc:
        msg = (
            f"Failed to parse credential file {path}: {exc}. "
            "Delete the file and re-add credentials via /auth if it is corrupt."
        )
        raise RuntimeError(msg) from exc
    if not isinstance(data, dict):
        msg = (
            f"Credential file {path} is not a JSON object. "
            "Delete it and re-add credentials via /auth."
        )
        # `RuntimeError` (not `TypeError`) is intentional: every corruption
        # path here surfaces the same error class so callers can render one
        # remediation hint regardless of the specific shape problem.
        raise RuntimeError(msg)  # noqa: TRY004
    version = data.get("version")
    if version != _STORAGE_VERSION:
        msg = (
            f"Credential file {path} has unsupported version {version!r} "
            f"(expected {_STORAGE_VERSION}). Delete it and re-add credentials via "
            "/auth."
        )
        raise RuntimeError(msg)
    return data


def _write_raw(data: dict) -> tuple[str, ...]:
    """Atomically write `data` as the new auth file with 0600 perms.

    Mirrors `mcp_auth.FileTokenStorage._write` so the security posture is
    consistent across both stores. If you change this, update
    `mcp_auth.FileTokenStorage._write` too — they share threat model.

    Returns:
        Tuple of warning strings for chmod failures the caller should
        surface to the user. Empty when permissions were locked down
        successfully (or on Windows where POSIX modes don't apply).
    """
    path = _auth_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    if hasattr(os, "chmod"):
        try:
            path.parent.chmod(stat.S_IRWXU)
        except OSError as exc:
            warnings.append(
                f"Could not set mode 0700 on {path.parent}: {exc}. "
                "Stored API keys may be readable by other local users."
            )
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(data, separators=(",", ":")).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if tmp.exists():
        # A leftover `.tmp` from a prior crashed write is the only path
        # `os.open(O_EXCL)` can fail without an actual conflict. Log so the
        # operator knows a stale write was cleaned up — silent suppression
        # masked recovery from a previous interrupted save.
        logger.warning(
            "Removing stale credential temp file %s left over from a prior write",
            tmp,
        )
        with contextlib.suppress(OSError):
            tmp.unlink()
    fd = os.open(str(tmp), flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
    except Exception:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
    try:
        tmp.replace(path)
    except Exception:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
    if hasattr(os, "chmod"):
        try:
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        except OSError as exc:
            warnings.append(
                f"Could not set mode 0600 on {path}: {exc}. "
                "Stored API keys may be world-readable."
            )
    for warning in warnings:
        logger.warning("%s", warning)
    return tuple(warnings)


def load_credentials() -> dict[str, StoredCredential]:
    """Return all stored credentials keyed by provider name.

    Returns:
        Mapping of provider name to its stored credential. Empty when no
        credentials are persisted yet.

    Raises:
        RuntimeError: If the file exists but is corrupt or has an unsupported
            schema version. Caller is expected to surface a remediation hint.
    """  # noqa: DOC502 - re-raised from `_read_raw`
    data = _read_raw()
    if data is None:
        return {}
    creds_raw = data.get("credentials")
    if not isinstance(creds_raw, dict):
        return {}
    result: dict[str, StoredCredential] = {}
    for provider, entry in creds_raw.items():
        coerced = _coerce_credential(entry)
        if coerced is not None:
            result[provider] = coerced
    return result


def _coerce_credential(raw: Any) -> StoredCredential | None:  # noqa: ANN401
    # `raw: Any` because entries come from `json.loads`; the body is
    # what enforces the `StoredCredential` contract.
    """Validate one raw credential entry, returning `None` on shape mismatch.

    Centralizes the runtime check against the `StoredCredential` union so
    `load_credentials` doesn't repeat the per-field guard logic and so a
    single helper can grow as new variants are added.

    Returns:
        The coerced `StoredCredential`, or `None` when the entry doesn't
        match any known variant's shape.
    """
    if not isinstance(raw, dict):
        return None
    cred_type = raw.get("type")
    if cred_type == "api_key":
        key = raw.get("key")
        if not isinstance(key, str) or not key:
            return None
        added_at = raw.get("added_at")
        if not isinstance(added_at, str):
            added_at = ""
        return ApiKeyCredential(type="api_key", key=key, added_at=added_at)
    # OAuth is reserved for a future PR — silently skip until the producer
    # path lands. `cred_type in {"oauth"}` falls through to None here.
    return None


def get_stored_key(provider: str) -> str | None:
    """Return the stored API key for `provider`, or `None` if unset.

    Returns `None` for stored OAuth credentials too — callers that need
    OAuth tokens should read `load_credentials()` directly and narrow on
    `type`.

    Raises:
        RuntimeError: If the credential file is corrupt.
    """  # noqa: DOC502 - re-raised from `_read_raw` via `load_credentials`
    creds = load_credentials()
    entry = creds.get(provider)
    if entry is None or entry["type"] != "api_key":
        return None
    return entry["key"] or None


def set_stored_key(provider: str, key: str) -> WriteOutcome:
    """Persist an API key for `provider`.

    Empty / whitespace-only keys are rejected so callers don't accidentally
    write a sentinel that masks a working environment variable (see
    `apply_stored_credentials` in `model_config` — a stored empty would
    unconditionally overwrite the env var).

    Args:
        provider: Provider identifier (e.g., `"anthropic"`).
        key: The API key value. Whitespace is stripped before storage.

    Returns:
        A `WriteOutcome` whose `warnings` tuple lists chmod failures the
        caller should surface to the user. Empty on a clean save.

    Raises:
        ValueError: If `provider` or the stripped `key` is empty.
        RuntimeError: If the credential file is corrupt and cannot be read.
    """  # noqa: DOC502 - `RuntimeError` is re-raised from `_read_raw`
    if not provider:
        msg = "Provider name cannot be empty"
        raise ValueError(msg)
    cleaned = key.strip()
    if not cleaned:
        msg = "API key cannot be empty"
        raise ValueError(msg)
    data = _read_raw() or {}
    creds = data.get("credentials")
    if not isinstance(creds, dict):
        creds = {}
    creds[provider] = {
        "type": "api_key",
        "key": cleaned,
        "added_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
    }
    data["version"] = _STORAGE_VERSION
    data["credentials"] = creds
    warnings = _write_raw(data)
    logger.debug("Stored credential for provider %s", provider)
    return WriteOutcome(warnings=warnings)


def delete_stored_key(provider: str) -> bool:
    """Remove a stored credential for `provider`.

    Args:
        provider: Provider identifier.

    Returns:
        `True` if a credential was removed, `False` if none was stored.

    Raises:
        RuntimeError: If the credential file is corrupt and cannot be read.
    """  # noqa: DOC502 - re-raised from `_read_raw`
    data = _read_raw()
    if data is None:
        return False
    creds = data.get("credentials")
    if not isinstance(creds, dict) or provider not in creds:
        return False
    del creds[provider]
    data["version"] = _STORAGE_VERSION
    data["credentials"] = creds
    _write_raw(data)
    logger.debug("Deleted credential for provider %s", provider)
    return True


def list_configured_providers() -> list[str]:
    """Return providers that currently have a stored credential, sorted.

    Raises:
        RuntimeError: If the credential file is corrupt.
    """  # noqa: DOC502 - re-raised from `_read_raw` via `load_credentials`
    return sorted(load_credentials().keys())
