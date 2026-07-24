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
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict

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

    base_url: NotRequired[str]
    """Optional provider endpoint paired with this key.

    Stored only when the user supplied one in `/auth`. A key and its endpoint
    form a coherent pair — applying the key also applies (or, when this is
    absent, resets to the provider default) the base URL, so a personal key is
    never sent to a gateway it doesn't belong to. Not treated as a secret (it is
    logged when malformed and surfaced in hints); avoid embedding credentials in
    the URL.
    """

    project: NotRequired[str]
    """Optional LangSmith project name paired with this credential.

    Set only for the `langsmith` tracing service when the user supplies a
    custom project in `/auth`; absent means traces fall back to the default
    (`deepagents-code`). Not a secret — it is shown in the `/auth` advanced
    panel and applied to `LANGSMITH_PROJECT` at startup.
    """


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


@dataclass(frozen=True, slots=True)
class DeleteOutcome:
    """Result of a credential delete that may have warnings to surface.

    A delete rewrites the whole store, so it can hit the same chmod failures as
    a write. `warnings` carries them symmetrically with `WriteOutcome` so a
    caller's "removed" confirmation doesn't paper over a store the delete
    failed to lock down to owner-only.
    """

    removed: bool
    """`True` if a credential was removed, `False` if none was stored."""

    warnings: tuple[str, ...] = field(default_factory=tuple)
    """chmod-failure warnings from the rewrite. Always empty for a no-op delete
    (`removed=False`), which performs no write."""

    def __post_init__(self) -> None:
        """Reject the one field combination the docstring forbids.

        A no-op delete performs no write, so it can raise no chmod warnings.
        Enforcing it here makes "no warnings when nothing was removed" a
        construction-time guarantee rather than a producer-side convention a
        future edit (or a second producer) could quietly break.

        Raises:
            ValueError: If `warnings` is non-empty while `removed` is `False`.
        """
        if self.warnings and not self.removed:
            msg = "DeleteOutcome cannot carry warnings when removed=False"
            raise ValueError(msg)


def auth_path() -> Path:
    """Return the resolved path to the credential store (`auth.json`).

    Resolved at call time (not import time) so tests can redirect storage by
    monkeypatching `deepagents_code.model_config.DEFAULT_STATE_DIR` — same
    pattern `mcp_auth._tokens_dir` uses.
    """
    from deepagents_code.model_config import DEFAULT_STATE_DIR

    return DEFAULT_STATE_DIR / "auth.json"


def _read_raw() -> dict | None:
    """Read and validate the on-disk auth file.

    Returns:
        The decoded JSON object, or `None` when the file is missing.

    Raises:
        RuntimeError: If the file exists but cannot be parsed or has an
            unsupported schema version.
    """
    path = auth_path()
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
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        # `UnicodeDecodeError` (a `ValueError`, not an `OSError`) escapes the
        # handler above when the file holds non-UTF-8 bytes; treat a decode
        # failure as corruption so callers get the same `RuntimeError` hint
        # instead of an unhandled traceback.
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
    path = auth_path()
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


def _write_raw_or_raise(data: dict) -> tuple[str, ...]:
    """Write `data` via `_write_raw`, converting write failures to `RuntimeError`.

    `_write_raw` lets `OSError` from the atomic write (no disk space, an
    unwritable state directory, a cross-device rename of the temp file)
    propagate. The public writers document `RuntimeError` for unrecoverable
    store failures, so translate here with a remediation hint instead of
    leaking a raw traceback to the caller (CLI or TUI). The message never
    includes the credential value.

    Returns:
        The chmod-warning tuple from `_write_raw` on success.

    Raises:
        RuntimeError: If the underlying write fails with an `OSError`.
    """
    try:
        return _write_raw(data)
    except OSError as exc:
        msg = (
            f"Failed to write credential file {auth_path()}: {exc}. "
            "Check available disk space and the permissions on the parent "
            "directory."
        )
        raise RuntimeError(msg) from exc


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
        credential = ApiKeyCredential(type="api_key", key=key, added_at=added_at)
        base_url = raw.get("base_url")
        if isinstance(base_url, str) and base_url:
            credential["base_url"] = base_url
        elif base_url is not None:
            # Present but not a usable string (e.g. a hand-edit left an int or
            # an empty value). Dropping it silently would pair the key with the
            # provider default — possibly the wrong endpoint — with no trace, so
            # log it. `base_url` is non-secret, so logging the value is safe.
            logger.warning(
                "Ignoring malformed base_url for a stored credential: %r", base_url
            )
        project = raw.get("project")
        if isinstance(project, str) and project:
            credential["project"] = project
        elif project is not None:
            # Same rationale as `base_url`: a hand-edited non-string value is
            # dropped with a trace. `project` is non-secret, so logging is safe.
            logger.warning(
                "Ignoring malformed project for a stored credential: %r", project
            )
        return credential
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


def get_stored_base_url(provider: str) -> str | None:
    """Return the base URL paired with `provider`'s stored key, or `None`.

    Returns `None` both when no key is stored and when a key is stored without
    an accompanying base URL (the user left the field blank, meaning "use the
    provider default"). Callers distinguish the two via `get_stored_key`.

    Raises:
        RuntimeError: If the credential file is corrupt.
    """  # noqa: DOC502 - re-raised from `_read_raw` via `load_credentials`
    creds = load_credentials()
    entry = creds.get(provider)
    if entry is None or entry["type"] != "api_key":
        return None
    return entry.get("base_url") or None


def get_stored_project(provider: str) -> str | None:
    """Return the LangSmith project paired with `provider`'s stored key, or `None`.

    Returns `None` when no key is stored and when a key is stored without a
    custom project (the user left the field blank, meaning "use the default").

    Raises:
        RuntimeError: If the credential file is corrupt.
    """  # noqa: DOC502 - re-raised from `_read_raw` via `load_credentials`
    creds = load_credentials()
    entry = creds.get(provider)
    if entry is None or entry["type"] != "api_key":
        return None
    return entry.get("project") or None


def set_stored_key(
    provider: str,
    key: str,
    *,
    base_url: str | None = None,
    project: str | None = None,
) -> WriteOutcome:
    """Persist an API key for `provider`.

    Empty / whitespace-only keys are rejected so callers don't accidentally
    write a sentinel that masks a working environment variable (see
    `apply_stored_credentials` in `model_config` — a stored empty would
    unconditionally overwrite the env var).

    This rewrites the whole credential record: `base_url` and `project` are
    *not* merged with any previously stored values. Passing blank/`None` for
    either clears it, so a caller rotating a key while wanting to keep the
    existing endpoint/project must read it back (e.g. via `get_stored_base_url`
    / `get_stored_project`) and pass it in again.

    Args:
        provider: Provider identifier (e.g., `"anthropic"`).
        key: The API key value. Whitespace is stripped before storage.
        base_url: Optional provider endpoint to pair with the key. Whitespace
            is stripped; blank/`None` stores no endpoint, meaning the key uses
            the provider default rather than any inherited (e.g. gateway) URL.
        project: Optional LangSmith project name to pair with the key. Valid
            only for the `langsmith` tracing service. Whitespace is stripped;
            blank/`None` stores no project, meaning traces use the default
            project.

    Returns:
        A `WriteOutcome` whose `warnings` tuple lists chmod failures the
        caller should surface to the user. Empty on a clean save.

    Raises:
        ValueError: If `provider` or the stripped `key` is empty, or a non-empty
            `project` is paired with a provider other than the `langsmith`
            service.
        RuntimeError: If the credential file is corrupt and cannot be read, or
            the new file cannot be written (e.g. no disk space or an
            unwritable state directory).
    """  # noqa: DOC502 - `RuntimeError` re-raised from `_read_raw`/`_write_raw_or_raise`
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
    entry: dict[str, str] = {
        "type": "api_key",
        "key": cleaned,
        "added_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
    }
    cleaned_base_url = base_url.strip() if base_url else ""
    if cleaned_base_url:
        entry["base_url"] = cleaned_base_url
    cleaned_project = project.strip() if project else ""
    if cleaned_project:
        # A project name is meaningful only for the LangSmith tracing service;
        # enforce the invariant at the write boundary so a stray project can
        # never be persisted onto an unrelated provider, regardless of caller.
        # Lazy import avoids a circular dependency (model_config imports this
        # module), matching the pattern in `auth_path`.
        from deepagents_code.model_config import is_langsmith

        if not is_langsmith(provider):
            msg = f"project is only valid for the langsmith service, not {provider!r}"
            raise ValueError(msg)
        entry["project"] = cleaned_project
    creds[provider] = entry
    data["version"] = _STORAGE_VERSION
    data["credentials"] = creds
    warnings = _write_raw_or_raise(data)
    logger.debug("Stored credential for provider %s", provider)
    return WriteOutcome(warnings=warnings)


def delete_stored_key(provider: str) -> DeleteOutcome:
    """Remove a stored credential for `provider`.

    Args:
        provider: Provider identifier.

    Returns:
        A `DeleteOutcome` whose `removed` flag reports whether a credential was
        present, and whose `warnings` tuple lists chmod failures from the
        rewrite (empty for a no-op delete, which performs no write).

    Raises:
        RuntimeError: If the credential file is corrupt and cannot be read, or
            the rewrite cannot be written (e.g. no disk space or an unwritable
            state directory).
    """  # noqa: DOC502 - re-raised from `_read_raw`/`_write_raw_or_raise`
    data = _read_raw()
    if data is None:
        return DeleteOutcome(removed=False)
    creds = data.get("credentials")
    if not isinstance(creds, dict) or provider not in creds:
        return DeleteOutcome(removed=False)
    del creds[provider]
    data["version"] = _STORAGE_VERSION
    data["credentials"] = creds
    warnings = _write_raw_or_raise(data)
    logger.debug("Deleted credential for provider %s", provider)
    return DeleteOutcome(removed=True, warnings=warnings)


def list_configured_providers() -> list[str]:
    """Return providers that currently have a stored credential, sorted.

    Raises:
        RuntimeError: If the credential file is corrupt.
    """  # noqa: DOC502 - re-raised from `_read_raw` via `load_credentials`
    return sorted(load_credentials().keys())
