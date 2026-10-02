"""Stderr marker emission used by the langgraph server graph entry point.

Lives in its own module so unit tests can exercise the marker contract
without triggering `server_graph.make_graph()` at import time.
"""

from __future__ import annotations

import json
import logging
import sys
import traceback

logger = logging.getLogger(__name__)

STARTUP_ERROR_MARKER = "DEEPAGENTS_STARTUP_ERROR:"
"""Stderr marker the parent app scans for in `server._extract_startup_error_marker`
to upgrade an opaque "Server process exited with code N" into a one-line summary.
Format is `{STARTUP_ERROR_MARKER}{single-line message}`."""

_STARTUP_ERROR_DETAILS_MARKER = "DEEPAGENTS_STARTUP_ERROR_DETAILS:"


def _recovery_details(exc: BaseException) -> dict[str, str | None] | None:
    """Allowlist recovery fields; never serialize an exception's full state.

    Returns:
        Recovery details for known errors, or `None` for other failures.
    """
    from deepagents_code.model_config import (
        MissingCredentialsError,
        MissingProviderPackageError,
    )

    if isinstance(exc, MissingCredentialsError):
        return {
            "type": "MissingCredentialsError",
            "message": str(exc),
            "provider": exc.provider,
            "env_var": exc.env_var,
        }
    if isinstance(exc, MissingProviderPackageError):
        return {
            "type": "MissingProviderPackageError",
            "message": str(exc),
            "provider": exc.provider,
            "package": exc.package,
        }
    return None


def _decode_recovery_error(payload: object) -> Exception | None:
    """Reconstruct only known recovery errors with validated string fields.

    Returns:
        A typed error, or `None` for unsupported or invalid payloads.
    """
    from deepagents_code.model_config import (
        MissingCredentialsError,
        MissingProviderPackageError,
    )

    if not isinstance(payload, dict):
        return None
    message, provider = payload.get("message"), payload.get("provider")
    if not isinstance(message, str) or not isinstance(provider, str) or not provider:
        return None
    if payload.get("type") == "MissingCredentialsError":
        env_var = payload.get("env_var")
        if env_var is None or isinstance(env_var, str):
            return MissingCredentialsError(message, provider=provider, env_var=env_var)
    if payload.get("type") == "MissingProviderPackageError":
        package = payload.get("package")
        if isinstance(package, str) and package:
            return MissingProviderPackageError(
                message, provider=provider, package=package
            )
    return None


def startup_error_from_output(output: str, fallback: str) -> Exception:
    """Restore a typed recovery error, falling back to the ordinary log summary.

    Args:
        output: Server subprocess output containing startup markers.
        fallback: Error text for legacy, unknown, or malformed markers.

    Returns:
        A known model recovery error or a `RuntimeError` with the fallback text.
    """
    for line in reversed(output.splitlines()):
        if _STARTUP_ERROR_DETAILS_MARKER in line:
            _, encoded = line.split(_STARTUP_ERROR_DETAILS_MARKER, 1)
            try:
                error = _decode_recovery_error(json.loads(encoded))
            except ValueError:
                break
            if error is not None:
                return error
            break
    return RuntimeError(fallback)


def emit_startup_failure(exc: BaseException) -> None:
    """Report a server graph startup failure to the parent app process.

    Emits the full traceback for logs/debugging, then a
    single-line `{STARTUP_ERROR_MARKER}{type}: {summary}` line that
    `server._extract_startup_error_marker` parses to upgrade an opaque
    "Server process exited with code N" into an actionable summary.
    Known credential/package errors also include JSON recovery fields so the
    client can restore their types without constructing a provider model.

    Args:
        exc: The exception raised during graph initialization.
    """
    logger.critical("Failed to initialize server graph", exc_info=exc)
    print(  # noqa: T201  # stderr fallback — logger may not reach parent process
        f"Failed to initialize server graph: {exc}\n{traceback.format_exc()}",
        file=sys.stderr,
    )
    # Marker contract is single-line; guard against multi-line/empty `str(exc)`
    # and include the type so e.g. `ValueError` and `RuntimeError` are
    # distinguishable in the parent's truncated summary.
    exc_lines = str(exc).splitlines()
    summary = exc_lines[0] if exc_lines else "<no message>"
    print(  # noqa: T201
        f"{STARTUP_ERROR_MARKER}{type(exc).__name__}: {summary}",
        file=sys.stderr,
    )
    details = _recovery_details(exc)
    if details is not None:
        print(  # noqa: T201  # parent process recovery protocol
            f"{_STARTUP_ERROR_DETAILS_MARKER}{json.dumps(details)}",
            file=sys.stderr,
        )
