"""Validation and environment-variable expansion for MCP server config.

Resolves `${VAR}` and `${VAR:-default}` references in the supported
configuration fields (`command`, `url`, `args`, `env`, `headers`) and
validates their types. A `${VAR:-default}` reference falls back to
`default` when `VAR` is unset *or* empty (POSIX `:-` semantics).
"""

from __future__ import annotations

import copy
import os
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

_ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
"""Matches a supported reference: `${VAR}` or `${VAR:-default}`.

Group 1 is the variable name; group 2 (present only for the `:-` form) is
the default. A bare `$VAR` and a literal `$` are intentionally not matched.
"""

_ENV_BRACE_RE = re.compile(r"\$\{")
"""Matches a `${` brace-open, used to catch malformed `${...}` references."""


def _interpolate_env(value: str, *, field: str) -> str:
    """Expand `${VAR}` / `${VAR:-default}` references in one config string.

    A bare `$VAR` (no braces) and a literal `$` pass through untouched;
    only the braced forms expand. `${VAR:-default}` uses `default` when
    `VAR` is unset or empty. A `${...}` that does not parse as one of the
    supported forms (e.g. `${VAR-default}` or an unterminated `${VAR`) is
    rejected rather than silently emitted, so a typo cannot inject a
    garbage value into a URL, command, or header.

    Args:
        value: Raw configuration string.
        field: Fully qualified field path for error messages.

    Returns:
        The interpolated string.

    Raises:
        RuntimeError: If a required environment variable is unset, or the
            string contains a malformed `${...}` reference.
    """

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(2)
        resolved = os.environ.get(name)
        # A non-empty value always wins, for both `${VAR}` and `${VAR:-default}`.
        if resolved:
            return resolved
        # `resolved` is now "" (set but empty) or None (unset).
        if default is not None:
            # `${VAR:-default}`: `:-` falls back for empty *and* unset (POSIX).
            return default
        if resolved is not None:
            # `${VAR}` set to "": no default, so emit the empty value.
            return resolved
        # `${VAR}` unset with no default: the only hard error.
        msg = (
            f"{field} references unset env var {name}. "
            f"Set {name} in the environment or provide a default."
        )
        raise RuntimeError(msg)

    # Reject any `${` that isn't the start of a well-formed reference. The
    # check is against the raw `value` (not the substituted result) so a
    # resolved value that happens to contain `${` never trips it.
    ref_spans = [match.span() for match in _ENV_REF_RE.finditer(value)]
    for brace in _ENV_BRACE_RE.finditer(value):
        if not any(start <= brace.start() < end for start, end in ref_spans):
            msg = (
                f"{field} contains a malformed '${{...}}' reference. "
                "Use '${VAR}' or '${VAR:-default}'."
            )
            raise RuntimeError(msg)

    return _ENV_REF_RE.sub(replace, value)


def _resolve_string(value: object, *, field: str) -> str:
    """Validate and interpolate one string field.

    Args:
        value: Raw field value.
        field: Fully qualified field path for error messages.

    Returns:
        The validated and interpolated string.

    Raises:
        TypeError: If the field value is not a string.
    """
    if not isinstance(value, str):
        msg = f"{field} must be a string, got {type(value).__name__}"
        raise TypeError(msg)
    return _interpolate_env(value, field=field)


def _resolve_mapping_values(
    values: Mapping[str, object],
    *,
    field: str,
) -> dict[str, str]:
    """Validate and interpolate string values in a mapping field.

    Args:
        values: Raw mapping values.
        field: Fully qualified field path for error messages.

    Returns:
        A new mapping with validated and interpolated values.
    """
    return {
        name: _resolve_string(value, field=f"{field}.{name}")
        for name, value in values.items()
    }


def resolve_mcp_server_env(
    server_name: str,
    server_config: Mapping[str, object],
) -> dict[str, Any]:
    """Resolve `${VAR}` references in one MCP server's supported fields.

    Interpolates the `command`, `url`, `args`, `env`, and `headers`
    fields (see `_interpolate_env` for the reference syntax); every other
    field is copied through verbatim. The input is not mutated.

    Args:
        server_name: Server name used in field-specific error messages.
        server_config: Raw server configuration.

    Returns:
        A resolved copy of the server configuration.

    Raises:
        TypeError: If a supported field contains a non-string value.
        RuntimeError: If a required environment variable is unset.
    """  # noqa: DOC502 - `RuntimeError` is raised by `_interpolate_env`
    resolved: dict[str, Any] = copy.deepcopy(dict(server_config))
    prefix = f"mcpServers.{server_name}"

    for name in ("command", "url"):
        if name in resolved:
            resolved[name] = _resolve_string(resolved[name], field=f"{prefix}.{name}")

    if "args" in resolved:
        args = resolved["args"]
        if not isinstance(args, list):
            msg = f"{prefix}.args must be a list, got {type(args).__name__}"
            raise TypeError(msg)
        resolved["args"] = [
            _resolve_string(value, field=f"{prefix}.args[{index}]")
            for index, value in enumerate(args)
        ]

    for name in ("env", "headers"):
        if name not in resolved:
            continue
        values = resolved[name]
        if not isinstance(values, dict):
            msg = f"{prefix}.{name} must be a dictionary, got {type(values).__name__}"
            raise TypeError(msg)
        resolved[name] = _resolve_mapping_values(values, field=f"{prefix}.{name}")

    return resolved
