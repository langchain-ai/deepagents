"""OAuth + bearer-token auth helpers for MCP servers."""

from __future__ import annotations

import os
import re


_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def resolve_headers(
    headers: dict[str, str],
    *,
    server_name: str | None = None,
) -> dict[str, str]:
    """Resolve `${VAR}` env-var references in header values.

    `$${VAR}` is the escape form and collapses to the literal `${VAR}` with no
    lookup. A dollar sign not followed by `{` or `$` is left untouched.

    Raises:
        TypeError: If a header value is not a string.
        RuntimeError: If a `${VAR}` references an unset environment variable.
    """
    resolved: dict[str, str] = {}
    for name, value in headers.items():
        if not isinstance(value, str):
            where = f"mcpServers.{server_name}.headers.{name}" if server_name else name
            msg = f"{where} must be a string, got {type(value).__name__}"
            raise TypeError(msg)
        resolved[name] = _interpolate(value, header=name, server_name=server_name)
    return resolved


def _interpolate(s: str, *, header: str, server_name: str | None) -> str:
    out: list[str] = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "$" and i + 1 < len(s):
            nxt = s[i + 1]
            if nxt == "$":
                # `$${...}` → literal `${...}` if a brace group follows;
                # otherwise literal `$$`.
                if i + 2 < len(s) and s[i + 2] == "{":
                    end = s.find("}", i + 3)
                    if end != -1:
                        out.append("$" + s[i + 2 : end + 1])
                        i = end + 1
                        continue
                out.append("$$")
                i += 2
                continue
            if nxt == "{":
                end = s.find("}", i + 2)
                if end != -1:
                    var_name = s[i + 2 : end]
                    if _IDENT_RE.fullmatch(var_name):
                        val = os.environ.get(var_name)
                        if val is None:
                            where = (
                                f"mcpServers.{server_name}.headers.{header}"
                                if server_name
                                else header
                            )
                            msg = (
                                f"{where} references unset env var {var_name}. "
                                f"Set {var_name} in the environment or remove "
                                "the reference."
                            )
                            raise RuntimeError(msg)
                        out.append(val)
                        i = end + 1
                        continue
        out.append(ch)
        i += 1
    return "".join(out)
