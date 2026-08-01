"""Redact live credential values from tool results before they reach the model.

Tool results such as an `execute` shell echo or a `write_file` preview can carry
a live API key or bearer token verbatim (e.g. verifying `TAVILY_API_KEY` by
echoing it, or persisting an MCP bearer token to `.mcp.json`). Once that value
is in the tool result it enters the model context and the trace, where it can be
logged, cached, or echoed downstream.

This module masks the credential *value* to `<REDACTED:CREDENTIAL>` while leaving
surrounding structure (variable names, key prefixes' shape) intact so the agent
can still reason about "the key is set". It only rewrites what the model/trace
sees; the on-disk file the user explicitly asked to create is untouched.

Intentionally lightweight so it can be imported on the tool-result path without
affecting startup performance.
"""

from __future__ import annotations

import re

REDACTION_PLACEHOLDER = "<REDACTED:CREDENTIAL>"

# Provider key prefixes with an opaque token body. Ordered longest-prefix-first
# so `tvly-dev-` / `sk-ant-` win over their shorter siblings.
_PROVIDER_KEY_PATTERN = re.compile(
    r"(?:tvly-dev-|tvly-|sk-ant-|sk-|AKIA|xoxb-|ghp_)[A-Za-z0-9_\-]{8,}"
)

# `Authorization: Bearer <token>` — keep the scheme, mask the token.
_BEARER_PATTERN = re.compile(
    r"(Authorization:\s*Bearer\s+)([A-Za-z0-9._\-]+)",
    re.IGNORECASE,
)

# `<NAME>_API_KEY=<value>` / `<NAME>_TOKEN=<value>` — keep the name, mask value.
_ASSIGNMENT_PATTERN = re.compile(
    r"([A-Za-z0-9_]*(?:_API_KEY|_TOKEN|API_KEY|TOKEN))(\s*[=:]\s*)"
    r"(\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)


def _mask_assignment(match: re.Match[str]) -> str:
    name, sep, value = match.group(1), match.group(2), match.group(3)
    if value and value[0] in {'"', "'"}:
        quote = value[0]
        return f"{name}{sep}{quote}{REDACTION_PLACEHOLDER}{quote}"
    return f"{name}{sep}{REDACTION_PLACEHOLDER}"


def redact_secrets(text: str) -> str:
    """Return `text` with known credential value shapes masked."""
    if not text:
        return text
    redacted = _PROVIDER_KEY_PATTERN.sub(REDACTION_PLACEHOLDER, text)
    redacted = _BEARER_PATTERN.sub(rf"\1{REDACTION_PLACEHOLDER}", redacted)
    return _ASSIGNMENT_PATTERN.sub(_mask_assignment, redacted)


def contains_secret(text: str) -> bool:
    """Return True when `text` differs from its redacted form."""
    return bool(text) and redact_secrets(text) != text
