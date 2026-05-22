"""Defensive scrubber for the `execute` tool's `command` argument.

The agent reads credentials from the inherited sandbox environment
(`LANGSMITH_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.). When
it nonetheless inlines a literal secret value into a shell command, that
value is persisted in the trace's `tool_calls[*].args` payload — which
the output-side secrets evaluator does not inspect.

This module provides a single high-confidence regex-based check that
scans the incoming `command` string for known secret-token shapes. On
match it raises `SecretInCommandError` whose message tells the model to
re-issue the command relying on the env-var by name.

Patterns intentionally stay narrow (full-token shapes only, anchored to
unambiguous prefixes) so that incidental matches on user payloads or
documentation strings don't break legitimate calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "SECRET_PATTERNS",
    "SecretInCommandError",
    "scan_command_for_secrets",
]


@dataclass(frozen=True)
class _SecretPattern:
    """A labelled regex matching a single secret-token shape."""

    name: str
    regex: re.Pattern[str]


# High-confidence patterns. Each anchors on an unambiguous prefix and
# requires enough trailing entropy that documentation/snippet text is
# unlikely to collide. Add new patterns sparingly — false positives
# break legitimate commands.
SECRET_PATTERNS: tuple[_SecretPattern, ...] = (
    # LangSmith personal tokens: `lsv2_pt_<32+ hex>_<10+ hex>`. Also
    # cover service-key variant `lsv2_sk_...`.
    _SecretPattern(
        name="LangSmith API key",
        regex=re.compile(r"lsv2_(?:pt|sk)_[a-f0-9]{16,}_[a-f0-9]{8,}"),
    ),
    # OpenAI API keys: `sk-...` (>=20 alnum/underscore/hyphen chars
    # after the prefix). Excludes the Anthropic `sk-ant-...` prefix via
    # a negative lookahead so each pattern reports its own label.
    _SecretPattern(
        name="OpenAI API key",
        regex=re.compile(r"sk-(?!ant-)[A-Za-z0-9_\-]{20,}"),
    ),
    # Anthropic API keys: `sk-ant-...`.
    _SecretPattern(
        name="Anthropic API key",
        regex=re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),
    ),
)


class SecretInCommandError(ValueError):
    """Raised when the execute tool's `command` argument inlines a secret.

    The message is surfaced verbatim to the model as a structured tool
    error so it can self-correct on the next turn.
    """

    def __init__(self, secret_name: str) -> None:
        message = (
            f"Refusing to execute: command appears to inline a literal "
            f"{secret_name}. Do not paste secret values as `KEY=value` "
            "command prefixes or as `--api-key <value>` arguments — "
            "the sandbox already inherits the relevant environment "
            "variable, so the CLI will pick it up automatically. "
            "Re-issue the command without the literal secret (and, if "
            "you genuinely need a different key, reference it by env-"
            "var name, e.g. `$ALT_KEY`)."
        )
        super().__init__(message)
        self.secret_name = secret_name


def scan_command_for_secrets(command: str) -> None:
    """Raise `SecretInCommandError` if `command` contains a known secret.

    Returns `None` (and leaves the caller to execute the command) when
    no high-confidence pattern matches.
    """
    if not command:
        return
    for pattern in SECRET_PATTERNS:
        if pattern.regex.search(command):
            raise SecretInCommandError(pattern.name)
