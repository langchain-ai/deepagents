"""Credential redaction helpers for untrusted tool results.

This module is intentionally lightweight so it can be imported in tool paths
without affecting startup performance.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from re import Pattern

MAX_REDACTION_CHARS = 1_000_000

CREDENTIAL_PATTERNS: tuple[tuple[str, Pattern[str]], ...] = (
    ("langsmith", re.compile(r"lsv2_(?:pt|sk)_[0-9a-f]{20,}_[0-9a-f]{6,}")),
    ("anthropic", re.compile(r"sk-ant-[A-Za-z0-9_-]{24,}")),
    ("openai", re.compile(r"sk-[A-Za-z0-9_-]{24,}")),
    ("github", re.compile(r"gh[pousr]_[A-Za-z0-9]{28,}")),
    ("github_pat", re.compile(r"github_pat_[A-Za-z0-9_]{40,}")),
    ("slack", re.compile(r"xox[bapre]-[A-Za-z0-9-]{20,}")),
    ("aws", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("google", re.compile(r"AIza[0-9A-Za-z_-]{35}")),
    ("tavily", re.compile(r"tvly-[A-Za-z0-9_-]{16,}")),
    (
        "jwt",
        re.compile(r"eyJ[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]+"),
    ),
)
"""Named, high-confidence credential patterns."""


@dataclass(frozen=True, slots=True)
class RedactionHit:
    """Describe a credential removed from tool-result text."""

    kind: str
    fingerprint: str


def redact_secrets(text: str) -> tuple[str, tuple[RedactionHit, ...]]:
    """Replace recognized credentials with stable, non-secret placeholders."""  # noqa: DOC201
    scan_text = text[:MAX_REDACTION_CHARS]
    matches: list[tuple[int, int, str]] = []
    for kind, pattern in CREDENTIAL_PATTERNS:
        matches.extend(
            (match.start(), match.end(), kind) for match in pattern.finditer(scan_text)
        )
    if not matches:
        return text, ()

    matches.sort(key=lambda item: (item[0], -item[1]))
    hits: list[RedactionHit] = []
    output: list[str] = []
    cursor = 0
    last_end = -1
    for start, end, kind in matches:
        if start < last_end:
            continue
        value = scan_text[start:end]
        fingerprint = hashlib.sha256(value.encode()).hexdigest()[:8]
        output.append(scan_text[cursor:start])
        replacement = f"[REDACTED_SECRET {kind} {fingerprint}]"
        output.append(replacement)
        hits.append(RedactionHit(kind=kind, fingerprint=fingerprint))
        cursor = end
        last_end = end
    output.extend((scan_text[cursor:], text[MAX_REDACTION_CHARS:]))
    return "".join(output), tuple(hits)
