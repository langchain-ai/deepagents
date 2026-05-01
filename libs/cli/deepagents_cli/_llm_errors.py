"""Helpers for surfacing LLM-provider auth errors with actionable guidance.

When the configured LLM provider rejects the request with an
`AuthenticationError` (typically a stale or invalid API key), the raw
exception bubbles up through `langchain_core` with a multi-line stack trace
and a body that includes the provider's error JSON. Mounting that as an
``ErrorMessage`` in the chat surface gives the user no clear next step.

This module provides:

- `LLM_AUTH_ERRORS`: a tuple of provider auth-error classes safe to use
  in an ``except`` clause. Optional providers that aren't installed are
  silently omitted, so the tuple is never empty-only when no providers
  are present.
- `format_llm_auth_error(exc)`: a short, user-facing string that names
  the provider, the env var to update, and where to get a new key. Does
  not include the stack trace.

This mirrors the Tavily handling pattern in ``deepagents_cli.tools``
(``test_web_search_handles_tavily_invalid_api_key``).
"""

from __future__ import annotations

from typing import Any

# Each entry: (exception class, env var name, provider display name, keys URL).
_PROVIDER_SPECS: list[tuple[type[BaseException], str, str, str]] = []

try:
    from openai import AuthenticationError as _OpenAIAuthError
except ImportError:  # pragma: no cover - openai not installed in this env
    pass
else:
    _PROVIDER_SPECS.append(
        (
            _OpenAIAuthError,
            "OPENAI_API_KEY",
            "OpenAI",
            "https://platform.openai.com/account/api-keys",
        )
    )

try:
    from anthropic import AuthenticationError as _AnthropicAuthError
except ImportError:  # pragma: no cover - anthropic not installed in this env
    pass
else:
    _PROVIDER_SPECS.append(
        (
            _AnthropicAuthError,
            "ANTHROPIC_API_KEY",
            "Anthropic",
            "https://console.anthropic.com/settings/keys",
        )
    )

#: Tuple of auth-error classes available at runtime. Use in ``except`` clauses.
#: Always contains at least :class:`PermissionError` so the tuple is never
#: empty (which would make the ``except`` clause a syntax error) and so a
#: future provider that re-raises ``PermissionError`` for credential issues
#: is also caught.
LLM_AUTH_ERRORS: tuple[type[BaseException], ...] = tuple(
    spec[0] for spec in _PROVIDER_SPECS
) or (PermissionError,)


def format_llm_auth_error(exc: BaseException) -> str:
    """Return a short, user-actionable message for an LLM auth error.

    The message names the provider, the env var to update, and the URL
    where the user can get a fresh key. The provider's underlying
    ``message`` attribute (if present) is appended for context, but the
    stack trace is intentionally omitted.
    """
    for cls, env_var, provider, keys_url in _PROVIDER_SPECS:
        if isinstance(exc, cls):
            detail = _extract_detail(exc)
            suffix = f" Provider said: {detail}" if detail else ""
            return (
                f"{provider} authentication failed. "
                f"Update the {env_var} environment variable "
                f"(get a new key at {keys_url}) and retry."
                f"{suffix}"
            )
    # Fall-through for the PermissionError shim or any other matched class.
    return f"LLM authentication failed: {exc}"


def _extract_detail(exc: BaseException) -> str:
    """Pull a short human-readable detail string out of a provider error.

    The OpenAI/Anthropic SDKs expose either a ``.message`` attribute or
    a ``.body`` dict with an ``error.message``. Trim to one line and
    cap the length so the chat surface doesn't render multi-paragraph
    JSON dumps.
    """
    detail: Any = getattr(exc, "message", None)
    if not detail:
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict):
                detail = err.get("message")
    if not detail:
        detail = str(exc)
    text = str(detail).splitlines()[0].strip()
    if len(text) > 200:
        text = text[:197] + "..."
    return text
