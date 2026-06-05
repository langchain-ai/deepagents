"""Helpers for turning provider LLM errors into user-actionable messages.

Provider SDKs (Groq, OpenAI, Anthropic) raise SDK-specific exception classes
when the assembled prompt exceeds a per-request or per-minute token budget.
Without dedicated handling these surface as `APIStatusError`/`RateLimitError`
repr in the UI and the user sees a blank, unactionable failure.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_HTTP_REQUEST_TOO_LARGE = 413


def _collect_rate_limit_classes() -> tuple[type[BaseException], ...]:
    """Best-effort import of provider rate-limit exception classes.

    Returns:
        Tuple of installed provider exception types that may carry a
        413/token-rate-limit payload. Empty when no provider SDK is
        installed.
    """
    classes: list[type[BaseException]] = []

    def _try(import_path: str, attr: str) -> None:
        try:
            module = __import__(import_path, fromlist=[attr])
        except ImportError:  # pragma: no cover — optional dependency
            return
        cls = getattr(module, attr, None)
        if isinstance(cls, type) and issubclass(cls, BaseException):
            classes.append(cls)

    # Groq raises APIStatusError(status_code=413) for TPM/request-size overflow
    # and RateLimitError for the 429 variant.
    _try("groq", "APIStatusError")
    _try("groq", "RateLimitError")
    _try("openai", "RateLimitError")
    _try("openai", "APIStatusError")
    _try("anthropic", "RateLimitError")
    _try("anthropic", "APIStatusError")

    return tuple(classes)


LLM_RATE_LIMIT_ERRORS: tuple[type[BaseException], ...] = _collect_rate_limit_classes()
"""Exception classes that signal a token/rate-limit overflow from a provider.

Empty when no provider SDK is installed; callers should guard against that
before using it in an `except` clause.
"""


_LIMIT_REQUESTED_RE = re.compile(
    r"Limit\s+(?P<limit>\d[\d,]*)[^\d]*Requested\s+(?P<requested>\d[\d,]*)",
    re.IGNORECASE,
)


def _is_token_rate_limit(exc: BaseException) -> bool:
    """Return True iff `exc` looks like a token/size overflow we should format.

    Args:
        exc: A provider exception caught from the model call.

    Returns:
        True when the exception carries a 413 status code or its message
        mentions a token-rate-limit; False otherwise.
    """
    status = getattr(exc, "status_code", None)
    if status == _HTTP_REQUEST_TOO_LARGE:
        return True
    text = str(exc).lower()
    return "token" in text and ("limit" in text or "rate_limit_exceeded" in text)


def format_llm_rate_limit_error(exc: BaseException, *, model: str | None = None) -> str:
    """Format a provider token/rate-limit error as a user-actionable string.

    Args:
        exc: The provider exception caught from the model call.
        model: Optional `provider:model` spec to mention in the message.

    Returns:
        A short user-facing string that names the model (when provided),
        the limit/requested token numbers (when present in the raw
        message), and two concrete remediation paths.
    """
    raw = str(exc) or type(exc).__name__
    match = _LIMIT_REQUESTED_RE.search(raw)
    limit_clause = ""
    if match:
        limit_clause = (
            f" Limit {match.group('limit')} tokens, "
            f"requested {match.group('requested')}."
        )

    model_clause = f" on {model}" if model else ""
    return (
        f"The assembled prompt exceeded the configured model's token "
        f"limit{model_clause}.{limit_clause} The harness adds overhead from "
        "skills, memory, and local-context middleware on top of your message. "
        "Pick a model with a larger context / TPM budget, or disable some "
        "skills or memory sources, then try again."
    )


def is_llm_rate_limit_error(exc: BaseException) -> bool:
    """Return True iff `exc` is a provider error we want to surface specially.

    Args:
        exc: The exception caught from the model call.

    Returns:
        True when `exc` is an instance of a detected provider rate-limit
        class AND its payload looks like a token/size overflow.
    """
    if not LLM_RATE_LIMIT_ERRORS:
        return False
    if not isinstance(exc, LLM_RATE_LIMIT_ERRORS):
        return False
    return _is_token_rate_limit(exc)
