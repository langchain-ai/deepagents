"""Optional Arize Phoenix tracing for the agent server process."""

from __future__ import annotations

import os
from typing import Protocol

from deepagents_code._env_vars import PHOENIX_TRACING, is_env_truthy

_DEFAULT_PROJECT_NAME = "deepagents-code"


class _FlushableTracerProvider(Protocol):
    """Subset of the OpenTelemetry provider used by this optional integration."""

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        """Export ended spans that are still queued."""
        ...


_provider: _FlushableTracerProvider | None = None


def flush_phoenix_tracing(*, timeout_millis: int) -> bool:
    """Flush queued Phoenix spans before the ephemeral agent server exits.

    Args:
        timeout_millis: Maximum time to wait for the batch exporter.

    Returns:
        `True` when tracing is inactive or queued spans were exported before
        the timeout, otherwise `False`.
    """
    if _provider is None:
        return True
    return _provider.force_flush(timeout_millis=timeout_millis)


def configure_phoenix_tracing() -> bool:
    """Configure Phoenix export and LangChain instrumentation when opted in.

    Phoenix is initialized in the agent server process, where model and tool
    calls run. Keeping the imports inside the enabled branch preserves startup
    performance and lets the integration remain an optional dependency.

    Returns:
        `True` when Phoenix tracing is configured, otherwise `False`.

    Raises:
        RuntimeError: If tracing is enabled without the Phoenix extra installed.
    """
    global _provider  # noqa: PLW0603

    if not is_env_truthy(PHOENIX_TRACING):
        return False
    if _provider is not None:
        return True

    # Never discover `.env.phoenix` from the working tree. A cloned repository
    # must not be able to redirect prompts, responses, or tool arguments to its
    # own collector. Trusted shell/global-dotenv Phoenix variables still work.
    os.environ["PHOENIX_DISCOVER_CONFIG"] = "false"

    try:
        from openinference.instrumentation.langchain import (  # ty: ignore[unresolved-import]
            LangChainInstrumentor,
        )
        from phoenix.otel import register  # ty: ignore[unresolved-import]
    except ImportError as exc:
        msg = (
            "Phoenix tracing is enabled but its optional dependencies are not "
            "installed. Install `deepagents-code[phoenix]` and restart dcode."
        )
        raise RuntimeError(msg) from exc

    project = (
        os.environ.get("PHOENIX_PROJECT")
        or os.environ.get("PHOENIX_PROJECT_NAME")
        or _DEFAULT_PROJECT_NAME
    )
    provider = register(
        project_name=project,
        protocol="http/protobuf",
        batch=True,
        auto_instrument=False,
        verbose=False,
    )
    # Generic OTLP export only controls transport. Phoenix's structured
    # input/output views and Span Replay require OpenInference attributes, so
    # instrument LangChain explicitly instead of relying on LangSmith's OTEL
    # exporter or Phoenix auto-discovery.
    LangChainInstrumentor().instrument(tracer_provider=provider)
    _provider = provider
    return True
