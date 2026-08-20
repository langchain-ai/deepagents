"""Optional Arize Phoenix tracing for the agent server process."""

from __future__ import annotations

import os

from deepagents_code._env_vars import PHOENIX_TRACING, is_env_truthy

_DEFAULT_PROJECT_NAME = "deepagents-code"
_provider: object | None = None


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
        # The bundled LangGraph server is an ephemeral child process. A batch
        # processor's default five-second export interval can outlive server
        # teardown, leaving only the early child spans in Phoenix while the
        # final model and root agent spans remain queued. Export completed spans
        # immediately so every normally completed dcode invocation is intact.
        batch=False,
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
