"""Lightweight runtime context type for app model overrides.

Extracted from `configurable_model` so hot-path modules (`app`,
`textual_adapter`) can import `CLIContext` without pulling in the langchain
middleware stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from typing_extensions import TypedDict

if TYPE_CHECKING:
    from collections.abc import Callable


class CLIContext(TypedDict, total=False):
    """Runtime context passed via `context=` to the LangGraph graph.

    Carries per-invocation overrides that `ConfigurableModelMiddleware`
    reads from `request.runtime.context`.
    """

    model: str | None
    """Model spec to swap at runtime (e.g. `'provider:model'`)."""

    model_params: dict[str, Any]
    """Invocation params (e.g. `temperature`, `max_tokens`) to merge
    into `model_settings`."""

    summarization_model: str | None
    """Model spec used for context-compaction summary generation.

    Read by the SDK's `SummarizationMiddleware` (not by
    `ConfigurableModelMiddleware`) to resolve the summarizer model per
    request without mutating shared middleware state. `None` falls back to
    `model` and, if that is also unset, the construction-time model.
    """

    summarization_model_params: dict[str, Any]
    """Optional kwargs forwarded to the summarization model factory.

    Reserved for future use; currently the summarization model is created
    with the same defaults as the main model.
    """

    model_resolver: Callable[[str], Any]
    """Callable that turns a `provider:model` spec into a `BaseChatModel`.

    Read by the SDK's `SummarizationMiddleware` when it encounters a string
    spec in `model` or `summarization_model`. Injected per-turn by the
    app so the SDK stays free of any CLI imports.
    """
