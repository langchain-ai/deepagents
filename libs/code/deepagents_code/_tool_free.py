"""Remove executable tool configuration from isolated model requests."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

_TOOL_OPTIONS = frozenset(
    {
        "tools",
        "tool_choice",
        "functions",
        "function_call",
        "parallel_tool_calls",
        "mcp_servers",
    }
)
_OPTION_CONTAINERS = ("model_kwargs", "extra_body")


def _tool_free_options[K](options: Mapping[K, object]) -> dict[K, object]:
    """Copy request defaults, including nested provider payload overrides.

    Returns:
        Options with tool configuration removed.
    """
    return {
        key: (
            _tool_free_options(value)
            if key in _OPTION_CONTAINERS and isinstance(value, Mapping)
            else deepcopy(value)
        )
        for key, value in options.items()
        if key not in _TOOL_OPTIONS
    }


def tool_free_model(model: BaseChatModel) -> BaseChatModel:
    """Isolate request defaults while sharing the provider's HTTP clients.

    Args:
        model: Unbound chat model whose defaults should be isolated.

    Returns:
        A model copy with tool-free provider defaults.
    """
    updates: dict[str, object] = {
        key: _tool_free_options(value)
        for key in _OPTION_CONTAINERS
        if isinstance(value := getattr(model, key, None), Mapping)
    }
    if hasattr(model, "mcp_servers"):
        updates["mcp_servers"] = None
    return model.model_copy(update=updates)


def tool_free_settings(
    model: object, settings: Mapping[str, object]
) -> dict[str, object]:
    """Flatten bound defaults with request overrides taking precedence.

    Args:
        model: Chat model, optionally wrapped in request bindings.
        settings: Request overrides to apply after bound defaults.

    Returns:
        Isolated generation settings without tool configuration.
    """
    from langchain_core.runnables import RunnableBinding

    merged = dict(settings)
    while isinstance(model, RunnableBinding):
        merged = {**model.kwargs, **merged}
        model = model.bound
    return _tool_free_options(merged)
