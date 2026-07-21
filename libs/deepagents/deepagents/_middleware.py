"""Shared middleware-composition helpers."""

from collections.abc import Sequence
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware


def apply_custom_middleware(
    base: list[AgentMiddleware[Any, Any, Any]],
    custom: Sequence[AgentMiddleware[Any, Any, Any]],
    *,
    core_names: set[str] | None = None,
) -> list[AgentMiddleware[Any, Any, Any]]:
    """Merge custom middleware into a base stack by name.

    A custom entry replaces a same-named base entry in place, preserving stack
    order. New entries are inserted after the final core middleware entry when
    ``core_names`` is supplied; otherwise, they are appended.
    """
    if not custom:
        return list(base)
    current_names = {middleware.name for middleware in base}
    replacements = {middleware.name: middleware for middleware in custom if middleware.name in current_names}
    to_append = [middleware for middleware in custom if middleware.name not in current_names]
    result = [replacements.get(middleware.name, middleware) for middleware in base]
    if to_append and core_names is not None:
        position = max((i for i, middleware in enumerate(result) if middleware.name in core_names), default=len(result) - 1) + 1
        result[position:position] = to_append
    else:
        result.extend(to_append)
    return result
