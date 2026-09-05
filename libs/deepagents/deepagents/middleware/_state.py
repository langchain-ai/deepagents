"""Helpers for working with Deep Agents state schemas."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated, get_args, get_origin, get_type_hints

from langchain.agents.middleware.types import PrivateStateAttr

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any


_EXCLUDED_STATE_KEYS = {
    "messages",
    "todos",
    "structured_response",
    "async_tasks",
}
"""State keys that are excluded when passing state to subagents and when
returning updates from subagents.

When returning updates:

1. The messages key is handled explicitly to ensure only the final message
    is included
2. The todos and `structured_response` keys are excluded as they do not have
    a defined reducer and no clear meaning for returning them from a subagent
    to the main agent.
3. Agent-private fields on middleware state schemas are excluded from both
    subagent output and subagent inputs.
"""


def prepare_subagent_state(
    state: Mapping[str, Any],
    *,
    private_state_keys: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Copy the public portion of parent state for a child subagent.

    Callers must supply the child-specific `messages` value after this helper
    returns. This allows synchronous and remote subagents to use their native
    message representations while sharing the same visibility policy.
    """
    excluded = _EXCLUDED_STATE_KEYS | private_state_keys
    return {key: value for key, value in state.items() if key not in excluded}


logger = logging.getLogger(__name__)


def private_state_field_names(*state_schemas: type[object]) -> frozenset[str]:
    """Return fields annotated with `PrivateStateAttr` across state schemas.

    Annotations are resolved at runtime, so a schema whose `PrivateStateAttr`
    annotation references a `TYPE_CHECKING`-only name cannot be inspected. That
    schema is skipped with a warning rather than failing the whole agent, because
    the caller may own several unrelated schemas -- but the warning matters: a
    skipped schema keeps none of its private fields, so they will be forwarded to
    and merged back from subagents.
    """
    names: set[str] = set()
    for state_schema in state_schemas:
        try:
            hints = get_type_hints(state_schema, include_extras=True)
        except (NameError, TypeError, AttributeError):
            logger.warning(
                "Could not resolve annotations for state schema %s; its "
                "PrivateStateAttr fields will NOT be kept private. Ensure every "
                "name used in those annotations is imported at runtime rather "
                "than only under TYPE_CHECKING.",
                getattr(state_schema, "__qualname__", state_schema),
            )
            continue
        for name, annotation in hints.items():
            if _has_marker(annotation, PrivateStateAttr):
                names.add(name)
    return frozenset(names)


def _has_marker(annotation: object, marker: object) -> bool:
    origin = get_origin(annotation)
    if origin is Annotated:
        args = get_args(annotation)
        return any(meta is marker for meta in args[1:])
    if origin is not None:
        return any(_has_marker(arg, marker) for arg in get_args(annotation))
    return False
