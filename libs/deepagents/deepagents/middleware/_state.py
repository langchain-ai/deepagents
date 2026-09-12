"""Helpers for working with Deep Agents state schemas."""

from __future__ import annotations

import logging
from typing import Annotated, get_args, get_origin, get_type_hints

from langchain.agents.middleware.types import PrivateStateAttr

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
