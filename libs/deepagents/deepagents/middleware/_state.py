"""Helpers for working with Deep Agents state schemas."""

from __future__ import annotations

import contextlib
from typing import Annotated, get_args, get_origin, get_type_hints

from langchain.agents.middleware.types import PrivateStateAttr


def private_state_field_names(*state_schemas: type[object]) -> frozenset[str]:
    """Return fields annotated with `PrivateStateAttr` across state schemas."""
    names: set[str] = set()
    for state_schema in state_schemas:
        with contextlib.suppress(Exception):
            hints = get_type_hints(state_schema, include_extras=True)
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
