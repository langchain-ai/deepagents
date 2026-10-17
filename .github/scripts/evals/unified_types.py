"""Shared identity types for the unified eval prep and aggregation scripts."""

from __future__ import annotations

from typing import NamedTuple


class LeafKey(NamedTuple):
    """Identity of one model, branch, config, and category leaf."""

    model: str
    branch: str
    config: str
    category: str


class RowKey(NamedTuple):
    """Identity of one scorecard row before its category leaves are attached."""

    model: str
    branch: str
    config: str
