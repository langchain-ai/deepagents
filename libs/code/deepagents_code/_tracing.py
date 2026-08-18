"""Helpers for grouping Deep Agents Code runs in traces."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

RESUME_TRACE_TAG = "dcode:resume"


def resume_trace_config(config: RunnableConfig) -> RunnableConfig:
    """Tag a resume round without mutating its turn-level trace metadata.

    Args:
        config: Stream config shared by every graph invocation in a user turn.

    Returns:
        A shallow config copy carrying the idempotent resume tag.
    """
    tags = list(config.get("tags") or ())
    if RESUME_TRACE_TAG not in tags:
        tags.append(RESUME_TRACE_TAG)
    updated = config.copy()
    updated["tags"] = tags
    return updated
