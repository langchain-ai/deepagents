"""Trace labels that mark interrupt-resume rounds within a turn.

Grouping itself comes from callers reusing one turn's stream config, so every
round carries the same `thread_id`/`turn_id`/`turn_number` (see
`build_stream_config`). This module only adds the marker that tells a resume
round apart from the initial run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

# Consumed by LangSmith, not by this repository: saved views and cost reports
# filter on this literal to fold a turn's sibling root runs back together.
# Nothing in-tree reads it, so treat the string as an external contract --
# renaming it silently breaks those saved filters. Shares the `dcode:` tag
# namespace with `dcode:auto` (see `deepagents_code.auto_mode`).
RESUME_TRACE_TAG = "dcode:resume"


def stream_trace_config(config: RunnableConfig, stream_input: Any) -> RunnableConfig:  # noqa: ANN401
    """Mark a graph invocation as an interrupt-resume round when it is one.

    A turn's initial run stays untagged, so the tag identifies continuations
    rather than the turn itself. Keep that asymmetry: tagging unconditionally
    would erase the distinction this exists to draw.

    LangGraph treats `tags` as inheritable and unions them onto every child
    config, so the tag also reaches the model, tool, and subagent runs beneath a
    resume. Pair it with an `is_root` filter to select resume roots alone.

    Args:
        config: Runnable config for this graph invocation. Callers pass the same
            object across a turn's rounds so the rounds share trace metadata.
            The `/offload` path passes a thread-only config that carries no
            `metadata`, so its rounds group by `thread_id` alone.
        stream_input: Graph input for this round. A `Command` marks a resume;
            anything else starts a run.

    Returns:
        `config` itself for an initial run. For a resume, a shallow copy whose
            `tags` carries `RESUME_TRACE_TAG` exactly once. The copy shares
            `metadata` and `configurable` with `config` by reference, so do not
            mutate those through the result.
    """
    from langgraph.types import Command

    if not isinstance(stream_input, Command):
        return config
    tags = list(config.get("tags") or ())
    if RESUME_TRACE_TAG not in tags:
        tags.append(RESUME_TRACE_TAG)
    updated = config.copy()
    updated["tags"] = tags
    return updated
