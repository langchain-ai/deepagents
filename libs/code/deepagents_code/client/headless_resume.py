"""Resolve `dcode -r <id> -n ...` into a thread the headless run may continue.

Headless resume never falls back to a new thread. Threads stay bound to the
workspace policy they were created with, so only threads started headless from
the same directory with the same flags can be resumed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class ResumeRequest:
    """An explicit `-r <id>` resume request from the command line."""

    thread_id: str

    adopt_agent: bool
    """Use the thread's agent; `False` when `-a` was given."""

    adopt_model: bool
    """Use the thread's last model; `False` when `-M` was given."""


@dataclass(frozen=True)
class ResumeTarget:
    """A thread that exists and passes the resume-age policy."""

    thread_id: str
    agent_name: str | None
    cwd: str | None


class ResumeError(Exception):
    """A user-facing reason the requested thread cannot be resumed."""


WORKSPACE_CONFLICT_MESSAGE = (
    "Couldn't resume this thread because it was created with different "
    "settings. A thread keeps the directory, agent, approval, shell, sandbox, "
    "and MCP settings it started with, so threads from the interactive UI "
    "can't be resumed with -n. Rerun from the same directory with the same "
    "flags as the run that created it."
)


async def resolve_resume_target(thread_id: str) -> ResumeTarget:
    """Look up a thread to resume and enforce the resume-age policy.

    Args:
        thread_id: The exact thread ID given to `-r`.

    Returns:
        The thread's persisted agent and directory.

    Raises:
        ResumeError: The thread is missing or blocked by policy.
    """
    from deepagents_code.resume_policy import resume_cutoff, thread_resume_block
    from deepagents_code.sessions import (
        find_similar_threads,
        get_thread_agent,
        get_thread_cwd,
        thread_exists,
    )

    if not await thread_exists(thread_id):
        similar = await find_similar_threads(thread_id)
        msg = f"Thread '{thread_id}' not found."
        if similar:
            msg += f" Did you mean: {', '.join(similar)}?"
        raise ResumeError(msg)
    if blocked := await thread_resume_block(thread_id, resume_cutoff()):
        raise ResumeError(blocked)
    return ResumeTarget(
        thread_id=thread_id,
        agent_name=await get_thread_agent(thread_id),
        cwd=await get_thread_cwd(thread_id),
    )


def cwd_conflict(target: ResumeTarget, current_cwd: str) -> str | None:
    """Explain why `target` cannot be resumed from `current_cwd`.

    The server refuses a directory change through the thread's workspace
    binding; checking first turns that generic refusal into a specific message.

    Returns:
        A user-facing message, or `None` when the directories match.
    """
    from deepagents_code.resume_policy import classify_thread_cwd

    if not target.cwd:
        return None
    status, path = classify_thread_cwd(target.cwd, current_cwd)
    if status == "mismatch":
        return (
            f"Thread {target.thread_id} belongs to {path}. "
            "Run from that directory to resume it."
        )
    if status == "unavailable":
        return (
            f"Thread {target.thread_id} was last used in {target.cwd!r}, "
            "which is no longer available, so it can't be resumed."
        )
    return None


def persisted_model(
    values: Mapping[str, object],
) -> tuple[str, dict[str, Any] | None] | None:
    """Read the model a thread last ran with from its checkpoint values.

    Returns:
        The `(spec, params)` pair, or `None` when no model is recorded.
    """
    from deepagents_code.resume_state import coerce_model_spec

    spec = coerce_model_spec(values.get("_model_spec"))
    if spec is None:
        return None
    raw_params = values.get("_model_params")
    params = (
        dict(cast("dict[str, Any]", raw_params))
        if isinstance(raw_params, dict)
        else None
    )
    return spec, params
