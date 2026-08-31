"""Helpers for attaching Hooks v2 session identity to graph context."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deepagents_code._cli_context import CLIContext
    from deepagents_code.hooks.runtime import HooksRuntime


def apply_hooks_context(
    context: CLIContext,
    runtime: HooksRuntime | None,
    *,
    prompt_id: str | None = None,
) -> CLIContext:
    """Attach Hooks v2 snapshot identity and server event gates to `context`.

    Args:
        context: Mutable per-run graph context.
        runtime: Session Hooks runtime, or `None` when hooks are unavailable.
        prompt_id: Optional per-turn prompt id.

    Returns:
        The same context mapping, updated in place.
    """
    if runtime is None:
        context.pop("hooks_snapshot_id", None)
        context.pop("hooks_server_events", None)
    else:
        context["hooks_snapshot_id"] = runtime.snapshot_id
        context["hooks_server_events"] = list(runtime.configured_server_events())
    if prompt_id is not None:
        context["prompt_id"] = prompt_id
    else:
        context.pop("prompt_id", None)
    return context
