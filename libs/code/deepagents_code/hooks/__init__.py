"""Hook contracts and compatibility dispatch."""

from deepagents_code.hooks.legacy import (
    HOOK_SUBPROCESS_TIMEOUT,
    HOOK_TOOL_OUTPUT_LIMIT,
    _background_tasks,
    _dispatch_hook_sync as _dispatch_hook_sync,
    _load_hooks as _load_hooks,
    dispatch_hook,
    dispatch_hook_fire_and_forget,
    drain_pending_hooks,
    has_pending_hooks,
    subprocess as subprocess,
)

__all__ = [
    "HOOK_SUBPROCESS_TIMEOUT",
    "HOOK_TOOL_OUTPUT_LIMIT",
    "_background_tasks",
    "dispatch_hook",
    "dispatch_hook_fire_and_forget",
    "drain_pending_hooks",
    "has_pending_hooks",
]
