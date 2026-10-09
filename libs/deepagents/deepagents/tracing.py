"""Opt-in tracing for Python scripts launched by `LocalShellBackend`."""

from __future__ import annotations

import json
import logging
import os
import re
from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal

from langsmith import Client, tracing_context
from langsmith.run_helpers import get_current_run_tree
from langsmith.run_trees import RunTree
from langsmith.utils import tracing_is_enabled

if TYPE_CHECKING:
    from collections.abc import Iterator

_CONTEXT_ENV = "DEEPAGENTS_TRACE_CONTEXT"
_MAX_CONTEXT_LENGTH = 16_384


def _execution_environment(env: dict[str, str]) -> dict[str, str]:
    """Copy the execution environment with allowlisted, invocation-local context."""
    result = env.copy()
    enabled = tracing_is_enabled()
    parent = get_current_run_tree() if enabled else None
    if enabled and (runnable_parent := RunTree.from_runnable_config(None)) and (parent is None or runnable_parent.dotted_order > parent.dotted_order):
        parent = runnable_parent
    payload: dict[str, str | bool] = {"enabled": False}
    if parent is not None and parent.session_name:
        payload = {"parent": parent.dotted_order, "project": parent.session_name, "enabled": enabled}
    encoded = json.dumps(payload)
    result[_CONTEXT_ENV] = encoded if len(encoded) <= _MAX_CONTEXT_LENGTH else '{"enabled": false}'
    return result


def _read_execution_context(raw: str) -> tuple[str | None, str | None, bool | Literal["local"]]:
    """Validate a bounded context envelope without accepting routing or auth data."""
    msg = "Invalid Deep Agents execution trace context."
    if len(raw) > _MAX_CONTEXT_LENGTH:
        raise ValueError(msg)
    try:
        payload = json.loads(raw)
    except (ValueError, RecursionError):
        raise ValueError(msg) from None
    if isinstance(payload, dict) and set(payload) == {"enabled"} and payload["enabled"] is False:
        return None, None, False
    if not isinstance(payload, dict) or set(payload) != {"parent", "project", "enabled"}:
        raise ValueError(msg)
    parent, project, enabled = payload["parent"], payload["project"], payload["enabled"]
    if not isinstance(parent, str) or not parent or not isinstance(project, str) or not project:
        raise ValueError(msg)
    segment = r"\d{8}T\d{12}Z[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}"
    if re.fullmatch(rf"{segment}(?:\.{segment})*", parent) is None or (enabled is not True and enabled != "local"):
        raise ValueError(msg)
    return parent, project, enabled


@contextmanager
def execution_tracing(*, client: Client | None = None) -> Iterator[None]:
    """Connect instrumented script operations to their originating execution.

    !!! warning "Experimental"
        Requires `LocalShellBackend(propagate_trace_context=True)`. This does
        not instrument arbitrary code or supply credentials. Configure a child
        client or explicit environment credentials for the parent's workspace.
        Inputs and outputs of instrumented operations may contain sensitive data;
        configure redaction on the client before using generated scripts.

    Args:
        client: Child-side LangSmith client; otherwise created from the child's
            environment. No parent credentials or endpoint are forwarded.

    Raises:
        ValueError: If an execution context exists but is malformed.

    Examples:
        ```python
        from deepagents.tracing import execution_tracing
        from langsmith import traceable


        @traceable
        def count_records(records):
            return len(records)


        with execution_tracing():
            count_records([1, 2, 3])
        ```

    Without an execution context this is a no-op. Disabled and local-only parent
    tracing remain disabled and local-only, respectively. Uploaded spans are
    flushed on exit, including errors; forceful process termination may lose
    pending spans. Only instrumented operations become spans, not every line.
    """
    raw = os.environ.get(_CONTEXT_ENV)
    if raw is None:
        yield
        return
    parent, project, enabled = _read_execution_context(raw)
    with tracing_context(enabled=False):
        tracing_client = client if client is not None else Client() if enabled else None
        try:
            parent_run = RunTree.from_dotted_order(parent, project_name=project, client=tracing_client) if parent else False
        except ValueError:
            msg = "Invalid Deep Agents execution trace context."
            raise ValueError(msg) from None
    try:
        with tracing_context(parent=parent_run, project_name=project, enabled=enabled, client=tracing_client):
            yield
    finally:
        if tracing_client is not None and enabled is True:
            try:
                tracing_client.flush(timeout=5)
            except Exception:  # noqa: BLE001  # Telemetry delivery must not mask script errors.
                logging.getLogger(__name__).warning("Could not flush execution traces.")


__all__ = ["execution_tracing"]
