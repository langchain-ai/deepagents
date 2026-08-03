"""Standalone orchestration for the Hooks v2 execution engine."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from deepagents_code.hooks.capabilities import get_event_spec
from deepagents_code.hooks.env import sanitize_hook_environ
from deepagents_code.hooks.envelope import HookEnvelopeAdapter
from deepagents_code.hooks.loading import PluginHooksSource
from deepagents_code.hooks.models.domain import HookDiagnostic
from deepagents_code.hooks.presenter import HookProgress
from deepagents_code.hooks.runner import (
    MAX_HOOK_OUTPUT_BYTES,
    HandlerResult,
    run_command_handler,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from deepagents_code.hooks.models.domain import HookDecision, HookInvocation
    from deepagents_code.hooks.snapshot import HookHandler, HooksSnapshot

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HookEngine:
    """Execute Hooks v2 invocations against one immutable snapshot."""

    snapshot: HooksSnapshot
    default_timeout: float | None = None
    max_output_bytes: int = MAX_HOOK_OUTPUT_BYTES
    adapter: HookEnvelopeAdapter = field(default_factory=HookEnvelopeAdapter)

    async def run(
        self,
        invocation: HookInvocation,
        *,
        transcript_path: Path,
        agent_transcript_path: Path | None = None,
        on_progress: Callable[[HookProgress], None] | None = None,
    ) -> HookDecision:
        """Execute matching handlers and return a normalized decision.

        Matching handlers run concurrently with independent timeouts. Results
        are reduced in stable configuration order, independent of completion
        order.

        The returned diagnostics are scoped to this invocation. Configuration
        diagnostics collected while the snapshot loaded belong to whoever owns
        the snapshot, which presents them once per load; repeating them here
        would re-surface the same warning on every hook that runs.

        Args:
            invocation: Native lifecycle invocation.
            transcript_path: Materialized client transcript path.
            agent_transcript_path: Materialized subagent transcript path.
            on_progress: Optional handler lifecycle callback.

        Returns:
            The event-specific decision produced by ordered hook reduction.
        """
        match = self.snapshot.match(invocation)
        try:
            payload = self.adapter.serialize_input(
                invocation,
                transcript_path=transcript_path,
                agent_transcript_path=agent_transcript_path,
            )
        except (TypeError, ValueError) as exc:
            diagnostic = HookDiagnostic(
                code="projection_failed",
                severity="warning",
                message=f"Could not project hook invocation: {exc}",
            )
            return self.adapter.to_domain_decision(
                invocation,
                (),
                diagnostics=(*match.diagnostics, diagnostic),
            )

        event = invocation.event.event
        event_default = (
            self.default_timeout
            if self.default_timeout is not None
            else get_event_spec(event).default_timeout_seconds
        )
        results = await asyncio.gather(
            *(
                _run_handler(
                    handler,
                    payload,
                    cwd=invocation.context.cwd,
                    default_timeout=event_default,
                    max_output_bytes=self.max_output_bytes,
                    operation_id=f"{id(invocation):x}:{handler.id}",
                    on_progress=on_progress,
                )
                for handler in match.handlers
            )
        )
        return self.adapter.to_domain_decision(
            invocation,
            results,
            diagnostics=match.diagnostics,
        )


async def _run_handler(
    handler: HookHandler,
    payload: bytes,
    *,
    cwd: Path,
    default_timeout: float,
    max_output_bytes: int,
    operation_id: str,
    on_progress: Callable[[HookProgress], None] | None,
) -> HandlerResult:
    message = (handler.status_message or "").strip()
    env = sanitize_hook_environ()
    if isinstance(handler.source, PluginHooksSource):
        env |= handler.source.env
    _report_progress(
        on_progress,
        HookProgress(
            operation_id=operation_id,
            handler_id=handler.id,
            event=handler.event,
            message=message,
            active=True,
        ),
    )
    try:
        return await run_command_handler(
            handler,
            payload,
            cwd=cwd,
            default_timeout=default_timeout,
            max_output_bytes=max_output_bytes,
            env=env,
        )
    finally:
        _report_progress(
            on_progress,
            HookProgress(
                operation_id=operation_id,
                handler_id=handler.id,
                event=handler.event,
                message=message,
                active=False,
            ),
        )


def _report_progress(
    callback: Callable[[HookProgress], None] | None,
    update: HookProgress,
) -> None:
    if callback is None:
        return
    try:
        callback(update)
    except Exception:
        logger.warning("Hook progress callback failed", exc_info=True)
