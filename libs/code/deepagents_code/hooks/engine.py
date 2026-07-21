"""Standalone orchestration for the Hooks v2 execution engine."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from deepagents_code.hooks.capabilities import get_event_spec
from deepagents_code.hooks.models.domain import HookDiagnostic
from deepagents_code.hooks.projection import (
    projection_diagnostics,
    serialize_hook_input,
)
from deepagents_code.hooks.reducer import reduce_hook_results
from deepagents_code.hooks.runner import (
    MAX_HOOK_OUTPUT_BYTES,
    run_command_handler,
)

if TYPE_CHECKING:
    from pathlib import Path

    from deepagents_code.hooks.models.domain import HookDecision, HookInvocation
    from deepagents_code.hooks.snapshot import HooksSnapshot


@dataclass(frozen=True, slots=True)
class HookEngine:
    """Execute Hooks v2 invocations against one immutable snapshot."""

    snapshot: HooksSnapshot
    default_timeout: float | None = None
    max_output_bytes: int = MAX_HOOK_OUTPUT_BYTES

    async def run(
        self,
        invocation: HookInvocation,
        *,
        transcript_path: Path | None = None,
        agent_transcript_path: Path | None = None,
    ) -> HookDecision:
        """Execute matching handlers and return a normalized decision.

        Matching handlers run concurrently with independent timeouts. Results
        are reduced in stable configuration order, independent of completion
        order.

        Args:
            invocation: Native lifecycle invocation.
            transcript_path: Materialized client transcript path.
            agent_transcript_path: Materialized subagent transcript path.

        Returns:
            The event-specific decision produced by ordered hook reduction.
        """
        match = self.snapshot.match(invocation)
        projected_diagnostics = projection_diagnostics(invocation)
        try:
            payload = serialize_hook_input(
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
            return reduce_hook_results(
                invocation,
                (),
                diagnostics=(
                    *self.snapshot.diagnostics,
                    *match.diagnostics,
                    *projected_diagnostics,
                    diagnostic,
                ),
            )

        event = invocation.event.event
        event_default = (
            self.default_timeout
            if self.default_timeout is not None
            else get_event_spec(event).default_timeout_seconds
        )
        results = await asyncio.gather(
            *(
                run_command_handler(
                    handler,
                    payload,
                    cwd=invocation.context.cwd,
                    event=event,
                    default_timeout=event_default,
                    max_output_bytes=self.max_output_bytes,
                )
                for handler in match.handlers
            )
        )
        return reduce_hook_results(
            invocation,
            results,
            diagnostics=(
                *self.snapshot.diagnostics,
                *match.diagnostics,
                *projected_diagnostics,
            ),
        )
