"""Standalone orchestration for the Hooks v2 execution engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from deepagents_code.hooks.models.domain import HookDiagnostic
from deepagents_code.hooks.projection import serialize_hook_input
from deepagents_code.hooks.reducer import reduce_hook_results
from deepagents_code.hooks.runner import (
    DEFAULT_HOOK_TIMEOUT,
    MAX_HOOK_OUTPUT_BYTES,
    HandlerResult,
    run_command_handler,
)

if TYPE_CHECKING:
    from deepagents_code.hooks.models.domain import HookDecision, HookInvocation
    from deepagents_code.hooks.snapshot import HooksSnapshot


@dataclass(frozen=True, slots=True)
class HookEngine:
    """Execute Hooks v2 invocations against one immutable snapshot."""

    snapshot: HooksSnapshot
    default_timeout: float = DEFAULT_HOOK_TIMEOUT
    max_output_bytes: int = MAX_HOOK_OUTPUT_BYTES

    async def run(self, invocation: HookInvocation) -> HookDecision:
        """Execute matching handlers and return a normalized decision.

        Args:
            invocation: Native lifecycle invocation.

        Returns:
            The event-specific decision produced by ordered hook reduction.
        """
        match = self.snapshot.match(invocation)
        try:
            payload = serialize_hook_input(invocation)
        except (TypeError, ValueError) as exc:
            diagnostic = HookDiagnostic(
                code="projection_failed",
                severity="warning",
                message=f"Could not project hook invocation: {exc}",
            )
            return reduce_hook_results(
                invocation,
                (),
                diagnostics=(*match.diagnostics, diagnostic),
            )

        results: list[HandlerResult] = []
        for handler in match.handlers:
            result = await run_command_handler(
                handler,
                payload,
                cwd=invocation.context.cwd,
                default_timeout=self.default_timeout,
                max_output_bytes=self.max_output_bytes,
            )
            results.append(result)
            if result.output is not None and not result.output.continue_:
                break
        return reduce_hook_results(
            invocation,
            results,
            diagnostics=match.diagnostics,
        )
