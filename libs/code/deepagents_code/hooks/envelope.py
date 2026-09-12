"""Canonical boundary between hook domain and wire models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents_code.hooks.projection import project_hook_input, serialize_hook_input
from deepagents_code.hooks.reducer import reduce_hook_results

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from deepagents_code.hooks.models.domain import (
        HookDecision,
        HookDiagnostic,
        HookInvocation,
    )
    from deepagents_code.hooks.models.wire import HookWireInput
    from deepagents_code.hooks.runner import HandlerResult


class HookEnvelopeAdapter:
    """Adapt hook invocations and handler output across the wire boundary."""

    @staticmethod
    def to_wire_input(
        invocation: HookInvocation,
        *,
        transcript_path: Path,
        agent_transcript_path: Path | None = None,
    ) -> HookWireInput:
        """Project a domain invocation into its validated wire model.

        Returns:
            Validated event-specific wire input.
        """
        return project_hook_input(
            invocation,
            transcript_path=transcript_path,
            agent_transcript_path=agent_transcript_path,
        )

    @staticmethod
    def serialize_input(
        invocation: HookInvocation,
        *,
        transcript_path: Path,
        agent_transcript_path: Path | None = None,
    ) -> bytes:
        """Serialize a domain invocation for command-handler stdin.

        Returns:
            Compact validated JSON bytes.
        """
        return serialize_hook_input(
            invocation,
            transcript_path=transcript_path,
            agent_transcript_path=agent_transcript_path,
        )

    @staticmethod
    def to_domain_decision(
        invocation: HookInvocation,
        results: Iterable[HandlerResult],
        *,
        diagnostics: Iterable[HookDiagnostic] = (),
    ) -> HookDecision:
        """Normalize ordered wire handler results into a domain decision.

        Returns:
            Event-specific normalized domain decision.
        """
        return reduce_hook_results(
            invocation,
            results,
            diagnostics=diagnostics,
        )
