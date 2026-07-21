"""Session-scoped client facade for the Hooks v2 runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import (  # noqa: TC003 - used in runtime fields and path joins
    Path,
)
from typing import TYPE_CHECKING

from deepagents_code.hooks.engine import HookEngine
from deepagents_code.hooks.loading import load_hooks_config
from deepagents_code.hooks.models.domain import (
    HookDecision,
    HookInvocation,
    SubagentStartEvent,
    SubagentStopEvent,
)
from deepagents_code.hooks.snapshot import HooksSnapshot
from deepagents_code.hooks.transcript import TranscriptStore

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.messages import BaseMessage


@dataclass(frozen=True, slots=True)
class PreparedHookInvocation:
    """Client-only materialization needed to build one hook wire envelope."""

    invocation: HookInvocation
    transcript_path: Path
    transcript_revision: str
    agent_transcript_path: Path | None = None
    agent_transcript_revision: str | None = None


@dataclass(frozen=True, slots=True)
class HooksRuntime:
    """Client-owned session runtime around an immutable Hooks snapshot.

    Owns configuration snapshot identity, transcript materialization, and the
    `HookEngine`. Lifecycle call sites are intentionally not wired here.
    """

    snapshot: HooksSnapshot
    transcripts: TranscriptStore
    engine: HookEngine
    cwd: Path

    @classmethod
    def create(
        cls,
        *,
        cwd: Path,
        config_dir: Path | None = None,
        transcript_root: Path | None = None,
    ) -> HooksRuntime:
        """Load configuration once and freeze a session runtime.

        Args:
            cwd: Session working directory.
            config_dir: Alternate user config directory for tests.
            transcript_root: Alternate transcript store root. Defaults to
                `{cwd}/.deepagents/transcripts`.

        Returns:
            A runtime ready to execute invocations for this session.
        """
        loaded = load_hooks_config(cwd=cwd, config_dir=config_dir)
        snapshot = HooksSnapshot.from_config(
            loaded.config,
            diagnostics=loaded.diagnostics,
            snapshot_id=loaded.snapshot_id,
        )
        store = TranscriptStore(
            transcript_root or (cwd / ".deepagents" / "transcripts")
        )
        engine = HookEngine(snapshot)
        return cls(snapshot=snapshot, transcripts=store, engine=engine, cwd=cwd)

    @property
    def snapshot_id(self) -> str:
        """Canonical configuration hash for this session."""
        return self.snapshot.snapshot_id

    def append_messages(
        self,
        thread_id: str,
        messages: Sequence[BaseMessage],
        *,
        agent_id: str | None = None,
    ) -> None:
        """Buffer conversation messages into the client transcript store.

        Args:
            thread_id: Conversation thread identifier.
            messages: LangChain messages to project.
            agent_id: Optional subagent scope.
        """
        self.transcripts.append_messages(thread_id, messages, agent_id=agent_id)

    async def invoke(self, invocation: HookInvocation) -> HookDecision:
        """Materialize transcripts, execute matching handlers, and return a decision.

        Args:
            invocation: Domain lifecycle invocation.

        Returns:
            Event-specific decision with notices, sequences, and diagnostics.
        """
        prepared = self.prepare_invocation(invocation)
        return await self.engine.run(
            prepared.invocation,
            transcript_path=prepared.transcript_path,
            agent_transcript_path=prepared.agent_transcript_path,
        )

    def prepare_invocation(
        self,
        invocation: HookInvocation,
    ) -> PreparedHookInvocation:
        """Materialize client-only transcript paths and revision identity.

        Args:
            invocation: Domain lifecycle invocation.

        Returns:
            A prepared value kept outside domain and graph state.
        """
        context = invocation.context
        thread_handle = self.transcripts.materialize(context.thread_id)
        agent_id: str | None = None
        if isinstance(invocation.event, SubagentStartEvent | SubagentStopEvent):
            agent_id = invocation.event.agent.id
        elif context.agent is not None:
            agent_id = context.agent.id

        agent_path: Path | None = None
        agent_revision: str | None = None
        if agent_id is not None:
            agent_handle = self.transcripts.materialize(
                context.thread_id,
                agent_id=agent_id,
            )
            agent_path = agent_handle.path
            agent_revision = agent_handle.revision

        return PreparedHookInvocation(
            invocation=invocation,
            transcript_path=thread_handle.path,
            transcript_revision=thread_handle.revision,
            agent_transcript_path=agent_path,
            agent_transcript_revision=agent_revision,
        )
