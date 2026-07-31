"""Session-scoped client facade for the Hooks v2 runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import (  # noqa: TC003 - used in runtime fields and path joins
    Path,
)
from typing import TYPE_CHECKING

from deepagents_code.hooks.client import HookFulfillmentLedger
from deepagents_code.hooks.engine import HookEngine
from deepagents_code.hooks.loading import load_hooks_config
from deepagents_code.hooks.models.domain import (
    HookDecision,
    HookEvent,
    HookInvocation,
    SubagentStartEvent,
    SubagentStopEvent,
)
from deepagents_code.hooks.presenter import HookPresenter
from deepagents_code.hooks.snapshot import HooksSnapshot
from deepagents_code.hooks.transcript import TranscriptStore
from deepagents_code.model_config import DEFAULT_CONFIG_DIR
from deepagents_code.project_utils import ProjectContext

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.messages import BaseMessage

    from deepagents_code.hooks.loading import PluginHooksSource
    from deepagents_code.hooks.models.domain import HookDiagnostic
    from deepagents_code.json_types import JsonValue


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
    `HookEngine`. Server-owned lifecycle events reach this runtime through the
    interrupt fulfill path in `hooks.client`.
    """

    snapshot: HooksSnapshot
    transcripts: TranscriptStore
    engine: HookEngine
    cwd: Path
    workspace_trusted: bool
    """Trust decision resolved for `cwd` when this runtime was frozen.

    Scoped to `cwd` by construction: a runtime is never reused across working
    directories, so `HooksManager` discards it and re-resolves trust whenever the
    session moves.
    """

    project_hooks_loaded: bool
    presenter: HookPresenter
    fulfillments: HookFulfillmentLedger

    @classmethod
    def create(
        cls,
        *,
        cwd: Path,
        workspace_trusted: bool = False,
        config_dir: Path | None = None,
        transcript_root: Path | None = None,
        presenter: HookPresenter | None = None,
        plugin_sources: Sequence[tuple[PluginHooksSource, JsonValue]] = (),
        plugin_diagnostics: Sequence[HookDiagnostic] = (),
    ) -> HooksRuntime:
        """Load configuration once and freeze a session runtime.

        Args:
            cwd: Session working directory.
            workspace_trusted: Whether project-scoped hooks may be loaded for
                `cwd`, already resolved by the caller from `WorkspaceTrust`. The
                runtime treats it as fixed for its lifetime.
            config_dir: Alternate user config directory for tests.
            transcript_root: Alternate transcript store root for tests.
                Defaults to `~/.deepagents/transcripts` regardless of
                `config_dir` (project and test hook configs must not relocate
                the global transcript store).
            presenter: Shared user-facing presenter. A private one is created
                when omitted, so output is logged rather than surfaced.
            plugin_sources: Hook documents contributed by enabled plugins, which
                the caller discovers so the runtime stays independent of plugin
                state. Merged last, holding the least authority.
            plugin_diagnostics: Diagnostics the caller collected while
                discovering `plugin_sources`.

        Returns:
            A runtime ready to execute invocations for this session.
        """
        project_context = ProjectContext.from_user_cwd(cwd)
        project_root = project_context.project_root or project_context.user_cwd
        loaded = load_hooks_config(
            project_root=project_root,
            workspace_trusted=workspace_trusted,
            config_dir=config_dir,
            documents=plugin_sources,
            document_diagnostics=plugin_diagnostics,
        )
        snapshot = HooksSnapshot.from_config(
            loaded.config,
            groups=loaded.groups,
            diagnostics=loaded.diagnostics,
            snapshot_id=loaded.snapshot_id,
        )
        store = TranscriptStore(
            transcript_root
            if transcript_root is not None
            else DEFAULT_CONFIG_DIR / "transcripts"
        )
        engine = HookEngine(snapshot)
        return cls(
            snapshot=snapshot,
            transcripts=store,
            engine=engine,
            cwd=project_context.user_cwd,
            workspace_trusted=workspace_trusted,
            project_hooks_loaded=loaded.project_source_loaded,
            presenter=presenter if presenter is not None else HookPresenter(),
            fulfillments=HookFulfillmentLedger(),
        )

    @property
    def snapshot_id(self) -> str:
        """Canonical configuration hash for this session."""
        return self.snapshot.snapshot_id

    def configured_server_events(self) -> tuple[str, ...]:
        """Stable event names the server should emit for this session.

        Returns:
            Sorted HookEvent values that have configured server-owned handlers.
        """
        return tuple(
            sorted(event.value for event in self.snapshot.configured_server_events())
        )

    def configured_events(self) -> frozenset[HookEvent]:
        """Return every event with at least one configured handler.

        Returns:
            Immutable configured event set.
        """
        return self.snapshot.configured_events()

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

        Raises:
            PermissionError: If project handlers were loaded without workspace trust.
        """
        if self.project_hooks_loaded and not self.workspace_trusted:
            msg = "Project hooks cannot execute before workspace trust is granted"
            raise PermissionError(msg)
        prepared = self.prepare_invocation(invocation)
        return await self.engine.run(
            prepared.invocation,
            transcript_path=prepared.transcript_path,
            agent_transcript_path=prepared.agent_transcript_path,
            on_progress=self.presenter.update_progress,
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
