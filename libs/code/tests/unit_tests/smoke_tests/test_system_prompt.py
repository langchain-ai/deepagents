"""Snapshot test for the CLI agent system prompt.

Snapshots the full system message the model receives on first invocation,
including middleware-injected sections (local context, memory, skills).
Machine-specific values (cwd, local-context detection output) are fixed
so the golden file is reproducible across machines.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from deepagents_code.agent import create_cli_agent
from deepagents_code.config import Settings

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence
    from pathlib import Path

    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.language_models import LanguageModelInput
    from langchain_core.outputs import ChatResult
    from langchain_core.runnables import Runnable
    from langchain_core.tools import BaseTool

# Fixed values so the snapshot is reproducible.
_FIXED_CWD = "/home/user/project"
_FIXED_MODEL_NAME = "claude-sonnet-4-20250514"
_FIXED_MODEL_PROVIDER = "anthropic"
_FIXED_CONTEXT_LIMIT = 200_000
# Mimics the markdown output of the local-context detection script.
_FIXED_LOCAL_CONTEXT = """## Local Context

**Current Directory**: `/home/user/project`

**Git**: branch `main`, 2 uncommitted changes

**Project**: python (uv), monorepo

**Runtimes**: Python 3.13.1, Node 24.14.0"""


class _SnapshotChatModel(GenericFakeChatModel):
    """Fake model that captures the first call's messages for snapshotting.

    Unlike the FixedGenericFakeChatModel in test_end_to_end.py, this model
    stores the system message separately so the snapshot test can access
    it without parsing captured_calls tuples.
    """

    captured_system_messages: list[SystemMessage] = Field(default_factory=list)

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],  # noqa: ARG002
        *,
        tool_choice: str | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not self.captured_system_messages:
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    self.captured_system_messages.append(msg)
                    break
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def _assert_snapshot(
    snapshot_path: Path, actual: str, *, update_snapshots: bool
) -> None:
    if update_snapshots or not snapshot_path.exists():
        snapshot_path.write_text(actual, encoding="utf-8")
        if update_snapshots:
            return
        msg = f"Created snapshot at {snapshot_path}. Re-run tests."
        raise AssertionError(msg)

    expected = snapshot_path.read_text(encoding="utf-8")
    assert actual == expected


@contextmanager
def _mock_settings(tmp_path: Path) -> Generator[None, None, None]:
    """Patch ``settings`` with temporary directories and fixed model identity.

    Mirrors the ``mock_settings`` pattern from ``test_end_to_end.py`` but
    fixes ``model_name``/``model_provider``/``model_context_limit`` so the
    model-identity section in the prompt is reproducible.
    """
    agent_dir = tmp_path / "agents" / "agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / "AGENTS.md").touch()

    skills_dir = tmp_path / "skills"
    skills_dir.mkdir(parents=True)

    with patch("deepagents_code.agent.settings") as mock_s:
        mock_s.ensure_agent_dir.return_value = agent_dir
        mock_s.ensure_user_skills_dir.return_value = skills_dir
        mock_s.get_project_skills_dir.return_value = None
        mock_s.get_built_in_skills_dir.return_value = Settings.get_built_in_skills_dir()
        mock_s.get_user_agent_md_path.return_value = agent_dir / "AGENTS.md"
        mock_s.get_project_agent_md_path.return_value = []
        mock_s.get_user_agents_dir.return_value = tmp_path / "agents"
        mock_s.get_project_agents_dir.return_value = None
        mock_s.get_user_agent_skills_dir.return_value = tmp_path / "agents_skills"
        mock_s.get_project_agent_skills_dir.return_value = None
        mock_s.get_user_claude_skills_dir.return_value = tmp_path / "claude_skills"
        mock_s.get_project_claude_skills_dir.return_value = None
        mock_s.model_name = _FIXED_MODEL_NAME
        mock_s.model_provider = _FIXED_MODEL_PROVIDER
        mock_s.model_context_limit = _FIXED_CONTEXT_LIMIT
        mock_s.model_unsupported_modalities = frozenset()
        mock_s.project_root = None
        mock_s.user_langchain_project = None
        mock_s.shell_allow_list = None

        # Patch generated backend roots so host-path mappings are deterministic
        # across machines and runs.
        def _fake_mkdtemp(prefix: str = "", **_kw: Any) -> str:
            return str(tmp_path / prefix.rstrip("_"))

        with (
            patch("deepagents_code.agent.tempfile.mkdtemp", _fake_mkdtemp),
            patch(
                "deepagents_code.agent._offload_fallback_root",
                return_value=tmp_path / ".deepagents",
            ),
        ):
            yield


def test_system_prompt_snapshot(
    tmp_path: Path,
    snapshots_dir: Path,
    *,
    update_snapshots: bool,
) -> None:
    """Snapshot the full interactive local-mode system prompt.

    The agent is created with default features (memory, skills, shell) in
    interactive local mode. A fake model captures the system message from
    the first model call. The local-context detection script output is
    mocked to keep the snapshot machine-independent.
    """
    model = _SnapshotChatModel(
        messages=iter([AIMessage(content="hello!") for _ in range(4)])
    )
    model.profile = {"max_input_tokens": _FIXED_CONTEXT_LIMIT}

    with _mock_settings(tmp_path):
        agent, _ = create_cli_agent(
            model=model,
            assistant_id="agent",
            checkpointer=InMemorySaver(),
            cwd=_FIXED_CWD,
        )

        # Mock the local-context detection script so the local-context
        # section in the prompt is reproducible.
        with patch(
            "deepagents_code.local_context.LocalContextMiddleware._run_detect_script",
            return_value=_FIXED_LOCAL_CONTEXT,
        ):
            try:
                agent.invoke(
                    {"messages": [HumanMessage(content="hi")]},
                    {"configurable": {"thread_id": str(uuid.uuid4())}},
                )
            except RuntimeError as exc:
                if "StopIteration" not in str(exc):
                    raise

    assert len(model.captured_system_messages) >= 1
    actual = str(model.captured_system_messages[0].text).rstrip("\n") + "\n"
    # Redact machine-specific paths so the snapshot is reproducible across
    # machines and CI checkouts with different repo roots.
    actual = actual.replace(str(tmp_path), "<tmp_path>")
    actual = actual.replace(
        str(Settings.get_built_in_skills_dir()), "<built_in_skills_dir>"
    )

    _assert_snapshot(
        snapshots_dir / "system_prompt_interactive_local.md",
        actual,
        update_snapshots=update_snapshots,
    )
