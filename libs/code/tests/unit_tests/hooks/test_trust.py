"""Tests for project-hook workspace trust."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from deepagents_code.approval_mode import ApprovalMode
from deepagents_code.hooks.models.domain import (
    HookContext,
    HookEvent,
    HookInvocation,
    StopEvent,
)
from deepagents_code.hooks.runtime import HooksRuntime
from deepagents_code.hooks.trust import (
    is_project_hooks_trusted,
    trust_project_hooks,
)

if TYPE_CHECKING:
    from pathlib import Path


def _project(root: Path) -> Path:
    root.mkdir()
    (root / ".git").mkdir()
    hooks_dir = root / ".deepagents"
    hooks_dir.mkdir()
    (hooks_dir / "hooks.json").write_text(
        '{"hooks":{"Stop":[{"hooks":[{"type":"command","command":"true"}]}]}}',
        encoding="utf-8",
    )
    return root


def test_trust_store_persists_canonical_project_root(tmp_path: Path) -> None:
    root = _project(tmp_path / "project")
    store = tmp_path / "state" / "hooks_trust.json"

    assert trust_project_hooks(root / ".", store_path=store)
    assert is_project_hooks_trusted(root, store_path=store)


def test_corrupt_trust_store_fails_closed_without_overwrite(tmp_path: Path) -> None:
    root = _project(tmp_path / "project")
    store = tmp_path / "hooks_trust.json"
    store.write_text("{invalid", encoding="utf-8")

    assert not is_project_hooks_trusted(root, store_path=store)
    assert not trust_project_hooks(root, store_path=store)
    assert store.read_text(encoding="utf-8") == "{invalid"


def test_runtime_discovers_project_hooks_from_repository_root(tmp_path: Path) -> None:
    root = _project(tmp_path / "project")
    cwd = root / "src"
    cwd.mkdir()
    config_dir = tmp_path / "user"
    config_dir.mkdir()

    trusted = HooksRuntime.create(
        cwd=cwd,
        workspace_trusted=True,
        config_dir=config_dir,
        transcript_root=tmp_path / "trusted-transcripts",
    )
    untrusted = HooksRuntime.create(
        cwd=cwd,
        workspace_trusted=False,
        config_dir=config_dir,
        transcript_root=tmp_path / "untrusted-transcripts",
    )

    assert HookEvent.STOP in trusted.configured_events()
    assert trusted.project_hooks_loaded
    assert HookEvent.STOP not in untrusted.configured_events()
    assert not untrusted.project_hooks_loaded


async def test_runtime_guard_blocks_loaded_project_hooks_without_trust(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path / "project")
    runtime = HooksRuntime.create(
        cwd=root,
        workspace_trusted=True,
        config_dir=tmp_path / "user",
        transcript_root=tmp_path / "transcripts",
    )
    runtime = replace(runtime, workspace_trusted=False)
    invocation = HookInvocation(
        context=HookContext(
            thread_id="thread",
            cwd=root,
            approval_mode=ApprovalMode.MANUAL,
        ),
        event=StopEvent(
            event=HookEvent.STOP,
            continuation_count=0,
            last_assistant_message="done",
        ),
    )

    with pytest.raises(PermissionError, match="workspace trust"):
        await runtime.invoke(invocation)


def test_interactive_prompt_remembers_project_hooks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepagents_code.hooks import trust
    from deepagents_code.main import (
        _check_project_hooks_trust,
        _ProjectMcpTrustAction,
    )

    root = _project(tmp_path / "project")
    store = tmp_path / "state" / "hooks_trust.json"
    monkeypatch.chdir(root)
    monkeypatch.setattr(trust, "_default_store_path", lambda: store)
    monkeypatch.setattr(
        "deepagents_code.main._select_project_mcp_trust_action",
        lambda _console: _ProjectMcpTrustAction.REMEMBER,
    )

    assert _check_project_hooks_trust() is True
    assert is_project_hooks_trusted(root, store_path=store)


def test_interactive_prompt_skips_untrusted_project_hooks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepagents_code.main import (
        _check_project_hooks_trust,
        _ProjectMcpTrustAction,
    )

    root = _project(tmp_path / "project")
    monkeypatch.chdir(root)
    monkeypatch.setattr(
        "deepagents_code.main._select_project_mcp_trust_action",
        lambda _console: _ProjectMcpTrustAction.DENY,
    )

    assert _check_project_hooks_trust() is False


async def test_textual_session_initialization_uses_project_trust() -> None:
    from deepagents_code.app import DeepAgentsApp

    app = DeepAgentsApp(
        agent=MagicMock(),
        thread_id="thread",
        trust_project_hooks=True,
    )
    runtime = MagicMock()
    with patch(
        "deepagents_code.hooks.runtime.HooksRuntime.create",
        return_value=runtime,
    ) as create:
        await app._init_session_state()

    assert app._session_state is not None
    assert app._session_state.hooks_runtime is runtime
    assert create.call_args.kwargs["workspace_trusted"] is True
