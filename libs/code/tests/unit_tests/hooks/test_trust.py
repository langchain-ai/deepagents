"""Tests for project-hook workspace trust."""

from __future__ import annotations

import json
import os
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
from deepagents_code.hooks.trust import is_project_hooks_trusted, trust_project_hooks

if TYPE_CHECKING:
    from pathlib import Path


def _write_project_hooks(root: Path) -> Path:
    (root / ".git").mkdir(parents=True)
    hooks_dir = root / ".deepagents"
    hooks_dir.mkdir()
    (hooks_dir / "hooks.json").write_text(
        json.dumps(
            {"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "true"}]}]}}
        ),
        encoding="utf-8",
    )
    return root


def test_trust_persists_under_canonical_key(tmp_path: Path) -> None:
    root = _write_project_hooks(tmp_path / "project")
    store = tmp_path / "state" / "hooks_trust.json"

    assert trust_project_hooks(root / ".", store_path=store)
    assert is_project_hooks_trusted(root, store_path=store)
    assert not is_project_hooks_trusted(tmp_path / "other", store_path=store)
    if os.name != "nt":
        assert (store.stat().st_mode & 0o777) == 0o600


def test_corrupt_store_fails_closed_without_overwrite(tmp_path: Path) -> None:
    root = _write_project_hooks(tmp_path / "project")
    store = tmp_path / "hooks_trust.json"
    store.write_text("{invalid", encoding="utf-8")

    assert not is_project_hooks_trusted(root, store_path=store)
    assert not trust_project_hooks(root, store_path=store)
    assert store.read_text(encoding="utf-8") == "{invalid"


def test_runtime_loads_project_hooks_only_when_trusted(tmp_path: Path) -> None:
    cwd = _write_project_hooks(tmp_path / "project") / "src"
    cwd.mkdir()
    config_dir = tmp_path / "user"
    config_dir.mkdir()

    def _create(*, trusted: bool) -> HooksRuntime:
        return HooksRuntime.create(
            cwd=cwd,
            workspace_trusted=trusted,
            config_dir=config_dir,
            transcript_root=tmp_path / f"transcripts-{trusted}",
        )

    trusted = _create(trusted=True)
    untrusted = _create(trusted=False)

    assert trusted.project_hooks_loaded
    assert HookEvent.STOP in trusted.configured_events()
    assert not untrusted.project_hooks_loaded
    assert HookEvent.STOP not in untrusted.configured_events()


async def test_runtime_refuses_loaded_project_hooks_without_trust(
    tmp_path: Path,
) -> None:
    root = _write_project_hooks(tmp_path / "project")
    runtime = replace(
        HooksRuntime.create(
            cwd=root,
            workspace_trusted=True,
            config_dir=tmp_path / "user",
            transcript_root=tmp_path / "transcripts",
        ),
        workspace_trusted=False,
    )
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


@pytest.mark.parametrize(
    ("action", "allowed", "persisted"),
    [("REMEMBER", True, True), ("ALLOW_ONCE", True, False), ("DENY", False, False)],
)
def test_interactive_prompt_applies_selected_action(
    action: str,
    allowed: bool,
    persisted: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepagents_code.hooks import trust
    from deepagents_code.main import _check_project_hooks_trust, _ProjectMcpTrustAction

    root = _write_project_hooks(tmp_path / "project")
    store = tmp_path / "state" / "hooks_trust.json"
    monkeypatch.chdir(root)
    monkeypatch.setattr(trust, "_default_store_path", lambda: store)
    monkeypatch.setattr(
        "deepagents_code.main._select_project_mcp_trust_action",
        lambda _console, **_kwargs: _ProjectMcpTrustAction[action],
    )

    assert _check_project_hooks_trust() is allowed
    assert is_project_hooks_trusted(root, store_path=store) is persisted


def test_persisted_trust_skips_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepagents_code.hooks import trust
    from deepagents_code.main import _check_project_hooks_trust

    root = _write_project_hooks(tmp_path / "project")
    store = tmp_path / "state" / "hooks_trust.json"
    monkeypatch.chdir(root)
    monkeypatch.setattr(trust, "_default_store_path", lambda: store)
    assert trust_project_hooks(root, store_path=store)
    monkeypatch.setattr(
        "deepagents_code.main._select_project_mcp_trust_action",
        lambda *_args, **_kwargs: pytest.fail("prompt ran despite persisted trust"),
    )

    assert _check_project_hooks_trust() is True


async def test_textual_app_forwards_project_trust() -> None:
    from deepagents_code.app import DeepAgentsApp

    app = DeepAgentsApp(agent=MagicMock(), thread_id="thread", trust_project_hooks=True)
    with patch(
        "deepagents_code.hooks.runtime.HooksRuntime.create",
        return_value=MagicMock(),
    ) as create:
        await app._init_session_state()

    assert create.call_args.kwargs["workspace_trusted"] is True
