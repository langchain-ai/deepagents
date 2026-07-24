"""Tests for project-hook workspace trust."""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from deepagents_code.approval_mode import ApprovalMode
from deepagents_code.hooks.loading import load_hooks_config
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


def _write_project_hooks(root: Path, command: str = "true") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / ".git").mkdir(exist_ok=True)
    hooks_dir = root / ".deepagents"
    hooks_dir.mkdir(exist_ok=True)
    (hooks_dir / "hooks.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "Stop": [
                        {
                            "hooks": [
                                {"type": "command", "command": command},
                            ]
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    return root


def test_trust_store_persists_canonical_project_root(tmp_path: Path) -> None:
    root = _write_project_hooks(tmp_path / "project")
    store = tmp_path / "state" / "hooks_trust.json"

    assert trust_project_hooks(root / ".", store_path=store)
    assert is_project_hooks_trusted(root, store_path=store)


def test_symlink_alias_shares_persisted_trust(tmp_path: Path) -> None:
    root = _write_project_hooks(tmp_path / "project")
    alias = tmp_path / "alias"
    alias.symlink_to(root)
    store = tmp_path / "hooks_trust.json"

    assert trust_project_hooks(root, store_path=store)
    assert is_project_hooks_trusted(alias, store_path=store)


def test_linked_worktree_style_roots_are_trusted_independently(
    tmp_path: Path,
) -> None:
    primary = _write_project_hooks(tmp_path / "main-worktree")
    linked = _write_project_hooks(tmp_path / "linked-worktree")
    store = tmp_path / "hooks_trust.json"

    assert trust_project_hooks(primary, store_path=store)
    assert is_project_hooks_trusted(primary, store_path=store)
    assert not is_project_hooks_trusted(linked, store_path=store)


def test_corrupt_trust_store_fails_closed_without_overwrite(tmp_path: Path) -> None:
    root = _write_project_hooks(tmp_path / "project")
    store = tmp_path / "hooks_trust.json"
    store.write_text("{invalid", encoding="utf-8")

    assert not is_project_hooks_trusted(root, store_path=store)
    assert not trust_project_hooks(root, store_path=store)
    assert store.read_text(encoding="utf-8") == "{invalid"


def test_partially_invalid_trust_store_keeps_valid_entries(tmp_path: Path) -> None:
    root = _write_project_hooks(tmp_path / "project")
    other = _write_project_hooks(tmp_path / "other")
    store = tmp_path / "hooks_trust.json"
    store.write_text(
        json.dumps(
            {
                "version": 1,
                "projects": {
                    str(root.resolve()): {"trusted_at": "2026-01-01T00:00:00+00:00"},
                    str(other.resolve()): "not-an-object",
                    str(tmp_path / "missing-fields"): {"unexpected": True},
                },
                "future_field": {"ignored": True},
            }
        ),
        encoding="utf-8",
    )

    assert is_project_hooks_trusted(root, store_path=store)
    assert not is_project_hooks_trusted(other, store_path=store)
    assert not is_project_hooks_trusted(tmp_path / "missing-fields", store_path=store)

    third = _write_project_hooks(tmp_path / "third")
    assert trust_project_hooks(third, store_path=store)
    payload = json.loads(store.read_text(encoding="utf-8"))
    assert str(root.resolve()) in payload["projects"]
    assert str(third.resolve()) in payload["projects"]
    assert str(other.resolve()) not in payload["projects"]


def test_malformed_projects_field_refuses_overwrite(tmp_path: Path) -> None:
    root = _write_project_hooks(tmp_path / "project")
    store = tmp_path / "hooks_trust.json"
    original = json.dumps({"version": 1, "projects": ["not", "a", "mapping"]})
    store.write_text(original, encoding="utf-8")

    assert not is_project_hooks_trusted(root, store_path=store)
    assert not trust_project_hooks(root, store_path=store)
    assert store.read_text(encoding="utf-8") == original


def test_concurrent_writers_preserve_all_entries(tmp_path: Path) -> None:
    store = tmp_path / "hooks_trust.json"
    roots = [_write_project_hooks(tmp_path / f"project-{index}") for index in range(8)]

    def _trust(root: Path) -> bool:
        return trust_project_hooks(root, store_path=store)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_trust, roots))

    assert all(results)
    for root in roots:
        assert is_project_hooks_trusted(root, store_path=store)


def test_trust_write_failure_surfaces_without_mutating_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepagents_code.hooks import trust as trust_mod

    root = _write_project_hooks(tmp_path / "project")
    store = tmp_path / "hooks_trust.json"
    store.write_text(
        json.dumps({"version": 1, "projects": {}}),
        encoding="utf-8",
    )

    def _boom(_path: object, _store: object) -> None:
        msg = "disk full"
        raise OSError(msg)

    monkeypatch.setattr(trust_mod, "_write_store", _boom)

    assert not trust_project_hooks(root, store_path=store)
    assert json.loads(store.read_text(encoding="utf-8")) == {
        "version": 1,
        "projects": {},
    }
    assert not is_project_hooks_trusted(root, store_path=store)


def test_loader_path_alias_does_not_misclassify_user_hooks_as_project(
    tmp_path: Path,
) -> None:
    """When user and project hooks paths alias, provenance must stay explicit."""
    shared = tmp_path / "shared-config"
    shared.mkdir()
    (shared / "hooks.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "Stop": [{"hooks": [{"type": "command", "command": "shared-hook"}]}]
                }
            }
        ),
        encoding="utf-8",
    )
    project = tmp_path / "project"
    project.mkdir()
    (project / ".git").mkdir()
    (project / ".deepagents").symlink_to(shared)

    untrusted = load_hooks_config(
        project_root=project,
        workspace_trusted=False,
        config_dir=shared,
    )
    assert not untrusted.project_source_loaded
    assert HookEvent.STOP in untrusted.config.hooks
    assert [
        group.hooks[0].command for group in untrusted.config.hooks[HookEvent.STOP]
    ] == ["shared-hook"]

    trusted = load_hooks_config(
        project_root=project,
        workspace_trusted=True,
        config_dir=shared,
    )
    assert trusted.project_source_loaded
    assert [
        group.hooks[0].command for group in trusted.config.hooks[HookEvent.STOP]
    ] == ["shared-hook"]


def test_runtime_home_as_git_alias_keeps_user_hooks_when_untrusted(
    tmp_path: Path,
) -> None:
    home_project = _write_project_hooks(tmp_path / "home-as-git", command="home-hook")
    config_dir = home_project / ".deepagents"

    untrusted = HooksRuntime.create(
        cwd=home_project,
        workspace_trusted=False,
        config_dir=config_dir,
        transcript_root=tmp_path / "untrusted-transcripts",
    )
    assert not untrusted.project_hooks_loaded
    assert HookEvent.STOP in untrusted.configured_events()

    trusted = HooksRuntime.create(
        cwd=home_project,
        workspace_trusted=True,
        config_dir=config_dir,
        transcript_root=tmp_path / "trusted-transcripts",
    )
    assert trusted.project_hooks_loaded
    assert HookEvent.STOP in trusted.configured_events()


def test_runtime_discovers_project_hooks_from_repository_root(tmp_path: Path) -> None:
    root = _write_project_hooks(tmp_path / "project")
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
    root = _write_project_hooks(tmp_path / "project")
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
        _PROJECT_HOOKS_REMEMBER_LABEL,
        _check_project_hooks_trust,
        _ProjectMcpTrustAction,
    )

    root = _write_project_hooks(tmp_path / "project")
    store = tmp_path / "state" / "hooks_trust.json"
    monkeypatch.chdir(root)
    monkeypatch.setattr(trust, "_default_store_path", lambda: store)
    captured: dict[str, str] = {}

    def _select(_console: object, *, remember_label: str) -> _ProjectMcpTrustAction:
        captured["remember_label"] = remember_label
        return _ProjectMcpTrustAction.REMEMBER

    monkeypatch.setattr(
        "deepagents_code.main._select_project_mcp_trust_action",
        _select,
    )

    assert _check_project_hooks_trust() is True
    assert is_project_hooks_trusted(root, store_path=store)
    assert captured["remember_label"] == _PROJECT_HOOKS_REMEMBER_LABEL
    assert captured["remember_label"] != "Allow for this project — until changed"


def test_interactive_prompt_allow_once_does_not_persist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepagents_code.hooks import trust
    from deepagents_code.main import (
        _check_project_hooks_trust,
        _ProjectMcpTrustAction,
    )

    root = _write_project_hooks(tmp_path / "project")
    store = tmp_path / "state" / "hooks_trust.json"
    monkeypatch.chdir(root)
    monkeypatch.setattr(trust, "_default_store_path", lambda: store)
    monkeypatch.setattr(
        "deepagents_code.main._select_project_mcp_trust_action",
        lambda _console, **_kwargs: _ProjectMcpTrustAction.ALLOW_ONCE,
    )

    assert _check_project_hooks_trust() is True
    assert not is_project_hooks_trusted(root, store_path=store)


def test_interactive_prompt_skips_untrusted_project_hooks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepagents_code.main import (
        _check_project_hooks_trust,
        _ProjectMcpTrustAction,
    )

    root = _write_project_hooks(tmp_path / "project")
    monkeypatch.chdir(root)
    monkeypatch.setattr(
        "deepagents_code.main._select_project_mcp_trust_action",
        lambda _console, **_kwargs: _ProjectMcpTrustAction.DENY,
    )

    assert _check_project_hooks_trust() is False


def test_interactive_prompt_cancelled_aborts_startup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepagents_code.main import (
        _check_project_hooks_trust,
        _ProjectMcpTrustPromptOutcome,
    )

    root = _write_project_hooks(tmp_path / "project")
    monkeypatch.chdir(root)
    monkeypatch.setattr(
        "deepagents_code.main._select_project_mcp_trust_action",
        lambda _console, **_kwargs: _ProjectMcpTrustPromptOutcome.CANCELLED,
    )

    assert _check_project_hooks_trust() is _ProjectMcpTrustPromptOutcome.CANCELLED


def test_persisted_trust_reused_on_later_run(
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

    called = False

    def _should_not_prompt(_console: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        msg = "prompt should not run when trust is persisted"
        raise AssertionError(msg)

    monkeypatch.setattr(
        "deepagents_code.main._select_project_mcp_trust_action",
        _should_not_prompt,
    )

    assert _check_project_hooks_trust() is True
    assert called is False


def test_headless_requires_explicit_opt_in_even_with_persisted_trust(
    tmp_path: Path,
) -> None:
    root = _write_project_hooks(tmp_path / "project")
    store = tmp_path / "hooks_trust.json"
    assert trust_project_hooks(root, store_path=store)
    assert is_project_hooks_trusted(root, store_path=store)

    # Headless/non-interactive paths pass workspace_trusted only from the
    # explicit CLI flag, never from the interactive trust store.
    runtime = HooksRuntime.create(
        cwd=root,
        workspace_trusted=False,
        config_dir=tmp_path / "user",
        transcript_root=tmp_path / "transcripts",
    )
    assert not runtime.project_hooks_loaded
    assert HookEvent.STOP not in runtime.configured_events()

    opted_in = HooksRuntime.create(
        cwd=root,
        workspace_trusted=True,
        config_dir=tmp_path / "user",
        transcript_root=tmp_path / "opted-in-transcripts",
    )
    assert opted_in.project_hooks_loaded


def test_interactive_remember_write_failure_surfaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from deepagents_code.hooks import trust
    from deepagents_code.main import (
        _check_project_hooks_trust,
        _ProjectMcpTrustAction,
    )

    root = _write_project_hooks(tmp_path / "project")
    store = tmp_path / "state" / "hooks_trust.json"
    monkeypatch.chdir(root)
    monkeypatch.setattr(trust, "_default_store_path", lambda: store)
    monkeypatch.setattr(trust, "trust_project_hooks", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        "deepagents_code.main._select_project_mcp_trust_action",
        lambda _console, **_kwargs: _ProjectMcpTrustAction.REMEMBER,
    )

    assert _check_project_hooks_trust() is True
    captured = capsys.readouterr()
    assert "could not be remembered" in captured.err
    assert not is_project_hooks_trusted(root, store_path=store)


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


def test_trust_store_lock_serializes_in_process_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepagents_code.hooks import trust as trust_mod

    store = tmp_path / "hooks_trust.json"
    root_a = _write_project_hooks(tmp_path / "a")
    root_b = _write_project_hooks(tmp_path / "b")
    hold = threading.Event()
    release = threading.Event()
    original_write = trust_mod._write_store

    def _gated_write(path: object, store_model: object) -> None:
        if not hold.is_set():
            hold.set()
            assert release.wait(timeout=5)
        original_write(path, store_model)

    monkeypatch.setattr(trust_mod, "_write_store", _gated_write)
    first = threading.Thread(
        target=trust_project_hooks,
        kwargs={"project_root": root_a, "store_path": store},
    )
    first.start()
    assert hold.wait(timeout=5)
    second_done = threading.Event()
    second_result: list[bool] = []

    def _second() -> None:
        second_result.append(trust_project_hooks(root_b, store_path=store))
        second_done.set()

    second = threading.Thread(target=_second)
    second.start()
    # Second writer should block on the in-process lock until first finishes.
    assert not second_done.wait(timeout=0.2)
    release.set()
    first.join(timeout=5)
    second.join(timeout=5)
    assert second_result == [True]
    assert is_project_hooks_trusted(root_a, store_path=store)
    assert is_project_hooks_trusted(root_b, store_path=store)


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits only")
def test_trust_store_written_with_restrictive_permissions(tmp_path: Path) -> None:
    root = _write_project_hooks(tmp_path / "project")
    store = tmp_path / "state" / "hooks_trust.json"
    assert trust_project_hooks(root, store_path=store)
    assert (store.stat().st_mode & 0o777) == 0o600
    assert (store.parent.stat().st_mode & 0o777) == 0o700
