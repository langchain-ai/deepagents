"""Tests for project-hook workspace trust."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from deepagents_code._env_vars import EXPERIMENTAL
from deepagents_code.approval_mode import ApprovalMode
from deepagents_code.hooks.manager import HookSessionIdentity, HooksManager
from deepagents_code.hooks.models.domain import (
    HookContext,
    HookEvent,
    HookInvocation,
    StopEvent,
)
from deepagents_code.hooks.runtime import HooksRuntime
from deepagents_code.hooks.trust import (
    WorkspaceTrust,
    is_project_hooks_trusted,
    trust_project_hooks,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _enable_hooks_v2(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hooks v2 only loads in experimental mode, which these tests exercise."""
    monkeypatch.setenv(EXPERIMENTAL, "1")


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


def test_non_utf8_store_fails_closed_without_overwrite(tmp_path: Path) -> None:
    """Decoding happens during the read, so it must fail closed like other I/O."""
    root = _write_project_hooks(tmp_path / "project")
    store = tmp_path / "hooks_trust.json"
    raw = b'{"version": 1, "projects": {"\xff\xfe": {}}}'
    store.write_bytes(raw)

    assert not is_project_hooks_trusted(root, store_path=store)
    assert not trust_project_hooks(root, store_path=store)
    assert store.read_bytes() == raw


def test_concurrent_writes_across_stores_preserve_every_entry(tmp_path: Path) -> None:
    """A single process-wide lock must not drop entries under contention."""
    roots = [_write_project_hooks(tmp_path / f"project-{index}") for index in range(8)]
    stores = [tmp_path / "state-a" / "trust.json", tmp_path / "state-b" / "trust.json"]

    with ThreadPoolExecutor(max_workers=len(roots)) as pool:
        results = list(
            pool.map(
                lambda pair: trust_project_hooks(pair[1], store_path=pair[0]),
                [
                    (stores[index % len(stores)], root)
                    for index, root in enumerate(roots)
                ],
            )
        )

    assert all(results)
    for index, root in enumerate(roots):
        assert is_project_hooks_trusted(root, store_path=stores[index % len(stores)])


def test_trust_is_resolved_per_workspace(tmp_path: Path) -> None:
    """Trust follows the workspace, so each directory resolves independently."""
    trusted = _write_project_hooks(tmp_path / "trusted")
    untrusted = _write_project_hooks(tmp_path / "untrusted")
    store = tmp_path / "state" / "hooks_trust.json"
    assert trust_project_hooks(trusted, store_path=store)

    nested = trusted / "src"
    nested.mkdir()
    policy = WorkspaceTrust(store_path=store)

    assert policy.allows(trusted)
    assert policy.allows(nested)
    assert not policy.allows(untrusted)


def test_session_grant_does_not_extend_to_other_workspaces(tmp_path: Path) -> None:
    """An unpersisted `allow once` grant covers only the workspace it was made in."""
    granted = _write_project_hooks(tmp_path / "granted")
    other = _write_project_hooks(tmp_path / "other")
    store = tmp_path / "state" / "hooks_trust.json"

    policy = WorkspaceTrust.for_session(granted, granted=True, store_path=store)

    assert policy.allows(granted)
    assert not policy.allows(other)
    assert not is_project_hooks_trusted(granted, store_path=store)


def test_explicit_only_policy_ignores_persisted_trust(tmp_path: Path) -> None:
    """Headless runs must not inherit a grant made in an interactive session."""
    root = _write_project_hooks(tmp_path / "project")
    store = tmp_path / "state" / "hooks_trust.json"
    assert trust_project_hooks(root, store_path=store)

    opted_out = WorkspaceTrust.explicit_only(root, granted=False, store_path=store)
    opted_in = WorkspaceTrust.explicit_only(root, granted=True, store_path=store)

    assert not opted_out.allows(root)
    assert opted_in.allows(root)
    # The interactive policy still honors the same persisted grant.
    assert WorkspaceTrust(store_path=store).allows(root)


def test_declined_trust_resolves_to_untrusted_everywhere(tmp_path: Path) -> None:
    root = _write_project_hooks(tmp_path / "project")
    store = tmp_path / "state" / "hooks_trust.json"

    policy = WorkspaceTrust.for_session(root, granted=False, store_path=store)

    assert not policy.allows(root)


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
    from deepagents_code.main import _check_project_hooks_trust, _TrustAction

    root = _write_project_hooks(tmp_path / "project")
    store = tmp_path / "state" / "hooks_trust.json"
    monkeypatch.chdir(root)
    monkeypatch.setattr(trust, "_default_store_path", lambda: store)
    monkeypatch.setattr(
        "deepagents_code.main._select_trust_action",
        lambda _console, **_kwargs: _TrustAction[action],
    )

    decision = _check_project_hooks_trust()
    assert isinstance(decision, WorkspaceTrust)
    assert decision.allows(root) is allowed
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
        "deepagents_code.main._select_trust_action",
        lambda *_args, **_kwargs: pytest.fail("prompt ran despite persisted trust"),
    )

    decision = _check_project_hooks_trust()
    assert isinstance(decision, WorkspaceTrust)
    assert decision.allows(root)


async def test_textual_app_forwards_hook_trust(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepagents_code.app import DeepAgentsApp

    root = _write_project_hooks(tmp_path / "project")
    monkeypatch.chdir(root)
    app = DeepAgentsApp(
        agent=MagicMock(),
        thread_id="thread",
        hook_trust=WorkspaceTrust.for_session(root, granted=True),
    )
    with patch(
        "deepagents_code.hooks.runtime.HooksRuntime.create",
        return_value=MagicMock(),
    ) as create:
        await app._init_session_state()

    assert create.call_args.kwargs["workspace_trusted"] is True


async def test_textual_app_defaults_to_untrusted_without_a_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.hooks import trust

    root = _write_project_hooks(tmp_path / "project")
    monkeypatch.chdir(root)
    monkeypatch.setattr(
        trust, "_default_store_path", lambda: tmp_path / "state" / "hooks_trust.json"
    )
    app = DeepAgentsApp(agent=MagicMock(), thread_id="thread")
    with patch(
        "deepagents_code.hooks.runtime.HooksRuntime.create",
        return_value=MagicMock(),
    ) as create:
        await app._init_session_state()

    assert create.call_args.kwargs["workspace_trusted"] is False


def _isolate_hook_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point user hooks and transcripts at `tmp_path` instead of the real home."""
    user_dir = tmp_path / "user"
    user_dir.mkdir()
    monkeypatch.setattr("deepagents_code.hooks.loading.DEFAULT_CONFIG_DIR", user_dir)
    monkeypatch.setattr(
        "deepagents_code.hooks.runtime.DEFAULT_CONFIG_DIR", tmp_path / "state"
    )


def _manager(cwd: Path, trust: WorkspaceTrust) -> HooksManager:
    return HooksManager.create(
        cwd=cwd,
        identity=lambda: HookSessionIdentity("thread", ApprovalMode.MANUAL),
        trust=trust,
    )


async def test_reload_drops_project_hooks_when_leaving_trusted_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A grant for one workspace must not survive a move into an untrusted one."""
    _isolate_hook_config(tmp_path, monkeypatch)
    trusted = _write_project_hooks(tmp_path / "trusted")
    untrusted = _write_project_hooks(tmp_path / "untrusted")
    store = tmp_path / "state" / "hooks_trust.json"

    manager = _manager(
        trusted, WorkspaceTrust.for_session(trusted, granted=True, store_path=store)
    )
    assert manager.has_handlers(HookEvent.STOP)

    await manager.reload(cwd=untrusted)

    assert not manager.has_handlers(HookEvent.STOP)


async def test_reload_picks_up_trust_when_entering_trusted_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Moving into a workspace the store already trusts loads its hooks."""
    _isolate_hook_config(tmp_path, monkeypatch)
    untrusted = _write_project_hooks(tmp_path / "untrusted")
    trusted = _write_project_hooks(tmp_path / "trusted")
    store = tmp_path / "state" / "hooks_trust.json"
    assert trust_project_hooks(trusted, store_path=store)

    manager = _manager(untrusted, WorkspaceTrust(store_path=store))
    assert not manager.has_handlers(HookEvent.STOP)

    await manager.reload(cwd=trusted)

    assert manager.has_handlers(HookEvent.STOP)


def test_headless_manager_ignores_persisted_trust(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The policy the headless runner builds must not load remembered hooks."""
    _isolate_hook_config(tmp_path, monkeypatch)
    root = _write_project_hooks(tmp_path / "project")
    store = tmp_path / "state" / "hooks_trust.json"
    assert trust_project_hooks(root, store_path=store)

    opted_out = _manager(
        root, WorkspaceTrust.explicit_only(root, granted=False, store_path=store)
    )
    opted_in = _manager(
        root, WorkspaceTrust.explicit_only(root, granted=True, store_path=store)
    )

    assert not opted_out.has_handlers(HookEvent.STOP)
    assert opted_in.has_handlers(HookEvent.STOP)
