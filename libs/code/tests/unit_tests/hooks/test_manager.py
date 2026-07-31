"""Tests for `HooksManager` ownership of the shared presenter."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from deepagents_code._env_vars import EXPERIMENTAL
from deepagents_code.approval_mode import ApprovalMode
from deepagents_code.hooks.manager import HookSessionIdentity, HooksManager
from deepagents_code.hooks.models.domain import PermissionEffect

if TYPE_CHECKING:
    from pathlib import Path

    from deepagents_code.hooks.presenter import HookNoticeSeverity, HookPresenter


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


def _isolate_hook_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    user_dir = tmp_path / "user"
    user_dir.mkdir()
    monkeypatch.setattr("deepagents_code.hooks.loading.DEFAULT_CONFIG_DIR", user_dir)
    monkeypatch.setattr(
        "deepagents_code.hooks.runtime.DEFAULT_CONFIG_DIR", tmp_path / "state"
    )


def _manager(cwd: Path) -> HooksManager:
    return HooksManager.create(
        cwd=cwd,
        identity=lambda: HookSessionIdentity("thread", ApprovalMode.MANUAL),
    )


async def test_reload_keeps_one_presenter_shared_with_the_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Output sinks bound once must survive a working-directory change."""
    _isolate_hook_config(tmp_path, monkeypatch)
    first = _write_project_hooks(tmp_path / "first")
    second = _write_project_hooks(tmp_path / "second")

    manager = _manager(first)
    presenter = manager.presenter
    assert _runtime_presenter(manager) is presenter

    await manager.reload(cwd=second)

    assert manager.presenter is presenter
    assert _runtime_presenter(manager) is presenter


def test_create_binds_sinks_to_the_manager_owned_presenter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Callers pass sinks, never a presenter; the manager builds and owns it."""
    _isolate_hook_config(tmp_path, monkeypatch)
    root = _write_project_hooks(tmp_path / "project")
    notices: list[tuple[str, str]] = []

    manager = HooksManager.create(
        cwd=root,
        identity=lambda: HookSessionIdentity("thread", ApprovalMode.MANUAL),
        notice=lambda message, severity: notices.append((message, severity)),
    )

    assert _runtime_presenter(manager) is manager.presenter
    manager.presenter.present_permission(
        "shell",
        PermissionEffect(behavior="deny", reason="nope"),
    )

    assert notices == [("PermissionRequest hook denied shell: nope", "warning")]


def test_attach_output_binds_sinks_and_replays_load_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A manager loaded before its UI must resurface what it could only log."""
    _isolate_hook_config(tmp_path, monkeypatch)
    (tmp_path / "user" / "hooks.json").write_text(
        json.dumps({"hooks": {"Stop": [{"hooks": [{"type": "command"}]}]}}),
        encoding="utf-8",
    )
    root = tmp_path / "project"
    root.mkdir()

    manager = _manager(root)
    notices: list[tuple[str, str]] = []

    def record(message: str, severity: HookNoticeSeverity) -> None:
        notices.append((message, severity))

    manager.attach_output(notice=record)

    assert notices
    assert all(severity in {"warning", "error"} for _, severity in notices)


def _runtime_presenter(manager: HooksManager) -> HookPresenter | None:
    runtime = manager._runtime  # asserting the shared-instance invariant
    return runtime.presenter if runtime is not None else None
