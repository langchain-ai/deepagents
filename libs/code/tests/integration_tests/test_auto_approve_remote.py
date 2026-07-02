"""Integration coverage for TUI auto-approve over the remote server path."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Literal

import pytest

from deepagents_code._testing_models import (
    DCA_TEST_DELEGATE_WRITE_MARKER,
    DCA_TEST_WRITE_FILE_MARKER,
    SUBAGENT_WRITE_CONTENT,
    TOP_LEVEL_WRITE_CONTENT,
)

if TYPE_CHECKING:
    from pathlib import Path


_WRITE_MODE = Literal["top_level", "subagent"]

# Per-mode prompt marker and the file content its write path produces. Distinct
# contents let the auto-approve assertion double as proof that subagent mode
# actually delegated through the `task` tool (only the subagent branch writes
# `SUBAGENT_WRITE_CONTENT`).
_MODE_WRITE: dict[_WRITE_MODE, tuple[str, str]] = {
    "top_level": (DCA_TEST_WRITE_FILE_MARKER, TOP_LEVEL_WRITE_CONTENT),
    "subagent": (DCA_TEST_DELEGATE_WRITE_MARKER, SUBAGENT_WRITE_CONTENT),
}


def _write_model_config(home_dir: Path) -> None:
    """Write a temp config that points the server subprocess at the tool-call model."""
    config_dir = home_dir / ".deepagents"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.toml").write_text(
        """
[models.providers.itest]
class_path = "deepagents_code._testing_models:ToolCallingIntegrationChatModel"
models = ["fake"]
""".strip()
        + "\n"
    )


async def _run_auto_approve_write(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: _WRITE_MODE,
    auto_approve: bool = True,
) -> None:
    """Drive an end-to-end gated write over the remote server and check HITL.

    Args:
        tmp_path: Pytest temp dir for the isolated home/project.
        monkeypatch: Pytest monkeypatch fixture.
        mode: Whether the write happens at the top level or via a delegated
            `task` subagent.
        auto_approve: When `True`, the gated `write_file` must be auto-approved
            (no approval prompt) and the file written. When `False`, the write
            must surface for approval; the callback here rejects it, so the file
            must not be written.
    """
    home_dir = tmp_path / "home"
    project_dir = tmp_path / "project"
    suffix = f"{mode}-{'auto' if auto_approve else 'manual'}"
    assistant_id = f"itest-auto-approve-{suffix}"

    home_dir.mkdir()
    project_dir.mkdir()
    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setenv("DEEPAGENTS_CODE_NO_UPDATE_CHECK", "1")
    monkeypatch.chdir(project_dir)
    _write_model_config(home_dir)

    from deepagents_code import model_config
    from deepagents_code.app import TextualSessionState
    from deepagents_code.config import create_model
    from deepagents_code.server_manager import server_session
    from deepagents_code.sessions import generate_thread_id
    from deepagents_code.textual_adapter import TextualUIAdapter, execute_task_textual

    config_path = home_dir / ".deepagents" / "config.toml"
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_DIR", config_path.parent)
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", config_path)

    approvals_requested: list[Any] = []

    async def request_approval(
        action_requests: list[dict[str, Any]],
        assistant_id: str | None,
    ) -> object:
        # The adapter awaits the return value a second time (`await future`), so
        # mirror the real UI contract: return an awaitable resolving to the
        # decision rather than the decision dict itself.
        await asyncio.sleep(0)
        approvals_requested.append((action_requests, assistant_id))
        decision: asyncio.Future[object] = asyncio.get_running_loop().create_future()
        decision.set_result({"type": "reject"})
        return decision

    async def mount_message(_: object) -> None:
        await asyncio.sleep(0)

    def update_status(_: str) -> None:
        return None

    marker, expected_content = _MODE_WRITE[mode]

    model_config.clear_caches()
    try:
        create_model("itest:fake").apply_to_settings()
        thread_id = generate_thread_id()
        target = project_dir / f"auto-approved-{suffix}.txt"

        async with server_session(
            assistant_id=assistant_id,
            model_name="itest:fake",
            no_mcp=True,
            enable_shell=False,
            interactive=True,
            sandbox_type="none",
        ) as (agent, _server_proc):
            adapter = TextualUIAdapter(
                mount_message=mount_message,
                update_status=update_status,
                request_approval=request_approval,
            )
            session_state = TextualSessionState(
                thread_id=thread_id,
                auto_approve=auto_approve,
            )

            await execute_task_textual(
                user_input=f"{marker}{target}",
                agent=agent,
                assistant_id=assistant_id,
                session_state=session_state,
                adapter=adapter,
            )

        if auto_approve:
            assert approvals_requested == []
            # Load-bearing: proves the gated write actually executed (and, in
            # subagent mode, that delegation occurred — only the subagent branch
            # writes `SUBAGENT_WRITE_CONTENT`).
            assert target.read_text() == expected_content
        else:
            # Without auto-approve the gated write must surface for approval, and
            # the reject decision must prevent the file from being written.
            assert approvals_requested, "expected a HITL approval request"
            assert not target.exists()
    finally:
        model_config.clear_caches()


@pytest.mark.timeout(180)
async def test_tui_auto_approve_suppresses_remote_hitl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default TUI remote path should not prompt for HITL in auto mode."""
    await _run_auto_approve_write(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        mode="top_level",
    )


@pytest.mark.timeout(180)
async def test_tui_auto_approve_suppresses_remote_subagent_hitl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default TUI remote path should not prompt for subagent HITL in auto mode."""
    await _run_auto_approve_write(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        mode="subagent",
    )


@pytest.mark.timeout(180)
async def test_tui_manual_approve_requests_remote_hitl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control: without auto-approve the remote path must gate the write via HITL.

    This anchors the auto-approve tests — it proves the `write_file` interrupt
    path is live, so their `approvals_requested == []` means the prompt was
    suppressed rather than never gated in the first place.
    """
    await _run_auto_approve_write(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        mode="top_level",
        auto_approve=False,
    )
