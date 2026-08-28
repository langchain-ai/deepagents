"""Tests for the hooks dispatch module."""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

import deepagents_code.hooks.legacy as hooks_mod


@pytest.fixture(autouse=True)
def _reset_hooks_cache() -> Generator[None]:
    """Clear module-level hooks cache and background tasks before each test."""
    hooks_mod._hooks_config = None
    hooks_mod._background_tasks.clear()
    yield
    hooks_mod._hooks_config = None
    hooks_mod._background_tasks.clear()


# ---------------------------------------------------------------------------
# _load_hooks
# ---------------------------------------------------------------------------


class TestLoadHooks:
    """Test lazy loading and caching of hook definitions."""

    def test_os_error(self, tmp_path):
        """Returns empty list on OS-level read failure."""
        (tmp_path / "hooks.json").write_text("{}")

        with (
            patch("deepagents_code.model_config.DEFAULT_CONFIG_DIR", tmp_path),
            patch("pathlib.Path.read_text", side_effect=OSError("permission denied")),
        ):
            result = hooks_mod._load_hooks()

        assert result == []


# ---------------------------------------------------------------------------
# dispatch_hook
# ---------------------------------------------------------------------------


class TestDispatchHook:
    """Test event dispatch to external hook commands."""

    async def test_event_key_auto_injected(self):
        """Event name is automatically added to the payload."""
        hooks_mod._hooks_config = [{"command": ["echo"]}]

        with patch("deepagents_code.hooks.subprocess.run") as mock_run:
            await hooks_mod.dispatch_hook("task.complete", {})

        stdin_bytes = mock_run.call_args[1]["input"]
        assert json.loads(stdin_bytes) == {"event": "task.complete"}

    async def test_permission_error_does_not_propagate(self):
        """PermissionError is caught and logged at warning, not raised."""
        hooks_mod._hooks_config = [{"command": ["/not/executable"]}]

        with patch(
            "deepagents_code.hooks.subprocess.run",
            side_effect=PermissionError("not executable"),
        ):
            # Should not raise.
            await hooks_mod.dispatch_hook("session.start", {})

    async def test_first_hook_failure_does_not_block_second(self):
        """A failing first hook does not prevent subsequent hooks from firing."""
        hooks_mod._hooks_config = [
            {"command": ["fail"]},
            {"command": ["succeed"]},
        ]

        calls: list[list[str]] = []

        def side_effect(cmd: list[str], **_: Any) -> None:
            calls.append(cmd)
            if cmd == ["fail"]:
                msg = "fail"
                raise FileNotFoundError(msg)

        with patch("deepagents_code.hooks.subprocess.run", side_effect=side_effect):
            await hooks_mod.dispatch_hook("session.start", {})

        assert ["fail"] in calls
        assert ["succeed"] in calls


# ---------------------------------------------------------------------------
# dispatch_hook_fire_and_forget
# ---------------------------------------------------------------------------


class TestDispatchHookFireAndForget:
    """Test the fire-and-forget task wrapper."""

    def test_no_running_loop_does_not_raise(self):
        """Gracefully skips when no event loop is running."""
        hooks_mod._hooks_config = [{"command": ["echo"]}]

        # Call from sync context with no running loop — should not raise
        hooks_mod.dispatch_hook_fire_and_forget("session.start", {})
        assert len(hooks_mod._background_tasks) == 0


# ---------------------------------------------------------------------------
# drain_pending_hooks
# ---------------------------------------------------------------------------


class TestDrainPendingHooks:
    """Test draining of in-flight fire-and-forget hook tasks."""

    async def test_drain_snapshots_once_and_ignores_later_scheduled_hooks(self):
        """A hook scheduled *during* the drain await is not awaited by that drain.

        `drain_pending_hooks` snapshots the in-flight set once; its documented
        precondition is that no further dispatches happen during the await. Pin
        that snapshot-once behavior: a task that schedules another task while the
        drain is in flight leaves the second one un-awaited by the same drain
        call, so a change to loop-until-empty semantics fails here.
        """
        loop = asyncio.get_running_loop()
        second_done = False

        async def _second() -> None:
            nonlocal second_done
            await asyncio.sleep(0.05)
            second_done = True

        async def _first() -> None:
            # Yield first so this runs inside the drain's gather, then schedule a
            # new hook task *after* the drain has already snapshotted the set.
            await asyncio.sleep(0)
            second = loop.create_task(_second())
            hooks_mod._background_tasks.add(second)
            second.add_done_callback(hooks_mod._background_tasks.discard)

        first = loop.create_task(_first())
        hooks_mod._background_tasks.add(first)
        first.add_done_callback(hooks_mod._background_tasks.discard)

        await hooks_mod.drain_pending_hooks()

        # The drain awaited `first` (now done) but not the task it spawned.
        assert first.done()
        assert not second_done
        assert hooks_mod._background_tasks  # the second task is still tracked

        # Clean up the straggler so it does not leak into other tests.
        await asyncio.gather(*hooks_mod._background_tasks, return_exceptions=True)
        assert second_done


# ---------------------------------------------------------------------------
# has_pending_hooks
# ---------------------------------------------------------------------------


class TestHasPendingHooks:
    """`has_pending_hooks` gates the TUI's drain-on-exit, so verify it directly.

    A wrong predicate here would silently skip the graceful-exit drain and drop
    the final `tool.result`, which a mock-only test could not catch.
    """
