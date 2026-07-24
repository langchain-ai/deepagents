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

    def test_missing_config_file(self, tmp_path):
        """Returns empty list when config file does not exist."""
        # tmp_path exists but has no hooks.json
        with patch("deepagents_code.model_config.DEFAULT_CONFIG_DIR", tmp_path):
            result = hooks_mod._load_hooks()

        assert result == []

    def test_valid_config(self, tmp_path):
        """Parses hooks array from well-formed config."""
        config = {"hooks": [{"command": ["echo", "hi"], "events": ["session.start"]}]}
        (tmp_path / "hooks.json").write_text(json.dumps(config))

        with patch("deepagents_code.model_config.DEFAULT_CONFIG_DIR", tmp_path):
            result = hooks_mod._load_hooks()

        assert result == config["hooks"]

    def test_malformed_json(self, tmp_path):
        """Returns empty list and logs warning on invalid JSON."""
        (tmp_path / "hooks.json").write_text("{not json!!")

        with patch("deepagents_code.model_config.DEFAULT_CONFIG_DIR", tmp_path):
            result = hooks_mod._load_hooks()

        assert result == []

    def test_missing_hooks_key(self, tmp_path):
        """Returns empty list when 'hooks' key is absent."""
        (tmp_path / "hooks.json").write_text(json.dumps({"other": "data"}))

        with patch("deepagents_code.model_config.DEFAULT_CONFIG_DIR", tmp_path):
            result = hooks_mod._load_hooks()

        assert result == []

    def test_caches_after_first_load(self, tmp_path):
        """Second call returns cached result without re-reading file."""
        config = {"hooks": [{"command": ["true"]}]}
        cfg_path = tmp_path / "hooks.json"
        cfg_path.write_text(json.dumps(config))

        with patch("deepagents_code.model_config.DEFAULT_CONFIG_DIR", tmp_path):
            first = hooks_mod._load_hooks()
            # Overwrite file — cached result should still be returned.
            cfg_path.write_text(json.dumps({"hooks": []}))
            second = hooks_mod._load_hooks()

        assert first is second
        assert first == config["hooks"]

    def test_os_error(self, tmp_path):
        """Returns empty list on OS-level read failure."""
        (tmp_path / "hooks.json").write_text("{}")

        with (
            patch("deepagents_code.model_config.DEFAULT_CONFIG_DIR", tmp_path),
            patch("pathlib.Path.read_text", side_effect=OSError("permission denied")),
        ):
            result = hooks_mod._load_hooks()

        assert result == []

    def test_non_dict_json(self, tmp_path):
        """Returns empty list when config root is not a JSON object."""
        (tmp_path / "hooks.json").write_text(json.dumps([1, 2, 3]))

        with patch("deepagents_code.model_config.DEFAULT_CONFIG_DIR", tmp_path):
            result = hooks_mod._load_hooks()

        assert result == []

    def test_non_list_hooks_value(self, tmp_path):
        """Returns empty list when 'hooks' value is not a list."""
        (tmp_path / "hooks.json").write_text(json.dumps({"hooks": "not-a-list"}))

        with patch("deepagents_code.model_config.DEFAULT_CONFIG_DIR", tmp_path):
            result = hooks_mod._load_hooks()

        assert result == []

    def test_null_json(self, tmp_path):
        """Returns empty list when config is JSON null."""
        (tmp_path / "hooks.json").write_text("null")

        with patch("deepagents_code.model_config.DEFAULT_CONFIG_DIR", tmp_path):
            result = hooks_mod._load_hooks()

        assert result == []


# ---------------------------------------------------------------------------
# dispatch_hook
# ---------------------------------------------------------------------------


class TestDispatchHook:
    """Test event dispatch to external hook commands."""

    async def test_no_hooks_configured(self):
        """Dispatch is a no-op when no hooks are loaded."""
        hooks_mod._hooks_config = []
        # Should not raise.
        await hooks_mod.dispatch_hook("session.start", {})

    async def test_matching_event(self):
        """Hook command is called when event matches."""
        hooks_mod._hooks_config = [
            {"command": ["echo", "hi"], "events": ["session.start"]}
        ]

        with patch("deepagents_code.hooks.subprocess.run") as mock_run:
            await hooks_mod.dispatch_hook("session.start", {"thread_id": "abc"})

        mock_run.assert_called_once()
        stdin_bytes = mock_run.call_args[1]["input"]
        assert json.loads(stdin_bytes) == {"event": "session.start", "thread_id": "abc"}

    async def test_event_key_auto_injected(self):
        """Event name is automatically added to the payload."""
        hooks_mod._hooks_config = [{"command": ["echo"]}]

        with patch("deepagents_code.hooks.subprocess.run") as mock_run:
            await hooks_mod.dispatch_hook("task.complete", {})

        stdin_bytes = mock_run.call_args[1]["input"]
        assert json.loads(stdin_bytes) == {"event": "task.complete"}

    async def test_non_matching_event_skipped(self):
        """Hook command is not called when event does not match."""
        hooks_mod._hooks_config = [
            {"command": ["echo", "hi"], "events": ["task.complete"]}
        ]

        with patch("deepagents_code.hooks.subprocess.run") as mock_run:
            await hooks_mod.dispatch_hook("session.start", {})

        mock_run.assert_not_called()

    async def test_empty_events_matches_everything(self):
        """Hook with no events filter receives all events."""
        hooks_mod._hooks_config = [{"command": ["echo", "hi"], "events": []}]

        with patch("deepagents_code.hooks.subprocess.run") as mock_run:
            await hooks_mod.dispatch_hook("any.event", {})

        mock_run.assert_called_once()

    async def test_missing_events_key_matches_everything(self):
        """Hook with omitted events key receives all events."""
        hooks_mod._hooks_config = [{"command": ["echo", "hi"]}]

        with patch("deepagents_code.hooks.subprocess.run") as mock_run:
            await hooks_mod.dispatch_hook("any.event", {})

        mock_run.assert_called_once()

    async def test_hook_without_command_skipped(self, caplog):
        """Hook entry missing 'command' is skipped and the misconfig is warned."""
        hooks_mod._hooks_config = [{"events": ["session.start"]}]

        with (
            patch("deepagents_code.hooks.subprocess.run") as mock_run,
            caplog.at_level(logging.WARNING, logger="deepagents_code.hooks"),
        ):
            await hooks_mod.dispatch_hook("session.start", {})

        mock_run.assert_not_called()
        # The config mistake must be greppable, not look like a hook that simply
        # never matched.
        assert "invalid `command`" in caplog.text

    async def test_hook_with_string_command_skipped(self, caplog):
        """Hook with string command (not list) is skipped and warned."""
        hooks_mod._hooks_config = [{"command": "echo hello"}]

        with (
            patch("deepagents_code.hooks.subprocess.run") as mock_run,
            caplog.at_level(logging.WARNING, logger="deepagents_code.hooks"),
        ):
            await hooks_mod.dispatch_hook("session.start", {})

        mock_run.assert_not_called()
        assert "invalid `command`" in caplog.text

    async def test_hook_with_empty_command_list_skipped(self, caplog):
        """Hook with empty command list is skipped and warned."""
        hooks_mod._hooks_config = [{"command": []}]

        with (
            patch("deepagents_code.hooks.subprocess.run") as mock_run,
            caplog.at_level(logging.WARNING, logger="deepagents_code.hooks"),
        ):
            await hooks_mod.dispatch_hook("session.start", {})

        mock_run.assert_not_called()
        assert "invalid `command`" in caplog.text

    async def test_timeout_does_not_propagate(self):
        """TimeoutExpired is caught and logged, not raised."""
        hooks_mod._hooks_config = [{"command": ["sleep", "999"]}]

        with patch(
            "deepagents_code.hooks.subprocess.run",
            side_effect=subprocess.TimeoutExpired("sleep", 5),
        ):
            # Should not raise.
            await hooks_mod.dispatch_hook("session.start", {})

    async def test_file_not_found_does_not_propagate(self):
        """FileNotFoundError is caught and logged at warning, not raised."""
        hooks_mod._hooks_config = [{"command": ["nonexistent"]}]

        with patch(
            "deepagents_code.hooks.subprocess.run",
            side_effect=FileNotFoundError("nonexistent"),
        ):
            # Should not raise.
            await hooks_mod.dispatch_hook("session.start", {})

    async def test_permission_error_does_not_propagate(self):
        """PermissionError is caught and logged at warning, not raised."""
        hooks_mod._hooks_config = [{"command": ["/not/executable"]}]

        with patch(
            "deepagents_code.hooks.subprocess.run",
            side_effect=PermissionError("not executable"),
        ):
            # Should not raise.
            await hooks_mod.dispatch_hook("session.start", {})

    async def test_generic_error_does_not_propagate(self):
        """Unexpected errors are caught and logged, not raised."""
        hooks_mod._hooks_config = [{"command": ["bad"]}]

        with patch(
            "deepagents_code.hooks.subprocess.run",
            side_effect=RuntimeError("unexpected"),
        ):
            # Should not raise.
            await hooks_mod.dispatch_hook("session.start", {})

    async def test_generic_error_logged_at_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Unexpected subprocess failures surface at WARNING, not hidden at DEBUG.

        The catch-all handles the least-understood failures (e.g. ENOEXEC for a
        non-executable hook file), so a silent debug log would hide a hook that
        never fires.
        """
        hooks_mod._hooks_config = [{"command": ["bad"]}]

        with (
            patch(
                "deepagents_code.hooks.subprocess.run",
                side_effect=RuntimeError("unexpected"),
            ),
            caplog.at_level("WARNING", logger="deepagents_code.hooks"),
        ):
            await hooks_mod.dispatch_hook("session.start", {})

        assert any(
            "failed unexpectedly" in r.getMessage() and r.levelname == "WARNING"
            for r in caplog.records
        )

    async def test_multiple_hooks_dispatched(self):
        """All matching hooks fire, not just the first."""
        hooks_mod._hooks_config = [
            {"command": ["first"]},
            {"command": ["second"]},
        ]

        with patch("deepagents_code.hooks.subprocess.run") as mock_run:
            await hooks_mod.dispatch_hook("session.start", {})

        assert mock_run.call_count == 2

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

    async def test_subprocess_run_called_with_correct_flags(self):
        """subprocess.run is called with detach and pipe config."""
        hooks_mod._hooks_config = [{"command": ["echo"]}]

        with patch("deepagents_code.hooks.subprocess.run") as mock_run:
            await hooks_mod.dispatch_hook("session.start", {})

        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["stdout"] == subprocess.DEVNULL
        assert call_kwargs["stderr"] == subprocess.DEVNULL
        assert call_kwargs["start_new_session"] is True
        assert call_kwargs["timeout"] == 5
        assert call_kwargs["check"] is False

    async def test_dispatch_hook_swallows_json_serialization_error(self):
        """Non-serializable payload does not propagate."""
        hooks_mod._hooks_config = [{"command": ["echo"]}]

        # Should not raise despite non-serializable payload.
        await hooks_mod.dispatch_hook("session.start", {"bad": object()})

    async def test_dispatch_hook_stringifies_non_serializable_value(self):
        """A non-JSON-serializable value is stringified and still delivered.

        Locks in the `default=str` behavior: the subprocess must still run with
        the value coerced to its string form, rather than the whole event being
        dropped (which is what happens if `default=str` is removed and the
        serialization error is swallowed by the outer guard).
        """
        hooks_mod._hooks_config = [{"command": ["cat"]}]

        class _Widget:
            def __str__(self) -> str:
                return "STRINGIFIED_WIDGET"

        with patch("deepagents_code.hooks.subprocess.run") as mock_run:
            await hooks_mod.dispatch_hook("tool.use", {"tool_args": _Widget()})

        mock_run.assert_called_once()
        stdin_bytes = mock_run.call_args[1]["input"]
        assert b"STRINGIFIED_WIDGET" in stdin_bytes

    async def test_dispatch_hook_drops_event_when_default_str_raises(self, caplog):
        """If `default=str` itself raises, the event is dropped (documented gap).

        `json.dumps(default=str)` degrades a non-serializable value to its string
        form, but if that value's own `__str__` raises, serialization fails and
        the outer guard drops the whole event rather than delivering a partial
        payload. This is a known non-guarantee — only `tool_output` is
        sentinel-protected upstream, not `tool_args` — so pin it here so any
        future change to the guard is a conscious one.
        """
        hooks_mod._hooks_config = [{"command": ["cat"]}]

        class _Exploding:
            def __str__(self) -> str:
                msg = "cannot stringify"
                raise RuntimeError(msg)

            __repr__ = __str__

        with (
            patch("deepagents_code.hooks.subprocess.run") as mock_run,
            caplog.at_level(logging.WARNING, logger="deepagents_code.hooks"),
        ):
            await hooks_mod.dispatch_hook("tool.use", {"tool_args": _Exploding()})

        # Serialization failed, so the subprocess never ran — the whole event is
        # dropped rather than partially delivered.
        mock_run.assert_not_called()
        assert "Unexpected error in dispatch_hook" in caplog.text


# ---------------------------------------------------------------------------
# dispatch_hook_fire_and_forget
# ---------------------------------------------------------------------------


class TestDispatchHookFireAndForget:
    """Test the fire-and-forget task wrapper."""

    async def test_creates_task_with_strong_reference(self):
        """Task is stored in _background_tasks to prevent GC."""
        hooks_mod._hooks_config = []

        hooks_mod.dispatch_hook_fire_and_forget("session.start", {})

        assert len(hooks_mod._background_tasks) == 1
        # Let the task complete.
        task = next(iter(hooks_mod._background_tasks))
        await task
        # done_callback should have removed it.
        assert len(hooks_mod._background_tasks) == 0

    async def test_task_removed_after_completion(self):
        """Completed tasks are discarded from the background set."""
        hooks_mod._hooks_config = [{"command": ["echo"]}]

        with patch("deepagents_code.hooks.subprocess.run"):
            hooks_mod.dispatch_hook_fire_and_forget("session.start", {})
            task = next(iter(hooks_mod._background_tasks))
            await task

        assert len(hooks_mod._background_tasks) == 0

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

    async def test_drains_pending_task_before_returning(self):
        """A scheduled hook runs to completion before drain returns."""
        hooks_mod._hooks_config = [{"command": ["echo"]}]

        with patch("deepagents_code.hooks.subprocess.run") as mock_run:
            hooks_mod.dispatch_hook_fire_and_forget("tool.result", {"tool_name": "x"})
            assert len(hooks_mod._background_tasks) == 1

            await hooks_mod.drain_pending_hooks()

        assert hooks_mod._background_tasks == set()
        mock_run.assert_called_once()

    async def test_no_pending_tasks_is_noop(self):
        """Draining with nothing pending returns immediately."""
        await hooks_mod.drain_pending_hooks()
        assert hooks_mod._background_tasks == set()

    async def test_real_dispatch_then_real_drain_completes_inflight_task(self):
        """End-to-end: the real drain runs a real, still-in-flight dispatch to done.

        Composes the real `dispatch_hook_fire_and_forget` with the real
        `drain_pending_hooks` (only `subprocess.run` is patched, to record). A
        freshly scheduled task has not run yet — no `await` has ceded control — so
        the hook subprocess has not been invoked at dispatch time; the drain is
        what carries it to completion. This is the composed seam that the
        per-surface tests (which patch the dispatch capture point) and the
        instant-return drain test do not exercise together: it pins that the drain
        is load-bearing for an in-flight hook rather than one already finished.
        """
        hooks_mod._hooks_config = [{"command": ["echo"]}]
        ran = {"count": 0}

        def record_run(*_args: object, **_kwargs: object) -> MagicMock:
            ran["count"] += 1
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("deepagents_code.hooks.subprocess.run", side_effect=record_run):
            hooks_mod.dispatch_hook_fire_and_forget("tool.result", {"tool_name": "x"})
            # Real task scheduled but not yet executed (no await has occurred), so
            # the hook subprocess has not been called.
            assert ran["count"] == 0
            assert len(hooks_mod._background_tasks) == 1

            await hooks_mod.drain_pending_hooks()

        assert ran["count"] == 1
        assert hooks_mod._background_tasks == set()

    async def test_drain_swallows_task_exceptions(self):
        """A task that raises does not propagate out of drain."""

        async def _boom() -> None:
            await asyncio.sleep(0)
            msg = "boom"
            raise RuntimeError(msg)

        loop = asyncio.get_running_loop()
        task = loop.create_task(_boom())
        hooks_mod._background_tasks.add(task)
        task.add_done_callback(hooks_mod._background_tasks.discard)

        # Must not raise despite the task erroring.
        await hooks_mod.drain_pending_hooks()

        assert hooks_mod._background_tasks == set()

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

    async def test_false_when_no_tasks(self):
        """No scheduled hooks means nothing to wait for."""
        assert hooks_mod.has_pending_hooks() is False

    async def test_true_while_task_in_flight_then_false_after_drain(self):
        """Reports True with a live task pending and False once it drains."""

        async def _slow() -> None:
            await asyncio.sleep(0.05)

        loop = asyncio.get_running_loop()
        task = loop.create_task(_slow())
        hooks_mod._background_tasks.add(task)
        task.add_done_callback(hooks_mod._background_tasks.discard)

        assert hooks_mod.has_pending_hooks() is True

        await hooks_mod.drain_pending_hooks()

        assert hooks_mod.has_pending_hooks() is False
        assert hooks_mod._background_tasks == set()
