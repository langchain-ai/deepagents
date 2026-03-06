"""Tests for the hooks dispatch module."""

from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

import deepagents_cli.hooks as hooks_mod


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
        with patch.object(hooks_mod, "_HOOKS_PATH", tmp_path / "nonexistent.json"):
            result = hooks_mod._load_hooks()

        assert result == []

    def test_valid_config(self, tmp_path):
        """Parses hooks array from well-formed config."""
        config = {"hooks": [{"command": ["echo", "hi"], "events": ["session.start"]}]}
        cfg_path = tmp_path / "hooks.json"
        cfg_path.write_text(json.dumps(config))

        with patch.object(hooks_mod, "_HOOKS_PATH", cfg_path):
            result = hooks_mod._load_hooks()

        assert result == config["hooks"]

    def test_malformed_json(self, tmp_path):
        """Returns empty list and logs warning on invalid JSON."""
        cfg_path = tmp_path / "hooks.json"
        cfg_path.write_text("{not json!!")

        with patch.object(hooks_mod, "_HOOKS_PATH", cfg_path):
            result = hooks_mod._load_hooks()

        assert result == []

    def test_missing_hooks_key(self, tmp_path):
        """Returns empty list when 'hooks' key is absent."""
        cfg_path = tmp_path / "hooks.json"
        cfg_path.write_text(json.dumps({"other": "data"}))

        with patch.object(hooks_mod, "_HOOKS_PATH", cfg_path):
            result = hooks_mod._load_hooks()

        assert result == []

    def test_caches_after_first_load(self, tmp_path):
        """Second call returns cached result without re-reading file."""
        config = {"hooks": [{"command": ["true"]}]}
        cfg_path = tmp_path / "hooks.json"
        cfg_path.write_text(json.dumps(config))

        with patch.object(hooks_mod, "_HOOKS_PATH", cfg_path):
            first = hooks_mod._load_hooks()
            # Overwrite file — cached result should still be returned.
            cfg_path.write_text(json.dumps({"hooks": []}))
            second = hooks_mod._load_hooks()

        assert first is second
        assert first == config["hooks"]

    def test_os_error(self, tmp_path):
        """Returns empty list on OS-level read failure."""
        cfg_path = tmp_path / "hooks.json"
        cfg_path.write_text("{}")

        with (
            patch.object(hooks_mod, "_HOOKS_PATH", cfg_path),
            patch("pathlib.Path.read_text", side_effect=OSError("permission denied")),
        ):
            result = hooks_mod._load_hooks()

        assert result == []

    def test_non_dict_json(self, tmp_path):
        """Returns empty list when config root is not a JSON object."""
        cfg_path = tmp_path / "hooks.json"
        cfg_path.write_text(json.dumps([1, 2, 3]))

        with patch.object(hooks_mod, "_HOOKS_PATH", cfg_path):
            result = hooks_mod._load_hooks()

        assert result == []

    def test_non_list_hooks_value(self, tmp_path):
        """Returns empty list when 'hooks' value is not a list."""
        cfg_path = tmp_path / "hooks.json"
        cfg_path.write_text(json.dumps({"hooks": "not-a-list"}))

        with patch.object(hooks_mod, "_HOOKS_PATH", cfg_path):
            result = hooks_mod._load_hooks()

        assert result == []

    def test_null_json(self, tmp_path):
        """Returns empty list when config is JSON null."""
        cfg_path = tmp_path / "hooks.json"
        cfg_path.write_text("null")

        with patch.object(hooks_mod, "_HOOKS_PATH", cfg_path):
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

        with patch("deepagents_cli.hooks.subprocess.run") as mock_run:
            await hooks_mod.dispatch_hook("session.start", {"thread_id": "abc"})

        mock_run.assert_called_once()
        stdin_bytes = mock_run.call_args[1]["input"]
        assert json.loads(stdin_bytes) == {"event": "session.start", "thread_id": "abc"}

    async def test_event_key_auto_injected(self):
        """Event name is automatically added to the payload."""
        hooks_mod._hooks_config = [{"command": ["echo"]}]

        with patch("deepagents_cli.hooks.subprocess.run") as mock_run:
            await hooks_mod.dispatch_hook("task.complete", {})

        stdin_bytes = mock_run.call_args[1]["input"]
        assert json.loads(stdin_bytes) == {"event": "task.complete"}

    async def test_non_matching_event_skipped(self):
        """Hook command is not called when event does not match."""
        hooks_mod._hooks_config = [
            {"command": ["echo", "hi"], "events": ["task.complete"]}
        ]

        with patch("deepagents_cli.hooks.subprocess.run") as mock_run:
            await hooks_mod.dispatch_hook("session.start", {})

        mock_run.assert_not_called()

    async def test_empty_events_matches_everything(self):
        """Hook with no events filter receives all events."""
        hooks_mod._hooks_config = [{"command": ["echo", "hi"], "events": []}]

        with patch("deepagents_cli.hooks.subprocess.run") as mock_run:
            await hooks_mod.dispatch_hook("any.event", {})

        mock_run.assert_called_once()

    async def test_missing_events_key_matches_everything(self):
        """Hook with omitted events key receives all events."""
        hooks_mod._hooks_config = [{"command": ["echo", "hi"]}]

        with patch("deepagents_cli.hooks.subprocess.run") as mock_run:
            await hooks_mod.dispatch_hook("any.event", {})

        mock_run.assert_called_once()

    async def test_hook_without_command_skipped(self):
        """Hook entry missing 'command' is silently skipped."""
        hooks_mod._hooks_config = [{"events": ["session.start"]}]

        with patch("deepagents_cli.hooks.subprocess.run") as mock_run:
            await hooks_mod.dispatch_hook("session.start", {})

        mock_run.assert_not_called()

    async def test_hook_with_string_command_skipped(self):
        """Hook with string command (not list) is skipped."""
        hooks_mod._hooks_config = [{"command": "echo hello"}]

        with patch("deepagents_cli.hooks.subprocess.run") as mock_run:
            await hooks_mod.dispatch_hook("session.start", {})

        mock_run.assert_not_called()

    async def test_hook_with_empty_command_list_skipped(self):
        """Hook with empty command list is skipped."""
        hooks_mod._hooks_config = [{"command": []}]

        with patch("deepagents_cli.hooks.subprocess.run") as mock_run:
            await hooks_mod.dispatch_hook("session.start", {})

        mock_run.assert_not_called()

    async def test_timeout_does_not_propagate(self):
        """TimeoutExpired is caught and logged, not raised."""
        hooks_mod._hooks_config = [{"command": ["sleep", "999"]}]

        with patch(
            "deepagents_cli.hooks.subprocess.run",
            side_effect=subprocess.TimeoutExpired("sleep", 5),
        ):
            # Should not raise.
            await hooks_mod.dispatch_hook("session.start", {})

    async def test_file_not_found_does_not_propagate(self):
        """FileNotFoundError is caught and logged at warning, not raised."""
        hooks_mod._hooks_config = [{"command": ["nonexistent"]}]

        with patch(
            "deepagents_cli.hooks.subprocess.run",
            side_effect=FileNotFoundError("nonexistent"),
        ):
            # Should not raise.
            await hooks_mod.dispatch_hook("session.start", {})

    async def test_permission_error_does_not_propagate(self):
        """PermissionError is caught and logged at warning, not raised."""
        hooks_mod._hooks_config = [{"command": ["/not/executable"]}]

        with patch(
            "deepagents_cli.hooks.subprocess.run",
            side_effect=PermissionError("not executable"),
        ):
            # Should not raise.
            await hooks_mod.dispatch_hook("session.start", {})

    async def test_generic_error_does_not_propagate(self):
        """Unexpected errors are caught and logged, not raised."""
        hooks_mod._hooks_config = [{"command": ["bad"]}]

        with patch(
            "deepagents_cli.hooks.subprocess.run",
            side_effect=RuntimeError("unexpected"),
        ):
            # Should not raise.
            await hooks_mod.dispatch_hook("session.start", {})

    async def test_multiple_hooks_dispatched(self):
        """All matching hooks fire, not just the first."""
        hooks_mod._hooks_config = [
            {"command": ["first"]},
            {"command": ["second"]},
        ]

        with patch("deepagents_cli.hooks.subprocess.run") as mock_run:
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

        with patch("deepagents_cli.hooks.subprocess.run", side_effect=side_effect):
            await hooks_mod.dispatch_hook("session.start", {})

        assert ["fail"] in calls
        assert ["succeed"] in calls

    async def test_subprocess_run_called_with_correct_flags(self):
        """subprocess.run is called with detach and pipe config."""
        hooks_mod._hooks_config = [{"command": ["echo"]}]

        with patch("deepagents_cli.hooks.subprocess.run") as mock_run:
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

        with patch("deepagents_cli.hooks.subprocess.run"):
            hooks_mod.dispatch_hook_fire_and_forget("session.start", {})
            task = next(iter(hooks_mod._background_tasks))
            await task

        assert len(hooks_mod._background_tasks) == 0
