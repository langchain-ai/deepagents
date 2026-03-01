"""Tests for the hooks dispatch module."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import Mock, patch

import pytest

import deepagents_cli.hooks as hooks_mod
from deepagents_cli.hooks import _load_hooks, dispatch_hook


@pytest.fixture(autouse=True)
def _reset_hooks_cache() -> None:
    """Clear the module-level hooks cache before each test."""
    hooks_mod._hooks_config = None
    yield
    hooks_mod._hooks_config = None


class TestLoadHooks:
    """Test lazy loading and caching of hook definitions."""

    def test_missing_config_file(self, tmp_path):
        """Returns empty list when config file does not exist."""
        with patch.object(hooks_mod, "_HOOKS_PATH", tmp_path / "nonexistent.json"):
            result = _load_hooks()

        assert result == []

    def test_valid_config(self, tmp_path):
        """Parses hooks array from well-formed config."""
        config = {"hooks": [{"command": ["echo", "hi"], "events": ["session.start"]}]}
        cfg_path = tmp_path / "hooks.json"
        cfg_path.write_text(json.dumps(config))

        with patch.object(hooks_mod, "_HOOKS_PATH", cfg_path):
            result = _load_hooks()

        assert result == config["hooks"]

    def test_malformed_json(self, tmp_path):
        """Returns empty list and logs warning on invalid JSON."""
        cfg_path = tmp_path / "hooks.json"
        cfg_path.write_text("{not json!!")

        with patch.object(hooks_mod, "_HOOKS_PATH", cfg_path):
            result = _load_hooks()

        assert result == []

    def test_missing_hooks_key(self, tmp_path):
        """Returns empty list when 'hooks' key is absent."""
        cfg_path = tmp_path / "hooks.json"
        cfg_path.write_text(json.dumps({"other": "data"}))

        with patch.object(hooks_mod, "_HOOKS_PATH", cfg_path):
            result = _load_hooks()

        assert result == []

    def test_caches_after_first_load(self, tmp_path):
        """Second call returns cached result without re-reading file."""
        config = {"hooks": [{"command": ["true"]}]}
        cfg_path = tmp_path / "hooks.json"
        cfg_path.write_text(json.dumps(config))

        with patch.object(hooks_mod, "_HOOKS_PATH", cfg_path):
            first = _load_hooks()
            # Overwrite file — cached result should still be returned.
            cfg_path.write_text(json.dumps({"hooks": []}))
            second = _load_hooks()

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
            result = _load_hooks()

        assert result == []


class TestDispatchHook:
    """Test event dispatch to external hook commands."""

    async def test_no_hooks_configured(self):
        """Dispatch is a no-op when no hooks are loaded."""
        hooks_mod._hooks_config = []
        # Should not raise.
        await dispatch_hook("session.start", {})

    async def test_matching_event(self):
        """Hook command is called when event matches."""
        hooks_mod._hooks_config = [
            {"command": ["echo", "hi"], "events": ["session.start"]}
        ]
        mock_proc = Mock()
        mock_proc.communicate = Mock()

        with patch(
            "deepagents_cli.hooks.subprocess.Popen", return_value=mock_proc
        ) as mock_popen:
            await dispatch_hook("session.start", {"thread_id": "abc"})

        mock_popen.assert_called_once()
        mock_proc.communicate.assert_called_once()
        stdin_bytes = mock_proc.communicate.call_args[1]["input"]
        assert json.loads(stdin_bytes) == {"event": "session.start", "thread_id": "abc"}

    async def test_event_key_auto_injected(self):
        """Event name is automatically added to the payload."""
        hooks_mod._hooks_config = [{"command": ["echo"]}]
        mock_proc = Mock()
        mock_proc.communicate = Mock()

        with patch("deepagents_cli.hooks.subprocess.Popen", return_value=mock_proc):
            await dispatch_hook("task.complete", {})

        stdin_bytes = mock_proc.communicate.call_args[1]["input"]
        assert json.loads(stdin_bytes) == {"event": "task.complete"}

    async def test_non_matching_event_skipped(self):
        """Hook command is not called when event does not match."""
        hooks_mod._hooks_config = [
            {"command": ["echo", "hi"], "events": ["task.complete"]}
        ]

        with patch("deepagents_cli.hooks.subprocess.Popen") as mock_popen:
            await dispatch_hook("session.start", {})

        mock_popen.assert_not_called()

    async def test_empty_events_matches_everything(self):
        """Hook with no events filter receives all events."""
        hooks_mod._hooks_config = [{"command": ["echo", "hi"], "events": []}]
        mock_proc = Mock()
        mock_proc.communicate = Mock()

        with patch(
            "deepagents_cli.hooks.subprocess.Popen", return_value=mock_proc
        ) as mock_popen:
            await dispatch_hook("any.event", {})

        mock_popen.assert_called_once()

    async def test_missing_events_key_matches_everything(self):
        """Hook with omitted events key receives all events."""
        hooks_mod._hooks_config = [{"command": ["echo", "hi"]}]
        mock_proc = Mock()
        mock_proc.communicate = Mock()

        with patch(
            "deepagents_cli.hooks.subprocess.Popen", return_value=mock_proc
        ) as mock_popen:
            await dispatch_hook("any.event", {})

        mock_popen.assert_called_once()

    async def test_hook_without_command_skipped(self):
        """Hook entry missing 'command' is silently skipped."""
        hooks_mod._hooks_config = [{"events": ["session.start"]}]

        with patch("deepagents_cli.hooks.subprocess.Popen") as mock_popen:
            await dispatch_hook("session.start", {})

        mock_popen.assert_not_called()

    async def test_timeout_does_not_propagate(self):
        """TimeoutExpired is caught and logged, not raised."""
        hooks_mod._hooks_config = [{"command": ["sleep", "999"]}]
        mock_proc = Mock()
        mock_proc.communicate = Mock(side_effect=subprocess.TimeoutExpired("sleep", 5))

        with patch("deepagents_cli.hooks.subprocess.Popen", return_value=mock_proc):
            # Should not raise.
            await dispatch_hook("session.start", {})

    async def test_generic_error_does_not_propagate(self):
        """Unexpected errors are caught and logged, not raised."""
        hooks_mod._hooks_config = [{"command": ["bad"]}]

        with patch(
            "deepagents_cli.hooks.subprocess.Popen",
            side_effect=FileNotFoundError("bad"),
        ):
            # Should not raise.
            await dispatch_hook("session.start", {})

    async def test_multiple_hooks_dispatched(self):
        """All matching hooks fire, not just the first."""
        hooks_mod._hooks_config = [
            {"command": ["first"]},
            {"command": ["second"]},
        ]
        mock_proc = Mock()
        mock_proc.communicate = Mock()

        with patch(
            "deepagents_cli.hooks.subprocess.Popen", return_value=mock_proc
        ) as mock_popen:
            await dispatch_hook("session.start", {})

        assert mock_popen.call_count == 2

    async def test_popen_called_with_detach_flags(self):
        """Subprocess is started detached with correct pipe config."""
        hooks_mod._hooks_config = [{"command": ["echo"]}]
        mock_proc = Mock()
        mock_proc.communicate = Mock()

        with patch(
            "deepagents_cli.hooks.subprocess.Popen", return_value=mock_proc
        ) as mock_popen:
            await dispatch_hook("session.start", {})

        call_kwargs = mock_popen.call_args[1]
        assert call_kwargs["stdin"] == subprocess.PIPE
        assert call_kwargs["stdout"] == subprocess.DEVNULL
        assert call_kwargs["stderr"] == subprocess.DEVNULL
        assert call_kwargs["start_new_session"] is True
