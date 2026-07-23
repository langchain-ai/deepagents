"""Unit tests for Hooks v2 configuration and snapshots."""

from __future__ import annotations

import io
import json
import sys
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from deepagents_code.hooks import migration
from deepagents_code.hooks.capabilities import (
    DEFAULT_COMMAND_TIMEOUT_SECONDS,
    get_event_spec,
)
from deepagents_code.hooks.loading import (
    canonical_hooks_bytes,
    compute_snapshot_id,
    load_hooks_config,
)
from deepagents_code.hooks.migration import (
    LEGACY_COMMAND_TIMEOUT_SECONDS,
    migrate_legacy_hooks,
)
from deepagents_code.hooks.models.config import HooksConfig
from deepagents_code.hooks.models.domain import HookEvent
from deepagents_code.hooks.snapshot import HooksSnapshot

if TYPE_CHECKING:
    from pathlib import Path


def test_registry_covers_all_hook_events() -> None:
    specs = {event: get_event_spec(event) for event in HookEvent}
    assert set(specs) == set(HookEvent)
    assert all(event is spec.event for event, spec in specs.items())
    assert (
        get_event_spec(HookEvent.SESSION_END).default_timeout_seconds
        == DEFAULT_COMMAND_TIMEOUT_SECONDS
    )
    assert get_event_spec(HookEvent.PERMISSION_REQUEST).matcher_field == "tool_name"
    assert get_event_spec(
        HookEvent.USER_PROMPT_SUBMIT
    ).default_timeout_seconds == pytest.approx(30.0)
    assert get_event_spec(HookEvent.PRE_COMPACT).matcher_field == "trigger"


def test_load_hooks_config_precedence_and_snapshot_hash(tmp_path: Path) -> None:
    user_dir = tmp_path / "user"
    project_dir = tmp_path / "project"
    user_dir.mkdir()
    (project_dir / ".deepagents").mkdir(parents=True)
    (user_dir / "hooks.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "SessionStart": [
                        {"hooks": [{"type": "command", "command": "user-hook"}]}
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    (project_dir / ".deepagents" / "hooks.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "SessionStart": [
                        {"hooks": [{"type": "command", "command": "project-hook"}]}
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    untrusted = load_hooks_config(
        project_root=project_dir,
        workspace_trusted=False,
        config_dir=user_dir,
    )
    assert [
        group.hooks[0].command
        for group in untrusted.config.hooks[HookEvent.SESSION_START]
    ] == ["user-hook"]
    assert untrusted.sources == (user_dir / "hooks.json",)

    loaded = load_hooks_config(
        project_root=project_dir,
        workspace_trusted=True,
        config_dir=user_dir,
    )
    groups = loaded.config.hooks[HookEvent.SESSION_START]

    assert [group.hooks[0].command for group in groups] == [
        "project-hook",
        "user-hook",
    ]
    assert loaded.snapshot_id == compute_snapshot_id(loaded.config)
    assert loaded.snapshot_id == compute_snapshot_id(
        HooksConfig.model_validate(
            {
                "hooks": {
                    "SessionStart": [
                        {"hooks": [{"type": "command", "command": "project-hook"}]},
                        {"hooks": [{"type": "command", "command": "user-hook"}]},
                    ]
                }
            }
        )
    )
    assert canonical_hooks_bytes(loaded.config).startswith(b'{"hooks":')


def test_legacy_migration_maps_equivalent_lifecycle_events(
    tmp_path: Path,
) -> None:
    migrated = migrate_legacy_hooks(
        [
            {
                "command": ["echo", "prompt"],
                "events": ["session.start", "user.prompt"],
            },
            {
                "command": ["echo", "compact"],
                "events": ["context.offload", "context.compact"],
            },
            {"command": ["echo", "complete"], "events": ["task.complete"]},
            {"command": ["echo", "tool"], "events": ["tool.use"]},
            {"command": ["echo", "end"], "events": ["session.end"]},
            {"command": ["echo", "input"], "events": ["input.required"]},
            {"command": ["echo", "perm"], "events": ["permission.request"]},
        ]
    )

    assert set(migrated.hooks) == {
        HookEvent.USER_PROMPT_SUBMIT,
        HookEvent.PRE_COMPACT,
        HookEvent.SESSION_END,
        HookEvent.NOTIFICATION,
    }
    # Distinct legacy event names stay as separate groups (no setdefault collapse).
    assert len(migrated.hooks[HookEvent.USER_PROMPT_SUBMIT]) == 2
    assert len(migrated.hooks[HookEvent.PRE_COMPACT]) == 2
    assert all(
        group.matcher == "manual" for group in migrated.hooks[HookEvent.PRE_COMPACT]
    )
    assert [group.matcher for group in migrated.hooks[HookEvent.NOTIFICATION]] == [
        "agent_completed",
        "agent_needs_input",
    ]
    prompt_legacy_events = [
        group.hooks[0].argv[3]
        for group in migrated.hooks[HookEvent.USER_PROMPT_SUBMIT]
        if group.hooks[0].argv is not None
    ]
    compact_legacy_events = [
        group.hooks[0].argv[3]
        for group in migrated.hooks[HookEvent.PRE_COMPACT]
        if group.hooks[0].argv is not None
    ]
    assert prompt_legacy_events == ["session.start", "user.prompt"]
    assert compact_legacy_events == ["context.offload", "context.compact"]
    assert HookEvent.SESSION_START not in migrated.hooks
    assert HookEvent.PRE_TOOL_USE not in migrated.hooks
    for groups in migrated.hooks.values():
        for group in groups:
            handler = group.hooks[0]
            assert handler.timeout == pytest.approx(
                LEGACY_COMMAND_TIMEOUT_SECONDS + 1.0
            )
            assert handler.inherit_environ is True
            assert handler.argv is not None
            assert handler.argv[1:3] == ["-m", "deepagents_code.hooks.migration"]
            assert "deepagents_code.hooks.migration" in handler.command
            assert "/dev/null" not in handler.command

    user_dir = tmp_path / "user"
    user_dir.mkdir()
    (user_dir / "hooks.json").write_text(
        json.dumps(
            {
                "hooks": [
                    {"command": ["echo", "start"], "events": ["session.start"]},
                    {"command": ["echo", "end"], "events": ["session.end"]},
                    {"command": ["echo", "tool"], "events": ["tool.use"]},
                ]
            }
        ),
        encoding="utf-8",
    )
    loaded = load_hooks_config(
        project_root=tmp_path,
        workspace_trusted=False,
        config_dir=user_dir,
    )

    assert HookEvent.SESSION_START not in loaded.config.hooks
    assert HookEvent.USER_PROMPT_SUBMIT in loaded.config.hooks
    assert HookEvent.SESSION_END in loaded.config.hooks
    assert HookEvent.PRE_TOOL_USE not in loaded.config.hooks
    assert loaded.diagnostics[0].code == "legacy_deprecated"
    assert "September 1, 2026" in loaded.diagnostics[0].message
    assert any(item.code == "legacy_migrated" for item in loaded.diagnostics)


def test_legacy_migration_prefers_argv_over_shell_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(migration.os, "name", "nt")
    monkeypatch.setattr(
        migration.sys,
        "executable",
        r"C:\Program Files\Python\python.exe",
    )

    migrated = migrate_legacy_hooks(
        [
            {
                "command": [
                    r"C:\Program Files\Hooks\a&b\observer.exe",
                    "arg with space",
                ]
            }
        ]
    )
    handlers = [
        group.hooks[0] for group in migrated.hooks[HookEvent.USER_PROMPT_SUBMIT]
    ]
    assert handlers
    for handler in handlers:
        assert handler.argv is not None
        assert handler.argv[0] == r"C:\Program Files\Python\python.exe"
        assert "&" not in "".join(handler.argv[1:3])
        # Shell form remains available for diagnostics; exec path uses argv.
        assert handler.command.startswith('"C:\\Program Files\\Python\\python.exe"')
        assert "'" not in handler.command


def test_legacy_migration_deduplicates_identical_event_names() -> None:
    migrated = migrate_legacy_hooks(
        [
            {
                "command": ["echo", "prompt"],
                "events": ["session.start", "session.start", "user.prompt"],
            }
        ]
    )

    legacy_events = [
        group.hooks[0].argv[3]
        for group in migrated.hooks[HookEvent.USER_PROMPT_SUBMIT]
        if group.hooks[0].argv is not None
    ]
    assert legacy_events == ["session.start", "user.prompt"]


def test_legacy_catch_all_migrates_only_safe_unique_targets() -> None:
    migrated = migrate_legacy_hooks([{"command": ["echo", "all"]}])

    assert set(migrated.hooks) == {
        HookEvent.USER_PROMPT_SUBMIT,
        HookEvent.SESSION_END,
        HookEvent.NOTIFICATION,
        HookEvent.PRE_COMPACT,
    }
    assert len(migrated.hooks[HookEvent.USER_PROMPT_SUBMIT]) == 2
    assert len(migrated.hooks[HookEvent.PRE_COMPACT]) == 2
    assert [group.matcher for group in migrated.hooks[HookEvent.NOTIFICATION]] == [
        "agent_completed",
        "agent_needs_input",
    ]
    assert HookEvent.PRE_TOOL_USE not in migrated.hooks
    assert HookEvent.PERMISSION_REQUEST not in migrated.hooks


def test_legacy_timeout_constant_matches_dispatcher() -> None:
    from deepagents_code.hooks.env import HOOK_SUBPROCESS_TIMEOUT_SECONDS
    from deepagents_code.hooks.legacy import HOOK_SUBPROCESS_TIMEOUT

    assert LEGACY_COMMAND_TIMEOUT_SECONDS == HOOK_SUBPROCESS_TIMEOUT_SECONDS
    assert HOOK_SUBPROCESS_TIMEOUT == HOOK_SUBPROCESS_TIMEOUT_SECONDS


def test_legacy_adapter_failures_are_nonzero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "stdin",
        io.TextIOWrapper(io.BytesIO(b"{"), encoding="utf-8"),
    )
    assert migration._run_adapter(["session.start"]) == 1
    assert migration._run_adapter(["session.start", "!!!not-b64!!!"]) == 1
    monkeypatch.setattr(
        sys,
        "stdin",
        io.TextIOWrapper(io.BytesIO(b"[]"), encoding="utf-8"),
    )
    encoded = migration.base64.urlsafe_b64encode(
        b'["/nonexistent-legacy-hook"]'
    ).decode()
    assert migration._run_adapter(["session.start", encoded]) == 1


def test_legacy_adapter_ignores_nested_hook_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    script = tmp_path / "hook.py"
    script.write_text("import sys; sys.exit(2)\n", encoding="utf-8")
    encoded = migration.base64.urlsafe_b64encode(
        json.dumps([sys.executable, str(script)]).encode()
    ).decode()
    monkeypatch.setattr(
        sys,
        "stdin",
        io.TextIOWrapper(io.BytesIO(b'{"session_id":"t1"}'), encoding="utf-8"),
    )
    assert migration._run_adapter(["session.start", encoded]) == 0


def test_legacy_adapter_keeps_nested_hook_in_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = MagicMock()
    monkeypatch.setattr(migration.subprocess, "run", run)
    monkeypatch.setattr(
        sys,
        "stdin",
        io.TextIOWrapper(io.BytesIO(b'{"session_id":"t1"}'), encoding="utf-8"),
    )
    encoded = migration.base64.urlsafe_b64encode(b'["legacy-hook"]').decode()

    assert migration._run_adapter(["session.start", encoded]) == 0

    run.assert_called_once()
    assert "start_new_session" not in run.call_args.kwargs


def test_invalid_config_is_diagnosed(tmp_path: Path) -> None:
    config_dir = tmp_path / "user"
    config_dir.mkdir()
    path = config_dir / "hooks.json"
    path.write_text(
        '{"hooks":{"Stop":[{"hooks":[{"type":"http"}]}]}}',
        encoding="utf-8",
    )

    loaded = load_hooks_config(
        project_root=tmp_path,
        workspace_trusted=False,
        config_dir=config_dir,
    )

    assert loaded.config.hooks == {}
    assert loaded.sources == ()
    assert [item.code for item in loaded.diagnostics] == ["invalid_config"]
    assert loaded.diagnostics[0].field == f"{path}:hooks.Stop[0].hooks[0]"


def test_invalid_handler_does_not_discard_valid_siblings(tmp_path: Path) -> None:
    path = tmp_path / "hooks.json"
    path.write_text(
        json.dumps(
            {
                "hooks": {
                    "Stop": [
                        {
                            "hooks": [
                                {"type": "command", "command": "valid"},
                                {"type": "http", "url": "https://example.com"},
                            ]
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    loaded = load_hooks_config(
        project_root=tmp_path,
        workspace_trusted=False,
        paths=[path],
    )

    handlers = loaded.config.hooks[HookEvent.STOP][0].hooks
    assert [handler.command for handler in handlers] == ["valid"]
    assert loaded.sources == (path.resolve(),)
    assert [item.code for item in loaded.diagnostics] == ["invalid_config"]
    assert loaded.diagnostics[0].field == f"{path.resolve()}:hooks.Stop[0].hooks[1]"


def test_source_paths_are_canonicalized_and_deduplicated(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    path = config_dir / "hooks.json"
    path.write_text(
        '{"hooks":{"Stop":[{"hooks":[{"type":"command","command":"once"}]}]}}',
        encoding="utf-8",
    )

    loaded = load_hooks_config(
        project_root=tmp_path,
        workspace_trusted=False,
        paths=[path, config_dir / ".." / "config" / "hooks.json"],
    )

    assert loaded.sources == (path.resolve(),)
    assert len(loaded.config.hooks[HookEvent.STOP]) == 1


def test_async_command_config_is_rejected() -> None:
    with pytest.raises(ValidationError, match="async"):
        HooksConfig.model_validate(
            {
                "hooks": {
                    "Stop": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "echo",
                                    "async": True,
                                }
                            ]
                        }
                    ]
                }
            }
        )


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("-inf"), float("nan")])
def test_command_timeout_must_be_positive_and_finite(timeout: float) -> None:
    with pytest.raises(ValidationError, match="timeout"):
        HooksConfig.model_validate(
            {
                "hooks": {
                    "Stop": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "echo",
                                    "timeout": timeout,
                                }
                            ]
                        }
                    ]
                }
            }
        )


def test_snapshot_id_is_immutable_and_stable() -> None:
    config = HooksConfig.model_validate(
        {
            "hooks": {
                "PreToolUse": [
                    {"hooks": [{"type": "command", "command": "policy"}]},
                ]
            }
        }
    )
    first = HooksSnapshot.from_config(config)
    second = HooksSnapshot.from_config(config)

    assert first.snapshot_id == second.snapshot_id
    assert len(first.snapshot_id) == 64

    with_false_async = HooksConfig.model_validate(
        {
            "hooks": {
                "PreToolUse": [
                    {
                        "hooks": [
                            {
                                "type": "command",
                                "command": "policy",
                                "async": False,
                            }
                        ]
                    }
                ]
            }
        }
    )
    assert compute_snapshot_id(with_false_async) == first.snapshot_id
    assert with_false_async.hooks[HookEvent.PRE_TOOL_USE][0].hooks[0].async_ is None

    with pytest.raises(ValueError, match="canonical"):
        HooksSnapshot.from_config(config, snapshot_id="not-the-canonical-id")


def test_snapshot_rejects_matcher_for_unmatchable_event() -> None:
    snapshot = HooksSnapshot.from_config(
        HooksConfig.model_validate(
            {
                "hooks": {
                    "Stop": [
                        {
                            "matcher": "Bash",
                            "hooks": [{"type": "command", "command": "stop"}],
                        }
                    ]
                }
            }
        )
    )

    assert snapshot.handlers[HookEvent.STOP] == ()
    assert [item.code for item in snapshot.diagnostics] == ["unsupported_matcher"]
