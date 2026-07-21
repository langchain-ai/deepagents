"""Unit tests for Hooks v2 configuration and snapshots."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from deepagents_code.hooks.capabilities import (
    DEFAULT_COMMAND_TIMEOUT_SECONDS,
    HOOK_EVENT_SPECS,
    assert_hook_event_registry_complete,
    get_event_spec,
)
from deepagents_code.hooks.loading import (
    canonical_hooks_bytes,
    compute_snapshot_id,
    load_hooks_config,
)
from deepagents_code.hooks.migration import migrate_legacy_hooks
from deepagents_code.hooks.models.config import HooksConfig
from deepagents_code.hooks.models.domain import HookEvent
from deepagents_code.hooks.snapshot import HooksSnapshot

if TYPE_CHECKING:
    from pathlib import Path


def test_registry_covers_all_hook_events() -> None:
    assert_hook_event_registry_complete()
    assert set(HOOK_EVENT_SPECS) == set(HookEvent)
    assert (
        get_event_spec(HookEvent.SESSION_END).default_timeout_seconds
        == DEFAULT_COMMAND_TIMEOUT_SECONDS
    )
    assert get_event_spec(HookEvent.PERMISSION_REQUEST).matcher_field == "tool_name"


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
        cwd=project_dir,
        workspace_trusted=False,
        config_dir=user_dir,
    )
    assert [
        group.hooks[0].command
        for group in untrusted.config.hooks[HookEvent.SESSION_START]
    ] == ["user-hook"]
    assert untrusted.sources == (user_dir / "hooks.json",)

    loaded = load_hooks_config(
        cwd=project_dir,
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


def test_legacy_migration_only_maps_exact_session_end_semantics(
    tmp_path: Path,
) -> None:
    migrated = migrate_legacy_hooks(
        [
            {"command": ["echo", "start"], "events": ["session.start"]},
            {"command": ["echo", "compact"], "events": ["context.compact"]},
            {"command": ["echo", "tool"], "events": ["tool.use"]},
            {"command": ["echo", "end"], "events": ["session.end"]},
            {"command": ["echo", "perm"], "events": ["permission.request"]},
        ]
    )

    assert set(migrated.hooks) == {HookEvent.SESSION_END}
    assert HookEvent.SESSION_START not in migrated.hooks
    assert HookEvent.PRE_TOOL_USE not in migrated.hooks

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
        cwd=tmp_path,
        workspace_trusted=False,
        config_dir=user_dir,
    )

    assert HookEvent.SESSION_START not in loaded.config.hooks
    assert HookEvent.SESSION_END in loaded.config.hooks
    assert HookEvent.PRE_TOOL_USE not in loaded.config.hooks
    assert any(item.code == "legacy_migrated" for item in loaded.diagnostics)


def test_invalid_config_is_diagnosed(tmp_path: Path) -> None:
    config_dir = tmp_path / "user"
    config_dir.mkdir()
    path = config_dir / "hooks.json"
    path.write_text(
        '{"hooks":{"Stop":[{"hooks":[{"type":"http"}]}]}}',
        encoding="utf-8",
    )

    loaded = load_hooks_config(
        cwd=tmp_path,
        workspace_trusted=False,
        config_dir=config_dir,
    )

    assert loaded.config.hooks == {}
    assert loaded.sources == ()
    assert [item.code for item in loaded.diagnostics] == ["invalid_config"]
    assert loaded.diagnostics[0].field == str(path)


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
