"""Tests for workspace refusal diagnostics and persisted snapshots."""

from __future__ import annotations

import sqlite3
from contextlib import closing

import pytest

from deepagents_code.workspace import (
    WorkspaceConflictError,
    bind_thread_workspace,
    require_thread_workspace,
)
from deepagents_code.workspace_diagnostics import (
    FieldChange,
    WorkspaceDiagnostics,
    diff_snapshots,
    format_diagnostics_content,
    snapshot_for_payload,
)


@pytest.fixture(autouse=True)
def workspace_database(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Point bindings at an isolated SQLite database."""
    database = tmp_path / "sessions.db"
    monkeypatch.setenv("DEEPAGENTS_CODE_SERVER_DB_PATH", str(database))
    return database


class TestSnapshotAllowlist:
    def test_snapshot_fields_stay_an_allowlisted_subset_of_the_payload(
        self,
    ) -> None:
        """A new payload field is never snapshotted until it is allowlisted."""
        from deepagents_code._server_config import ServerConfig
        from deepagents_code.workspace_diagnostics import SAFE_SNAPSHOT_FIELDS

        payload_keys = set(ServerConfig().to_workspace_payload())
        assert payload_keys > SAFE_SNAPSHOT_FIELDS
        # Path-valued payload fields are never snapshotted: they can carry
        # user-identifying directory names.
        assert SAFE_SNAPSHOT_FIELDS.isdisjoint(
            {
                "sandbox_setup",
                "mcp_config_path",
                "extension_paths",
                "sandbox_id",
                "sandbox_snapshot_name",
            }
        )

    def test_snapshot_records_only_allowlisted_fields(self) -> None:
        snapshot = snapshot_for_payload(
            {
                "auto_approve": True,
                "sandbox_type": "daytona",
                "sandbox_setup": "/home/user/private/setup.sh",
                "extension_paths": ["/home/user/ext.py"],
                "mcp_config_path": "/home/user/mcp.json",
            }
        )

        assert snapshot.fields == {"auto_approve": True, "sandbox_type": "daytona"}
        assert "setup.sh" not in snapshot.to_json()
        assert "ext.py" not in snapshot.to_json()
        assert "mcp.json" not in snapshot.to_json()

    def test_snapshot_omits_unbounded_values(self) -> None:
        snapshot = snapshot_for_payload(
            {"sandbox_type": "x" * 4096, "shell_allow_list": ["y" * 257]}
        )

        assert snapshot.fields == {}

    def test_snapshot_truncation_never_rejects_large_policies(self) -> None:
        snapshot = snapshot_for_payload(
            {
                "auto_approve": True,
                "shell_allow_list": [str(index) * 256 for index in range(100)],
            }
        )

        assert snapshot.fields["auto_approve"] is True
        assert len(snapshot.to_json()) < 16_100

    def test_snapshot_round_trip(self) -> None:
        snapshot = snapshot_for_payload(
            {"auto_approve": True, "shell_allow_list": ["git status", "ls"]}
        )

        from deepagents_code.workspace_diagnostics import WorkspaceSnapshot

        parsed = WorkspaceSnapshot.from_json(snapshot.to_json())

        assert parsed == snapshot
        assert parsed.fields["shell_allow_list"] == ["git status", "ls"]

    def test_snapshot_rejects_malformed_json(self) -> None:
        from deepagents_code.workspace_diagnostics import WorkspaceSnapshot

        with pytest.raises(ValueError, match="not valid JSON"):
            WorkspaceSnapshot.from_json("not-json")
        with pytest.raises(ValueError, match="unsupported format"):
            WorkspaceSnapshot.from_json('{"version": 99, "fields": {}}')


async def test_binding_persists_the_allowlisted_snapshot(
    tmp_path, workspace_database
) -> None:
    """The snapshot lands in the session DB beside the binding."""
    config = {
        "auto_approve": True,
        "sandbox_setup": "/home/user/private/setup.sh",
    }
    await bind_thread_workspace("thread-1", str(tmp_path), config)

    with closing(sqlite3.connect(workspace_database)) as conn, conn:
        row = conn.execute(
            "SELECT snapshot_version, snapshot_json FROM dcode_workspace_snapshots"
        ).fetchone()

    assert row is not None
    assert row[0] == 1
    assert '"auto_approve":true' in row[1]
    assert "setup.sh" not in row[1]


async def test_binding_rejection_carries_field_diagnostics(tmp_path) -> None:
    """A refused rebind names the allowlisted fields that changed."""
    await bind_thread_workspace(
        "thread-1", str(tmp_path), {"auto_approve": False, "sandbox_type": None}
    )

    with pytest.raises(WorkspaceConflictError) as exc_info:
        await bind_thread_workspace(
            "thread-1",
            str(tmp_path),
            {"auto_approve": True, "sandbox_type": "daytona"},
        )

    diagnostics = exc_info.value.diagnostics
    assert diagnostics is not None
    assert diagnostics.category == "config_drift"
    assert diagnostics.snapshot_status == "current"
    changes = {change.name: change for change in diagnostics.changes}
    assert changes["auto_approve"].bound is False
    assert changes["auto_approve"].current is True
    assert changes["sandbox_type"].bound is None
    assert changes["sandbox_type"].current == "daytona"
    # Path-valued fields are never named, even when present in the payloads.
    assert "sandbox_setup" not in changes


async def test_binding_rejection_does_not_overwrite_the_snapshot(
    tmp_path, workspace_database
) -> None:
    """A rejected binding never replaces the recorded comparison snapshot."""
    await bind_thread_workspace("thread-1", str(tmp_path), {"auto_approve": False})
    with closing(sqlite3.connect(workspace_database)) as conn, conn:
        before = conn.execute(
            "SELECT snapshot_json FROM dcode_workspace_snapshots"
        ).fetchone()[0]

    with pytest.raises(WorkspaceConflictError):
        await bind_thread_workspace("thread-1", str(tmp_path), {"auto_approve": True})

    with closing(sqlite3.connect(workspace_database)) as conn, conn:
        after = conn.execute(
            "SELECT snapshot_json FROM dcode_workspace_snapshots"
        ).fetchone()[0]
    assert after == before


async def test_binding_rejection_is_logged(caplog, tmp_path) -> None:
    """Binding refusals log the reason and changed field names, not values."""
    import logging

    await bind_thread_workspace("thread-1", str(tmp_path), {"auto_approve": False})

    with (
        caplog.at_level(logging.WARNING, logger="deepagents_code.workspace"),
        pytest.raises(WorkspaceConflictError),
    ):
        await bind_thread_workspace("thread-1", str(tmp_path), {"auto_approve": True})

    messages = [record.getMessage() for record in caplog.records]
    assert any("binding refused" in message.lower() for message in messages)
    assert any("auto_approve" in message for message in messages)


async def test_legacy_binding_reports_snapshot_unavailable(tmp_path) -> None:
    """A pre-snapshot binding refuses with an explicit unavailable status."""
    await bind_thread_workspace("thread-1", str(tmp_path), {"auto_approve": False})
    import deepagents_code.workspace as workspace_mod

    db = workspace_mod._database_path()
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.execute("DELETE FROM dcode_workspace_snapshots")

    with pytest.raises(WorkspaceConflictError) as exc_info:
        await bind_thread_workspace("thread-1", str(tmp_path), {"auto_approve": True})

    diagnostics = exc_info.value.diagnostics
    assert diagnostics is not None
    assert diagnostics.snapshot_status == "unavailable"
    # Legacy rows still name drifted fields, without claiming values.
    assert any(change.name == "auto_approve" for change in diagnostics.changes)


async def test_unbound_thread_and_schema_diagnostics(tmp_path) -> None:
    """Structural refusals carry their categories for client display."""
    binding = await bind_thread_workspace("thread-1", str(tmp_path), {})

    with pytest.raises(WorkspaceConflictError) as exc_info:
        await require_thread_workspace("thread-unknown", binding.to_payload())
    assert exc_info.value.diagnostics is not None
    assert exc_info.value.diagnostics.category == "unbound_thread"

    payload = binding.to_payload()
    payload["cwd"] = "/tmp/elsewhere"
    with pytest.raises(WorkspaceConflictError) as exc_info:
        await require_thread_workspace("thread-1", payload)
    assert exc_info.value.diagnostics is not None
    assert exc_info.value.diagnostics.category == "context_mismatch"


class TestDiagnosticsWireFormat:
    def test_round_trip(self) -> None:
        diagnostics = WorkspaceDiagnostics(
            category="policy_drift",
            reason="policy changed",
            changes=diff_snapshots(
                snapshot_for_payload({"auto_approve": False}),
                snapshot_for_payload({"auto_approve": True}),
                changed_names=["sandbox_setup"],
            ),
        )

        parsed = WorkspaceDiagnostics.from_dict(diagnostics.to_dict())

        assert parsed is not None
        assert parsed.category == "policy_drift"
        by_name = {change.name: change for change in parsed.changes}
        assert by_name["auto_approve"].bound is False
        assert by_name["auto_approve"].current is True
        assert by_name["sandbox_setup"].state == "values_unavailable"

    def test_from_dict_tolerates_absent_and_malformed_payloads(self) -> None:
        assert WorkspaceDiagnostics.from_dict(None) is None
        assert WorkspaceDiagnostics.from_dict("nope") is None
        assert WorkspaceDiagnostics.from_dict({}) is None
        assert (
            WorkspaceDiagnostics.from_dict(
                {"category": 1, "reason": "x", "snapshot_status": "current"}
            )
            is None
        )
        assert (
            WorkspaceDiagnostics.from_dict(
                {"category": "config_drift", "reason": "x", "snapshot_status": []}
            )
            is None
        )

    def test_log_summary_names_fields_not_values(self) -> None:
        diagnostics = WorkspaceDiagnostics(
            category="config_drift",
            reason="config changed",
            changes=diff_snapshots(
                snapshot_for_payload({"shell_allow_list": ["git secret-cmd"]}),
                snapshot_for_payload({"shell_allow_list": ["ls"]}),
            ),
        )

        summary = diagnostics.log_summary()

        assert "shell_allow_list" in summary
        assert "secret-cmd" not in summary


class TestDiagnosticsContent:
    @pytest.mark.parametrize(
        ("field", "bound", "current", "instruction"),
        [
            (
                "interpreter_ptc",
                "safe",
                None,
                "Set interpreter_ptc to safe (currently unset)",
            ),
            ("interpreter_ptc", None, "safe", "Unset interpreter_ptc (currently safe)"),
            ("auto_approve", False, True, "Set auto_approve to off (currently on)"),
            ("recursion_limit", 100, 200, "Set recursion_limit to 100 (currently 200)"),
            (
                "shell_allow_list",
                ["ls", "pwd"],
                ["git status"],
                "Set shell_allow_list to ls, pwd (currently git status)",
            ),
        ],
    )
    def test_restore_instructions_use_bound_values(
        self, field: str, bound: object, current: object, instruction: str
    ) -> None:
        diagnostics = WorkspaceDiagnostics(
            category="config_drift",
            reason="config changed",
            changes=diff_snapshots(
                snapshot_for_payload({field: bound}),
                snapshot_for_payload({field: current}),
            ),
        )

        text = format_diagnostics_content(diagnostics).plain

        assert "To resume this thread, restore these settings and relaunch:" in text
        assert instruction in text

    @pytest.mark.parametrize(
        "omitted",
        ["x" * 257, ["x" * 257], ["x" * 256] * 64],
        ids=["long-scalar", "long-command", "snapshot-size-limit"],
    )
    @pytest.mark.parametrize("omit_bound", [True, False])
    def test_omitted_values_are_unavailable(
        self, omitted: object, *, omit_bound: bool
    ) -> None:
        bound, current = (omitted, ["ls"]) if omit_bound else (["ls"], omitted)
        diagnostics = WorkspaceDiagnostics(
            category="config_drift",
            reason="config changed",
            changes=diff_snapshots(
                snapshot_for_payload({"shell_allow_list": bound}),
                snapshot_for_payload({"shell_allow_list": current}),
            ),
        )

        wire = diagnostics.to_dict()
        assert wire["changes"] == [
            {"name": "shell_allow_list", "state": "values_unavailable"}
        ]
        parsed = WorkspaceDiagnostics.from_dict(wire)
        assert parsed is not None
        text = format_diagnostics_content(parsed).plain
        assert "Restore shell_allow_list to its original value (unavailable)" in text
        assert "Unset shell_allow_list" not in text
        assert "currently unset" not in text

    def test_legacy_omission_is_not_treated_as_unset(self) -> None:
        diagnostics = WorkspaceDiagnostics(
            category="config_drift",
            reason="config changed",
            changes=diff_snapshots(
                snapshot_for_payload({}),
                snapshot_for_payload({"interpreter_ptc": "safe"}),
            ),
        )

        text = format_diagnostics_content(diagnostics).plain
        assert "Restore interpreter_ptc to its original value (unavailable)" in text
        assert "Unset interpreter_ptc" not in text

    def test_unknown_original_value_is_not_treated_as_unset(self) -> None:
        diagnostics = WorkspaceDiagnostics(
            category="config_drift",
            reason="config changed",
            changes=(FieldChange(name="interpreter_ptc", state="values_unavailable"),),
            snapshot_status="unavailable",
        )

        text = format_diagnostics_content(diagnostics).plain

        assert "Restore interpreter_ptc to its original value (unavailable)" in text
        assert "Unset interpreter_ptc" not in text

    def test_renders_changes_without_markup_injection(self) -> None:
        """Bracket-shaped values cannot break Rich markup parsing."""
        diagnostics = WorkspaceDiagnostics(
            category="policy_drift",
            reason="policy changed",
            changes=diff_snapshots(
                snapshot_for_payload({"sandbox_type": "da[y]tona[/]\u202e"}),
                snapshot_for_payload({"sandbox_type": None}),
            ),
        )

        content = format_diagnostics_content(diagnostics)

        text = content.plain
        assert "Set sandbox_type to da[y]tona[/] (currently unset)" in text
        assert "\u202e" not in text
        assert "policy changed" in text

    def test_unavailable_snapshot_is_explained(self) -> None:
        diagnostics = WorkspaceDiagnostics(
            category="config_drift",
            reason="config changed",
            snapshot_status="unavailable",
        )

        content = format_diagnostics_content(diagnostics)

        assert "predates recorded configuration snapshots" in content.plain
