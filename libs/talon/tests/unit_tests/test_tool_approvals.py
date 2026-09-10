"""Storage, snapshot, and authorization contracts for tool approvals."""

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path
from threading import Barrier
from unittest.mock import patch

import pytest

from deepagents_talon.tool_approvals import (
    ACTIVE_APPROVALS,
    APPROVAL_OPERATOR,
    ApprovalSnapshot,
    ToolApprovalStore,
)


def test_ensure_defaults_and_preserve(tmp_path):
    path = tmp_path / "nested" / "tools.json"
    store = ToolApprovalStore(path)
    with pytest.raises(FileNotFoundError):
        store.read()
    snapshot = store.ensure()
    assert snapshot.approvals == {
        "update_tool_approvals": True,
        "delete_conversations": True,
        "update_mcp_server": True,
        "start_async_task": True,
    }
    assert snapshot.approvals.get("execute", False) is False
    assert snapshot.interrupt_on["update_tool_approvals"] == {
        "allowed_decisions": ["approve", "reject"]
    }
    raw = b'{"custom": false}\n'
    path.write_bytes(raw)
    assert store.ensure().approvals == {"custom": False}
    assert path.read_bytes() == raw


def test_snapshot_immutable():
    policy = {"one": True, "two": False}
    snapshot = ApprovalSnapshot("revision", policy)
    policy["one"] = False
    assert snapshot.approvals["one"] is True
    with pytest.raises(TypeError):
        snapshot.approvals["one"] = False
    with pytest.raises(FrozenInstanceError):
        snapshot.revision = "changed"
    interrupts = snapshot.interrupt_on
    interrupts["one"]["allowed_decisions"].append("edit")
    assert snapshot.interrupt_on == {"one": {"allowed_decisions": ["approve", "reject"]}}


@pytest.mark.parametrize(
    "raw",
    [
        b"not-json-secret",
        b"[]",
        b"null",
        b'{"tool": 1}',
        b'{"tool": "true"}',
        b'{"tool": null}',
        b'{"tool": {}}',
        b'{"tool": true, "tool": false}',
        b'{"": true}',
        b'{" tool": true}',
        b'{"tool ": true}',
        b'{"to*ol": true}',
        b'{"to?ol": true}',
        b'{"to[ol]": true}',
        b'{"to\\u0000ol": true}',
        b'{"to\\u007fol": true}',
        b'{"to\\u200bol": true}',
        json.dumps({"a" * 257: True}).encode(),
        json.dumps(dict.fromkeys((f"tool{i}" for i in range(4097)), True)).encode(),
        b" " * 1_048_577,
    ],
)
def test_invalid_storage_fails_without_overwrite(tmp_path, raw):
    path = tmp_path / "tools.json"
    path.write_bytes(raw)
    store = ToolApprovalStore(path)
    for read in (store.read, store.ensure):
        with pytest.raises((ValueError, TypeError)):
            read()
        assert path.read_bytes() == raw
    active = ApprovalSnapshot("old", {"execute": True})
    view = store.view(active)
    assert view["status"] == "error"
    assert view["active_revision"] == "old"
    assert view["active_tools"] == {"execute": True}
    assert view["persisted_revision"] is None
    assert view["saved_changes_inactive"] is None
    assert "secret" not in str(view)
    assert store.update({"execute": False}, "old")["status"] == "error"
    assert path.read_bytes() == raw


def test_revisions_batch_and_active_view(tmp_path):
    path = tmp_path / "tools.json"
    store = ToolApprovalStore(path)
    active = store.ensure()
    assert active.revision == hashlib.sha256(path.read_bytes()).hexdigest()
    assert store.view(active)["saved_changes_inactive"] is False
    result = store.update({"mcp.server/tool": True, "delete_conversations": False}, active.revision)
    assert result["status"] == "updated"
    saved = store.read()
    assert saved.revision != active.revision
    assert saved.approvals["update_mcp_server"] is True
    assert saved.approvals["delete_conversations"] is False
    assert saved.approvals["mcp.server/tool"] is True
    assert active.approvals["delete_conversations"] is True
    view = store.view(active)
    assert view["persisted_revision"] == saved.revision
    assert view["active_revision"] == active.revision
    assert view["saved_changes_inactive"] is True
    assert store.update({}, saved.revision)["persisted_revision"] == saved.revision
    path.write_bytes(path.read_bytes() + b" \n")
    assert store.read().approvals == saved.approvals
    assert store.read().revision != saved.revision
    assert store.update({"execute": True}, saved.revision)["status"] == "conflict"


@pytest.mark.parametrize("invalid", [1, "false", None, [], {}, False])
def test_batch_invalid_rollback(tmp_path, invalid):
    path = tmp_path / "tools.json"
    store = ToolApprovalStore(path)
    active = store.ensure()
    raw = path.read_bytes()
    updates = {"execute": True, "bad*name" if invalid is False else "other": invalid}
    assert store.update(updates, active.revision)["status"] == "error"
    assert path.read_bytes() == raw


@pytest.mark.parametrize("operation", ["pathlib.Path.replace", "os.fsync"])
def test_atomic_write_failure_preserves_policy(tmp_path: Path, operation: str) -> None:
    path = tmp_path / "tools.json"
    raw = b'{ "execute": true, "custom": false }\n'
    path.write_bytes(raw)
    store = ToolApprovalStore(path)
    active = store.read()

    with patch(operation, side_effect=OSError("injected write failure")) as failure:
        result = store.update({"execute": False, "new_tool": True}, active.revision)

    failure.assert_called_once()
    assert result["status"] == "error"
    assert path.read_bytes() == raw
    assert store.read() == active
    assert not list(tmp_path.glob(".tools-*"))


def test_concurrent_stores_compare_and_swap(tmp_path):
    path = tmp_path / "tools.json"
    active = ToolApprovalStore(path).ensure()
    barrier = Barrier(2)

    def update(name):
        store = ToolApprovalStore(path)
        barrier.wait(timeout=5)
        return store.update({name: True}, active.revision)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(update, ("one", "two")))
    assert sorted(result["status"] for result in results) == ["conflict", "updated"]
    saved = ToolApprovalStore(path).read()
    assert sum(name in saved.approvals for name in ("one", "two")) == 1


@pytest.mark.parametrize("dangling", [True, False])
def test_symlink_rejected(tmp_path, dangling):
    target = tmp_path / "target"
    if not dangling:
        target.write_text('{"secret": true}')
    path = tmp_path / "tools.json"
    path.symlink_to(target)
    store = ToolApprovalStore(path)
    for read in (store.read, store.ensure):
        with pytest.raises(OSError, match="Too many levels"):
            read()
    assert store.update({"execute": True}, "revision")["status"] == "error"
    assert path.is_symlink()
    assert not target.exists() if dangling else target.read_text() == '{"secret": true}'


@pytest.mark.parametrize("kind", ["directory", "fifo"])
def test_nonregular_rejected(tmp_path, kind):
    path = tmp_path / "tools.json"
    if kind == "directory":
        path.mkdir()
    else:
        os.mkfifo(path)
    with pytest.raises((OSError, ValueError)):
        ToolApprovalStore(path).read()


@pytest.mark.parametrize(("operator", "invocation"), [(False, False), (False, True), (True, False)])
def test_tool_denies_without_operator_and_invocation(tmp_path, operator, invocation):
    path = tmp_path / "tools.json"
    path.write_text('{"update_tool_approvals": false}')
    store = ToolApprovalStore(path)
    active = store.read()
    _, update = store.tools(active)
    operator_token = APPROVAL_OPERATOR.set(operator)
    active_token = ACTIVE_APPROVALS.set(active if invocation else None)
    try:
        result = update.invoke(
            {"updates": {"execute": False}, "expected_revision": active.revision}
        )
    finally:
        APPROVAL_OPERATOR.reset(operator_token)
        ACTIVE_APPROVALS.reset(active_token)
    assert result["status"] == "error"
    assert store.read() == active


def test_authorized_tools_keep_old_snapshot(tmp_path):
    store = ToolApprovalStore(tmp_path / "tools.json")
    active = store.ensure()
    read, update = store.tools(active)
    operator_token = APPROVAL_OPERATOR.set(True)
    active_token = ACTIVE_APPROVALS.set(active)
    try:
        result = update.invoke(
            {"updates": {"update_tool_approvals": False}, "expected_revision": active.revision}
        )
    finally:
        APPROVAL_OPERATOR.reset(operator_token)
        ACTIVE_APPROVALS.reset(active_token)
    assert result["status"] == "updated"
    assert result["available"] == "next_invocation"
    assert result["saved_changes_inactive"] is True
    assert read.invoke({})["active_tools"]["update_tool_approvals"] is True
    assert read.invoke({})["tools"]["update_tool_approvals"] is False
