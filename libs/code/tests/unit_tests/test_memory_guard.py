"""Tests for the managed onboarding-name memory guard middleware."""

from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING, Any, cast

import pytest
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from deepagents_code.memory_guard import ManagedMemoryGuardMiddleware
from deepagents_code.onboarding import (
    ONBOARDING_NAME_MEMORY_END,
    ONBOARDING_NAME_MEMORY_START,
    extract_onboarding_name_block,
)

if TYPE_CHECKING:
    from pathlib import Path

    from langgraph.types import Command


def _managed_file(path: Path, name: str = "Ada", *, extra: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "## User Preferences\n\n"
        f"{ONBOARDING_NAME_MEMORY_START}\n"
        f'- The user\'s preferred name is "{name}".\n'
        f"{ONBOARDING_NAME_MEMORY_END}\n"
        f"{extra}",
        encoding="utf-8",
    )


def _request(tool_name: str, file_path: str, **args: Any) -> ToolCallRequest:
    return ToolCallRequest(
        runtime=cast("Any", None),
        tool_call={
            "id": "call-1",
            "name": tool_name,
            "args": {"file_path": file_path, **args},
        },
        state={},
        tool=None,
    )


def _success(name: str = "edit_file") -> ToolMessage:
    return ToolMessage(content="ok", name=name, tool_call_id="call-1", status="success")


def test_edit_inside_managed_block_is_reverted(tmp_path) -> None:
    """An edit that rewrites the managed block is restored and reported as error."""
    path = tmp_path / "agent" / "AGENTS.md"
    _managed_file(path, "Ada", extra="\nKeep this note.\n")
    middleware = ManagedMemoryGuardMiddleware([str(path)])

    def handler(_request: ToolCallRequest) -> ToolMessage:
        path.write_text(
            path.read_text(encoding="utf-8").replace("Ada", "Mallory"),
            encoding="utf-8",
        )
        return _success()

    result = middleware.wrap_tool_call(_request("edit_file", str(path)), handler)

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    content = path.read_text(encoding="utf-8")
    assert '- The user\'s preferred name is "Ada".' in content
    assert "Mallory" not in content
    assert "Keep this note." in content


def test_edit_outside_managed_block_passes_through(tmp_path) -> None:
    """Edits that leave the managed block intact are not disturbed."""
    path = tmp_path / "agent" / "AGENTS.md"
    _managed_file(path, "Ada", extra="\nOld note.\n")
    middleware = ManagedMemoryGuardMiddleware([str(path)])

    def handler(_request: ToolCallRequest) -> ToolMessage:
        path.write_text(
            path.read_text(encoding="utf-8").replace("Old note.", "New note."),
            encoding="utf-8",
        )
        return _success()

    result = middleware.wrap_tool_call(_request("edit_file", str(path)), handler)

    assert isinstance(result, ToolMessage)
    assert result.status == "success"
    content = path.read_text(encoding="utf-8")
    assert "New note." in content
    assert extract_onboarding_name_block(content) is not None
    assert '- The user\'s preferred name is "Ada".' in content


def test_other_edits_preserved_when_block_reverted(tmp_path) -> None:
    """The model's unrelated edits survive even when the managed block is restored."""
    path = tmp_path / "agent" / "AGENTS.md"
    _managed_file(path, "Ada", extra="\nKeep this note.\n")
    middleware = ManagedMemoryGuardMiddleware([str(path)])

    def handler(_request: ToolCallRequest) -> ToolMessage:
        text = path.read_text(encoding="utf-8")
        text = text.replace("Ada", "Mallory").replace(
            "Keep this note.", "Added a real learning."
        )
        path.write_text(text, encoding="utf-8")
        return _success()

    result = middleware.wrap_tool_call(_request("edit_file", str(path)), handler)

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    content = path.read_text(encoding="utf-8")
    assert '- The user\'s preferred name is "Ada".' in content
    assert "Mallory" not in content
    assert "Added a real learning." in content


@pytest.mark.skipif(
    not hasattr(os, "O_NOFOLLOW"),
    reason="symlink hardening requires O_NOFOLLOW",
)
def test_restore_does_not_follow_replaced_guarded_file_symlink(tmp_path) -> None:
    """A symlink swap during restore must not overwrite the symlink target."""
    path = tmp_path / "agent" / "AGENTS.md"
    target = tmp_path / "target.txt"
    _managed_file(path, "Ada")
    target.write_text("do not overwrite\n", encoding="utf-8")
    middleware = ManagedMemoryGuardMiddleware([str(path)])

    def handler(_request: ToolCallRequest) -> ToolMessage:
        path.unlink()
        path.symlink_to(target)
        return _success()

    result = middleware.wrap_tool_call(_request("edit_file", str(path)), handler)

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert target.read_text(encoding="utf-8") == "do not overwrite\n"
    assert path.is_symlink()


def test_unguarded_file_passes_through(tmp_path) -> None:
    """A guarded middleware ignores writes to other files."""
    guarded = tmp_path / "agent" / "AGENTS.md"
    _managed_file(guarded, "Ada")
    other = tmp_path / "project" / "AGENTS.md"
    other.parent.mkdir(parents=True)
    other.write_text("project notes\n", encoding="utf-8")
    middleware = ManagedMemoryGuardMiddleware([str(guarded)])

    def handler(_request: ToolCallRequest) -> ToolMessage:
        other.write_text("rewritten\n", encoding="utf-8")
        return _success("write_file")

    result = middleware.wrap_tool_call(
        _request("write_file", str(other), content="rewritten\n"), handler
    )

    assert isinstance(result, ToolMessage)
    assert result.status == "success"
    assert other.read_text(encoding="utf-8") == "rewritten\n"


def test_non_write_tool_passes_through(tmp_path) -> None:
    """Read-only tools targeting the guarded file are never intercepted."""
    path = tmp_path / "agent" / "AGENTS.md"
    _managed_file(path, "Ada")
    middleware = ManagedMemoryGuardMiddleware([str(path)])

    sentinel = ToolMessage(
        content="contents", name="read_file", tool_call_id="call-1", status="success"
    )
    result = middleware.wrap_tool_call(
        _request("read_file", str(path)), lambda _r: sentinel
    )

    assert result is sentinel


def test_file_without_managed_block_passes_through(tmp_path) -> None:
    """When no managed block exists, edits are left untouched."""
    path = tmp_path / "agent" / "AGENTS.md"
    path.parent.mkdir(parents=True)
    path.write_text("## Notes\n\nfreeform\n", encoding="utf-8")
    middleware = ManagedMemoryGuardMiddleware([str(path)])

    def handler(_request: ToolCallRequest) -> ToolMessage:
        path.write_text("## Notes\n\nedited\n", encoding="utf-8")
        return _success()

    result = middleware.wrap_tool_call(_request("edit_file", str(path)), handler)

    assert isinstance(result, ToolMessage)
    assert result.status == "success"
    assert "edited" in path.read_text(encoding="utf-8")


async def test_async_edit_inside_managed_block_is_reverted(tmp_path) -> None:
    """The async wrapper reverts managed-block edits like the sync path."""
    path = tmp_path / "agent" / "AGENTS.md"
    _managed_file(path, "Ada")
    middleware = ManagedMemoryGuardMiddleware([str(path)])

    async def handler(_request: ToolCallRequest) -> ToolMessage:  # noqa: RUF029
        path.write_text(
            path.read_text(encoding="utf-8").replace("Ada", "Mallory"),
            encoding="utf-8",
        )
        return _success()

    result = await middleware.awrap_tool_call(_request("edit_file", str(path)), handler)

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "Mallory" not in path.read_text(encoding="utf-8")


def test_edit_removing_block_is_restored(tmp_path) -> None:
    """Dropping the managed block entirely restores it and reports an error."""
    path = tmp_path / "agent" / "AGENTS.md"
    _managed_file(path, "Ada", extra="\nKeep this note.\n")
    middleware = ManagedMemoryGuardMiddleware([str(path)])

    def handler(_request: ToolCallRequest) -> ToolMessage:
        text = path.read_text(encoding="utf-8")
        block = extract_onboarding_name_block(text)
        assert block is not None
        path.write_text(text.replace(block, "").rstrip() + "\n", encoding="utf-8")
        return _success()

    result = middleware.wrap_tool_call(_request("edit_file", str(path)), handler)

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    content = path.read_text(encoding="utf-8")
    assert extract_onboarding_name_block(content) is not None
    assert '- The user\'s preferred name is "Ada".' in content
    assert "Keep this note." in content


def test_partial_marker_edit_is_restored(tmp_path) -> None:
    """Deleting one marker still restores a clean block without orphan markers."""
    path = tmp_path / "agent" / "AGENTS.md"
    _managed_file(path, "Ada", extra="\nOld note.\n")
    middleware = ManagedMemoryGuardMiddleware([str(path)])

    def handler(_request: ToolCallRequest) -> ToolMessage:
        text = path.read_text(encoding="utf-8")
        text = text.replace(f"{ONBOARDING_NAME_MEMORY_END}\n", "")
        text = text.replace("Ada", "Mallory")
        text = text.replace("Old note.", "New note.")
        path.write_text(text, encoding="utf-8")
        return _success()

    result = middleware.wrap_tool_call(_request("edit_file", str(path)), handler)

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    content = path.read_text(encoding="utf-8")
    assert extract_onboarding_name_block(content) is not None
    assert content.count(ONBOARDING_NAME_MEMORY_START) == 1
    assert content.count(ONBOARDING_NAME_MEMORY_END) == 1
    assert content.count('- The user\'s preferred name is "Ada".') == 1
    assert "Mallory" not in content
    assert "New note." in content


def test_write_file_altering_block_is_reverted(tmp_path) -> None:
    """`write_file` clobbering the block is reverted like `edit_file`."""
    path = tmp_path / "agent" / "AGENTS.md"
    _managed_file(path, "Ada")
    middleware = ManagedMemoryGuardMiddleware([str(path)])

    def handler(_request: ToolCallRequest) -> ToolMessage:
        path.write_text(
            path.read_text(encoding="utf-8").replace("Ada", "Mallory"),
            encoding="utf-8",
        )
        return _success("write_file")

    result = middleware.wrap_tool_call(
        _request("write_file", str(path), content="ignored"), handler
    )

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    content = path.read_text(encoding="utf-8")
    assert "Mallory" not in content
    assert '- The user\'s preferred name is "Ada".' in content


def test_file_created_with_block_passes_through(tmp_path) -> None:
    """Creating the guarded file with a fresh block is not treated as an edit."""
    path = tmp_path / "agent" / "AGENTS.md"  # does not exist yet
    middleware = ManagedMemoryGuardMiddleware([str(path)])

    def handler(_request: ToolCallRequest) -> ToolMessage:
        _managed_file(path, "Ada")
        return _success("write_file")

    result = middleware.wrap_tool_call(
        _request("write_file", str(path), content="ignored"), handler
    )

    assert isinstance(result, ToolMessage)
    assert result.status == "success"
    assert '- The user\'s preferred name is "Ada".' in path.read_text(encoding="utf-8")


def test_error_message_propagates_call_metadata(tmp_path) -> None:
    """The error result carries the originating tool name and call id."""
    path = tmp_path / "agent" / "AGENTS.md"
    _managed_file(path, "Ada")
    middleware = ManagedMemoryGuardMiddleware([str(path)])

    def handler(_request: ToolCallRequest) -> ToolMessage:
        path.write_text(
            path.read_text(encoding="utf-8").replace("Ada", "Mallory"),
            encoding="utf-8",
        )
        return _success()

    result = middleware.wrap_tool_call(_request("edit_file", str(path)), handler)

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert result.name == "edit_file"
    assert result.tool_call_id == "call-1"


async def test_async_edit_outside_block_passes_through(tmp_path) -> None:
    """The async wrapper passes through edits that leave the block intact."""
    path = tmp_path / "agent" / "AGENTS.md"
    _managed_file(path, "Ada", extra="\nOld note.\n")
    middleware = ManagedMemoryGuardMiddleware([str(path)])

    async def handler(_request: ToolCallRequest) -> ToolMessage:  # noqa: RUF029
        path.write_text(
            path.read_text(encoding="utf-8").replace("Old note.", "New note."),
            encoding="utf-8",
        )
        return _success()

    result = await middleware.awrap_tool_call(_request("edit_file", str(path)), handler)

    assert isinstance(result, ToolMessage)
    assert result.status == "success"
    content = path.read_text(encoding="utf-8")
    assert "New note." in content
    assert extract_onboarding_name_block(content) is not None


async def test_async_guard_filesystem_helpers_run_off_event_loop(
    tmp_path, monkeypatch
) -> None:
    """The async wrapper offloads guard filesystem work to worker threads."""
    path = tmp_path / "agent" / "AGENTS.md"
    _managed_file(path, "Ada")
    middleware = ManagedMemoryGuardMiddleware([str(path)])
    event_loop_thread = threading.get_ident()
    helper_threads: list[int] = []

    original_guarded_path = middleware._guarded_path
    original_read = middleware._read
    original_result_after_restore = middleware._result_after_restore

    def guarded_path(request: ToolCallRequest) -> Path | None:
        helper_threads.append(threading.get_ident())
        return original_guarded_path(request)

    def read(path: Path) -> str | None:
        helper_threads.append(threading.get_ident())
        return original_read(path)

    def result_after_restore(
        request: ToolCallRequest,
        path: Path,
        before: str,
        before_block: str,
        result: ToolMessage | Command[Any],
    ) -> ToolMessage | Command[Any]:
        helper_threads.append(threading.get_ident())
        return original_result_after_restore(
            request, path, before, before_block, result
        )

    monkeypatch.setattr(middleware, "_guarded_path", guarded_path)
    monkeypatch.setattr(middleware, "_read", read)
    monkeypatch.setattr(middleware, "_result_after_restore", result_after_restore)

    async def handler(_request: ToolCallRequest) -> ToolMessage:  # noqa: RUF029
        path.write_text(
            path.read_text(encoding="utf-8").replace("Ada", "Mallory"),
            encoding="utf-8",
        )
        return _success()

    result = await middleware.awrap_tool_call(_request("edit_file", str(path)), handler)

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert helper_threads
    assert all(thread != event_loop_thread for thread in helper_threads)
