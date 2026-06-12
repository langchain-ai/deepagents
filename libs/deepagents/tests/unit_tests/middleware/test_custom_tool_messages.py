"""Tests for `custom_tool_messages` on FilesystemMiddleware (#2941)."""

from pathlib import Path

import pytest
from langchain.tools import ToolRuntime

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.filesystem import FilesystemMiddleware, FilesystemState


def _middleware(tmp_path: Path, **kwargs: object) -> FilesystemMiddleware:
    return FilesystemMiddleware(backend=FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True), **kwargs)  # type: ignore[arg-type]


def _runtime() -> ToolRuntime:
    return ToolRuntime(
        state=FilesystemState(messages=[], files={}),
        context=None,
        tool_call_id="call-1",
        store=None,
        stream_writer=lambda _: None,
        config={},
    )


def _tool(middleware: FilesystemMiddleware, name: str):
    return next(tool for tool in middleware.tools if tool.name == name)


def test_write_file_default_message_unchanged(tmp_path: Path) -> None:
    middleware = _middleware(tmp_path)
    result = _tool(middleware, "write_file").invoke({"file_path": "/memory.md", "content": "hi", "runtime": _runtime()})
    assert result.content == "Updated file /memory.md"
    assert result.status == "success"


def test_write_file_custom_message(tmp_path: Path) -> None:
    middleware = _middleware(
        tmp_path,
        custom_tool_messages={"write_file": "Created {path}. Refer to it by this exact absolute path."},
    )
    result = _tool(middleware, "write_file").invoke({"file_path": "/memory.md", "content": "hi", "runtime": _runtime()})
    assert result.content == "Created /memory.md. Refer to it by this exact absolute path."
    assert result.status == "success"


def test_edit_file_custom_message_with_occurrences(tmp_path: Path) -> None:
    middleware = _middleware(tmp_path, custom_tool_messages={"edit_file": "Replaced {occurrences}x in {path}"})
    runtime = _runtime()
    _tool(middleware, "write_file").invoke({"file_path": "/notes.txt", "content": "a b a", "runtime": runtime})
    result = _tool(middleware, "edit_file").invoke(
        {
            "file_path": "/notes.txt",
            "old_string": "a",
            "new_string": "c",
            "replace_all": True,
            "runtime": runtime,
        }
    )
    assert result.content == "Replaced 2x in /notes.txt"
    assert result.status == "success"


def test_error_messages_not_affected_by_custom_message(tmp_path: Path) -> None:
    (tmp_path / "adir").mkdir()
    middleware = _middleware(tmp_path, custom_tool_messages={"write_file": "Created {path}"})
    result = _tool(middleware, "write_file").invoke({"file_path": "/adir", "content": "hi", "runtime": _runtime()})
    assert result.status == "error"
    assert "Created" not in result.content


def test_unsupported_tool_key_raises() -> None:
    with pytest.raises(ValueError, match="unsupported tool 'read_file'"):
        FilesystemMiddleware(custom_tool_messages={"read_file": "nope"})


def test_unknown_placeholder_raises() -> None:
    with pytest.raises(ValueError, match=r"unknown placeholder\(s\) \['size'\]"):
        FilesystemMiddleware(custom_tool_messages={"write_file": "Wrote {path} ({size} bytes)"})


async def test_write_file_custom_message_async(tmp_path: Path) -> None:
    middleware = _middleware(tmp_path, custom_tool_messages={"write_file": "Saved {path}"})
    result = await _tool(middleware, "write_file").ainvoke({"file_path": "/memory.md", "content": "hi", "runtime": _runtime()})
    assert result.content == "Saved /memory.md"
    assert result.status == "success"
