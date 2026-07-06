"""Unit tests for FilesystemMiddleware `custom_tool_messages`."""

from pathlib import Path

import pytest
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage

from deepagents.backends import FilesystemBackend, StateBackend
from deepagents.middleware.filesystem import FilesystemMiddleware


def _runtime(tool_call_id: str = "tc") -> ToolRuntime:
    return ToolRuntime(
        state={"messages": [], "files": {}},
        context=None,
        tool_call_id=tool_call_id,
        store=None,
        stream_writer=lambda _: None,
        config={},
    )


def _middleware(tmp_path: Path, **kwargs: object) -> FilesystemMiddleware:
    return FilesystemMiddleware(
        backend=FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True),
        **kwargs,
    )


def _tool(middleware: FilesystemMiddleware, name: str):
    return next(tool for tool in middleware.tools if tool.name == name)


class TestCustomToolMessagesValidation:
    """Templates are validated at initialization."""

    def test_unknown_tool_key_rejected(self) -> None:
        with pytest.raises(ValueError, match="does not support tool 'read_file'"):
            FilesystemMiddleware(
                backend=StateBackend(),
                custom_tool_messages={"read_file": "Read {path}"},
            )

    def test_unsupported_placeholder_rejected(self) -> None:
        with pytest.raises(ValueError, match="unsupported placeholder"):
            FilesystemMiddleware(
                backend=StateBackend(),
                custom_tool_messages={"write_file": "Wrote {bogus}"},
            )

    def test_write_file_rejects_occurrences_placeholder(self) -> None:
        # {occurrences} is valid for edit_file only.
        with pytest.raises(ValueError, match="unsupported placeholder"):
            FilesystemMiddleware(
                backend=StateBackend(),
                custom_tool_messages={"write_file": "Wrote {occurrences}"},
            )

    def test_conversion_rejected(self) -> None:
        with pytest.raises(ValueError, match="format specifiers or conversions"):
            FilesystemMiddleware(
                backend=StateBackend(),
                custom_tool_messages={"write_file": "Wrote {path!r}"},
            )

    def test_format_spec_rejected(self) -> None:
        with pytest.raises(ValueError, match="format specifiers or conversions"):
            FilesystemMiddleware(
                backend=StateBackend(),
                custom_tool_messages={"write_file": "Wrote {path:.20}"},
            )

    def test_auto_numbering_rejected(self) -> None:
        with pytest.raises(ValueError, match="unsupported placeholder"):
            FilesystemMiddleware(
                backend=StateBackend(),
                custom_tool_messages={"write_file": "Wrote {}"},
            )

    def test_edit_file_allows_both_placeholders(self) -> None:
        # Should not raise.
        FilesystemMiddleware(
            backend=StateBackend(),
            custom_tool_messages={"edit_file": "Edited {path} ({occurrences})"},
        )

    def test_literal_braces_allowed(self) -> None:
        # Escaped braces are literal text, not placeholders.
        FilesystemMiddleware(
            backend=StateBackend(),
            custom_tool_messages={"write_file": "Wrote {path} {{literal}}"},
        )


class TestWriteFileSuccessMessage:
    """`write_file` honors the custom template on success only."""

    def test_default_message_unchanged_when_unset(self, tmp_path: Path) -> None:
        mw = _middleware(tmp_path)
        result = _tool(mw, "write_file").invoke({"file_path": "/memory.md", "content": "hi", "runtime": _runtime()})
        assert isinstance(result, ToolMessage)
        assert result.status == "success"
        assert result.content == "Updated file /memory.md"

    def test_custom_message_rendered(self, tmp_path: Path) -> None:
        mw = _middleware(
            tmp_path,
            custom_tool_messages={
                "write_file": "Created or updated {path}. Use this exact absolute path in later tool calls.",
            },
        )
        result = _tool(mw, "write_file").invoke({"file_path": "/memory.md", "content": "hi", "runtime": _runtime()})
        assert result.status == "success"
        assert result.content == "Created or updated /memory.md. Use this exact absolute path in later tool calls."

    async def test_custom_message_rendered_async(self, tmp_path: Path) -> None:
        mw = _middleware(tmp_path, custom_tool_messages={"write_file": "Saved {path}"})
        result = await _tool(mw, "write_file").ainvoke({"file_path": "/notes.md", "content": "hi", "runtime": _runtime()})
        assert result.status == "success"
        assert result.content == "Saved /notes.md"

    def test_error_message_not_overridden(self, tmp_path: Path) -> None:
        mw = _middleware(tmp_path, custom_tool_messages={"write_file": "Saved {path}"})
        result = _tool(mw, "write_file").invoke({"file_path": "../escape.md", "content": "hi", "runtime": _runtime()})
        assert result.status == "error"
        assert "Saved" not in str(result.content)


class TestEditFileSuccessMessage:
    """`edit_file` honors the custom template with `{path}` and `{occurrences}`."""

    def _seed(self, mw: FilesystemMiddleware, content: str) -> None:
        _tool(mw, "write_file").invoke({"file_path": "/doc.md", "content": content, "runtime": _runtime()})

    def test_default_message_unchanged_when_unset(self, tmp_path: Path) -> None:
        mw = _middleware(tmp_path)
        self._seed(mw, "a b c")
        result = _tool(mw, "edit_file").invoke(
            {
                "file_path": "/doc.md",
                "old_string": "a",
                "new_string": "x",
                "runtime": _runtime(),
            }
        )
        assert result.status == "success"
        assert result.content == "Successfully replaced 1 instance(s) of the string in '/doc.md'"

    def test_custom_message_rendered_with_occurrences(self, tmp_path: Path) -> None:
        mw = _middleware(
            tmp_path,
            custom_tool_messages={"edit_file": "Edited {path}: {occurrences} change(s)"},
        )
        self._seed(mw, "a a a")
        result = _tool(mw, "edit_file").invoke(
            {
                "file_path": "/doc.md",
                "old_string": "a",
                "new_string": "b",
                "replace_all": True,
                "runtime": _runtime(),
            }
        )
        assert result.status == "success"
        assert result.content == "Edited /doc.md: 3 change(s)"
