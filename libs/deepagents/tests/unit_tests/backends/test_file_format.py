"""Tests for the new str-based file format with encoding support.

Covers:
- Text round-trip through create_file_data / file_data_to_string
- Binary (base64) round-trip
- Legacy list[str] backwards compatibility (with DeprecationWarning)
- Store backend upload/download for binary and text
- State backend legacy read
- Grep with new and legacy formats
- Encoding inference via utf-8 decode attempt
"""

import base64
import warnings

from langchain.tools import ToolRuntime
from langgraph.store.memory import InMemoryStore

from deepagents.backends.state import StateBackend
from deepagents.backends.store import StoreBackend
from deepagents.backends.utils import (
    _to_legacy_file_data,
    create_file_data,
    file_data_to_string,
    grep_matches_from_files,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store_runtime():
    return ToolRuntime(
        state={"messages": []},
        context=None,
        tool_call_id="t1",
        store=InMemoryStore(),
        stream_writer=lambda _: None,
        config={},
    )


def _make_state_runtime(files=None):
    return ToolRuntime(
        state={"messages": [], "files": files or {}},
        context=None,
        tool_call_id="t1",
        store=None,
        stream_writer=lambda _: None,
        config={},
    )


# ---------------------------------------------------------------------------
# 1. Text round-trip
# ---------------------------------------------------------------------------


def test_text_round_trip():
    fd = create_file_data("hello\nworld")
    assert isinstance(fd["content"], str)
    assert fd["content"] == "hello\nworld"
    assert fd["encoding"] == "utf-8"
    assert file_data_to_string(fd) == "hello\nworld"


# ---------------------------------------------------------------------------
# 2. Binary round-trip
# ---------------------------------------------------------------------------


def test_binary_round_trip():
    original = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
    b64_str = base64.standard_b64encode(original).decode("ascii")
    fd = create_file_data(b64_str, encoding="base64")
    assert fd["content"] == b64_str
    assert fd["encoding"] == "base64"
    assert base64.standard_b64decode(fd["content"]) == original


# ---------------------------------------------------------------------------
# 3. Legacy backwards compat — emits DeprecationWarning
# ---------------------------------------------------------------------------


def test_legacy_list_content_emits_warning():
    legacy_fd = {
        "content": ["line1", "line2"],
        "encoding": "utf-8",
        "created_at": "2025-01-01T00:00:00+00:00",
        "modified_at": "2025-01-01T00:00:00+00:00",
    }
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = file_data_to_string(legacy_fd)
        assert result == "line1\nline2"
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "list[str]" in str(w[0].message)


# ---------------------------------------------------------------------------
# 4. New format — no warning
# ---------------------------------------------------------------------------


def test_new_format_no_warning():
    fd = {
        "content": "hello",
        "encoding": "utf-8",
        "created_at": "2025-01-01T00:00:00+00:00",
        "modified_at": "2025-01-01T00:00:00+00:00",
    }
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = file_data_to_string(fd)
        assert result == "hello"
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) == 0


# ---------------------------------------------------------------------------
# 5. Store backend upload binary
# ---------------------------------------------------------------------------


def test_store_upload_binary():
    rt = _make_store_runtime()
    be = StoreBackend(rt, namespace=lambda _ctx: ("filesystem",), file_format="v2")

    png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
    responses = be.upload_files([("/images/test.png", png_bytes)])

    assert len(responses) == 1
    assert responses[0].error is None

    # Verify stored with base64 encoding
    store = rt.store
    item = store.get(("filesystem",), "/images/test.png")
    assert item is not None
    assert item.value["encoding"] == "base64"
    assert isinstance(item.value["content"], str)


# ---------------------------------------------------------------------------
# 6. Store backend download binary round-trip
# ---------------------------------------------------------------------------


def test_store_upload_download_binary_round_trip():
    rt = _make_store_runtime()
    be = StoreBackend(rt, namespace=lambda _ctx: ("filesystem",), file_format="v2")

    original_bytes = b"\x89PNG\r\n\x1a\n" + bytes(range(256))
    be.upload_files([("/images/photo.png", original_bytes)])

    responses = be.download_files(["/images/photo.png"])
    assert len(responses) == 1
    assert responses[0].error is None
    assert responses[0].content == original_bytes


# ---------------------------------------------------------------------------
# 7. Store backend upload text
# ---------------------------------------------------------------------------


def test_store_upload_text():
    rt = _make_store_runtime()
    be = StoreBackend(rt, namespace=lambda _ctx: ("filesystem",), file_format="v2")

    text_bytes = b"Hello, world!\nLine 2"
    responses = be.upload_files([("/docs/readme.txt", text_bytes)])

    assert len(responses) == 1
    assert responses[0].error is None

    store = rt.store
    item = store.get(("filesystem",), "/docs/readme.txt")
    assert item is not None
    assert item.value["encoding"] == "utf-8"
    assert item.value["content"] == "Hello, world!\nLine 2"


# ---------------------------------------------------------------------------
# 8. Store backend legacy read
# ---------------------------------------------------------------------------


def test_store_legacy_list_content_read():
    rt = _make_store_runtime()
    be = StoreBackend(rt, namespace=lambda _ctx: ("filesystem",))
    store = rt.store

    # Manually put legacy list[str] item
    store.put(
        ("filesystem",),
        "/legacy/file.txt",
        {
            "content": ["line1", "line2", "line3"],
            "encoding": "utf-8",
            "created_at": "2025-01-01T00:00:00+00:00",
            "modified_at": "2025-01-01T00:00:00+00:00",
        },
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = be.read("/legacy/file.txt")
        assert "line1" in result
        assert "line2" in result
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= 1

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        responses = be.download_files(["/legacy/file.txt"])
        assert responses[0].content == b"line1\nline2\nline3"
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= 1


# ---------------------------------------------------------------------------
# 9. State backend legacy read
# ---------------------------------------------------------------------------


def test_state_legacy_list_content_read():
    legacy_fd = {
        "content": ["alpha", "beta"],
        "encoding": "utf-8",
        "created_at": "2025-01-01T00:00:00+00:00",
        "modified_at": "2025-01-01T00:00:00+00:00",
    }
    rt = _make_state_runtime(files={"/old/file.txt": legacy_fd})
    be = StateBackend(rt)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = be.read("/old/file.txt")
        assert "alpha" in result
        assert "beta" in result
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= 1

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        responses = be.download_files(["/old/file.txt"])
        assert responses[0].content == b"alpha\nbeta"
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= 1


# ---------------------------------------------------------------------------
# 10. Grep with new format
# ---------------------------------------------------------------------------


def test_grep_new_format():
    fd = create_file_data("import os\nprint('hello')\nimport sys")
    files = {"/src/main.py": fd}
    matches = grep_matches_from_files(files, "import", path="/")
    assert isinstance(matches, list)
    assert len(matches) == 2
    assert matches[0]["line"] == 1
    assert matches[0]["text"] == "import os"
    assert matches[1]["line"] == 3
    assert matches[1]["text"] == "import sys"


# ---------------------------------------------------------------------------
# 11. Grep with legacy format
# ---------------------------------------------------------------------------


def test_grep_legacy_format():
    legacy_fd = {
        "content": ["def foo():", "    return 42", "def bar():", "    return 0"],
        "encoding": "utf-8",
        "created_at": "2025-01-01T00:00:00+00:00",
        "modified_at": "2025-01-01T00:00:00+00:00",
    }
    files = {"/src/funcs.py": legacy_fd}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        matches = grep_matches_from_files(files, "def", path="/")
        assert isinstance(matches, list)
        assert len(matches) == 2
        assert matches[0]["text"] == "def foo():"
        assert matches[1]["text"] == "def bar():"
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= 1


# ---------------------------------------------------------------------------
# 12. Encoding inference — utf-8 attempt, fallback to base64
# ---------------------------------------------------------------------------


def test_store_upload_utf8_content_stored_as_text():
    """Valid utf-8 bytes are stored with encoding='utf-8'."""
    rt = _make_store_runtime()
    be = StoreBackend(rt, namespace=lambda _ctx: ("filesystem",), file_format="v2")

    be.upload_files([("/docs/notes.txt", b"Hello, world!")])

    item = rt.store.get(("filesystem",), "/docs/notes.txt")
    assert item.value["encoding"] == "utf-8"
    assert item.value["content"] == "Hello, world!"


def test_store_upload_non_utf8_content_stored_as_base64():
    """Non-utf-8 bytes are stored with encoding='base64'."""
    rt = _make_store_runtime()
    be = StoreBackend(rt, namespace=lambda _ctx: ("filesystem",), file_format="v2")

    raw = b"\x89PNG\r\n\x1a\n" + b"\xff\xfe" + b"\x00" * 20
    be.upload_files([("/images/photo.png", raw)])

    item = rt.store.get(("filesystem",), "/images/photo.png")
    assert item.value["encoding"] == "base64"
    assert base64.standard_b64decode(item.value["content"]) == raw


# ---------------------------------------------------------------------------
# 13. file_format="v1" flag — StoreBackend
# ---------------------------------------------------------------------------


def test_store_write_as_list():
    """StoreBackend with file_format="v1" stores content as list[str]."""
    rt = _make_store_runtime()
    be = StoreBackend(rt, namespace=lambda _ctx: ("filesystem",), file_format="v1")

    be.write("/docs/readme.txt", "line1\nline2\nline3")

    item = rt.store.get(("filesystem",), "/docs/readme.txt")
    assert item is not None
    assert item.value["content"] == ["line1", "line2", "line3"]
    assert "encoding" not in item.value


def test_store_edit_as_list():
    """StoreBackend with file_format="v1" preserves list format after edit."""
    rt = _make_store_runtime()
    be = StoreBackend(rt, namespace=lambda _ctx: ("filesystem",), file_format="v1")

    be.write("/docs/readme.txt", "hello\nworld")
    result = be.edit("/docs/readme.txt", "world", "there")

    assert result.error is None
    item = rt.store.get(("filesystem",), "/docs/readme.txt")
    assert item.value["content"] == ["hello", "there"]
    assert "encoding" not in item.value


def test_store_write_as_list_readable():
    """Files stored as list[str] are still readable via the same backend."""
    rt = _make_store_runtime()
    be = StoreBackend(rt, namespace=lambda _ctx: ("filesystem",), file_format="v1")

    be.write("/file.txt", "aaa\nbbb")
    result = be.read("/file.txt")
    assert "aaa" in result
    assert "bbb" in result


# ---------------------------------------------------------------------------
# 14. file_format="v1" flag — StateBackend
# ---------------------------------------------------------------------------


def test_state_write_as_list():
    """StateBackend with file_format="v1" stores content as list[str]."""
    rt = _make_state_runtime()
    be = StateBackend(rt, file_format="v1")

    result = be.write("/docs/readme.txt", "alpha\nbeta")
    assert result.error is None
    fd = result.files_update["/docs/readme.txt"]
    assert fd["content"] == ["alpha", "beta"]
    assert "encoding" not in fd


def test_state_edit_as_list():
    """StateBackend with file_format="v1" preserves list format after edit."""
    # Seed with a new-format file (as create_file_data produces)
    legacy = _to_legacy_file_data(create_file_data("hello\nworld"))
    rt = _make_state_runtime(files={"/file.txt": legacy})
    be = StateBackend(rt, file_format="v1")

    result = be.edit("/file.txt", "world", "there")
    assert result.error is None
    fd = result.files_update["/file.txt"]
    assert fd["content"] == ["hello", "there"]
    assert "encoding" not in fd
