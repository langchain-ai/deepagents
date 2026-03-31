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
from contextlib import contextmanager

from langchain_core.runnables.config import var_child_runnable_config
from langgraph._internal._constants import CONFIG_KEY_READ, CONFIG_KEY_SEND
from langgraph.store.memory import InMemoryStore

from deepagents.backends.protocol import ReadResult
from deepagents.backends.state import StateBackend
from deepagents.backends.store import StoreBackend
from deepagents.backends.utils import (
    _to_legacy_file_data,
    create_file_data,
    file_data_to_string,
    grep_matches_from_files,
)
from deepagents.middleware.filesystem import _file_data_reducer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state_config(files=None):
    store = {"files": files or {}}

    def read(select, *, _fresh=False):
        if isinstance(select, str):
            return store.get(select)
        return {k: store.get(k) for k in select}

    def send(writes):
        for channel, value in writes:
            if channel == "files":
                store["files"] = _file_data_reducer(store.get("files"), value)

    config = {"configurable": {CONFIG_KEY_SEND: send, CONFIG_KEY_READ: read}}
    return config, store


@contextmanager
def state_config_context(files=None):
    config, store = _make_state_config(files)
    token = var_child_runnable_config.set(config)
    try:
        yield store
    finally:
        var_child_runnable_config.reset(token)


@contextmanager
def _config_context(config=None):
    """Set a LangGraph config context so get_config() works in tests."""
    token = var_child_runnable_config.set(config or {})
    try:
        yield
    finally:
        var_child_runnable_config.reset(token)


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
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, file_format="v2")

    with _config_context():
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
        responses = be.upload_files([("/images/test.png", png_bytes)])

    assert len(responses) == 1
    assert responses[0].error is None

    # Verify stored with base64 encoding
    item = mem_store.get(("filesystem",), "/images/test.png")
    assert item is not None
    assert item.value["encoding"] == "base64"
    assert isinstance(item.value["content"], str)


# ---------------------------------------------------------------------------
# 6. Store backend download binary round-trip
# ---------------------------------------------------------------------------


def test_store_upload_download_binary_round_trip():
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, file_format="v2")

    with _config_context():
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
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, file_format="v2")

    with _config_context():
        text_bytes = b"Hello, world!\nLine 2"
        responses = be.upload_files([("/docs/readme.txt", text_bytes)])

    assert len(responses) == 1
    assert responses[0].error is None

    item = mem_store.get(("filesystem",), "/docs/readme.txt")
    assert item is not None
    assert item.value["encoding"] == "utf-8"
    assert item.value["content"] == "Hello, world!\nLine 2"


# ---------------------------------------------------------------------------
# 8. Store backend legacy read
# ---------------------------------------------------------------------------


def test_store_legacy_list_content_read():
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store)

    # Manually put legacy list[str] item
    mem_store.put(
        ("filesystem",),
        "/legacy/file.txt",
        {
            "content": ["line1", "line2", "line3"],
            "encoding": "utf-8",
            "created_at": "2025-01-01T00:00:00+00:00",
            "modified_at": "2025-01-01T00:00:00+00:00",
        },
    )

    with _config_context():
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = be.read("/legacy/file.txt")
            assert isinstance(result, ReadResult)
            assert result.file_data is not None
            assert "line1" in result.file_data["content"]
            assert "line2" in result.file_data["content"]
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
    with state_config_context(files={"/old/file.txt": legacy_fd}):
        be = StateBackend()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = be.read("/old/file.txt")
            assert isinstance(result, ReadResult)
            assert result.file_data is not None
            assert "alpha" in result.file_data["content"]
            assert "beta" in result.file_data["content"]
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
    result = grep_matches_from_files(files, "import", path="/")
    assert result.matches is not None
    assert len(result.matches) == 2
    assert result.matches[0]["line"] == 1
    assert result.matches[0]["text"] == "import os"
    assert result.matches[1]["line"] == 3
    assert result.matches[1]["text"] == "import sys"


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
        result = grep_matches_from_files(files, "def", path="/")
        assert result.matches is not None
        assert len(result.matches) == 2
        assert result.matches[0]["text"] == "def foo():"
        assert result.matches[1]["text"] == "def bar():"
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= 1


# ---------------------------------------------------------------------------
# 12. Encoding inference — utf-8 attempt, fallback to base64
# ---------------------------------------------------------------------------


def test_store_upload_utf8_content_stored_as_text():
    """Valid utf-8 bytes are stored with encoding='utf-8'."""
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, file_format="v2")

    with _config_context():
        be.upload_files([("/docs/notes.txt", b"Hello, world!")])

    item = mem_store.get(("filesystem",), "/docs/notes.txt")
    assert item.value["encoding"] == "utf-8"
    assert item.value["content"] == "Hello, world!"


def test_store_upload_non_utf8_content_stored_as_base64():
    """Non-utf-8 bytes are stored with encoding='base64'."""
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, file_format="v2")

    with _config_context():
        raw = b"\x89PNG\r\n\x1a\n" + b"\xff\xfe" + b"\x00" * 20
        be.upload_files([("/images/photo.png", raw)])

    item = mem_store.get(("filesystem",), "/images/photo.png")
    assert item.value["encoding"] == "base64"
    assert base64.standard_b64decode(item.value["content"]) == raw


# ---------------------------------------------------------------------------
# 13. file_format="v1" flag — StoreBackend
# ---------------------------------------------------------------------------


def test_store_write_as_list():
    """StoreBackend with file_format="v1" stores content as list[str]."""
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, file_format="v1")

    with _config_context():
        be.write("/docs/readme.txt", "line1\nline2\nline3")

    item = mem_store.get(("filesystem",), "/docs/readme.txt")
    assert item is not None
    assert item.value["content"] == ["line1", "line2", "line3"]
    assert "encoding" not in item.value


def test_store_edit_as_list():
    """StoreBackend with file_format="v1" preserves list format after edit."""
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, file_format="v1")

    with _config_context():
        be.write("/docs/readme.txt", "hello\nworld")
        result = be.edit("/docs/readme.txt", "world", "there")

    assert result.error is None
    item = mem_store.get(("filesystem",), "/docs/readme.txt")
    assert item.value["content"] == ["hello", "there"]
    assert "encoding" not in item.value


def test_store_write_as_list_readable():
    """Files stored as list[str] are still readable via the same backend."""
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, file_format="v1")

    with _config_context():
        be.write("/file.txt", "aaa\nbbb")
        result = be.read("/file.txt")
    assert isinstance(result, ReadResult)
    assert result.file_data is not None
    assert "aaa" in result.file_data["content"]
    assert "bbb" in result.file_data["content"]


# ---------------------------------------------------------------------------
# 14. file_format="v1" flag — StateBackend
# ---------------------------------------------------------------------------


def test_state_write_as_list():
    """StateBackend with file_format="v1" stores content as list[str]."""
    with state_config_context() as store:
        be = StateBackend(file_format="v1")

        result = be.write("/docs/readme.txt", "alpha\nbeta")
        assert result.error is None
        fd = store["files"]["/docs/readme.txt"]
        assert fd["content"] == ["alpha", "beta"]
        assert "encoding" not in fd


def test_state_edit_as_list():
    """StateBackend with file_format="v1" preserves list format after edit."""
    # Seed with a new-format file (as create_file_data produces)
    legacy = _to_legacy_file_data(create_file_data("hello\nworld"))
    with state_config_context(files={"/file.txt": legacy}) as store:
        be = StateBackend(file_format="v1")

        result = be.edit("/file.txt", "world", "there")
        assert result.error is None
        fd = store["files"]["/file.txt"]
        assert fd["content"] == ["hello", "there"]
        assert "encoding" not in fd
