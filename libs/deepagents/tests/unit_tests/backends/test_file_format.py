"""Tests for current file data storage format and helpers."""

import base64

import pytest
from langgraph.store.memory import InMemoryStore

from deepagents.backends.store import StoreBackend
from deepagents.backends.utils import (
    compile_grep_include_glob,
    create_file_data,
    file_data_to_string,
    grep_matches_from_files,
)

NS = lambda _rt: ("filesystem",)  # noqa: E731


def test_text_round_trip() -> None:
    fd = create_file_data("hello\nworld")
    assert isinstance(fd["content"], str)
    assert fd["content"] == "hello\nworld"
    assert fd["encoding"] == "utf-8"
    assert file_data_to_string(fd) == "hello\nworld"


def test_legacy_list_content_is_readable() -> None:
    fd = {"content": ["hello", "world", ""]}

    assert file_data_to_string(fd) == "hello\nworld\n"  # type: ignore[arg-type]


def test_empty_legacy_list_content_is_readable() -> None:
    fd = {"content": []}

    assert file_data_to_string(fd) == ""  # type: ignore[arg-type]


def test_legacy_list_content_rejects_non_string_items() -> None:
    fd = {"content": ["hello", 1]}

    with pytest.raises(TypeError, match="got list"):
        file_data_to_string(fd)  # type: ignore[arg-type]


def test_binary_round_trip() -> None:
    original = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
    b64_str = base64.standard_b64encode(original).decode("ascii")
    fd = create_file_data(b64_str, encoding="base64")
    assert fd["content"] == b64_str
    assert fd["encoding"] == "base64"
    assert base64.standard_b64decode(fd["content"]) == original


def test_store_upload_binary() -> None:
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, namespace=NS)

    png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
    responses = be.upload_files([("/images/test.png", png_bytes)])

    assert len(responses) == 1
    assert responses[0].error is None
    item = mem_store.get(("filesystem",), "/images/test.png")
    assert item is not None
    assert item.value["encoding"] == "base64"
    assert isinstance(item.value["content"], str)


def test_store_upload_download_binary_round_trip() -> None:
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, namespace=NS)

    original_bytes = b"\x89PNG\r\n\x1a\n" + bytes(range(256))
    be.upload_files([("/images/photo.png", original_bytes)])

    responses = be.download_files(["/images/photo.png"])
    assert len(responses) == 1
    assert responses[0].error is None
    assert responses[0].content == original_bytes


def test_store_upload_text() -> None:
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, namespace=NS)

    text_bytes = b"Hello, world!\nLine 2"
    responses = be.upload_files([("/docs/readme.txt", text_bytes)])

    assert len(responses) == 1
    assert responses[0].error is None
    item = mem_store.get(("filesystem",), "/docs/readme.txt")
    assert item is not None
    assert item.value["encoding"] == "utf-8"
    assert item.value["content"] == "Hello, world!\nLine 2"


def test_grep_new_format() -> None:
    fd = create_file_data("import os\nprint('hello')\nimport sys")
    files = {"/src/main.py": fd}
    result = grep_matches_from_files(files, "import", path="/")
    assert result.matches is not None
    assert len(result.matches) == 2
    assert result.matches[0]["line"] == 1
    assert result.matches[0]["text"] == "import os"
    assert result.matches[1]["line"] == 3
    assert result.matches[1]["text"] == "import sys"


def test_grep_glob_filters_by_filename() -> None:
    files = {
        "/src/main.py": create_file_data("import os"),
        "/src/notes.txt": create_file_data("import os"),
    }
    result = grep_matches_from_files(files, "import", path="/", glob="*.py")
    assert result.matches is not None
    assert len(result.matches) == 1
    assert result.matches[0]["path"] == "/src/main.py"


def test_grep_glob_brace_expansion() -> None:
    files = {
        "/a.py": create_file_data("hit"),
        "/b.md": create_file_data("hit"),
        "/c.txt": create_file_data("hit"),
    }
    result = grep_matches_from_files(files, "hit", path="/", glob="*.{py,md}")
    assert result.matches is not None
    paths = sorted(m["path"] for m in result.matches)
    assert paths == ["/a.py", "/b.md"]


def test_grep_glob_matches_nothing() -> None:
    files = {
        "/a.py": create_file_data("hit"),
        "/b.md": create_file_data("hit"),
    }
    result = grep_matches_from_files(files, "hit", path="/", glob="*.rs")
    assert result.matches == []


def test_compile_glob_is_cached() -> None:
    assert compile_grep_include_glob("*.py") is compile_grep_include_glob("*.py")
    assert compile_grep_include_glob("*.py") is not compile_grep_include_glob("*.md")


def test_grep_glob_repeated_pattern_stays_correct() -> None:
    first = {"/x.py": create_file_data("hit"), "/x.md": create_file_data("hit")}
    second = {"/y.py": create_file_data("hit"), "/y.txt": create_file_data("hit")}
    r1 = grep_matches_from_files(first, "hit", path="/", glob="*.py")
    r2 = grep_matches_from_files(second, "hit", path="/", glob="*.py")
    assert r1.matches is not None
    assert r2.matches is not None
    assert [m["path"] for m in r1.matches] == ["/x.py"]
    assert [m["path"] for m in r2.matches] == ["/y.py"]


def test_store_upload_utf8_content_stored_as_text() -> None:
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, namespace=NS)

    be.upload_files([("/docs/notes.txt", b"Hello, world!")])

    item = mem_store.get(("filesystem",), "/docs/notes.txt")
    assert item is not None
    assert item.value["encoding"] == "utf-8"
    assert item.value["content"] == "Hello, world!"


def test_store_upload_non_utf8_content_stored_as_base64() -> None:
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, namespace=NS)

    raw = b"\x89PNG\r\n\x1a\n" + b"\xff\xfe" + b"\x00" * 20
    be.upload_files([("/images/photo.png", raw)])

    item = mem_store.get(("filesystem",), "/images/photo.png")
    assert item is not None
    assert item.value["encoding"] == "base64"
    assert base64.standard_b64decode(item.value["content"]) == raw
