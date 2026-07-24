from typing import Any, Never

import pytest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.store.base import PutOp
from langgraph.store.memory import InMemoryStore

from deepagents.backends.protocol import EditResult, ReadResult, WriteResult
from deepagents.backends.store import StoreBackend, _validate_namespace
from deepagents.middleware.filesystem import FilesystemMiddleware


def test_store_backend_crud_and_search():
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem",))

    # write new file
    msg = be.write("/docs/readme.md", "hello store")
    assert isinstance(msg, WriteResult) and msg.error is None and msg.path == "/docs/readme.md"

    # read
    read_result = be.read("/docs/readme.md")
    assert isinstance(read_result, ReadResult) and read_result.file_data is not None
    assert "hello store" in read_result.file_data["content"]

    # edit
    msg2 = be.edit("/docs/readme.md", "hello", "hi", replace_all=False)
    assert isinstance(msg2, EditResult) and msg2.error is None and msg2.occurrences == 1

    # ls_info (path prefix filter)
    infos = be.ls("/docs/").entries
    assert infos is not None
    assert any(i["path"] == "/docs/readme.md" for i in infos)

    # grep
    matches = be.grep("hi", path="/").matches
    assert matches is not None and any(m["path"] == "/docs/readme.md" for m in matches)

    # glob
    g = be.glob("*.md", path="/").matches
    assert len(g) == 0

    g2 = be.glob("**/*.md", path="/").matches
    assert any(i["path"] == "/docs/readme.md" for i in g2)


def test_store_backend_read_surfaces_pagination_metadata():
    """`StoreBackend.read` propagates the pagination metadata from `slice_read_response`."""
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem",))
    be.write("/notes.txt", "one\ntwo\nthree\nfour\nfive")

    result = be.read("/notes.txt", offset=1, limit=2)

    assert result.error is None
    assert result.total_lines == 5
    assert result.start_line == 2
    assert result.end_line == 3
    assert result.next_offset == 3


def test_store_backend_reads_mkv_as_binary_without_slicing():
    """`.mkv` reads bypass text line-slicing so binary bytes are returned intact.

    Regression: `.mkv` is not in the shared multimodal map, so a text
    classification here would run `slice_read_response` and raise a spurious
    "line offset exceeds file length" error for any offset past the content.
    """
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem",))
    be.write("/clip.mkv", "single line of bytes")

    # An offset past the 1-line content would raise a text-slice error if `.mkv`
    # were misclassified as text; the binary path returns the file unsliced.
    read_result = be.read("/clip.mkv", offset=5, limit=10)

    assert isinstance(read_result, ReadResult)
    assert read_result.error is None
    assert read_result.file_data is not None
    assert read_result.file_data["content"] == "single line of bytes"


def test_store_backend_reads_legacy_list_content() -> None:
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem",))
    legacy_content = ["hello", "world", ""]
    mem_store.put(
        ("filesystem",),
        "/legacy.txt",
        {
            "content": legacy_content,
            "created_at": "2025-01-01T00:00:00+00:00",
            "modified_at": "2025-01-02T00:00:00+00:00",
        },
    )

    read_result = be.read("/legacy.txt")
    assert read_result.file_data is not None
    assert read_result.file_data["content"] == "hello\nworld\n"
    assert read_result.file_data["encoding"] == "utf-8"

    listing = be.ls("/").entries
    assert listing is not None
    assert listing[0]["size"] == len("hello\nworld\n")

    matches = be.grep("world", path="/").matches
    assert matches is not None
    assert matches[0]["text"] == "world"

    glob_matches = be.glob("*.txt", path="/").matches
    assert glob_matches is not None
    assert glob_matches[0]["size"] == len("hello\nworld\n")

    download = be.download_files(["/legacy.txt"])[0]
    assert download.content == b"hello\nworld\n"

    stored = mem_store.get(("filesystem",), "/legacy.txt")
    assert stored is not None
    assert stored.value["content"] == legacy_content
    assert "encoding" not in stored.value


def test_store_backend_rejects_legacy_lists_with_non_string_items() -> None:
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem",))
    mem_store.put(("filesystem",), "/invalid.txt", {"content": ["hello", 1]})

    with pytest.raises(TypeError, match="got list"):
        be.read("/invalid.txt")


def test_store_backend_write_migrates_legacy_list_content() -> None:
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem",))
    created_at = "2025-01-01T00:00:00+00:00"
    mem_store.put(
        ("filesystem",),
        "/legacy.txt",
        {"content": ["hello", "world"], "created_at": created_at},
    )

    result = be.write("/legacy.txt", "replacement")

    assert result.error is None
    stored = mem_store.get(("filesystem",), "/legacy.txt")
    assert stored is not None
    assert stored.value["content"] == "replacement"
    assert stored.value["encoding"] == "utf-8"
    assert stored.value["created_at"] == created_at


def test_store_backend_edit_migrates_legacy_list_content() -> None:
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem",))
    mem_store.put(("filesystem",), "/legacy.txt", {"content": ["hello", "world"]})

    result = be.edit("/legacy.txt", "world", "there")

    assert result.error is None
    stored = mem_store.get(("filesystem",), "/legacy.txt")
    assert stored is not None
    assert stored.value["content"] == "hello\nthere"
    assert stored.value["encoding"] == "utf-8"


def test_store_backend_write_overwrites_existing_file():
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, namespace=lambda _ctx: ("filesystem",))

    # Create the file
    result = be.write("/charmander.txt", "Hello world")
    assert result.error is None

    # Overwrite should succeed
    result = be.write("/charmander.txt", "New content")
    assert result.error is None

    # New content visible
    read_result = be.read("/charmander.txt")
    assert read_result.file_data is not None
    assert "New content" in read_result.file_data["content"]


def test_store_backend_ls_nested_directories():
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem",))

    files = {
        "/src/main.py": "main code",
        "/src/utils/helper.py": "helper code",
        "/src/utils/common.py": "common code",
        "/docs/readme.md": "readme",
        "/docs/api/reference.md": "api reference",
        "/config.json": "config",
    }

    for path, content in files.items():
        res = be.write(path, content)
        assert res.error is None

    root_listing = be.ls("/").entries
    assert root_listing is not None
    root_paths = [fi["path"] for fi in root_listing]
    assert "/config.json" in root_paths
    assert "/src/" in root_paths
    assert "/docs/" in root_paths
    assert "/src/main.py" not in root_paths
    assert "/src/utils/helper.py" not in root_paths
    assert "/docs/readme.md" not in root_paths
    assert "/docs/api/reference.md" not in root_paths

    src_listing = be.ls("/src/").entries
    assert src_listing is not None
    src_paths = [fi["path"] for fi in src_listing]
    assert "/src/main.py" in src_paths
    assert "/src/utils/" in src_paths
    assert "/src/utils/helper.py" not in src_paths

    utils_listing = be.ls("/src/utils/").entries
    assert utils_listing is not None
    utils_paths = [fi["path"] for fi in utils_listing]
    assert "/src/utils/helper.py" in utils_paths
    assert "/src/utils/common.py" in utils_paths
    assert len(utils_paths) == 2

    empty_listing = be.ls("/nonexistent/")
    assert empty_listing.entries == []


def test_store_backend_ls_trailing_slash():
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem",))

    files = {
        "/file.txt": "content",
        "/dir/nested.txt": "nested",
    }

    for path, content in files.items():
        res = be.write(path, content)
        assert res.error is None

    listing_from_root = be.ls("/").entries
    assert listing_from_root is not None
    assert len(listing_from_root) > 0

    listing1 = be.ls("/dir/").entries
    listing2 = be.ls("/dir").entries
    assert listing1 is not None
    assert listing2 is not None
    assert len(listing1) == len(listing2)
    assert [fi["path"] for fi in listing1] == [fi["path"] for fi in listing2]


def test_store_backend_intercept_large_tool_result():
    """Test that StoreBackend properly handles large tool result interception."""
    mem_store = InMemoryStore()
    middleware = FilesystemMiddleware(
        backend=StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem",)),
        tool_token_limit_before_evict=1000,
    )

    large_content = "y" * 5000
    tool_message = ToolMessage(content=large_content, tool_call_id="test_456")
    result = middleware._intercept_large_tool_result(tool_message)

    assert isinstance(result, ToolMessage)
    assert "Tool result too large" in result.content
    assert "/large_tool_results/test_456" in result.content

    stored_content = mem_store.get(("filesystem",), "/large_tool_results/test_456")
    assert stored_content is not None
    assert stored_content.value["content"] == large_content


def test_store_backend_namespace_user_scoped() -> None:
    """Test namespace factory with user_id captured in closure."""
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem", "alice"))

    # Write a file
    be.write("/test.txt", "hello alice")

    # Verify it's stored in the correct namespace
    items = mem_store.search(("filesystem", "alice"))
    assert len(items) == 1
    assert items[0].key == "/test.txt"

    # Read it back
    read_result = be.read("/test.txt")
    assert read_result.file_data is not None
    assert "hello alice" in read_result.file_data["content"]


def test_store_backend_namespace_multi_level() -> None:
    """Test namespace factory with multiple values."""
    mem_store = InMemoryStore()
    be = StoreBackend(
        store=mem_store,
        namespace=lambda _rt: (
            "workspace",
            "ws-123",
            "user",
            "bob",
        ),
    )

    # Write a file
    be.write("/doc.md", "workspace doc")

    # Verify it's stored in the correct namespace
    items = mem_store.search(("workspace", "ws-123", "user", "bob"))
    assert len(items) == 1
    assert items[0].key == "/doc.md"


def test_store_backend_namespace_isolation() -> None:
    """Test that different users have isolated namespaces."""
    mem_store = InMemoryStore()

    # User alice
    be_alice = StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem", "alice"))
    be_alice.write("/notes.txt", "alice notes")

    # User bob
    be_bob = StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem", "bob"))
    be_bob.write("/notes.txt", "bob notes")

    # Verify isolation
    alice_result = be_alice.read("/notes.txt")
    assert alice_result.file_data is not None
    assert "alice notes" in alice_result.file_data["content"]

    bob_result = be_bob.read("/notes.txt")
    assert bob_result.file_data is not None
    assert "bob notes" in bob_result.file_data["content"]

    # Verify they're in different namespaces
    alice_items = mem_store.search(("filesystem", "alice"))
    assert len(alice_items) == 1
    bob_items = mem_store.search(("filesystem", "bob"))
    assert len(bob_items) == 1


def test_store_backend_namespace_error_handling() -> None:
    """Test that factory errors propagate correctly."""

    def bad_factory(_rt: Runtime[Any]) -> Never:
        msg = "user_id"
        raise KeyError(msg)

    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, namespace=bad_factory)

    # Errors from the factory propagate
    with pytest.raises(KeyError):
        be.write("/test.txt", "content")


def test_store_backend_namespace_is_required() -> None:
    """`namespace` is a required keyword-only argument."""
    mem_store = InMemoryStore()
    with pytest.raises(TypeError):
        StoreBackend(store=mem_store)  # type: ignore[call-arg]


def test_store_backend_runtime_reading_factory_outside_graph() -> None:
    """A factory that reads the runtime outside a graph raises a clear error.

    `get_runtime()` fails outside graph execution, so the factory receives
    `None`. A factory that reads runtime attributes should surface an
    actionable `RuntimeError` rather than an opaque `AttributeError`.
    """
    mem_store = InMemoryStore()
    be = StoreBackend(
        store=mem_store,
        namespace=lambda rt: (rt.server_info.user.identity, "filesystem"),
    )

    with pytest.raises(RuntimeError, match="outside a LangGraph graph execution"):
        be.write("/test.txt", "content")


def test_store_backend_arg_ignoring_factory_works_outside_graph() -> None:
    """A factory that ignores its argument still works outside a graph."""
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem",))

    result = be.write("/test.txt", "content")
    assert result.error is None
    assert mem_store.get(("filesystem",), "/test.txt") is not None


def test_store_backend_namespace_with_context() -> None:
    """Test that namespace factory receives values and stores correctly."""
    mem_store = InMemoryStore()

    def namespace_from_user(uid: str):
        return lambda _rt: ("threads", uid)

    be = StoreBackend(store=mem_store, namespace=namespace_from_user("ctx-user"))

    # Write a file
    be.write("/test.txt", "content")

    # Verify it's stored in the correct namespace
    items = mem_store.search(("threads", "ctx-user"))
    assert len(items) == 1
    assert items[0].key == "/test.txt"


@pytest.mark.parametrize(
    ("pattern", "expected_file"),
    [
        ("def __init__(", "code.py"),  # Parentheses (not regex grouping)
        ("str | int", "types.py"),  # Pipe (not regex OR)
        ("[a-z]", "regex.py"),  # Brackets (not character class)
        ("(.*)", "regex.py"),  # Multiple special chars
        ("api.key", "config.json"),  # Dot (not "any character")
        ("x * y", "math.py"),  # Asterisk (not "zero or more")
        ("a^2", "math.py"),  # Caret (not line anchor)
    ],
)
def test_store_backend_grep_literal_search_special_chars(pattern: str, expected_file: str) -> None:
    """Test that grep performs literal search with regex special characters."""
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem",))

    # Create files with various special regex characters
    files = {
        "/code.py": "def __init__(self, arg):\n    pass",
        "/types.py": "def func(x: str | int) -> None:\n    return x",
        "/regex.py": "pattern = r'[a-z]+'\nchars = '(.*)'",
        "/config.json": '{"api.key": "value", "url": "https://example.com"}',
        "/math.py": "result = x * y + z\nformula = a^2 + b^2",
    }

    for path, content in files.items():
        res = be.write(path, content)
        assert res.error is None

    # Test literal search with the pattern
    matches = be.grep(pattern, path="/").matches
    assert matches is not None
    assert any(expected_file in m["path"] for m in matches), f"Pattern '{pattern}' not found in {expected_file}"


# --- _validate_namespace tests ---


class TestValidateNamespace:
    """Tests for the _validate_namespace function."""

    @pytest.mark.parametrize(
        "namespace",
        [
            ("filesystem",),
            ("filesystem", "alice"),
            ("workspace", "ws-123", "user", "bob"),
            ("user@example.com",),
            ("user+tag@example.com",),
            ("550e8400-e29b-41d4-a716-446655440000",),
            ("asst-123", "filesystem"),
            ("threads", "thread-abc"),
            ("org:team:project",),
            ("~user",),
            ("v1.2.3",),
        ],
    )
    def test_valid_namespaces(self, namespace: tuple[str, ...]) -> None:
        assert _validate_namespace(namespace) == namespace

    @pytest.mark.parametrize(
        ("namespace", "match_msg"),
        [
            ((), "must not be empty"),
            (("",), "must not be empty"),
            (("ok", ""), "must not be empty"),
            (("*",), "disallowed characters"),
            (("file*system",), "disallowed characters"),
            (("user?",), "disallowed characters"),
            (("ns[0]",), "disallowed characters"),
            (("ns{a}",), "disallowed characters"),
            (("a b",), "disallowed characters"),
            (("path/to",), "disallowed characters"),
            (("$var",), "disallowed characters"),
            (("hello!",), "disallowed characters"),
            (("semi;colon",), "disallowed characters"),
            (("back\\slash",), "disallowed characters"),
            (("a\nb",), "disallowed characters"),
        ],
    )
    def test_invalid_namespaces(self, namespace: tuple[str, ...], match_msg: str) -> None:
        with pytest.raises(ValueError, match=match_msg):
            _validate_namespace(namespace)

    def test_non_string_component(self) -> None:
        with pytest.raises(TypeError, match="must be a string"):
            _validate_namespace(("ok", 123))  # type: ignore[arg-type]


def test_store_backend_rejects_wildcard_namespace() -> None:
    """Ensure StoreBackend rejects namespace tuples with wildcard characters."""
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem", "*"))

    with pytest.raises(ValueError, match="disallowed characters"):
        be.write("/test.txt", "content")


def test_store_backend_delete() -> None:
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem",))

    be.write("/docs/readme.md", "hello store")
    assert be.read("/docs/readme.md").error is None

    result = be.delete("/docs/readme.md")
    assert result.error is None
    assert result.path == "/docs/readme.md"
    # File is gone
    assert be.read("/docs/readme.md").error is not None


def test_store_backend_delete_directory_recursive() -> None:
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem",))
    be.write("/work/a.txt", "a")
    be.write("/work/sub/b.txt", "b")
    be.write("/keep.txt", "k")

    result = be.delete("/work")
    assert result.error is None
    assert result.path == "/work"
    # Every key at or under /work is gone.
    assert be.read("/work/a.txt").error is not None
    assert be.read("/work/sub/b.txt").error is not None
    # A sibling outside the subtree is untouched.
    assert be.read("/keep.txt").error is None


async def test_store_backend_adelete_directory_recursive() -> None:
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem",))
    await be.awrite("/work/a.txt", "a")
    await be.awrite("/work/sub/b.txt", "b")
    await be.awrite("/keep.txt", "k")

    result = await be.adelete("/work")
    assert result.error is None
    assert (await be.aread("/work/a.txt")).error is not None
    assert (await be.aread("/work/sub/b.txt")).error is not None
    assert (await be.aread("/keep.txt")).error is None


class _RecordingStore(InMemoryStore):
    """InMemoryStore that records the ops passed to each batch/abatch call.

    `batch`/`abatch` are the single primitive every store op routes through, so
    recording them lets a test assert how many round-trips a delete performs.
    """

    def __init__(self) -> None:
        super().__init__()
        self.batch_calls: list[list] = []

    def batch(self, ops):
        ops = list(ops)
        self.batch_calls.append(ops)
        return super().batch(ops)

    async def abatch(self, ops):
        ops = list(ops)
        self.batch_calls.append(ops)
        return await super().abatch(ops)


def _delete_op_batches(batch_calls: list[list]) -> list[list]:
    """Batches that carried at least one delete (PutOp with value=None)."""
    return [ops for ops in batch_calls if any(isinstance(op, PutOp) and op.value is None for op in ops)]


def test_store_backend_delete_uses_single_batch_call() -> None:
    """A recursive delete issues one batched store write, not one call per key."""
    store = _RecordingStore()
    be = StoreBackend(store=store, namespace=lambda _rt: ("filesystem",))
    be.write("/work/a.txt", "a")
    be.write("/work/sub/b.txt", "b")
    be.write("/keep.txt", "k")
    store.batch_calls.clear()  # ignore the setup writes

    result = be.delete("/work")
    assert result.error is None

    # All the deletes happened in exactly one batch (not one call per key).
    delete_batches = _delete_op_batches(store.batch_calls)
    assert len(delete_batches) == 1
    ops = delete_batches[0]
    assert all(isinstance(op, PutOp) and op.value is None for op in ops)
    assert {op.key for op in ops} == {"/work/a.txt", "/work/sub/b.txt"}


async def test_store_backend_adelete_uses_single_batch_call() -> None:
    """Async recursive delete issues one batched store write, not one per key."""
    store = _RecordingStore()
    be = StoreBackend(store=store, namespace=lambda _rt: ("filesystem",))
    await be.awrite("/work/a.txt", "a")
    await be.awrite("/work/sub/b.txt", "b")
    await be.awrite("/keep.txt", "k")
    store.batch_calls.clear()

    result = await be.adelete("/work")
    assert result.error is None

    delete_batches = _delete_op_batches(store.batch_calls)
    assert len(delete_batches) == 1
    ops = delete_batches[0]
    assert all(isinstance(op, PutOp) and op.value is None for op in ops)
    assert {op.key for op in ops} == {"/work/a.txt", "/work/sub/b.txt"}


def test_store_backend_delete_missing_returns_error() -> None:
    be = StoreBackend(store=InMemoryStore(), namespace=lambda _rt: ("filesystem",))
    result = be.delete("/nope.md")
    assert result.path is None
    assert result.error is not None
    assert "not found" in result.error


async def test_store_backend_adelete() -> None:
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem",))

    await be.awrite("/a.md", "x")
    result = await be.adelete("/a.md")
    assert result.error is None
    assert result.path == "/a.md"
    assert (await be.aread("/a.md")).error is not None

    missing = await be.adelete("/a.md")
    assert missing.error is not None and "not found" in missing.error


def test_store_backend_delete_treats_wildcard_as_literal_key() -> None:
    """`*` is used as an exact store key, not a wildcard.

    Deleting "*" must only remove an entry whose key is literally "*" and must
    never expand to match/remove other files in the namespace.
    """
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem",))

    be.write("/a.md", "a")
    be.write("/b.md", "b")
    be.write("*", "literal star")

    # Deleting "*" removes only the literally-named entry.
    result = be.delete("*")
    assert result.error is None
    assert result.path == "*"

    # The other files are untouched — "*" did not expand to a wildcard.
    assert be.read("/a.md").error is None
    assert be.read("/b.md").error is None
    assert be.read("*").error is not None


def test_store_backend_delete_wildcard_missing_does_not_match_files() -> None:
    """Deleting "*" when no literal "*" key exists is a not-found, not a purge."""
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem",))

    be.write("/a.md", "a")
    be.write("/b.md", "b")

    result = be.delete("*")
    assert result.path is None
    assert result.error is not None and "not found" in result.error

    # Nothing was deleted by the wildcard-looking key.
    assert be.read("/a.md").error is None
    assert be.read("/b.md").error is None


async def test_store_backend_adelete_treats_wildcard_as_literal_key() -> None:
    """Async delete also treats `*` as an exact key, not a wildcard."""
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem",))

    await be.awrite("/a.md", "a")
    await be.awrite("*", "literal star")

    result = await be.adelete("*")
    assert result.error is None
    assert result.path == "*"

    assert (await be.aread("/a.md")).error is None
    assert (await be.aread("*")).error is not None

    missing = await be.adelete("*")
    assert missing.error is not None and "not found" in missing.error
