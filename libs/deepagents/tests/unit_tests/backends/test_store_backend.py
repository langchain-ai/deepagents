import warnings
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Never

import pytest
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore

from deepagents.backends.protocol import EditResult, ReadResult, WriteResult
from deepagents.backends.store import BackendContext, StoreBackend, _NamespaceRuntimeCompat, _validate_namespace
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


@pytest.mark.parametrize("file_format", ["v1", "v2"])
def test_store_backend_intercept_large_tool_result(file_format):
    """Test that StoreBackend properly handles large tool result interception."""
    mem_store = InMemoryStore()
    middleware = FilesystemMiddleware(
        backend=StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem",), file_format=file_format),
        tool_token_limit_before_evict=1000,
    )

    large_content = "y" * 5000
    tool_message = ToolMessage(content=large_content, tool_call_id="test_456")
    rt = ToolRuntime(
        state={"messages": []},
        context=None,
        tool_call_id="t2",
        store=mem_store,
        stream_writer=lambda _: None,
        config={},
    )
    result = middleware._intercept_large_tool_result(tool_message, rt)

    assert isinstance(result, ToolMessage)
    assert "Tool result too large" in result.content
    assert "/large_tool_results/test_456" in result.content

    stored_content = mem_store.get(("filesystem",), "/large_tool_results/test_456")
    assert stored_content is not None
    expected = [large_content] if file_format == "v1" else large_content
    assert stored_content.value["content"] == expected


@dataclass
class UserContext:
    """Simple context object for testing."""

    user_id: str
    workspace_id: str | None = None


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


def test_store_backend_namespace_legacy_mode() -> None:
    """Test that legacy mode still works when no namespace is provided, but emits deprecation warning."""
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store)  # No namespace - uses legacy mode

    # Should emit deprecation warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        be.write("/legacy.txt", "legacy content")
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "namespace" in str(w[0].message)

    # Should be in default namespace (no assistant_id in context config)
    items = mem_store.search(("filesystem",))
    assert len(items) == 1
    assert items[0].key == "/legacy.txt"


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


# --- Backwards compatibility tests for _NamespaceRuntimeCompat ---


def test_compat_wrapper_old_style_runtime_access_warns() -> None:
    """Old-style factories accessing .runtime get a deprecation warning."""
    compat = _NamespaceRuntimeCompat(runtime=None)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = compat.runtime
        assert result is None
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert ".runtime" in str(w[0].message)
        assert "v0.7" in str(w[0].message)


def test_compat_wrapper_old_style_state_access_warns() -> None:
    """Old-style factories accessing .state get a deprecation warning."""
    compat = _NamespaceRuntimeCompat(runtime=None, state={"messages": []})

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = compat.state
        assert result == {"messages": []}
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert ".state" in str(w[0].message)
        assert "v0.7" in str(w[0].message)


def test_compat_wrapper_proxies_runtime_attrs() -> None:
    """New-style factories can access Runtime attributes directly through the wrapper."""

    @dataclass
    class Ctx:
        user_id: str

    rt = Runtime(context=Ctx(user_id="alice"))
    compat = _NamespaceRuntimeCompat(runtime=rt)

    # New-style access: no warning, proxied to Runtime
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert compat.context.user_id == "alice"  # type: ignore[union-attr]
        assert compat.store is None  # type: ignore[union-attr]
        # No deprecation warnings for direct Runtime attr access
        assert len(w) == 0


def test_compat_wrapper_old_style_factory_end_to_end() -> None:
    """An old-style namespace factory using ctx.runtime.context still works."""

    @dataclass
    class Ctx:
        user_id: str

    rt = Runtime(context=Ctx(user_id="bob"))
    compat = _NamespaceRuntimeCompat(runtime=rt)

    # Old-style factory
    def old_factory(ctx: BackendContext) -> tuple[str, ...]:  # type: ignore[type-arg]
        return (ctx.runtime.context.user_id, "filesystem")  # type: ignore[union-attr]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = old_factory(compat)  # type: ignore[arg-type]
        assert result == ("bob", "filesystem")
        assert len(w) == 1  # one warning from .runtime access


def test_compat_wrapper_new_style_factory_end_to_end() -> None:
    """A new-style namespace factory using rt.context works without warnings."""

    rt = Runtime(
        context=None,
        server_info=SimpleNamespace(user=SimpleNamespace(identity="carol")),  # type: ignore[arg-type]
    )
    compat = _NamespaceRuntimeCompat(runtime=rt)

    # New-style factory
    def new_factory(rt: Runtime) -> tuple[str, ...]:  # type: ignore[type-arg]
        return (rt.server_info.user.identity, "filesystem")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = new_factory(compat)  # type: ignore[arg-type]
        assert result == ("carol", "filesystem")
        assert len(w) == 0  # no warnings


def test_compat_wrapper_no_runtime_raises_on_attr_access() -> None:
    """Accessing Runtime attrs when runtime is None raises AttributeError."""
    compat = _NamespaceRuntimeCompat(runtime=None)

    with pytest.raises(AttributeError, match="running outside graph execution"):
        _ = compat.context  # type: ignore[union-attr]


def test_compat_wrapper_unknown_attr_on_real_runtime_raises() -> None:
    """Accessing a non-existent attribute when runtime is set propagates AttributeError."""
    rt = Runtime(context=None)
    compat = _NamespaceRuntimeCompat(runtime=rt)

    with pytest.raises(AttributeError, match="nonexistent_attribute_xyz"):
        _ = compat.nonexistent_attribute_xyz  # type: ignore[union-attr]


def test_compat_wrapper_both_deprecated_attrs_warn_independently() -> None:
    """Accessing both .runtime and .state emits two separate DeprecationWarnings."""
    rt = Runtime(context=None)
    compat = _NamespaceRuntimeCompat(runtime=rt, state={"key": "val"})

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = compat.runtime
        _ = compat.state
        assert len(w) == 2
        categories = [warning.category for warning in w]
        assert all(c is DeprecationWarning for c in categories)
        messages = [str(warning.message) for warning in w]
        assert any(".runtime" in m for m in messages)
        assert any(".state" in m for m in messages)


def test_compat_wrapper_old_factory_none_runtime_chained_access() -> None:
    """Old-style factory with runtime=None that chains .runtime.context raises AttributeError."""
    compat = _NamespaceRuntimeCompat(runtime=None)

    def old_factory_chaining(ctx: BackendContext) -> tuple[str, ...]:  # type: ignore[type-arg]
        return (ctx.runtime.context.user_id, "filesystem")  # type: ignore[union-attr]

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        # .runtime returns None (with deprecation warning), chaining .context on None raises
        with pytest.raises(AttributeError):
            old_factory_chaining(compat)  # type: ignore[arg-type]


def test_get_namespace_outside_graph_execution_with_factory() -> None:
    """_get_namespace() works with a namespace factory outside graph execution (no runtime)."""
    mem_store = InMemoryStore()
    # This factory ignores the runtime arg, so runtime=None is fine
    be = StoreBackend(store=mem_store, namespace=lambda _rt: ("myns",))

    # Should not raise — factory ignores rt and runtime=None is gracefully handled
    be.write("/file.txt", "content")
    read = be.read("/file.txt")
    assert read.error is None
    assert read.file_data is not None
    assert "content" in read.file_data["content"]
    items = mem_store.search(("myns",))
    assert len(items) == 1
    assert items[0].key == "/file.txt"


def test_compat_wrapper_factory_returning_none_raises() -> None:
    """A namespace factory returning None triggers ValueError from _validate_namespace."""
    mem_store = InMemoryStore()

    def none_returning_factory(_rt: object) -> None:  # type: ignore[return]
        return None  # type: ignore[return-value]

    be = StoreBackend(store=mem_store, namespace=none_returning_factory)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="must not be empty"):
        be.write("/test.txt", "content")


def test_backend_context_still_instantiatable() -> None:
    """BackendContext class is still importable and instantiatable for backwards compat."""
    rt = Runtime(context=None)
    ctx = BackendContext(state=None, runtime=rt)
    assert ctx.runtime is rt
    assert ctx.state is None


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
