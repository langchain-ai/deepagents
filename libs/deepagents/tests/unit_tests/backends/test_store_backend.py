import warnings
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Never
from unittest.mock import patch

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
    txt = be.read("/docs/readme.md")
    assert "hello store" in txt

    # edit
    msg2 = be.edit("/docs/readme.md", "hello", "hi", replace_all=False)
    assert isinstance(msg2, EditResult) and msg2.error is None and msg2.occurrences == 1

    # ls_info (path prefix filter)
    infos = be.ls_info("/docs/")
    assert any(i["path"] == "/docs/readme.md" for i in infos)

    # grep_raw
    matches = be.grep_raw("hi", path="/")
    assert isinstance(matches, list) and any(m["path"] == "/docs/readme.md" for m in matches)

    # glob_info
    g = be.glob_info("*.md", path="/")
    assert len(g) == 0

    g2 = be.glob_info("**/*.md", path="/")
    assert any(i["path"] == "/docs/readme.md" for i in g2)


def test_store_backend_write_rejects_existing_file():
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, namespace=lambda _ctx: ("filesystem",))

    # Create the file
    result = be.write("/charmander.txt", "Hello world")
    assert result.error is None

    # Attempt to overwrite — should fail
    result = be.write("/charmander.txt", "New content")
    assert result.error is not None
    assert "Cannot write" in result.error
    assert "already exists" in result.error

    # Original content preserved
    read_result = be.read("/charmander.txt")
    assert read_result.file_data is not None
    assert "Hello world" in read_result.file_data["content"]


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

    root_listing = be.ls_info("/")
    root_paths = [fi["path"] for fi in root_listing]
    assert "/config.json" in root_paths
    assert "/src/" in root_paths
    assert "/docs/" in root_paths
    assert "/src/main.py" not in root_paths
    assert "/src/utils/helper.py" not in root_paths
    assert "/docs/readme.md" not in root_paths
    assert "/docs/api/reference.md" not in root_paths

    src_listing = be.ls_info("/src/")
    src_paths = [fi["path"] for fi in src_listing]
    assert "/src/main.py" in src_paths
    assert "/src/utils/" in src_paths
    assert "/src/utils/helper.py" not in src_paths

    utils_listing = be.ls_info("/src/utils/")
    utils_paths = [fi["path"] for fi in utils_listing]
    assert "/src/utils/helper.py" in utils_paths
    assert "/src/utils/common.py" in utils_paths
    assert len(utils_paths) == 2

    empty_listing = be.ls_info("/nonexistent/")
    assert empty_listing == []


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

    listing_from_root = be.ls_info("/")
    assert len(listing_from_root) > 0

    listing1 = be.ls_info("/dir/")
    listing2 = be.ls_info("/dir")
    assert len(listing1) == len(listing2)
    assert [fi["path"] for fi in listing1] == [fi["path"] for fi in listing2]


def test_store_backend_intercept_large_tool_result():
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
    assert stored_content.value["content"] == [large_content]


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
    content = be.read("/test.txt")
    assert "hello alice" in content


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
    alice_content = be_alice.read("/notes.txt")
    assert "alice notes" in alice_content

    bob_content = be_bob.read("/notes.txt")
    assert "bob notes" in bob_content

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
    matches = be.grep_raw(pattern, path="/")
    assert isinstance(matches, list)
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


def test_store_backend_legacy_path_rejects_malicious_assistant_id() -> None:
    """Ensure the legacy namespace path validates assistant_id from config metadata.

    A wildcard or otherwise malformed assistant_id must raise ValueError and
    never reach the store, closing the injection gap in _get_namespace_legacy.
    """
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store)  # No namespace factory — triggers legacy path

    malicious_ids = ["*", "user*", "a b", "path/to", "$var", "ns[0]", "ns{a}"]

    for bad_id in malicious_ids:
        fake_cfg = {"metadata": {"assistant_id": bad_id}}
        with patch("deepagents.backends.store.get_config", return_value=fake_cfg), warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with pytest.raises(ValueError, match="disallowed characters"):
                be.write("/test.txt", "content")
