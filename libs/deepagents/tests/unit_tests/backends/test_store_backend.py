import warnings
from dataclasses import dataclass
from typing import Any, Never

import pytest
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langgraph.store.memory import InMemoryStore

from deepagents.backends.protocol import EditResult, WriteResult
from deepagents.backends.store import BackendContext, StoreBackend, _validate_namespace
from deepagents.middleware.filesystem import FilesystemMiddleware


def make_runtime():
    return ToolRuntime(
        state={"messages": []},
        context=None,
        tool_call_id="t2",
        store=InMemoryStore(),
        stream_writer=lambda _: None,
        config={},
    )


def test_store_backend_crud_and_search():
    rt = make_runtime()
    be = StoreBackend(rt, namespace=lambda _ctx: ("filesystem",))

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


def test_store_backend_ls_nested_directories():
    rt = make_runtime()
    be = StoreBackend(rt, namespace=lambda _ctx: ("filesystem",))

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
    rt = make_runtime()
    be = StoreBackend(rt, namespace=lambda _ctx: ("filesystem",))

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
    rt = make_runtime()
    middleware = FilesystemMiddleware(backend=lambda r: StoreBackend(r, namespace=lambda _ctx: ("filesystem",)), tool_token_limit_before_evict=1000)

    large_content = "y" * 5000
    tool_message = ToolMessage(content=large_content, tool_call_id="test_456")
    result = middleware._intercept_large_tool_result(tool_message, rt)

    assert isinstance(result, ToolMessage)
    assert "Tool result too large" in result.content
    assert "/large_tool_results/test_456" in result.content

    stored_content = rt.store.get(("filesystem",), "/large_tool_results/test_456")
    assert stored_content is not None
    assert stored_content.value["content"] == [large_content]


@dataclass
class UserContext:
    """Simple context object for testing."""

    user_id: str
    workspace_id: str | None = None


def test_store_backend_namespace_user_scoped() -> None:
    """Test namespace factory with user_id from context."""
    store = InMemoryStore()
    rt = ToolRuntime(
        state={"messages": []},
        context=UserContext(user_id="alice"),
        tool_call_id="t1",
        store=store,
        stream_writer=lambda _: None,
        config={},
    )
    be = StoreBackend(rt, namespace=lambda ctx: ("filesystem", ctx.runtime.context.user_id))

    # Write a file
    be.write("/test.txt", "hello alice")

    # Verify it's stored in the correct namespace
    items = store.search(("filesystem", "alice"))
    assert len(items) == 1
    assert items[0].key == "/test.txt"

    # Read it back
    content = be.read("/test.txt")
    assert "hello alice" in content


def test_store_backend_namespace_multi_level() -> None:
    """Test namespace factory with multiple values from context."""
    store = InMemoryStore()
    rt = ToolRuntime(
        state={"messages": []},
        context=UserContext(user_id="bob", workspace_id="ws-123"),
        tool_call_id="t1",
        store=store,
        stream_writer=lambda _: None,
        config={},
    )
    be = StoreBackend(
        rt,
        namespace=lambda ctx: (
            "workspace",
            ctx.runtime.context.workspace_id,
            "user",
            ctx.runtime.context.user_id,
        ),
    )

    # Write a file
    be.write("/doc.md", "workspace doc")

    # Verify it's stored in the correct namespace
    items = store.search(("workspace", "ws-123", "user", "bob"))
    assert len(items) == 1
    assert items[0].key == "/doc.md"


def test_store_backend_namespace_isolation() -> None:
    """Test that different users have isolated namespaces."""
    store = InMemoryStore()

    def user_namespace(ctx: BackendContext[Any, Any]) -> tuple[str, ...]:
        return ("filesystem", ctx.runtime.context.user_id)

    # User alice
    rt_alice = ToolRuntime(
        state={"messages": []},
        context=UserContext(user_id="alice"),
        tool_call_id="t1",
        store=store,
        stream_writer=lambda _: None,
        config={},
    )
    be_alice = StoreBackend(rt_alice, namespace=user_namespace)
    be_alice.write("/notes.txt", "alice notes")

    # User bob
    rt_bob = ToolRuntime(
        state={"messages": []},
        context=UserContext(user_id="bob"),
        tool_call_id="t2",
        store=store,
        stream_writer=lambda _: None,
        config={},
    )
    be_bob = StoreBackend(rt_bob, namespace=user_namespace)
    be_bob.write("/notes.txt", "bob notes")

    # Verify isolation
    alice_content = be_alice.read("/notes.txt")
    assert "alice notes" in alice_content

    bob_content = be_bob.read("/notes.txt")
    assert "bob notes" in bob_content

    # Verify they're in different namespaces
    alice_items = store.search(("filesystem", "alice"))
    assert len(alice_items) == 1
    bob_items = store.search(("filesystem", "bob"))
    assert len(bob_items) == 1


def test_store_backend_namespace_error_handling() -> None:
    """Test that factory errors propagate correctly."""

    def bad_factory(_ctx: BackendContext[Any, Any]) -> Never:
        msg = "user_id"
        raise KeyError(msg)

    rt = ToolRuntime(
        state={"messages": []},
        context=None,
        tool_call_id="t1",
        store=InMemoryStore(),
        stream_writer=lambda _: None,
        config={"configurable": {}},
    )
    be = StoreBackend(rt, namespace=bad_factory)

    # Errors from the factory propagate
    with pytest.raises(KeyError):
        be.write("/test.txt", "content")


def test_store_backend_namespace_legacy_mode() -> None:
    """Test that legacy mode still works when no namespace is provided, but emits deprecation warning."""
    store = InMemoryStore()
    rt = ToolRuntime(
        state={"messages": []},
        context=None,
        tool_call_id="t1",
        store=store,
        stream_writer=lambda _: None,
        config={"metadata": {"assistant_id": "asst-123"}},
    )
    be = StoreBackend(rt)  # No namespace - uses legacy mode

    # Should emit deprecation warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        be.write("/legacy.txt", "legacy content")
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "namespace" in str(w[0].message)

    # Should be in legacy namespace (assistant_id, filesystem)
    items = store.search(("asst-123", "filesystem"))
    assert len(items) == 1
    assert items[0].key == "/legacy.txt"


def test_store_backend_namespace_with_state() -> None:
    """Test that namespace factory receives state via BackendContext."""
    store = InMemoryStore()

    def namespace_from_state(ctx: BackendContext[Any, Any]) -> tuple[str, ...]:
        # Use something from state to build namespace
        thread_id = ctx.state.get("thread_id", "default")
        return ("threads", thread_id)

    rt = ToolRuntime(
        state={"messages": [], "thread_id": "thread-abc"},
        context=None,
        tool_call_id="t1",
        store=store,
        stream_writer=lambda _: None,
        config={},
    )
    be = StoreBackend(rt, namespace=namespace_from_state)

    # Write a file
    be.write("/test.txt", "content")

    # Verify it's stored in the correct namespace
    items = store.search(("threads", "thread-abc"))
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
    rt = make_runtime()
    be = StoreBackend(rt)

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
    store = InMemoryStore()
    rt = ToolRuntime(
        state={"messages": []},
        context=UserContext(user_id="*"),
        tool_call_id="t1",
        store=store,
        stream_writer=lambda _: None,
        config={},
    )
    be = StoreBackend(rt, namespace=lambda ctx: ("filesystem", ctx.runtime.context.user_id))

    with pytest.raises(ValueError, match="disallowed characters"):
        be.write("/test.txt", "content")
