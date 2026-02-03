from langchain.tools import ToolRuntime
from langgraph.store.memory import InMemoryStore

from deepagents.backends.protocol import EditResult, WriteResult
from deepagents.backends.store import StoreBackend


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
    be = StoreBackend(rt)

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
    be = StoreBackend(rt)

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
    be = StoreBackend(rt)

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
    from langchain_core.messages import ToolMessage

    from deepagents.middleware.filesystem import FilesystemMiddleware

    rt = make_runtime()
    middleware = FilesystemMiddleware(backend=lambda r: StoreBackend(r), tool_token_limit_before_evict=1000)

    large_content = "y" * 5000
    tool_message = ToolMessage(content=large_content, tool_call_id="test_456")
    result = middleware._intercept_large_tool_result(tool_message, rt)

    assert isinstance(result, ToolMessage)
    assert "Tool result too large" in result.content
    assert "/large_tool_results/test_456" in result.content

    stored_content = rt.store.get(("filesystem",), "/large_tool_results/test_456")
    assert stored_content is not None
    assert stored_content.value["content"] == [large_content]


def test_store_backend_namespace_template_user_scoped() -> None:
    """Test namespace template with user_id variable."""
    store = InMemoryStore()
    rt = ToolRuntime(
        state={"messages": []},
        context=None,
        tool_call_id="t1",
        store=store,
        stream_writer=lambda _: None,
        config={"configurable": {"user_id": "alice"}},
    )
    be = StoreBackend(rt, namespace_template=("filesystem", "{user_id}"))

    # Write a file
    be.write("/test.txt", "hello alice")

    # Verify it's stored in the correct namespace
    items = store.search(("filesystem", "alice"))
    assert len(items) == 1
    assert items[0].key == "/test.txt"

    # Read it back
    content = be.read("/test.txt")
    assert "hello alice" in content


def test_store_backend_namespace_template_multi_level() -> None:
    """Test namespace template with multiple variables."""
    store = InMemoryStore()
    rt = ToolRuntime(
        state={"messages": []},
        context=None,
        tool_call_id="t1",
        store=store,
        stream_writer=lambda _: None,
        config={"configurable": {"workspace_id": "ws-123", "user_id": "bob"}},
    )
    be = StoreBackend(rt, namespace_template=("workspace", "{workspace_id}", "user", "{user_id}"))

    # Write a file
    be.write("/doc.md", "workspace doc")

    # Verify it's stored in the correct namespace
    items = store.search(("workspace", "ws-123", "user", "bob"))
    assert len(items) == 1
    assert items[0].key == "/doc.md"


def test_store_backend_namespace_template_isolation() -> None:
    """Test that different users have isolated namespaces."""
    store = InMemoryStore()

    # User alice
    rt_alice = ToolRuntime(
        state={"messages": []},
        context=None,
        tool_call_id="t1",
        store=store,
        stream_writer=lambda _: None,
        config={"configurable": {"user_id": "alice"}},
    )
    be_alice = StoreBackend(rt_alice, namespace_template=("filesystem", "{user_id}"))
    be_alice.write("/notes.txt", "alice notes")

    # User bob
    rt_bob = ToolRuntime(
        state={"messages": []},
        context=None,
        tool_call_id="t2",
        store=store,
        stream_writer=lambda _: None,
        config={"configurable": {"user_id": "bob"}},
    )
    be_bob = StoreBackend(rt_bob, namespace_template=("filesystem", "{user_id}"))
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


def test_store_backend_namespace_template_missing_variable() -> None:
    """Test error handling when required variable is missing from config."""
    import pytest

    rt = ToolRuntime(
        state={"messages": []},
        context=None,
        tool_call_id="t1",
        store=InMemoryStore(),
        stream_writer=lambda _: None,
        config={"configurable": {}},  # No user_id provided
    )
    be = StoreBackend(rt, namespace_template=("filesystem", "{user_id}"))

    # Should raise ValueError with helpful message
    with pytest.raises(ValueError, match="Missing namespace variable 'user_id'"):
        be.write("/test.txt", "content")


def test_store_backend_namespace_template_legacy_mode() -> None:
    """Test that legacy mode still works when no template is provided."""
    store = InMemoryStore()
    rt = ToolRuntime(
        state={"messages": []},
        context=None,
        tool_call_id="t1",
        store=store,
        stream_writer=lambda _: None,
        config={"metadata": {"assistant_id": "asst-123"}},
    )
    be = StoreBackend(rt)  # No template - uses legacy mode

    # Write a file
    be.write("/legacy.txt", "legacy content")

    # Should be in legacy namespace (assistant_id, filesystem)
    items = store.search(("asst-123", "filesystem"))
    assert len(items) == 1
    assert items[0].key == "/legacy.txt"


def test_store_backend_namespace_template_validation_wildcard() -> None:
    """Test that wildcards are rejected in namespace templates."""
    import pytest

    rt = ToolRuntime(
        state={"messages": []},
        context=None,
        tool_call_id="t1",
        store=InMemoryStore(),
        stream_writer=lambda _: None,
        config={"configurable": {"user_id": "alice"}},
    )

    # Should reject wildcards
    with pytest.raises(ValueError, match="Invalid namespace template segment"):
        StoreBackend(rt, namespace_template=("filesystem", "*"))


def test_store_backend_namespace_template_validation_special_chars() -> None:
    """Test that special characters are rejected in namespace templates."""
    import pytest

    rt = ToolRuntime(
        state={"messages": []},
        context=None,
        tool_call_id="t1",
        store=InMemoryStore(),
        stream_writer=lambda _: None,
        config={"configurable": {"user_id": "alice"}},
    )

    # Should reject paths with slashes
    with pytest.raises(ValueError, match="Invalid namespace template segment"):
        StoreBackend(rt, namespace_template=("filesystem", "users/{user_id}"))

    # Should reject other special chars like @
    with pytest.raises(ValueError, match="Invalid namespace template segment"):
        StoreBackend(rt, namespace_template=("filesystem", "user@domain"))


def test_store_backend_namespace_template_validation_valid() -> None:
    """Test that valid templates are accepted."""
    store = InMemoryStore()
    rt = ToolRuntime(
        state={"messages": []},
        context=None,
        tool_call_id="t1",
        store=store,
        stream_writer=lambda _: None,
        config={"configurable": {"userId": "alice"}},
    )

    # Should accept valid templates
    be = StoreBackend(rt, namespace_template=("filesystem", "{userId}"))
    assert be is not None

    # Should work with mixed case and numbers
    rt2 = ToolRuntime(
        state={"messages": []},
        context=None,
        tool_call_id="t2",
        store=store,
        stream_writer=lambda _: None,
        config={"configurable": {"workspaceId": "ws123"}},
    )
    be2 = StoreBackend(rt2, namespace_template=("workspace", "{workspaceId}"))
    assert be2 is not None

    # Should accept hyphens (for UUIDs)
    rt3 = ToolRuntime(
        state={"messages": []},
        context=None,
        tool_call_id="t3",
        store=store,
        stream_writer=lambda _: None,
        config={"configurable": {"user_id": "alice", "session_id": "abc-123"}},
    )
    be3 = StoreBackend(rt3, namespace_template=("user-sessions", "{user_id}", "{session_id}"))
    assert be3 is not None
