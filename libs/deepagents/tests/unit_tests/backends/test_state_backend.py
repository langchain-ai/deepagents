from contextlib import contextmanager

import pytest
from langchain_core.runnables.config import var_child_runnable_config
from langgraph._internal._constants import CONFIG_KEY_READ, CONFIG_KEY_SEND

from deepagents.backends.protocol import EditResult, WriteResult
from deepagents.backends.state import StateBackend
from deepagents.middleware.filesystem import FilesystemMiddleware, _file_data_reducer


def _make_state_config(files=None):
    """Create a mock config with CONFIG_KEY_SEND and CONFIG_KEY_READ.

    Returns (config, store) where store is a mutable dict so tests can
    inspect the resulting state.
    """
    store = {"files": files or {}}

    def read(select, fresh=False):  # noqa: ARG001
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
    """Context manager that activates a mock config for StateBackend."""
    config, store = _make_state_config(files)
    token = var_child_runnable_config.set(config)
    try:
        yield store
    finally:
        var_child_runnable_config.reset(token)


def test_write_read_edit_ls_grep_glob_state_backend():
    with state_config_context() as store:
        be = StateBackend()

        # write
        res = be.write("/notes.txt", "hello world")
        assert isinstance(res, WriteResult)
        assert res.error is None
        assert "/notes.txt" in store["files"]

        # read
        read_result = be.read("/notes.txt")
        assert "hello world" in read_result.file_data["content"]

        # edit unique occurrence
        res2 = be.edit("/notes.txt", "hello", "hi", replace_all=False)
        assert isinstance(res2, EditResult)
        assert res2.error is None

        read_result2 = be.read("/notes.txt")
        assert "hi world" in read_result2.file_data["content"]

        # ls_info should include the file
        listing = be.ls("/").entries
        assert listing is not None
        assert any(fi["path"] == "/notes.txt" for fi in listing)

        # grep
        matches = be.grep("hi", path="/").matches
        assert matches is not None and any(m["path"] == "/notes.txt" for m in matches)

        # special characters are treated literally, not regex
        result = be.grep("[", path="/")
        assert result.matches is not None  # Returns empty list, not error

        # glob
        infos = be.glob("*.txt", path="/").matches
        assert any(i["path"] == "/notes.txt" for i in infos)


def test_state_backend_errors():
    with state_config_context():
        be = StateBackend()

        # edit missing file
        err = be.edit("/missing.txt", "a", "b")
        assert isinstance(err, EditResult) and err.error and "not found" in err.error

        # write duplicate
        res = be.write("/dup.txt", "x")
        assert isinstance(res, WriteResult) and res.error is None
        dup_err = be.write("/dup.txt", "y")
        assert isinstance(dup_err, WriteResult) and dup_err.error and "already exists" in dup_err.error


def test_state_backend_ls_nested_directories():
    with state_config_context():
        be = StateBackend()

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


def test_state_backend_ls_trailing_slash():
    with state_config_context():
        be = StateBackend()

        files = {
            "/file.txt": "content",
            "/dir/nested.txt": "nested",
        }

        for path, content in files.items():
            res = be.write(path, content)
            assert res.error is None

        listing_with_slash = be.ls("/").entries
        assert listing_with_slash is not None
        assert len(listing_with_slash) == 2
        assert "/file.txt" in [fi["path"] for fi in listing_with_slash]
        assert "/dir/" in [fi["path"] for fi in listing_with_slash]

        listing_from_dir = be.ls("/dir/").entries
        assert listing_from_dir is not None
        assert len(listing_from_dir) == 1
        assert listing_from_dir[0]["path"] == "/dir/nested.txt"


@pytest.mark.parametrize("file_format", ["v1", "v2"])
def test_state_backend_intercept_large_tool_result(file_format):
    """Test that StateBackend properly handles large tool result interception."""
    from langchain.tools import ToolRuntime
    from langchain_core.messages import ToolMessage

    backend = StateBackend(file_format=file_format)
    middleware = FilesystemMiddleware(backend=backend, tool_token_limit_before_evict=1000)

    rt = ToolRuntime(
        state={"messages": [], "files": {}},
        context=None,
        tool_call_id="t1",
        store=None,
        stream_writer=lambda _: None,
        config=_make_state_config()[0],
    )

    with state_config_context() as store:
        large_content = "x" * 5000
        tool_message = ToolMessage(content=large_content, tool_call_id="test_123")
        result = middleware._intercept_large_tool_result(tool_message, rt)

        # With CONFIG_KEY_SEND, the write goes directly to state — result is just a ToolMessage
        assert isinstance(result, ToolMessage)
        assert "Tool result too large" in result.content
        # Verify the file was written to state via send
        assert "/large_tool_results/test_123" in store["files"]
        content = store["files"]["/large_tool_results/test_123"]["content"]
        if file_format == "v1":
            assert content == [large_content]
        else:
            assert content == large_content


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
def test_state_backend_grep_literal_search_special_chars(pattern: str, expected_file: str) -> None:
    """Test that grep performs literal search with regex special characters."""
    with state_config_context():
        be = StateBackend()

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


def test_state_backend_grep_exact_file_path() -> None:
    """Test that grep works with exact file paths (no trailing slash)."""
    with state_config_context():
        be = StateBackend()

        # Simulate an evicted large tool result
        evicted_path = "/large_tool_results/toolu_01ABC123XYZ"
        content = """Task Results:
Project Alpha - Status: Active
Project Beta - Status: Pending
Project Gamma - Status: Completed
Total projects: 3
"""

        res = be.write(evicted_path, content)
        assert res.error is None

        # Test 1: Grep with parent directory path works
        matches_parent = be.grep("Project Beta", path="/large_tool_results/").matches
        assert matches_parent is not None
        assert len(matches_parent) == 1
        assert matches_parent[0]["path"] == evicted_path
        assert "Project Beta" in matches_parent[0]["text"]

        # Test 2: Grep with exact file path should also work
        matches_exact = be.grep("Project Beta", path=evicted_path).matches
        assert matches_exact is not None, "Expected list but got None"
        assert len(matches_exact) == 1, f"Expected 1 match but got {len(matches_exact)} matches"
        assert matches_exact[0]["path"] == evicted_path
        assert "Project Beta" in matches_exact[0]["text"]

        # Test 3: Verify glob also works with exact file paths
        glob_matches = be.glob("*", path=evicted_path).matches
        assert glob_matches is not None
        assert len(glob_matches) == 1
        assert glob_matches[0]["path"] == evicted_path


def test_state_backend_path_edge_cases() -> None:
    """Test edge cases in path handling for grep and glob operations."""
    with state_config_context():
        be = StateBackend()

        # Create test files
        files = {
            "/file.txt": "root content",
            "/dir/nested.txt": "nested content",
            "/dir/subdir/deep.txt": "deep content",
        }

        for path, content in files.items():
            res = be.write(path, content)
            assert res.error is None

        # Test 1: Grep with None path should default to root
        matches = be.grep("content", path=None).matches
        assert matches is not None
        assert len(matches) == 3

        # Test 2: Grep with trailing slash on directory
        matches_slash = be.grep("nested", path="/dir/").matches
        assert matches_slash is not None
        assert len(matches_slash) == 1
        assert matches_slash[0]["path"] == "/dir/nested.txt"

        # Test 3: Grep with no trailing slash on directory
        matches_no_slash = be.grep("nested", path="/dir").matches
        assert matches_no_slash is not None
        assert len(matches_no_slash) == 1
        assert matches_no_slash[0]["path"] == "/dir/nested.txt"

        # Test 4: Glob with exact file path
        glob_exact = be.glob("*.txt", path="/file.txt").matches
        assert glob_exact is not None
        assert len(glob_exact) == 1
        assert glob_exact[0]["path"] == "/file.txt"

        # Test 5: Glob with directory and pattern
        glob_dir = be.glob("*.txt", path="/dir/").matches
        assert glob_dir is not None
        assert len(glob_dir) == 1  # Only nested.txt, not deep.txt (non-recursive)
        assert glob_dir[0]["path"] == "/dir/nested.txt"

        # Test 6: Glob with recursive pattern
        glob_recursive = be.glob("**/*.txt", path="/dir/").matches
        assert glob_recursive is not None
        assert len(glob_recursive) == 2  # Both nested.txt and deep.txt
        paths = {g["path"] for g in glob_recursive}
        assert "/dir/nested.txt" in paths
        assert "/dir/subdir/deep.txt" in paths


@pytest.mark.parametrize(
    ("path", "expected_count", "expected_paths"),
    [
        ("/app/main.py/", 1, ["/app/main.py"]),  # Exact file with trailing slash
        ("/app", 2, ["/app/main.py", "/app/utils.py"]),  # Directory without slash
        ("/app/", 2, ["/app/main.py", "/app/utils.py"]),  # Directory with slash
    ],
)
def test_state_backend_grep_with_path_variations(path: str, expected_count: int, expected_paths: list[str]) -> None:
    """Test grep with various path input formats."""
    with state_config_context():
        be = StateBackend()

        # Create nested structure
        res1 = be.write("/app/main.py", "import os\nprint('main')")
        res2 = be.write("/app/utils.py", "import sys\nprint('utils')")
        res3 = be.write("/tests/test_main.py", "import pytest")

        for res in [res1, res2, res3]:
            assert res.error is None

        # Test the path variation
        matches = be.grep("import", path=path).matches
        assert matches is not None
        assert len(matches) == expected_count
        match_paths = {m["path"] for m in matches}
        assert match_paths == set(expected_paths)


def test_state_backend_runtime_deprecation_warning():
    """Passing runtime= to StateBackend should emit a DeprecationWarning."""
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        StateBackend(runtime="ignored_value")
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) == 1
        assert "v0.7" in str(deprecation_warnings[0].message)
        assert "runtime" in str(deprecation_warnings[0].message)


def test_state_backend_no_deprecation_without_runtime():
    """StateBackend() without runtime should NOT emit a DeprecationWarning."""
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        StateBackend()
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) == 0


def test_state_backend_raises_outside_graph_context():
    """StateBackend operations outside a graph context should raise RuntimeError."""
    be = StateBackend()
    with pytest.raises(RuntimeError, match="inside a LangGraph graph execution"):
        be.read("/anything.txt")
