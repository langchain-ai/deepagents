import pytest
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from deepagents.backends.protocol import EditResult, WriteResult
from deepagents.backends.state import StateBackend
from deepagents.middleware.filesystem import FilesystemMiddleware


def make_runtime(files=None):
    return ToolRuntime(
        state={
            "messages": [],
            "files": files or {},
        },
        context=None,
        tool_call_id="t1",
        store=None,
        stream_writer=lambda _: None,
        config={},
    )


def test_write_read_edit_ls_grep_glob_state_backend():
    rt = make_runtime()
    be = StateBackend(rt)

    # write
    res = be.write("/notes.txt", "hello world")
    assert isinstance(res, WriteResult)
    assert res.error is None and res.files_update is not None
    # apply state update
    rt.state["files"].update(res.files_update)

    # read
    content = be.read("/notes.txt")
    assert "hello world" in content

    # edit unique occurrence
    res2 = be.edit("/notes.txt", "hello", "hi", replace_all=False)
    assert isinstance(res2, EditResult)
    assert res2.error is None and res2.files_update is not None
    rt.state["files"].update(res2.files_update)

    content2 = be.read("/notes.txt")
    assert "hi world" in content2

    # ls_info should include the file
    listing = be.ls_info("/")
    assert any(fi["path"] == "/notes.txt" for fi in listing)

    # grep_raw
    matches = be.grep_raw("hi", path="/")
    assert isinstance(matches, list) and any(m["path"] == "/notes.txt" for m in matches)

    # special characters are treated literally, not regex
    result = be.grep_raw("[", path="/")
    assert isinstance(result, list)  # Returns empty list, not error

    # glob_info
    infos = be.glob_info("*.txt", path="/")
    assert any(i["path"] == "/notes.txt" for i in infos)


def test_state_backend_errors():
    rt = make_runtime()
    be = StateBackend(rt)

    # edit missing file
    err = be.edit("/missing.txt", "a", "b")
    assert isinstance(err, EditResult) and err.error and "not found" in err.error

    # write duplicate
    res = be.write("/dup.txt", "x")
    assert isinstance(res, WriteResult) and res.files_update is not None
    rt.state["files"].update(res.files_update)
    dup_err = be.write("/dup.txt", "y")
    assert isinstance(dup_err, WriteResult) and dup_err.error and "already exists" in dup_err.error


def test_state_backend_ls_nested_directories():
    rt = make_runtime()
    be = StateBackend(rt)

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
        rt.state["files"].update(res.files_update)

    root_listing = be.ls_info("/")
    root_paths = [fi["path"] for fi in root_listing]
    assert "/config.json" in root_paths
    assert "/src/" in root_paths
    assert "/docs/" in root_paths
    assert "/src/main.py" not in root_paths
    assert "/src/utils/helper.py" not in root_paths

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


def test_state_backend_ls_trailing_slash():
    rt = make_runtime()
    be = StateBackend(rt)

    files = {
        "/file.txt": "content",
        "/dir/nested.txt": "nested",
    }

    for path, content in files.items():
        res = be.write(path, content)
        assert res.error is None
        rt.state["files"].update(res.files_update)

    listing_with_slash = be.ls_info("/")
    assert len(listing_with_slash) == 2
    assert "/file.txt" in [fi["path"] for fi in listing_with_slash]
    assert "/dir/" in [fi["path"] for fi in listing_with_slash]

    listing_from_dir = be.ls_info("/dir/")
    assert len(listing_from_dir) == 1
    assert listing_from_dir[0]["path"] == "/dir/nested.txt"


def test_state_backend_intercept_large_tool_result():
    """Test that StateBackend properly handles large tool result interception."""
    rt = make_runtime()
    middleware = FilesystemMiddleware(backend=StateBackend, tool_token_limit_before_evict=1000)

    large_content = "x" * 5000
    tool_message = ToolMessage(content=large_content, tool_call_id="test_123")
    result = middleware._intercept_large_tool_result(tool_message, rt)

    assert isinstance(result, Command)
    assert "/large_tool_results/test_123" in result.update["files"]
    assert result.update["files"]["/large_tool_results/test_123"]["content"] == [large_content]
    assert "Tool result too large" in result.update["messages"][0].content


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
    rt = make_runtime()
    be = StateBackend(rt)

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
        rt.state["files"].update(res.files_update)

    # Test literal search with the pattern
    matches = be.grep_raw(pattern, path="/")
    assert isinstance(matches, list)
    assert any(expected_file in m["path"] for m in matches), f"Pattern '{pattern}' not found in {expected_file}"


def test_state_backend_grep_exact_file_path() -> None:
    """Test that grep works with exact file paths (no trailing slash).

    This reproduces the bug where validate_path adds a trailing slash to all paths,
    causing exact file path matching to fail with startswith filter.

    Bug: When grep is called with an exact file path like "/data/result_abc123",
    validate_path adds a trailing slash making it "/data/result_abc123/",
    which doesn't match the key in state (which has no trailing slash).
    """
    rt = make_runtime()
    be = StateBackend(rt)

    # Simulate an evicted large tool result (like what happens with large API responses)
    evicted_path = "/large_tool_results/toolu_01ABC123XYZ"
    content = """Task Results:
Project Alpha - Status: Active
Project Beta - Status: Pending
Project Gamma - Status: Completed
Total projects: 3
"""

    res = be.write(evicted_path, content)
    assert res.error is None
    rt.state["files"].update(res.files_update)

    # Test 1: Grep with parent directory path works (establishes baseline)
    matches_parent = be.grep_raw("Project Beta", path="/large_tool_results/")
    assert isinstance(matches_parent, list)
    assert len(matches_parent) == 1
    assert matches_parent[0]["path"] == evicted_path
    assert "Project Beta" in matches_parent[0]["text"]

    # Test 2: Grep with exact file path should also work (THIS IS THE BUG)
    matches_exact = be.grep_raw("Project Beta", path=evicted_path)
    assert isinstance(matches_exact, list), f"Expected list but got: {matches_exact}"
    assert len(matches_exact) == 1, f"Expected 1 match but got {len(matches_exact)} matches"
    assert matches_exact[0]["path"] == evicted_path
    assert "Project Beta" in matches_exact[0]["text"]

    # Test 3: Verify glob also works with exact file paths
    glob_matches = be.glob_info("*", path=evicted_path)
    assert len(glob_matches) == 1
    assert glob_matches[0]["path"] == evicted_path


def test_state_backend_path_edge_cases() -> None:
    """Test edge cases in path handling for grep and glob operations."""
    rt = make_runtime()
    be = StateBackend(rt)

    # Create test files
    files = {
        "/file.txt": "root content",
        "/dir/nested.txt": "nested content",
        "/dir/subdir/deep.txt": "deep content",
    }

    for path, content in files.items():
        res = be.write(path, content)
        assert res.error is None
        rt.state["files"].update(res.files_update)

    # Test 1: Grep with None path should default to root
    matches = be.grep_raw("content", path=None)
    assert isinstance(matches, list)
    assert len(matches) == 3

    # Test 2: Grep with trailing slash on directory
    matches_slash = be.grep_raw("nested", path="/dir/")
    assert isinstance(matches_slash, list)
    assert len(matches_slash) == 1
    assert matches_slash[0]["path"] == "/dir/nested.txt"

    # Test 3: Grep with no trailing slash on directory
    matches_no_slash = be.grep_raw("nested", path="/dir")
    assert isinstance(matches_no_slash, list)
    assert len(matches_no_slash) == 1
    assert matches_no_slash[0]["path"] == "/dir/nested.txt"

    # Test 4: Glob with exact file path
    glob_exact = be.glob_info("*.txt", path="/file.txt")
    assert len(glob_exact) == 1
    assert glob_exact[0]["path"] == "/file.txt"

    # Test 5: Glob with directory and pattern
    glob_dir = be.glob_info("*.txt", path="/dir/")
    assert len(glob_dir) == 1  # Only nested.txt, not deep.txt (non-recursive)
    assert glob_dir[0]["path"] == "/dir/nested.txt"

    # Test 6: Glob with recursive pattern
    glob_recursive = be.glob_info("**/*.txt", path="/dir/")
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
    rt = make_runtime()
    be = StateBackend(rt)

    # Create nested structure
    res1 = be.write("/app/main.py", "import os\nprint('main')")
    res2 = be.write("/app/utils.py", "import sys\nprint('utils')")
    res3 = be.write("/tests/test_main.py", "import pytest")

    for res in [res1, res2, res3]:
        assert res.error is None
        rt.state["files"].update(res.files_update)

    # Test the path variation
    matches = be.grep_raw("import", path=path)
    assert isinstance(matches, list)
    assert len(matches) == expected_count
    match_paths = {m["path"] for m in matches}
    assert match_paths == set(expected_paths)
