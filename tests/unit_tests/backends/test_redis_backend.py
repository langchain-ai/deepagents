import pytest
from unittest.mock import MagicMock, patch
from deepagents.backends.redis_backend import RedisBackend


@pytest.fixture
def mock_redis():
    with patch("deepagents.backends.redis_backend.redis.Redis") as mock:
        client = MagicMock()
        mock.return_value = client
        yield client


def test_write_creates_file(mock_redis):
    backend = RedisBackend(host="localhost", port=6379, namespace="test")
    mock_redis.hexists.return_value = False
    mock_redis.hset.return_value = 1

    result = backend.write("/workspace/hello.txt", "hello world")

    assert result.error is None
    assert result.path == "/workspace/hello.txt"
    assert result.files_update is None  # 外部存储，不走 LangGraph state


def test_write_fails_on_conflict(mock_redis):
    backend = RedisBackend(host="localhost", port=6379, namespace="test")
    mock_redis.hexists.return_value = True

    result = backend.write("/workspace/hello.txt", "hello world")

    assert result.error is not None
    assert "already exists" in result.error


def test_read_returns_content_with_line_numbers(mock_redis):
    backend = RedisBackend(host="localhost", port=6379, namespace="test")
    mock_redis.hget.return_value = b"line1\nline2\nline3"

    content = backend.read("/workspace/hello.txt")

    assert "1\tline1" in content
    assert "2\tline2" in content
    assert "3\tline3" in content


def test_read_missing_file_returns_error(mock_redis):
    backend = RedisBackend(host="localhost", port=6379, namespace="test")
    mock_redis.hget.return_value = None

    content = backend.read("/workspace/missing.txt")

    assert "not found" in content.lower()


def test_write_with_ttl_calls_expire(mock_redis):
    backend = RedisBackend(namespace="test", ttl_seconds=300)
    mock_redis.hexists.return_value = False

    backend.write("/ws/file.txt", "data")

    mock_redis.expire.assert_called_once()


def test_edit_replaces_unique_string(mock_redis):
    backend = RedisBackend(namespace="test")
    mock_redis.hget.return_value = b"hello world"

    result = backend.edit("/ws/file.txt", "hello", "hi")

    assert result.error is None
    assert result.path == "/ws/file.txt"
    assert result.occurrences == 1
    mock_redis.hset.assert_called_once()


def test_edit_fails_when_file_not_found(mock_redis):
    backend = RedisBackend(namespace="test")
    mock_redis.hget.return_value = None

    result = backend.edit("/ws/missing.txt", "x", "y")

    assert result.error is not None
    assert "not found" in result.error.lower()


def test_edit_fails_when_string_not_found(mock_redis):
    backend = RedisBackend(namespace="test")
    mock_redis.hget.return_value = b"hello world"

    result = backend.edit("/ws/file.txt", "xyz", "abc")

    assert result.error is not None
    assert "not found" in result.error.lower()


def test_edit_fails_on_multiple_occurrences_without_replace_all(mock_redis):
    backend = RedisBackend(namespace="test")
    mock_redis.hget.return_value = b"foo foo foo"

    result = backend.edit("/ws/file.txt", "foo", "bar", replace_all=False)

    assert result.error is not None
    assert "3" in result.error


def test_edit_replace_all_succeeds(mock_redis):
    backend = RedisBackend(namespace="test")
    mock_redis.hget.return_value = b"foo foo foo"

    result = backend.edit("/ws/file.txt", "foo", "bar", replace_all=True)

    assert result.error is None
    assert result.occurrences == 3


def test_ls_info_returns_direct_children(mock_redis):
    backend = RedisBackend(namespace="test")
    mock_redis.hkeys.return_value = [
        b"/ws/a.txt",
        b"/ws/b.txt",
        b"/ws/sub/c.txt",
    ]

    entries = backend.ls_info("/ws")
    paths = [e["path"] for e in entries]

    assert "/ws/a.txt" in paths
    assert "/ws/b.txt" in paths
    assert "/ws/sub/" in paths
    assert "/ws/sub/c.txt" not in paths  # 不递归


def test_ls_info_empty_directory(mock_redis):
    backend = RedisBackend(namespace="test")
    mock_redis.hkeys.return_value = []

    entries = backend.ls_info("/ws")
    assert entries == []


def test_grep_raw_finds_matches(mock_redis):
    backend = RedisBackend(namespace="test")
    mock_redis.hgetall.return_value = {
        b"/ws/file.txt": b"apple\nbanana\napricot",
    }

    matches = backend.grep_raw(r"^ap")

    texts = [m["text"] for m in matches]
    assert "apple" in texts
    assert "apricot" in texts
    assert "banana" not in texts


def test_grep_raw_invalid_pattern_returns_error(mock_redis):
    backend = RedisBackend(namespace="test")

    result = backend.grep_raw(r"[invalid")

    assert isinstance(result, str)
    assert "Invalid regex" in result


def test_grep_raw_filters_by_path(mock_redis):
    backend = RedisBackend(namespace="test")
    mock_redis.hgetall.return_value = {
        b"/ws/file.txt": b"match",
        b"/other/file.txt": b"match",
    }

    matches = backend.grep_raw("match", path="/ws")

    paths = [m["path"] for m in matches]
    assert "/ws/file.txt" in paths
    assert "/other/file.txt" not in paths


def test_glob_info_matches_pattern(mock_redis):
    backend = RedisBackend(namespace="test")
    mock_redis.hkeys.return_value = [
        b"/ws/notes.txt",
        b"/ws/code.py",
        b"/ws/data.txt",
    ]

    results = backend.glob_info("/ws/*.txt")

    paths = [r["path"] for r in results]
    assert "/ws/notes.txt" in paths
    assert "/ws/data.txt" in paths
    assert "/ws/code.py" not in paths
