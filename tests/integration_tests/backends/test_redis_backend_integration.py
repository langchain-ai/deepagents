"""
集成测试：需要真实 Redis 实例（localhost:6379）。
若 Redis 不可用则自动跳过。
"""
import pytest
from deepagents.backends.redis_backend import RedisBackend

REDIS_AVAILABLE = True
try:
    import redis as _r
    _r.Redis(host="localhost", port=6379).ping()
except Exception:
    REDIS_AVAILABLE = False

pytestmark = pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis not available")


@pytest.fixture
def backend():
    b = RedisBackend(namespace="test_integration")
    # 清理测试数据
    b._client.delete(b._hash_key)
    yield b
    b._client.delete(b._hash_key)


def test_write_read_roundtrip(backend):
    backend.write("/ws/note.txt", "hello\nworld")
    content = backend.read("/ws/note.txt")
    assert "hello" in content
    assert "world" in content


def test_ls_info_lists_direct_children(backend):
    backend.write("/ws/a.txt", "a")
    backend.write("/ws/b.txt", "b")
    backend.write("/ws/sub/c.txt", "c")

    entries = backend.ls_info("/ws")
    paths = [e["path"] for e in entries]
    assert "/ws/a.txt" in paths
    assert "/ws/b.txt" in paths
    assert "/ws/sub/" in paths
    assert "/ws/sub/c.txt" not in paths  # 不递归


def test_edit_replaces_content(backend):
    backend.write("/ws/edit.txt", "foo bar foo")
    result = backend.edit("/ws/edit.txt", "foo", "baz", replace_all=True)
    assert result.error is None
    content = backend.read("/ws/edit.txt")
    assert "baz bar baz" in content


def test_grep_finds_pattern(backend):
    backend.write("/ws/grep.txt", "apple\nbanana\napricot")
    matches = backend.grep_raw(r"^ap", "/ws")
    paths = [m["path"] for m in matches]
    lines = [m["text"] for m in matches]
    assert "/ws/grep.txt" in paths
    assert "apple" in lines
    assert "apricot" in lines
    assert "banana" not in lines
