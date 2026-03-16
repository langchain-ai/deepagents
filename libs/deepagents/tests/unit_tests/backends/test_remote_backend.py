"""Unit tests for RemoteBackend backed by a moto-mock S3 server.

``ThreadedMotoServer`` starts a real HTTP server that intercepts all AWS API
calls, so ``s3fs`` / ``aiobotocore`` run exactly as they would against real S3.
No real AWS credentials or network access is required.

The ``endpoint_url`` in ``_backend()`` is test infrastructure only — it
redirects calls to the local mock server.  Real-world usage omits it entirely:

    from deepagents.backends.remote import RemoteBackend

    # Against real AWS S3 (uses default credential chain — no endpoint_url)
    backend = RemoteBackend(
        "s3://my-bucket/workspace/",
        storage_options={"client_kwargs": {"region_name": "us-east-1"}},
    )
"""

import socket
import uuid

import boto3
import pytest
from moto.server import ThreadedMotoServer

from deepagents.backends.protocol import EditResult, WriteResult
from deepagents.backends.remote import RemoteBackend

_BUCKET = "test-deepagents"
_REGION = "us-east-1"

# Set by the session fixture; each xdist worker picks its own free port.
_server_endpoint: str = ""

pytestmark = pytest.mark.allow_hosts(["localhost", "127.0.0.1"])


@pytest.fixture(scope="session", autouse=True)
def _moto_s3_server():
    """Start a moto S3 server once per worker session."""
    global _server_endpoint  # noqa: PLW0603

    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]

    endpoint = f"http://localhost:{port}"
    server = ThreadedMotoServer(port=port)
    server.start()

    boto3.client(
        "s3",
        region_name=_REGION,
        endpoint_url=endpoint,
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    ).create_bucket(Bucket=_BUCKET)

    _server_endpoint = endpoint
    yield
    server.stop()


def _backend(suffix: str = "") -> RemoteBackend:
    """Create a RemoteBackend pointed at a unique S3 prefix inside the mocked bucket."""
    uid = uuid.uuid4().hex
    prefix = f"{suffix.strip('/')}/{uid}" if suffix else uid
    return RemoteBackend(
        f"s3://{_BUCKET}/{prefix}/",
        storage_options={
            "key": "testing",
            "secret": "testing",
            "client_kwargs": {"region_name": _REGION, "endpoint_url": _server_endpoint},
            "skip_instance_cache": True,
        },
    )


def test_write_read_edit_roundtrip():
    be = _backend()

    # write
    result = be.write("/hello.txt", "hello world")
    assert isinstance(result, WriteResult)
    assert result.error is None
    assert result.path == "/hello.txt"
    assert result.files_update is None  # external storage — never touches LangGraph state

    # read
    txt = be.read("/hello.txt")
    assert "hello world" in txt

    # edit
    edit_result = be.edit("/hello.txt", "world", "remote", replace_all=False)
    assert isinstance(edit_result, EditResult)
    assert edit_result.error is None
    assert edit_result.occurrences == 1
    assert edit_result.files_update is None

    updated = be.read("/hello.txt")
    assert "hello remote" in updated


def test_write_duplicate_returns_error():
    be = _backend()
    be.write("/dup.txt", "first")
    result = be.write("/dup.txt", "second")
    assert result.error is not None
    assert "already exists" in result.error


def test_read_missing_file_returns_error():
    be = _backend()
    result = be.read("/nonexistent.txt")
    assert result.startswith("Error:")


def test_edit_missing_file_returns_error():
    be = _backend()
    result = be.edit("/nonexistent.txt", "old", "new")
    assert result.error is not None
    assert "not found" in result.error


def test_edit_replace_all():
    be = _backend()
    be.write("/multi.txt", "foo bar foo baz")

    # replace_all=False with multiple occurrences should fail
    res1 = be.edit("/multi.txt", "foo", "qux", replace_all=False)
    assert res1.error is not None
    assert "appears 2 times" in res1.error

    # replace_all=True replaces all
    res2 = be.edit("/multi.txt", "foo", "qux", replace_all=True)
    assert res2.error is None
    assert res2.occurrences == 2
    assert "qux bar qux baz" in be.read("/multi.txt")


def test_read_offset_and_limit():
    be = _backend()
    lines = "\n".join([f"Line {i}" for i in range(1, 11)])
    be.write("/multi.txt", lines)

    content = be.read("/multi.txt", offset=2, limit=3)
    assert "Line 3" in content
    assert "Line 4" in content
    assert "Line 5" in content
    assert "Line 1" not in content
    assert "Line 6" not in content


# ---------------------------------------------------------------------------
# ls_info
# ---------------------------------------------------------------------------


def test_ls_info_root():
    be = _backend()
    be.write("/a.txt", "a")
    be.write("/b.txt", "b")
    be.write("/src/main.py", "code")

    infos = be.ls_info("/")
    paths = {i["path"] for i in infos}
    assert "/a.txt" in paths
    assert "/b.txt" in paths
    assert "/src/" in paths  # directory
    assert "/src/main.py" not in paths  # not a direct child


def test_ls_info_nested_directories():
    be = _backend()
    be.write("/config.json", "config")
    be.write("/src/main.py", "code")
    be.write("/src/utils/helper.py", "utils")
    be.write("/src/utils/common.py", "common")
    be.write("/docs/readme.md", "docs")
    be.write("/docs/api/reference.md", "api docs")

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


def test_ls_info_trailing_slash():
    be = _backend()
    be.write("/file.txt", "content")
    be.write("/dir/nested.txt", "nested")

    listing_with_slash = be.ls_info("/")
    assert len(listing_with_slash) > 0

    listing = be.ls_info("/")
    paths = [fi["path"] for fi in listing]
    assert paths == sorted(paths)

    listing1 = be.ls_info("/dir/")
    listing2 = be.ls_info("/dir")
    assert len(listing1) == len(listing2)
    assert [fi["path"] for fi in listing1] == [fi["path"] for fi in listing2]

    empty = be.ls_info("/nonexistent/")
    assert empty == []


def test_ls_info_nonexistent_returns_empty():
    be = _backend()
    assert be.ls_info("/nonexistent/") == []


def test_ls_info_sorted():
    be = _backend()
    be.write("/z.txt", "z")
    be.write("/a.txt", "a")
    be.write("/m.txt", "m")

    infos = be.ls_info("/")
    paths = [i["path"] for i in infos]
    assert paths == sorted(paths)


# ---------------------------------------------------------------------------
# grep_raw
# ---------------------------------------------------------------------------


def test_grep_raw_finds_match():
    be = _backend()
    be.write("/a.txt", "hello world\nsecond line")
    be.write("/b.txt", "unrelated content")

    matches = be.grep_raw("hello", path="/")
    assert isinstance(matches, list)
    assert any(m["path"] == "/a.txt" for m in matches)
    assert not any(m["path"] == "/b.txt" for m in matches)


def test_grep_raw_restricted_to_path():
    be = _backend()
    be.write("/src/main.py", "import os")
    be.write("/docs/readme.md", "import nothing")

    matches = be.grep_raw("import", path="/src")
    paths = [m["path"] for m in matches]
    assert any("main.py" in p for p in paths)
    assert not any("readme" in p for p in paths)


def test_grep_raw_glob_filter():
    be = _backend()
    be.write("/test.py", "import os")
    be.write("/test.txt", "import nothing")
    be.write("/main.py", "import sys")

    matches = be.grep_raw("import", path="/", glob="*.py")
    py_paths = [m["path"] for m in matches]
    assert any("test.py" in p for p in py_paths)
    assert any("main.py" in p for p in py_paths)
    assert not any("test.txt" in p for p in py_paths)


@pytest.mark.parametrize(
    ("pattern", "expected_file"),
    [
        ("def __init__(", "code.py"),  # Parentheses (not regex grouping)
        ("str | int", "types.py"),  # Pipe (not regex OR)
        ("[a-z]", "regex.py"),  # Brackets (not character class)
        ("(.*)", "regex.py"),  # Multiple special chars
    ],
)
def test_grep_raw_literal_search_with_special_chars(pattern: str, expected_file: str) -> None:
    """Test that grep treats patterns as literal strings, not regex."""
    be = _backend()
    be.write("/code.py", "def __init__(self):\n    pass")
    be.write("/types.py", "x: str | int")
    be.write("/regex.py", "pattern = r'[a-z]+'\nregex_chars = '(.*)'")

    matches = be.grep_raw(pattern, path="/")
    assert isinstance(matches, list)
    assert any(expected_file in m["path"] for m in matches), f"Pattern {pattern!r} not found in {expected_file}"


def test_grep_raw_no_match_returns_empty():
    be = _backend()
    be.write("/file.txt", "some content")

    matches = be.grep_raw("zzz_no_match", path="/")
    assert isinstance(matches, list)
    assert len(matches) == 0


# ---------------------------------------------------------------------------
# glob_info
# ---------------------------------------------------------------------------


def test_glob_info_simple_pattern():
    be = _backend()
    be.write("/src/main.py", "code")
    be.write("/src/utils/helper.py", "utils")
    be.write("/readme.txt", "docs")

    result = be.glob_info("**/*.py", path="/")
    py_paths = [i["path"] for i in result]
    assert any("main.py" in p for p in py_paths)
    assert any("helper.py" in p for p in py_paths)
    assert not any("readme.txt" in p for p in py_paths)


def test_glob_info_sorted():
    be = _backend()
    be.write("/z/file.py", "z")
    be.write("/a/file.py", "a")

    infos = be.glob_info("**/*.py", path="/")
    paths = [i["path"] for i in infos]
    assert paths == sorted(paths)


# ---------------------------------------------------------------------------
# upload_files / download_files
# ---------------------------------------------------------------------------


def test_upload_single_file():
    be = _backend()
    responses = be.upload_files([("/upload.bin", b"binary content")])
    assert len(responses) == 1
    assert responses[0].path == "/upload.bin"
    assert responses[0].error is None


def test_upload_multiple_files():
    be = _backend()
    files = [
        ("/file1.bin", b"Content 1"),
        ("/file2.bin", b"Content 2"),
        ("/subdir/file3.bin", b"Content 3"),
    ]
    responses = be.upload_files(files)
    assert len(responses) == 3
    for i, (path, _) in enumerate(files):
        assert responses[i].path == path
        assert responses[i].error is None


def test_download_single_file():
    be = _backend()
    be.upload_files([("/data.bin", b"Download me!")])

    responses = be.download_files(["/data.bin"])
    assert len(responses) == 1
    assert responses[0].path == "/data.bin"
    assert responses[0].content == b"Download me!"
    assert responses[0].error is None


def test_download_multiple_files():
    be = _backend()
    files = [
        ("/file1.txt", b"File 1"),
        ("/file2.txt", b"File 2"),
        ("/subdir/file3.txt", b"File 3"),
    ]
    be.upload_files(files)

    responses = be.download_files(["/file1.txt", "/file2.txt", "/subdir/file3.txt"])
    assert len(responses) == 3
    assert responses[0].content == b"File 1" and responses[0].error is None
    assert responses[1].content == b"File 2" and responses[1].error is None
    assert responses[2].content == b"File 3" and responses[2].error is None


def test_upload_download_roundtrip():
    be = _backend()
    content = bytes(range(256))  # All possible byte values

    be.upload_files([("/roundtrip.bin", content)])
    responses = be.download_files(["/roundtrip.bin"])
    assert responses[0].error is None
    assert responses[0].content == content


def test_download_errors():
    be = _backend()

    # file_not_found
    responses = be.download_files(["/nonexistent.txt"])
    assert len(responses) == 1
    assert responses[0].path == "/nonexistent.txt"
    assert responses[0].content is None
    assert responses[0].error == "file_not_found"

    # invalid_path (path traversal)
    responses = be.download_files(["/../etc/passwd"])
    assert len(responses) == 1
    assert responses[0].error == "invalid_path"
    assert responses[0].content is None


def test_upload_errors():
    be = _backend()

    # invalid_path (path traversal)
    responses = be.upload_files([("/../bad/path.txt", b"content")])
    assert len(responses) == 1
    assert responses[0].error == "invalid_path"


def test_partial_success_upload():
    be = _backend()
    files = [
        ("/valid1.txt", b"Valid content 1"),
        ("/../invalid.txt", b"Invalid path"),
        ("/valid2.txt", b"Valid content 2"),
    ]
    responses = be.upload_files(files)
    assert len(responses) == 3
    assert responses[0].error is None
    assert responses[1].error == "invalid_path"
    assert responses[2].error is None


def test_partial_success_download():
    be = _backend()
    be.upload_files([("/exists.txt", b"I exist!")])

    responses = be.download_files(["/exists.txt", "/doesnotexist.txt", "/../invalid"])
    assert len(responses) == 3
    assert responses[0].error is None
    assert responses[0].content == b"I exist!"
    assert responses[1].error == "file_not_found"
    assert responses[1].content is None
    assert responses[2].error == "invalid_path"
    assert responses[2].content is None


# ---------------------------------------------------------------------------
# Path traversal is blocked
# ---------------------------------------------------------------------------


def test_read_blocks_traversal():
    be = _backend()
    result = be.read("/../etc/passwd")
    assert result.startswith("Error:")


def test_write_blocks_traversal():
    be = _backend()
    result = be.write("/../etc/passwd", "content")
    assert result.error is not None


def test_edit_blocks_traversal():
    be = _backend()
    result = be.edit("/../etc/passwd", "old", "new")
    assert result.error is not None


# ---------------------------------------------------------------------------
# _to_virtual_path
# ---------------------------------------------------------------------------


class TestToVirtualPath:
    """Tests for RemoteBackend._to_virtual_path."""

    def test_returns_virtual_path(self):
        be = _backend("prefix")
        result = be._to_virtual_path(be._root + "/src/file.py")
        assert result == "/src/file.py"

    def test_root_itself_returns_slash(self):
        be = _backend()
        result = be._to_virtual_path(be._root)
        assert result == "/"

    def test_outside_root_raises_value_error(self):
        be = _backend()
        with pytest.raises(ValueError, match="outside root"):
            be._to_virtual_path("/completely/different/path")

    def test_resolve_path_blocks_traversal(self):
        be = _backend()
        with pytest.raises(ValueError, match="traversal"):
            be._resolve_path("/../etc/passwd")
