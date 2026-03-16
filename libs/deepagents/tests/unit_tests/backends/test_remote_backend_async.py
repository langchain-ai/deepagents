"""Async tests for RemoteBackend backed by a moto-mock S3 server.

See ``test_remote_backend.py`` for notes on real-world usage and why
``endpoint_url`` is test infrastructure only.
"""

import socket
import uuid

import boto3
import pytest
from moto.server import ThreadedMotoServer

from deepagents.backends.protocol import EditResult, WriteResult
from deepagents.backends.remote import RemoteBackend

_BUCKET = "test-deepagents-async"
_REGION = "us-east-1"

# Set by the session fixture; each xdist worker picks its own free port.
_server_endpoint: str = ""

pytestmark = pytest.mark.allow_hosts(["localhost", "127.0.0.1"])


@pytest.fixture(scope="session", autouse=True)
def _moto_s3_server_async():
    """Start a moto S3 server once per worker session (async test module)."""
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


async def test_remote_backend_async_write_read_edit():
    be = _backend()

    result = await be.awrite("/hello.txt", "hello async")
    assert isinstance(result, WriteResult) and result.error is None and result.files_update is None

    txt = await be.aread("/hello.txt")
    assert "hello async" in txt

    edit = await be.aedit("/hello.txt", "async", "remote", replace_all=False)
    assert isinstance(edit, EditResult) and edit.error is None and edit.occurrences == 1

    updated = await be.aread("/hello.txt")
    assert "hello remote" in updated


async def test_remote_backend_async_ls_info():
    be = _backend()
    await be.awrite("/a.txt", "a")
    await be.awrite("/src/main.py", "code")

    infos = await be.als_info("/")
    paths = {i["path"] for i in infos}
    assert "/a.txt" in paths
    assert "/src/" in paths
    assert "/src/main.py" not in paths

    src = await be.als_info("/src/")
    assert any(i["path"] == "/src/main.py" for i in src)


async def test_remote_backend_als_nested_directories():
    """Test async ls with nested directories."""
    be = _backend()
    await be.awrite("/config.json", "config")
    await be.awrite("/src/main.py", "code")
    await be.awrite("/src/utils/helper.py", "utils")
    await be.awrite("/src/utils/common.py", "common")
    await be.awrite("/docs/readme.md", "docs")
    await be.awrite("/docs/api/reference.md", "api docs")

    root_listing = await be.als_info("/")
    root_paths = [fi["path"] for fi in root_listing]
    assert "/config.json" in root_paths
    assert "/src/" in root_paths
    assert "/docs/" in root_paths
    assert "/src/main.py" not in root_paths
    assert "/src/utils/helper.py" not in root_paths

    src_listing = await be.als_info("/src/")
    src_paths = [fi["path"] for fi in src_listing]
    assert "/src/main.py" in src_paths
    assert "/src/utils/" in src_paths
    assert "/src/utils/helper.py" not in src_paths

    utils_listing = await be.als_info("/src/utils/")
    utils_paths = [fi["path"] for fi in utils_listing]
    assert "/src/utils/helper.py" in utils_paths
    assert "/src/utils/common.py" in utils_paths
    assert len(utils_paths) == 2

    empty_listing = await be.als_info("/nonexistent/")
    assert empty_listing == []


async def test_remote_backend_als_trailing_slash():
    """Test async ls_info edge cases with trailing slashes."""
    be = _backend()
    await be.awrite("/file.txt", "content")
    await be.awrite("/dir/nested.txt", "nested")

    listing_with_slash = await be.als_info("/")
    assert len(listing_with_slash) > 0

    listing = await be.als_info("/")
    paths = [fi["path"] for fi in listing]
    assert paths == sorted(paths)

    listing1 = await be.als_info("/dir/")
    listing2 = await be.als_info("/dir")
    assert len(listing1) == len(listing2)
    assert [fi["path"] for fi in listing1] == [fi["path"] for fi in listing2]

    empty = await be.als_info("/nonexistent/")
    assert empty == []


async def test_remote_backend_async_grep_raw():
    be = _backend()
    await be.awrite("/a.txt", "hello world")
    await be.awrite("/b.txt", "unrelated")

    matches = await be.agrep_raw("hello", path="/")
    assert isinstance(matches, list)
    assert any(m["path"] == "/a.txt" for m in matches)
    assert not any(m["path"] == "/b.txt" for m in matches)


async def test_remote_backend_async_grep_glob_filter():
    be = _backend()
    await be.awrite("/test.py", "import os")
    await be.awrite("/test.txt", "import nothing")

    matches = await be.agrep_raw("import", path="/", glob="*.py")
    py_paths = [m["path"] for m in matches]
    assert any("test.py" in p for p in py_paths)
    assert not any("test.txt" in p for p in py_paths)


async def test_remote_backend_async_glob_info():
    be = _backend()
    await be.awrite("/src/main.py", "code")
    await be.awrite("/src/utils/helper.py", "utils")
    await be.awrite("/readme.txt", "docs")

    infos = await be.aglob_info("**/*.py", path="/")
    py_paths = [i["path"] for i in infos]
    assert any("main.py" in p for p in py_paths)
    assert any("helper.py" in p for p in py_paths)
    assert not any("readme.txt" in p for p in py_paths)


async def test_remote_backend_async_upload_download():
    be = _backend()

    responses = await be.aupload_files([("/data.bin", b"async binary")])
    assert responses[0].error is None

    responses = await be.adownload_files(["/data.bin"])
    assert responses[0].content == b"async binary"
    assert responses[0].error is None


async def test_remote_backend_aupload_multiple_files():
    """Test async uploading multiple files in one call."""
    be = _backend()

    files = [
        ("/file1.bin", b"Content 1"),
        ("/file2.bin", b"Content 2"),
        ("/subdir/file3.bin", b"Content 3"),
    ]
    responses = await be.aupload_files(files)

    assert len(responses) == 3
    for i, (path, _) in enumerate(files):
        assert responses[i].path == path
        assert responses[i].error is None


async def test_remote_backend_adownload_multiple_files():
    """Test async downloading multiple files in one call."""
    be = _backend()

    files = [
        ("/file1.txt", b"File 1"),
        ("/file2.txt", b"File 2"),
        ("/subdir/file3.txt", b"File 3"),
    ]
    await be.aupload_files(files)

    responses = await be.adownload_files(["/file1.txt", "/file2.txt", "/subdir/file3.txt"])
    assert len(responses) == 3
    assert responses[0].content == b"File 1" and responses[0].error is None
    assert responses[1].content == b"File 2" and responses[1].error is None
    assert responses[2].content == b"File 3" and responses[2].error is None


async def test_remote_backend_async_upload_download_roundtrip():
    be = _backend()
    content = bytes(range(256))

    await be.aupload_files([("/roundtrip.bin", content)])
    responses = await be.adownload_files(["/roundtrip.bin"])
    assert responses[0].error is None
    assert responses[0].content == content


async def test_remote_backend_async_download_errors():
    be = _backend()

    responses = await be.adownload_files(["/nonexistent.txt"])
    assert responses[0].error == "file_not_found"
    assert responses[0].content is None

    responses = await be.adownload_files(["/../etc/passwd"])
    assert responses[0].error == "invalid_path"
    assert responses[0].content is None


async def test_remote_backend_aupload_errors():
    """Test async upload error handling."""
    be = _backend()

    responses = await be.aupload_files([("/../bad/path.txt", b"content")])
    assert len(responses) == 1
    assert responses[0].error == "invalid_path"


async def test_remote_backend_partial_success_aupload():
    """Test partial success in async batch upload."""
    be = _backend()

    files = [
        ("/valid1.txt", b"Valid content 1"),
        ("/../invalid.txt", b"Invalid path"),
        ("/valid2.txt", b"Valid content 2"),
    ]
    responses = await be.aupload_files(files)

    assert len(responses) == 3
    assert responses[0].error is None
    assert responses[1].error == "invalid_path"
    assert responses[2].error is None


async def test_remote_backend_partial_success_adownload():
    """Test partial success in async batch download."""
    be = _backend()
    await be.aupload_files([("/exists.txt", b"I exist!")])

    responses = await be.adownload_files(["/exists.txt", "/doesnotexist.txt", "/../invalid"])
    assert len(responses) == 3
    assert responses[0].error is None
    assert responses[0].content == b"I exist!"
    assert responses[1].error == "file_not_found"
    assert responses[1].content is None
    assert responses[2].error == "invalid_path"
    assert responses[2].content is None


async def test_remote_backend_async_traversal_blocked():
    be = _backend()

    result = await be.aread("/../etc/passwd")
    assert result.startswith("Error:")

    result = await be.awrite("/../etc/passwd", "x")
    assert result.error is not None

    result = await be.aedit("/../etc/passwd", "old", "new")
    assert result.error is not None


async def test_remote_backend_async_edit_replace_all():
    be = _backend()
    await be.awrite("/multi.txt", "foo bar foo baz")

    res1 = await be.aedit("/multi.txt", "foo", "qux", replace_all=False)
    assert res1.error is not None
    assert "appears 2 times" in res1.error

    res2 = await be.aedit("/multi.txt", "foo", "qux", replace_all=True)
    assert res2.error is None
    assert res2.occurrences == 2
    content = await be.aread("/multi.txt")
    assert "qux bar qux baz" in content


async def test_remote_backend_async_read_offset_limit():
    be = _backend()
    lines = "\n".join([f"Line {i}" for i in range(1, 11)])
    await be.awrite("/multi.txt", lines)

    content = await be.aread("/multi.txt", offset=2, limit=3)
    assert "Line 3" in content
    assert "Line 4" in content
    assert "Line 5" in content
    assert "Line 1" not in content
    assert "Line 6" not in content
