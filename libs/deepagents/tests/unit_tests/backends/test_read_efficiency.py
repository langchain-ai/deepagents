from pathlib import Path

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.utils import EMPTY_CONTENT_WARNING


def test_read_lines_islice(tmp_path: Path):
    # Create a file with multiple lines
    content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
    file_path = tmp_path / "test.txt"
    file_path.write_text(content)

    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=False)

    # Read first 2 lines
    response = backend.read("test.txt", offset=0, limit=2)
    assert response.file_data["content"] == "Line 1\nLine 2"

    # Read middle lines
    response = backend.read("test.txt", offset=2, limit=2)
    assert response.file_data["content"] == "Line 3\nLine 4"

    # Read last line
    response = backend.read("test.txt", offset=4, limit=10)
    assert response.file_data["content"] == "Line 5"


def test_read_empty_file(tmp_path: Path):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("")

    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=False)
    response = backend.read("empty.txt")
    assert response.file_data["content"] == EMPTY_CONTENT_WARNING


def test_read_offset_out_of_bounds(tmp_path: Path):
    file_path = tmp_path / "test.txt"
    file_path.write_text("line 1")

    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=False)
    response = backend.read("test.txt", offset=10)
    assert "Line offset 10 exceeds file length" in response.error


def test_read_binary_limit(tmp_path: Path):
    # Use a small limit for testing
    backend = FilesystemBackend(root_dir=tmp_path, max_file_size_mb=0.0001, virtual_mode=False)  # ~100 bytes

    # Create a "binary" file
    binary_path = tmp_path / "test.png"
    # Null bytes make it binary (and .png extension triggers binary branch)
    binary_path.write_bytes(b"\x00" * 200)

    response = backend.read("test.png")
    assert response.error is not None
    assert "exceeds binary read limit" in response.error
