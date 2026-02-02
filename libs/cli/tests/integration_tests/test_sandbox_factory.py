"""Test sandbox integrations with upload/download functionality.

This module tests sandbox backends (RunLoop, Daytona, Modal) with support for
optional sandbox reuse to reduce test execution time.

Set REUSE_SANDBOX=1 environment variable to reuse sandboxes across tests within
a class. Otherwise, a fresh sandbox is created for each test method.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator

import pytest
from deepagents.backends.protocol import SandboxBackendProtocol
from deepagents.backends.sandbox import BaseSandbox

from deepagents_cli.integrations.sandbox_factory import create_sandbox


class BaseSandboxIntegrationTest(ABC):
    """Base class for sandbox integration tests.

    Subclasses must implement the `sandbox` fixture to provide a sandbox instance.
    All test methods are defined here and will be inherited by concrete test classes.
    """

    @pytest.fixture(scope="class")
    @abstractmethod
    def sandbox(self) -> Iterator[SandboxBackendProtocol]:
        """Provide a sandbox instance for testing."""
        ...

    def test_sandbox_creation(self, sandbox: SandboxBackendProtocol) -> None:
        """Test basic sandbox creation and command execution."""
        assert sandbox.id is not None
        result = sandbox.execute("echo 'hello'")
        assert result.output.strip() == "hello"

    def test_upload_single_file(self, sandbox: SandboxBackendProtocol) -> None:
        """Test uploading a single file."""
        test_path = "/tmp/test_upload_single.txt"
        test_content = b"Hello, Sandbox!"
        upload_responses = sandbox.upload_files([(test_path, test_content)])

        assert len(upload_responses) == 1
        assert upload_responses[0].path == test_path
        assert upload_responses[0].error is None

        # Verify file exists via command execution
        result = sandbox.execute(f"cat {test_path}")
        assert result.output.strip() == test_content.decode()

    def test_download_single_file(self, sandbox: SandboxBackendProtocol) -> None:
        """Test downloading a single file."""
        test_path = "/tmp/test_download_single.txt"
        test_content = b"Download test content"
        # Create file first
        sandbox.upload_files([(test_path, test_content)])

        # Download and verify
        download_responses = sandbox.download_files([test_path])

        assert len(download_responses) == 1
        assert download_responses[0].path == test_path
        assert download_responses[0].content == test_content
        assert download_responses[0].error is None

    def test_upload_download_roundtrip(self, sandbox: SandboxBackendProtocol) -> None:
        """Test upload followed by download for data integrity."""
        test_path = "/tmp/test_roundtrip.txt"
        test_content = b"Roundtrip test: special chars \n\t\r\x00"

        # Upload
        upload_responses = sandbox.upload_files([(test_path, test_content)])
        assert upload_responses[0].error is None

        # Download
        download_responses = sandbox.download_files([test_path])
        assert download_responses[0].error is None
        assert download_responses[0].content == test_content

    def test_upload_multiple_files(self, sandbox: SandboxBackendProtocol) -> None:
        """Test uploading multiple files in a single batch."""
        files = [
            ("/tmp/test_multi_1.txt", b"Content 1"),
            ("/tmp/test_multi_2.txt", b"Content 2"),
            ("/tmp/test_multi_3.txt", b"Content 3"),
        ]

        upload_responses = sandbox.upload_files(files)

        assert len(upload_responses) == 3
        for i, resp in enumerate(upload_responses):
            assert resp.path == files[i][0]
            assert resp.error is None

    def test_download_multiple_files(self, sandbox: SandboxBackendProtocol) -> None:
        """Test downloading multiple files in a single batch."""
        files = [
            ("/tmp/test_batch_1.txt", b"Batch 1"),
            ("/tmp/test_batch_2.txt", b"Batch 2"),
            ("/tmp/test_batch_3.txt", b"Batch 3"),
        ]

        # Upload files first
        sandbox.upload_files(files)

        # Download all at once
        paths = [f[0] for f in files]
        download_responses = sandbox.download_files(paths)

        assert len(download_responses) == 3
        for i, resp in enumerate(download_responses):
            assert resp.path == files[i][0]
            assert resp.content == files[i][1]
            assert resp.error is None

    @pytest.mark.skip(reason="Error handling not implemented yet.")
    def test_download_nonexistent_file(self, sandbox: SandboxBackendProtocol) -> None:
        """Test that downloading a non-existent file returns an error."""
        nonexistent_path = "/tmp/does_not_exist.txt"

        download_responses = sandbox.download_files([nonexistent_path])

        assert len(download_responses) == 1
        assert download_responses[0].path == nonexistent_path
        assert download_responses[0].content is None
        assert download_responses[0].error is not None

    def test_upload_binary_content(self, sandbox: SandboxBackendProtocol) -> None:
        """Test uploading binary content (not valid UTF-8)."""
        test_path = "/tmp/binary_file.bin"
        # Create binary content with all byte values
        test_content = bytes(range(256))

        upload_responses = sandbox.upload_files([(test_path, test_content)])

        assert len(upload_responses) == 1
        assert upload_responses[0].error is None

        # Verify by downloading
        download_responses = sandbox.download_files([test_path])
        assert download_responses[0].content == test_content

    def test_partial_success_upload(self, sandbox: SandboxBackendProtocol) -> None:
        """Test that batch upload supports partial success."""
        files = [
            ("/tmp/valid_upload.txt", b"Valid content"),
            ("/tmp/another_valid.txt", b"Another valid"),
        ]

        upload_responses = sandbox.upload_files(files)

        # Should get a response for each file
        assert len(upload_responses) == len(files)
        # At least verify we got responses with proper paths
        for i, resp in enumerate(upload_responses):
            assert resp.path == files[i][0]

    @pytest.mark.skip(reason="Error handling not implemented yet.")
    def test_partial_success_download(self, sandbox: SandboxBackendProtocol) -> None:
        """Test that batch download supports partial success."""
        # Create one valid file
        valid_path = "/tmp/valid_file.txt"
        valid_content = b"Valid"
        sandbox.upload_files([(valid_path, valid_content)])

        # Request both valid and invalid files
        paths = [valid_path, "/tmp/does_not_exist.txt"]
        download_responses = sandbox.download_files(paths)

        assert len(download_responses) == 2
        # First should succeed
        assert download_responses[0].path == valid_path
        assert download_responses[0].content == valid_content
        assert download_responses[0].error is None
        # Second should fail
        assert download_responses[1].path == "/tmp/does_not_exist.txt"
        assert download_responses[1].content is None
        assert download_responses[1].error is not None

    @pytest.mark.skip(
        reason="Error handling not yet implemented in sandbox providers - requires implementation"
    )
    def test_download_error_file_not_found(self, sandbox: SandboxBackendProtocol) -> None:
        """Test downloading a non-existent file returns file_not_found error.

        Expected behavior: download_files should return FileDownloadResponse with
        error='file_not_found' when the requested file doesn't exist.
        """
        responses = sandbox.download_files(["/tmp/nonexistent_test_file.txt"])

        assert len(responses) == 1
        assert responses[0].path == "/tmp/nonexistent_test_file.txt"
        assert responses[0].content is None
        assert responses[0].error == "file_not_found"

    @pytest.mark.skip(
        reason="Error handling not yet implemented in sandbox providers - requires implementation"
    )
    def test_download_error_is_directory(self, sandbox: SandboxBackendProtocol) -> None:
        """Test downloading a directory returns is_directory error.

        Expected behavior: download_files should return FileDownloadResponse with
        error='is_directory' when trying to download a directory as a file.
        """
        # Create a directory
        sandbox.execute("mkdir -p /tmp/test_directory")

        responses = sandbox.download_files(["/tmp/test_directory"])

        assert len(responses) == 1
        assert responses[0].path == "/tmp/test_directory"
        assert responses[0].content is None
        assert responses[0].error == "is_directory"

    @pytest.mark.skip(
        reason="Error handling not yet implemented in sandbox providers - requires implementation"
    )
    def test_upload_error_parent_not_found(self, sandbox: SandboxBackendProtocol) -> None:
        """Test uploading to a path with non-existent parent returns parent_not_found error.

        Expected behavior: upload_files should return FileUploadResponse with
        error='parent_not_found' when the parent directory doesn't exist and
        can't be created automatically.

        Note: This test may need adjustment based on whether sandbox providers
        auto-create parent directories or not.
        """
        # Try to upload to a path where the parent is a file, not a directory
        # First create a file
        sandbox.upload_files([("/tmp/parent_is_file.txt", b"I am a file")])

        # Now try to upload as if parent_is_file.txt were a directory
        responses = sandbox.upload_files([("/tmp/parent_is_file.txt/child.txt", b"child")])

        assert len(responses) == 1
        assert responses[0].path == "/tmp/parent_is_file.txt/child.txt"
        # Could be parent_not_found or invalid_path depending on implementation
        assert responses[0].error in ("parent_not_found", "invalid_path")

    @pytest.mark.skip(
        reason="Error handling not yet implemented in sandbox providers - requires implementation"
    )
    def test_upload_error_invalid_path(self, sandbox: SandboxBackendProtocol) -> None:
        """Test uploading with invalid path returns invalid_path error.

        Expected behavior: upload_files should return FileUploadResponse with
        error='invalid_path' for malformed paths (null bytes, invalid chars, etc).
        """
        # Test with null byte (invalid in most filesystems)
        responses = sandbox.upload_files([("/tmp/file\x00name.txt", b"content")])

        assert len(responses) == 1
        assert responses[0].path == "/tmp/file\x00name.txt"
        assert responses[0].error == "invalid_path"

    @pytest.mark.skip(
        reason="Error handling not yet implemented in sandbox providers - requires implementation"
    )
    def test_download_error_invalid_path(self, sandbox: SandboxBackendProtocol) -> None:
        """Test downloading with invalid path returns invalid_path error.

        Expected behavior: download_files should return FileDownloadResponse with
        error='invalid_path' for malformed paths (null bytes, invalid chars, etc).
        """
        # Test with null byte (invalid in most filesystems)
        responses = sandbox.download_files(["/tmp/file\x00name.txt"])

        assert len(responses) == 1
        assert responses[0].path == "/tmp/file\x00name.txt"
        assert responses[0].content is None
        assert responses[0].error == "invalid_path"

    @pytest.mark.skip(
        reason="Error handling not yet implemented in sandbox providers - requires implementation"
    )
    def test_upload_to_existing_directory_path(self, sandbox: SandboxBackendProtocol) -> None:
        """Test uploading to a path that is an existing directory.

        Expected behavior: This should either succeed by overwriting or return
        an appropriate error. The exact behavior depends on the sandbox provider.
        """
        # Create a directory
        sandbox.execute("mkdir -p /tmp/test_dir_upload")

        # Try to upload a file with the same name as the directory
        responses = sandbox.upload_files([("/tmp/test_dir_upload", b"file content")])

        assert len(responses) == 1
        assert responses[0].path == "/tmp/test_dir_upload"
        # Behavior depends on implementation - just verify we get a response


class TestRunLoopIntegration(BaseSandboxIntegrationTest):
    """Test RunLoop backend integration."""

    @pytest.fixture(scope="class")
    def sandbox(self) -> Iterator[BaseSandbox]:
        """Provide a RunLoop sandbox instance."""
        with create_sandbox("runloop") as sandbox:
            yield sandbox


class TestDaytonaIntegration(BaseSandboxIntegrationTest):
    """Test Daytona backend integration."""

    @pytest.fixture(scope="class")
    def sandbox(self) -> Iterator[BaseSandbox]:
        """Provide a Daytona sandbox instance."""
        with create_sandbox("daytona") as sandbox:
            yield sandbox


class TestModalIntegration(BaseSandboxIntegrationTest):
    """Test Modal backend integration."""

    @pytest.fixture(scope="class")
    def sandbox(self) -> Iterator[BaseSandbox]:
        """Provide a Modal sandbox instance."""
        with create_sandbox("modal") as sandbox:
            yield sandbox
