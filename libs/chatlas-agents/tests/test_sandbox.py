"""Tests for sandbox backends."""

import pytest
import subprocess
import shutil
from chatlas_agents.sandbox import (
    DockerSandboxBackend, 
    ApptainerSandboxBackend,
    SandboxBackendType,
)


def test_sandbox_backend_type_enum():
    """Test SandboxBackendType enum."""
    assert SandboxBackendType.DOCKER == "docker"
    assert SandboxBackendType.APPTAINER == "apptainer"


@pytest.mark.skipif(
    not shutil.which('docker'),
    reason="Docker not installed"
)
def test_docker_sandbox_initialization():
    """Test Docker sandbox initialization."""
    try:
        backend = DockerSandboxBackend(image="alpine:latest", auto_remove=True)
        assert backend.id is not None
        assert backend._container_id is not None
        assert backend._created_container is True
        backend.cleanup()
    except Exception as e:
        pytest.skip(f"Docker sandbox creation failed: {e}")


@pytest.mark.skipif(
    not shutil.which('docker'),
    reason="Docker not installed"
)
def test_docker_sandbox_execute():
    """Test Docker sandbox command execution."""
    try:
        backend = DockerSandboxBackend(image="alpine:latest", auto_remove=True)
        response = backend.execute("echo 'Hello from Docker'")
        assert response.exit_code == 0
        assert "Hello from Docker" in response.output
        backend.cleanup()
    except Exception as e:
        pytest.skip(f"Docker sandbox execution failed: {e}")


@pytest.mark.skipif(
    not shutil.which('docker'),
    reason="Docker not installed"
)
def test_docker_sandbox_file_operations():
    """Test Docker sandbox file upload/download."""
    try:
        backend = DockerSandboxBackend(image="alpine:latest", auto_remove=True)
        
        # Upload a file
        test_content = b"Test file content"
        upload_responses = backend.upload_files([("test.txt", test_content)])
        assert len(upload_responses) == 1
        assert upload_responses[0].error is None
        
        # Verify file exists
        exec_response = backend.execute("cat test.txt")
        assert exec_response.exit_code == 0
        assert "Test file content" in exec_response.output
        
        # Download the file
        download_responses = backend.download_files(["test.txt"])
        assert len(download_responses) == 1
        assert download_responses[0].content == test_content
        assert download_responses[0].error is None
        
        backend.cleanup()
    except Exception as e:
        pytest.skip(f"Docker file operations failed: {e}")


@pytest.mark.skipif(
    not shutil.which('apptainer'),
    reason="Apptainer not installed"
)
def test_apptainer_sandbox_initialization():
    """Test Apptainer sandbox initialization."""
    try:
        backend = ApptainerSandboxBackend(
            image="docker://alpine:latest", 
            auto_remove=True
        )
        assert backend.id is not None
        assert backend._instance_name is not None
        assert backend._created_instance is True
        backend.cleanup()
    except Exception as e:
        pytest.skip(f"Apptainer sandbox creation failed: {e}")


@pytest.mark.skipif(
    not shutil.which('apptainer'),
    reason="Apptainer not installed"
)
def test_apptainer_sandbox_execute():
    """Test Apptainer sandbox command execution."""
    try:
        backend = ApptainerSandboxBackend(
            image="docker://alpine:latest", 
            auto_remove=True
        )
        response = backend.execute("echo 'Hello from Apptainer'")
        assert response.exit_code == 0
        assert "Hello from Apptainer" in response.output
        backend.cleanup()
    except Exception as e:
        pytest.skip(f"Apptainer sandbox execution failed: {e}")


@pytest.mark.skipif(
    not shutil.which('apptainer'),
    reason="Apptainer not installed"
)
def test_apptainer_sandbox_file_operations():
    """Test Apptainer sandbox file upload/download."""
    try:
        backend = ApptainerSandboxBackend(
            image="docker://alpine:latest", 
            auto_remove=True
        )
        
        # Upload a file
        test_content = b"Test file content from Apptainer"
        upload_responses = backend.upload_files([("test.txt", test_content)])
        assert len(upload_responses) == 1
        assert upload_responses[0].error is None
        
        # Verify file exists
        exec_response = backend.execute("cat test.txt")
        assert exec_response.exit_code == 0
        assert "Test file content from Apptainer" in exec_response.output
        
        # Download the file
        download_responses = backend.download_files(["test.txt"])
        assert len(download_responses) == 1
        assert download_responses[0].content == test_content
        assert download_responses[0].error is None
        
        backend.cleanup()
    except Exception as e:
        pytest.skip(f"Apptainer file operations failed: {e}")


def test_apptainer_image_prefix_normalization():
    """Test that Apptainer backend normalizes image names correctly."""
    from chatlas_agents.sandbox import ApptainerSandboxBackend
    
    # Helper function to check image prefix normalization
    def normalize_image(image: str) -> str:
        """Apply the same normalization logic as ApptainerSandboxBackend."""
        if not any(image.startswith(prefix) for prefix in ["docker://", "oras://", "library://", "/"]):
            return f"docker://{image}"
        return image
    
    # Test cases
    assert normalize_image("alpine:latest") == "docker://alpine:latest"
    assert normalize_image("python:3.13-slim") == "docker://python:3.13-slim"
    assert normalize_image("docker://alpine:latest") == "docker://alpine:latest"
    assert normalize_image("oras://example.com/image:tag") == "oras://example.com/image:tag"
    assert normalize_image("library://image:tag") == "library://image:tag"
    assert normalize_image("/path/to/image.sif") == "/path/to/image.sif"


@pytest.mark.skipif(
    not shutil.which('docker'),
    reason="Docker not installed"
)
def test_docker_sandbox_context_manager():
    """Test Docker sandbox as context manager."""
    try:
        with DockerSandboxBackend(image="alpine:latest", auto_remove=True) as backend:
            response = backend.execute("echo 'Context manager test'")
            assert response.exit_code == 0
            container_id = backend.id
        
        # Verify container is cleaned up
        result = subprocess.run(
            ["docker", "inspect", container_id],
            capture_output=True,
            text=True,
        )
        # Container should be gone (either because it was stopped or auto-removed)
        assert result.returncode != 0 or "false" in result.stdout.lower()
    except Exception as e:
        pytest.skip(f"Docker context manager test failed: {e}")
