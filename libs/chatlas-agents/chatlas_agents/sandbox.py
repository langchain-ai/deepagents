"""Docker and Apptainer sandbox backends for DeepAgents."""

import logging
import subprocess
import sys
from typing import Any
from enum import Enum

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
    SandboxBackendProtocol,
)
from deepagents.backends.sandbox import BaseSandbox

logger = logging.getLogger(__name__)


class SandboxBackendType(str, Enum):
    """Supported sandbox backend types."""
    
    DOCKER = "docker"
    APPTAINER = "apptainer"


class DockerSandboxBackend(BaseSandbox):
    """Docker-based sandbox backend for executing agent code in isolation.
    
    This backend spawns a Docker container and executes commands within it,
    providing file system isolation and security boundaries for agent operations.
    
    Args:
        container_id: Docker container ID or name to use. If not provided,
            a new container will be created.
        image: Docker image to use for the container (default: python:3.13-slim).
        auto_remove: Whether to automatically remove the container on exit.
        working_dir: Working directory inside the container (default: /workspace).
    """

    def __init__(
        self,
        container_id: str | None = None,
        image: str = "python:3.13-slim",
        auto_remove: bool = True,
        working_dir: str = "/workspace",
    ):
        """Initialize Docker sandbox backend."""
        self._container_id = container_id
        self._image = image
        self._auto_remove = auto_remove
        self._working_dir = working_dir
        self._created_container = False

        if not self._container_id:
            self._create_container()
        else:
            self._verify_container()

    def _create_container(self) -> None:
        """Create a new Docker container for the sandbox."""
        try:
            # Create container with working directory
            cmd = ["docker", "run", "-d"]
            if self._auto_remove:
                cmd.append("--rm")

            # Note: some Docker-compatible CLIs (podman) error if -w points
            # to a non-existent directory. We avoid passing -w here and set
            # the working directory later via `docker exec -w` when running
            # commands inside the container.
            cmd.extend([self._image, "sleep", "infinity"])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            self._container_id = result.stdout.strip()
            self._created_container = True
            logger.info(f"Created Docker container: {self._container_id[:12]}")

            # Create working directory inside the container if it doesn't exist
            # Don't use -w flag here since the directory doesn't exist yet
            subprocess.run(
                [
                    "docker",
                    "exec",
                    self._container_id,
                    "sh",
                    "-c",
                    f"mkdir -p {self._working_dir}",
                ],
                capture_output=True,
                check=True,
            )

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create Docker container: {e.stderr}")
            raise RuntimeError(f"Failed to create Docker container: {e.stderr}") from e

    def _verify_container(self) -> None:
        """Verify that the specified container exists and is running."""
        try:
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", self._container_id],
                capture_output=True,
                text=True,
                check=True,
            )
            is_running = result.stdout.strip() == "true"
            if not is_running:
                raise RuntimeError(f"Container {self._container_id} is not running")
            logger.info(f"Using existing Docker container: {self._container_id[:12]}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Container {self._container_id} not found") from e

    def execute(self, command: str) -> ExecuteResponse:
        """Execute a command in the Docker container.
        
        Args:
            command: Shell command to execute.
            
        Returns:
            ExecuteResponse with output, exit code, and metadata.
        """
        try:
            result = subprocess.run(
                [
                    "docker",
                    "exec",
                    "-w",
                    self._working_dir,
                    self._container_id,
                    "sh",
                    "-c",
                    command,
                ],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            # Combine stdout and stderr
            output = result.stdout
            if result.stderr:
                output += result.stderr

            return ExecuteResponse(
                output=output,
                exit_code=result.returncode,
                truncated=False,
            )

        except subprocess.TimeoutExpired:
            return ExecuteResponse(
                output="Command execution timed out after 300 seconds",
                exit_code=124,  # Standard timeout exit code
                truncated=True,
            )
        except Exception as e:
            logger.error(f"Failed to execute command in container: {e}")
            return ExecuteResponse(
                output=f"Error executing command: {str(e)}",
                exit_code=1,
                truncated=False,
            )

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files to the Docker container.
        
        Args:
            files: List of (path, content) tuples to upload.
            
        Returns:
            List of FileUploadResponse objects indicating success/failure.
        """
        responses: list[FileUploadResponse] = []

        for path, content in files:
            try:
                # Write content to a temporary file
                import tempfile

                with tempfile.NamedTemporaryFile(delete=False, mode="wb") as tmp_file:
                    tmp_file.write(content)
                    tmp_path = tmp_file.name

                # Copy file into container
                dest_path = f"{self._working_dir}/{path}".replace("//", "/")
                
                # Create parent directory
                parent_dir = dest_path.rsplit("/", 1)[0]
                self.execute(f"mkdir -p {parent_dir}")

                subprocess.run(
                    ["docker", "cp", tmp_path, f"{self._container_id}:{dest_path}"],
                    check=True,
                    capture_output=True,
                )

                # Clean up temp file
                import os

                os.unlink(tmp_path)

                responses.append(FileUploadResponse(path=path, error=None))
                logger.debug(f"Uploaded file to container: {path}")

            except Exception as e:
                logger.error(f"Failed to upload file {path}: {e}")
                responses.append(FileUploadResponse(path=path, error=str(e)))

        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files from the Docker container.
        
        Args:
            paths: List of file paths to download.
            
        Returns:
            List of FileDownloadResponse objects with file contents or errors.
        """
        responses: list[FileDownloadResponse] = []

        for path in paths:
            try:
                # Copy file from container to temp location
                import tempfile

                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_path = tmp_file.name

                source_path = f"{self._working_dir}/{path}".replace("//", "/")
                subprocess.run(
                    ["docker", "cp", f"{self._container_id}:{source_path}", tmp_path],
                    check=True,
                    capture_output=True,
                )

                # Read content
                with open(tmp_path, "rb") as f:
                    content = f.read()

                # Clean up temp file
                import os

                os.unlink(tmp_path)

                responses.append(FileDownloadResponse(path=path, content=content, error=None))
                logger.debug(f"Downloaded file from container: {path}")

            except Exception as e:
                logger.error(f"Failed to download file {path}: {e}")
                responses.append(FileDownloadResponse(path=path, content=None, error=str(e)))

        return responses

    @property
    def id(self) -> str:
        """Get the Docker container ID."""
        return self._container_id or "no-container"

    def cleanup(self) -> None:
        """Clean up the Docker container."""
        if self._created_container and self._container_id:
            try:
                subprocess.run(
                    ["docker", "stop", self._container_id],
                    capture_output=True,
                    check=False,
                )
                # Avoid logging during interpreter shutdown where logging handlers
                # (like rich) may be torn down and cause ImportError. Check for
                # sys.meta_path being None which indicates shutdown.
                try:
                    if getattr(sys, "meta_path", None) is not None:
                        logger.info(f"Stopped Docker container: {self._container_id[:12]}")
                except Exception:
                    # Swallow any errors during logging to avoid crashes during shutdown
                    pass
            except Exception as e:
                try:
                    if getattr(sys, "meta_path", None) is not None:
                        logger.warning(f"Failed to stop container: {e}")
                except Exception:
                    pass

    def __enter__(self) -> "DockerSandboxBackend":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit with cleanup."""
        self.cleanup()

    def __del__(self) -> None:
        """Destructor with cleanup."""
        if self._created_container:
            try:
                # Avoid logging in destructor during interpreter shutdown
                self.cleanup()
            except Exception:
                # Be silent — don't raise during garbage collection/ interpreter teardown
                pass


class ApptainerSandboxBackend(BaseSandbox):
    """Apptainer-based sandbox backend for executing agent code in isolation.
    
    This backend spawns an Apptainer instance and executes commands within it,
    providing file system isolation and security boundaries for agent operations.
    Apptainer (formerly Singularity) is designed for HPC environments and doesn't
    require root privileges or a daemon.
    
    Args:
        instance_name: Apptainer instance name to use. If not provided,
            a unique instance name will be generated.
        image: Container image to use (default: docker://python:3.13-slim).
            Can be docker://, oras://, or a local .sif file.
        auto_remove: Whether to automatically stop the instance on exit.
        working_dir: Working directory inside the container (default: /workspace).
    """

    def __init__(
        self,
        instance_name: str | None = None,
        image: str = "docker://python:3.13-slim",
        auto_remove: bool = True,
        working_dir: str = "/workspace",
    ):
        """Initialize Apptainer sandbox backend."""
        self._instance_name = instance_name or f"chatlas-agent-{id(self)}"
        self._image = image
        self._auto_remove = auto_remove
        self._working_dir = working_dir
        self._created_instance = False

        # Ensure image has a transport prefix
        if not any(self._image.startswith(prefix) for prefix in ["docker://", "oras://", "library://", "/"]):
            self._image = f"docker://{self._image}"

        # Always create a new instance if instance_name was not provided
        if not instance_name:
            self._create_instance()
        else:
            self._verify_instance()

    def _create_instance(self) -> None:
        """Create a new Apptainer instance for the sandbox."""
        try:
            # Start an instance from the image
            # apptainer instance start [options] <container> <instance name>
            cmd = ["apptainer", "instance", "start"]
            
            # Use writable-tmpfs to make the container filesystem writable
            # This creates a temporary writable overlay that is discarded when the instance stops
            cmd.append("--writable-tmpfs")
            
            # Note: Apptainer automatically binds $HOME, /tmp, and other common dirs
            
            cmd.extend([self._image, self._instance_name])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            self._created_instance = True
            logger.info(f"Created Apptainer instance: {self._instance_name}")

            # Create working directory inside the instance if it doesn't exist
            # Don't use the execute method here since it tries to cd to the working directory
            subprocess.run(
                [
                    "apptainer",
                    "exec",
                    f"instance://{self._instance_name}",
                    "sh",
                    "-c",
                    f"mkdir -p {self._working_dir}",
                ],
                capture_output=True,
                check=True,
            )

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create Apptainer instance: {e.stderr}")
            raise RuntimeError(f"Failed to create Apptainer instance: {e.stderr}") from e

    def _verify_instance(self) -> None:
        """Verify that the specified instance exists and is running."""
        try:
            result = subprocess.run(
                ["apptainer", "instance", "list"],
                capture_output=True,
                text=True,
                check=True,
            )
            # Check if instance name appears in the list
            if self._instance_name not in result.stdout:
                raise RuntimeError(f"Instance {self._instance_name} is not running")
            logger.info(f"Using existing Apptainer instance: {self._instance_name}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to verify instance {self._instance_name}") from e

    def execute(self, command: str) -> ExecuteResponse:
        """Execute a command in the Apptainer instance.
        
        Args:
            command: Shell command to execute.
            
        Returns:
            ExecuteResponse with output, exit code, and metadata.
        """
        try:
            result = subprocess.run(
                [
                    "apptainer",
                    "exec",
                    f"instance://{self._instance_name}",
                    "sh",
                    "-c",
                    f"cd {self._working_dir} && {command}",
                ],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            # Combine stdout and stderr
            output = result.stdout
            if result.stderr:
                output += result.stderr

            return ExecuteResponse(
                output=output,
                exit_code=result.returncode,
                truncated=False,
            )

        except subprocess.TimeoutExpired:
            return ExecuteResponse(
                output="Command execution timed out after 300 seconds",
                exit_code=124,  # Standard timeout exit code
                truncated=True,
            )
        except Exception as e:
            logger.error(f"Failed to execute command in instance: {e}")
            return ExecuteResponse(
                output=f"Error executing command: {str(e)}",
                exit_code=1,
                truncated=False,
            )

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files to the Apptainer instance.
        
        Args:
            files: List of (path, content) tuples to upload.
            
        Returns:
            List of FileUploadResponse objects indicating success/failure.
        """
        responses: list[FileUploadResponse] = []

        for path, content in files:
            try:
                # For Apptainer, we write to the bound working directory
                # Since working_dir is bound, we can write directly to the host filesystem
                import tempfile
                import os
                import shutil

                # Create full destination path
                dest_path = f"{self._working_dir}/{path}".replace("//", "/")
                
                # Create parent directory
                parent_dir = dest_path.rsplit("/", 1)[0]
                self.execute(f"mkdir -p {parent_dir}")

                # Write to temp file then copy via exec
                with tempfile.NamedTemporaryFile(delete=False, mode="wb") as tmp_file:
                    tmp_file.write(content)
                    tmp_path = tmp_file.name

                # Use cat to write content into the instance
                # Read the temp file and pipe it into the instance
                with open(tmp_path, "rb") as f:
                    file_content = f.read()
                
                # Use exec with shell redirection to write file
                result = subprocess.run(
                    [
                        "apptainer",
                        "exec",
                        f"instance://{self._instance_name}",
                        "sh",
                        "-c",
                        f"cat > {dest_path}",
                    ],
                    input=file_content,
                    capture_output=True,
                    text=False,  # Binary mode for input
                )

                # Clean up temp file
                os.unlink(tmp_path)

                if result.returncode == 0:
                    responses.append(FileUploadResponse(path=path, error=None))
                    logger.debug(f"Uploaded file to instance: {path}")
                else:
                    error_msg = result.stderr.decode() if isinstance(result.stderr, bytes) else result.stderr
                    responses.append(FileUploadResponse(path=path, error=error_msg))

            except Exception as e:
                logger.error(f"Failed to upload file {path}: {e}")
                responses.append(FileUploadResponse(path=path, error=str(e)))

        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files from the Apptainer instance.
        
        Args:
            paths: List of file paths to download.
            
        Returns:
            List of FileDownloadResponse objects with file contents or errors.
        """
        responses: list[FileDownloadResponse] = []

        for path in paths:
            try:
                source_path = f"{self._working_dir}/{path}".replace("//", "/")
                
                # Use exec with cat to read the file
                result = subprocess.run(
                    [
                        "apptainer",
                        "exec",
                        f"instance://{self._instance_name}",
                        "cat",
                        source_path,
                    ],
                    capture_output=True,
                )

                if result.returncode == 0:
                    content = result.stdout
                    responses.append(FileDownloadResponse(path=path, content=content, error=None))
                    logger.debug(f"Downloaded file from instance: {path}")
                else:
                    error_msg = result.stderr if result.stderr else "File not found"
                    responses.append(FileDownloadResponse(path=path, content=None, error=error_msg))

            except Exception as e:
                logger.error(f"Failed to download file {path}: {e}")
                responses.append(FileDownloadResponse(path=path, content=None, error=str(e)))

        return responses

    @property
    def id(self) -> str:
        """Get the Apptainer instance name."""
        return self._instance_name or "no-instance"

    def cleanup(self) -> None:
        """Clean up the Apptainer instance."""
        if self._created_instance and self._instance_name:
            try:
                subprocess.run(
                    ["apptainer", "instance", "stop", self._instance_name],
                    capture_output=True,
                    check=False,
                )
                # Avoid logging during interpreter shutdown where logging handlers
                # (like rich) may be torn down and cause ImportError. Check for
                # sys.meta_path being None which indicates shutdown.
                try:
                    if getattr(sys, "meta_path", None) is not None:
                        logger.info(f"Stopped Apptainer instance: {self._instance_name}")
                except Exception:
                    # Swallow any errors during logging to avoid crashes during shutdown
                    pass
            except Exception as e:
                try:
                    if getattr(sys, "meta_path", None) is not None:
                        logger.warning(f"Failed to stop instance: {e}")
                except Exception:
                    pass

    def __enter__(self) -> "ApptainerSandboxBackend":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit with cleanup."""
        self.cleanup()

    def __del__(self) -> None:
        """Destructor with cleanup."""
        if self._created_instance:
            try:
                # Avoid logging in destructor during interpreter shutdown
                self.cleanup()
            except Exception:
                # Be silent — don't raise during garbage collection/ interpreter teardown
                pass
