"""Harbor environment backed by LangSmith sandboxes."""

from __future__ import annotations

import logging
import shlex
from pathlib import Path
from typing import TYPE_CHECKING

from dockerfile_parse import DockerfileParser
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.trial.paths import EnvironmentPaths, TrialPaths

if TYPE_CHECKING:
    from harbor.models.environment_type import EnvironmentType
    from harbor.models.task.config import EnvironmentConfig
    from langsmith.sandbox import Sandbox, SandboxClient

logger = logging.getLogger(__name__)

_DEFAULT_EXEC_TIMEOUT_SEC = 30 * 60
_MB_PER_GB = 1024


class LangSmithEnvironment(BaseEnvironment):
    """Harbor environment backed by LangSmith sandboxes.

    Uses `--environment-import-path` because harbor's `EnvironmentType` enum
    does not include `langsmith` yet. Example:

        harbor run --environment-import-path \
            deepagents_harbor.langsmith_environment:LangSmithEnvironment ...

    The environment reads the task's Dockerfile to extract the base image,
    creates a LangSmith template + sandbox with the appropriate resources, and
    delegates all exec/file operations to the LangSmith sandbox SDK.
    """

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
    ) -> None:
        """Initialize a LangSmith harbor environment.

        Args:
            environment_dir: Path to the task's environment directory.
            environment_name: Task name used for the sandbox.
            session_id: Unique trial session identifier.
            trial_paths: Local paths for trial artifacts.
            task_env_config: Resource and network configuration.
        """
        self._sandbox: Sandbox | None = None
        self._client: SandboxClient | None = None
        self._template_name: str | None = None
        super().__init__(
            environment_dir,
            environment_name,
            session_id,
            trial_paths,
            task_env_config,
        )

    # -- Required abstract properties / methods --------------------------------

    @staticmethod
    def type() -> EnvironmentType:
        """Not applicable — this environment is loaded via import path."""
        msg = (
            "LangSmithEnvironment is used via --environment-import-path, "
            "not --env. It has no EnvironmentType enum member."
        )
        raise NotImplementedError(msg)

    @property
    def is_mounted(self) -> bool:
        """Whether the environment mounts host logging directories."""
        return False

    @property
    def supports_gpus(self) -> bool:
        """Whether LangSmith sandboxes support GPU allocation."""
        return False

    @property
    def can_disable_internet(self) -> bool:
        """Whether LangSmith sandboxes support network isolation."""
        return False

    # -- Validation overrides --------------------------------------------------
    # Override base-class validators so they never call self.type(), which
    # would raise NotImplementedError.

    def _validate_definition(self) -> None:
        """Validate that the task provides a usable image source.

        Accepts either a prebuilt `docker_image` in the task config or a
        Dockerfile in the environment directory.
        """
        if self.task_env_config.docker_image:
            return
        dockerfile_path = self.environment_dir / "Dockerfile"
        if not dockerfile_path.exists():
            msg = (
                f"LangSmith environment requires either a Dockerfile at "
                f"'{dockerfile_path}' or a 'docker_image' in the task config."
            )
            raise FileNotFoundError(msg)

    def _validate_gpu_support(self) -> None:
        if self.task_env_config.gpus > 0:
            msg = "LangSmith sandbox does not support GPU allocation."
            raise RuntimeError(msg)

    def _validate_internet_config(self) -> None:
        if not self.task_env_config.allow_internet:
            msg = "LangSmith sandbox does not support disabling internet access."
            raise ValueError(msg)

    # -- Image resolution ------------------------------------------------------

    def _resolve_image(self) -> str:
        """Return the container image for the LangSmith template.

        Prefers `docker_image` from the task config; falls back to parsing
        the `FROM` instruction in the environment Dockerfile.
        """
        if self.task_env_config.docker_image:
            return self.task_env_config.docker_image

        dockerfile_path = self.environment_dir / "Dockerfile"
        parser = DockerfileParser(path=str(dockerfile_path))
        base = parser.baseimage
        if not base:
            msg = f"Could not extract FROM image from {dockerfile_path}"
            raise ValueError(msg)
        return base

    # -- Lifecycle -------------------------------------------------------------

    async def start(self, force_build: bool) -> None:  # noqa: ARG002
        """Provision a LangSmith sandbox from the task's Dockerfile image.

        Args:
            force_build: Ignored — LangSmith templates are image-only.
        """
        from langsmith.sandbox import SandboxClient

        image = self._resolve_image()

        self._client = SandboxClient()
        self._template_name = f"harbor-{self.session_id}"

        # Convert task resource specs to LangSmith format
        cpu = f"{self.task_env_config.cpus * 1000}m"
        memory_mb = self.task_env_config.memory_mb
        memory = f"{memory_mb}Mi" if memory_mb < _MB_PER_GB else f"{memory_mb // _MB_PER_GB}Gi"
        storage_mb = self.task_env_config.storage_mb
        storage = f"{storage_mb}Mi" if storage_mb < _MB_PER_GB else f"{storage_mb // _MB_PER_GB}Gi"

        self._client.create_template(
            name=self._template_name,
            image=image,
            cpu=cpu,
            memory=memory,
            storage=storage,
        )
        logger.info(
            "Created LangSmith template '%s' (image=%s, cpu=%s, mem=%s, disk=%s)",
            self._template_name,
            image,
            cpu,
            memory,
            storage,
        )

        self._sandbox = self._client.sandbox(
            template_name=self._template_name,
            name=self.session_id,
        )
        logger.info("Created LangSmith sandbox '%s'", self._sandbox.name)

        # Create required harbor directory structure
        self._sandbox.run(
            f"mkdir -p {EnvironmentPaths.agent_dir} {EnvironmentPaths.verifier_dir}",
            timeout=30,
        )

    async def stop(self, delete: bool) -> None:
        """Tear down the LangSmith sandbox and template.

        Args:
            delete: If True, delete sandbox and template (default behavior
                for ephemeral CI runs).
        """
        if self._sandbox and self._client and delete:
            try:
                self._client.delete_sandbox(self._sandbox.name)
                logger.info("Deleted LangSmith sandbox '%s'", self._sandbox.name)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Failed to delete sandbox '%s'",
                    self._sandbox.name,
                    exc_info=True,
                )
        if self._template_name and self._client and delete:
            try:
                self._client.delete_template(self._template_name)
                logger.info("Deleted LangSmith template '%s'", self._template_name)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Failed to delete template '%s'",
                    self._template_name,
                    exc_info=True,
                )
        if self._client:
            self._client.close()
        self._sandbox = None
        self._client = None
        self._template_name = None

    # -- Command execution -----------------------------------------------------

    def _require_sandbox(self) -> Sandbox:
        if self._sandbox is None:
            msg = "Sandbox not started. Call start() first."
            raise RuntimeError(msg)
        return self._sandbox

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
    ) -> ExecResult:
        """Execute a command inside the LangSmith sandbox.

        Args:
            command: Shell command string to execute.
            cwd: Working directory for command execution.
            env: Environment variables to set.
            timeout_sec: Timeout in seconds.
        """
        sandbox = self._require_sandbox()
        effective_timeout = timeout_sec or _DEFAULT_EXEC_TIMEOUT_SEC

        result = sandbox.run(
            command,
            timeout=effective_timeout,
            cwd=cwd,
            env=env,
        )
        return ExecResult(
            stdout=result.stdout or "",
            stderr=result.stderr or "",
            return_code=result.exit_code,
        )

    # -- File operations -------------------------------------------------------

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        """Upload a local file to the sandbox.

        Args:
            source_path: Local file path.
            target_path: Destination path inside the sandbox.
        """
        sandbox = self._require_sandbox()
        content = Path(source_path).read_bytes()

        # Ensure parent directory exists
        parent = str(Path(target_path).parent)
        if parent != "/":
            sandbox.run(f"mkdir -p {shlex.quote(parent)}", timeout=30)

        sandbox.write(target_path, content)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        """Upload a local directory to the sandbox recursively.

        Args:
            source_dir: Local directory path.
            target_dir: Destination directory inside the sandbox.
        """
        source = Path(source_dir)
        for file_path in source.rglob("*"):
            if file_path.is_file():
                relative = file_path.relative_to(source)
                target = str(Path(target_dir) / relative)
                await self.upload_file(file_path, target)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        """Download a file from the sandbox to the local machine.

        Args:
            source_path: File path inside the sandbox.
            target_path: Local destination path.
        """
        sandbox = self._require_sandbox()
        data = sandbox.read(source_path)
        local = Path(target_path)
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        """Download a directory from the sandbox to the local machine.

        Args:
            source_dir: Directory path inside the sandbox.
            target_dir: Local destination directory.
        """
        local_dir = Path(target_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        # List files via shell find
        result = await self.exec(
            f"find {shlex.quote(source_dir)} -type f",
            timeout_sec=60,
        )
        if result.return_code != 0 or not result.stdout:
            logger.warning("Failed to list files in '%s': %s", source_dir, result.stderr)
            return

        for file_path in result.stdout.strip().split("\n"):
            if not file_path:
                continue
            relative = Path(file_path).relative_to(source_dir)
            local_file = local_dir / relative
            try:
                await self.download_file(file_path, local_file)
            except Exception:  # noqa: BLE001
                logger.debug("Failed to download %s", file_path, exc_info=True)
