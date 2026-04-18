"""Harbor environment backed by LangSmith sandboxes."""

from __future__ import annotations

import asyncio
import hashlib
import re
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dockerfile_parse import DockerfileParser
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.trial.paths import EnvironmentPaths, TrialPaths
from harbor.utils.logger import logger
from langsmith.sandbox import AsyncSandboxClient

from deepagents_harbor.langsmith import resolve_langsmith_api_key

if TYPE_CHECKING:
    from harbor.models.environment_type import EnvironmentType
    from harbor.models.task.config import EnvironmentConfig
    from langsmith.sandbox import AsyncSandbox


_DEFAULT_EXEC_TIMEOUT_SEC = 30 * 60
_MB_PER_GB = 1024
_MAX_NAME_LEN = 63
_SESSION_HASH_LEN = 8


class LangSmithEnvironment(BaseEnvironment):
    r"""Harbor environment backed by LangSmith sandboxes.

    Uses `--environment-import-path` because harbor's `EnvironmentType` enum
    does not include `langsmith` yet. Example:

        harbor run --environment-import-path \
            deepagents_harbor.langsmith_environment:LangSmithEnvironment ...

    The environment reads the task's Dockerfile to extract the base image,
    creates a LangSmith template + sandbox with the appropriate resources, and
    delegates all exec/file operations to the LangSmith sandbox SDK.

    Each trial gets its own template (named with the session ID) so that
    concurrent trials never race on shared resources during creation or
    teardown.
    """

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        **kwargs: Any,
    ) -> None:
        """Initialize a LangSmith harbor environment.

        Args:
            environment_dir: Path to the task's environment directory.
            environment_name: Logical name for this environment.
            session_id: Unique trial session identifier.
            trial_paths: Local paths for trial artifacts.
            task_env_config: Resource and network configuration.
            **kwargs: Forwarded to `BaseEnvironment` (e.g. `logger`,
                `override_cpus`, `override_memory_mb`).
        """
        self._session_id = session_id
        self._sandbox: AsyncSandbox | None = None
        self._client: AsyncSandboxClient | None = None
        self._template_name: str | None = None
        self._default_cwd: str | None = None
        super().__init__(
            environment_dir,
            environment_name,
            session_id,
            trial_paths,
            task_env_config,
            **kwargs,
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

        Raises:
            FileNotFoundError: If neither a Dockerfile nor `docker_image`
                is available.
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
        """Override base to avoid calling `self.type()`."""
        if self.task_env_config.gpus > 0:
            msg = "LangSmith sandbox does not support GPU allocation."
            raise RuntimeError(msg)

    def _validate_internet_config(self) -> None:
        """Override base to avoid calling `self.type()`."""
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

    # -- Name helpers ----------------------------------------------------------

    @staticmethod
    def _sanitize_name(raw: str) -> str:
        """Sanitize a string for use as a LangSmith resource name.

        LangSmith requires names that start with a lowercase letter, contain
        only lowercase letters, numbers, and hyphens, and do not end with a
        hyphen.  Max 63 characters.
        """
        name = raw.lower()
        name = re.sub(r"[^a-z0-9-]", "-", name)
        name = re.sub(r"-{2,}", "-", name)
        name = name.strip("-")
        if not name or not name[0].isalpha():
            name = f"h-{name}"
        return name[:_MAX_NAME_LEN].rstrip("-")

    # -- Template naming -------------------------------------------------------

    @staticmethod
    def _build_template_name(image: str, session_id: str) -> str:
        """Build a unique LangSmith template name within the 63-char limit.

        Previous implementation concatenated ``harbor-{image}-{session}`` and
        truncated to 63 characters.  When the image name was long the unique
        session suffix was silently chopped off, causing name collisions across
        concurrent matrix jobs.

        This version reserves 8 hex characters (from a SHA-256 of the full
        session ID) as a suffix so uniqueness is guaranteed regardless of
        image-name length.

        Args:
            image: Container image reference (e.g. ``alexgshaw/foo:tag``).
            session_id: Unique trial session identifier.

        Returns:
            A sanitized name of at most 63 characters, guaranteed unique per
            ``session_id``.
        """
        sanitized_image = LangSmithEnvironment._sanitize_name(image)
        session_suffix = hashlib.sha256(session_id.encode()).hexdigest()[:_SESSION_HASH_LEN]
        # "harbor-" (7) + "-" (1) + suffix (8) = 16 reserved chars
        max_image_len = _MAX_NAME_LEN - len("harbor-") - 1 - _SESSION_HASH_LEN
        truncated_image = sanitized_image[:max_image_len].rstrip("-")
        return f"harbor-{truncated_image}-{session_suffix}"

    # -- Lifecycle -------------------------------------------------------------

    async def start(self, force_build: bool) -> None:
        """Provision a LangSmith sandbox from the task's Dockerfile image.

        Args:
            force_build: Accepted for interface compatibility but unused.
                Each trial creates its own template, so there is nothing
                to force-rebuild.
        """
        image = self._resolve_image()

        resolved = resolve_langsmith_api_key()
        if not resolved:
            msg = (
                "No LangSmith API key found. Set one of: "
                "LANGSMITH_SANDBOX_API_KEY, LANGSMITH_API_KEY, LANGCHAIN_API_KEY."
            )
            raise ValueError(msg)

        api_key, key_source = resolved
        if key_source == "LANGSMITH_SANDBOX_API_KEY":
            logger.info("Using LangSmith API key from %s", key_source)

        client = AsyncSandboxClient(api_key=api_key)
        self._client = client

        template_name = self._build_template_name(image, self._session_id)

        cpu = f"{self.task_env_config.cpus * 1000}m"
        memory_mb = self.task_env_config.memory_mb
        memory = f"{memory_mb}Mi" if memory_mb < _MB_PER_GB else f"{memory_mb // _MB_PER_GB}Gi"
        storage_gi = max(1, (self.task_env_config.storage_mb + _MB_PER_GB - 1) // _MB_PER_GB)
        storage = f"{storage_gi}Gi"

        await client.create_template(
            name=template_name,
            image=image,
            cpu=cpu,
            memory=memory,
            storage=storage,
        )
        logger.info(
            "Created LangSmith template '%s' (image=%s, cpu=%s, mem=%s, disk=%s)",
            template_name,
            image,
            cpu,
            memory,
            storage,
        )
        self._template_name = template_name

        sandbox = await client.create_sandbox(
            template_name=template_name,
            timeout=120,
        )
        self._sandbox = sandbox
        logger.info("Created LangSmith sandbox '%s'", sandbox.name)

        await sandbox.run(
            f"mkdir -p {EnvironmentPaths.agent_dir} {EnvironmentPaths.verifier_dir}"
            f" {EnvironmentPaths.artifacts_dir}",
            timeout=30,
        )

        self._default_cwd = await self._detect_workdir(sandbox)
        logger.info(
            "LangSmith sandbox '%s' default cwd resolved to '%s'",
            sandbox.name,
            self._default_cwd,
        )

    @staticmethod
    async def _detect_workdir(sandbox: AsyncSandbox) -> str:
        """Resolve the container's Dockerfile ``WORKDIR`` at runtime.

        LangSmith's `sandbox.run(cwd=None)` spawns each command from the
        dataplane daemon's cwd, not the image's `WORKDIR`. Terminal-bench
        verifier scripts rely on `WORKDIR` (e.g. `/app`) — many include
        `if [ "$PWD" = "/" ]; then exit 1; fi` as a guard and abort without
        writing `/logs/verifier/reward.txt` when this assumption is violated.

        PID 1 (the container entrypoint) inherits the image's `WORKDIR` as
        its cwd, so `readlink /proc/1/cwd` yields the correct directory
        without requiring access to the image metadata. Falls back to `/app`
        (terminal-bench convention) when the probe is inconclusive.

        Args:
            sandbox: The active LangSmith sandbox.

        Returns:
            Absolute path to use as the default working directory for
            subsequent command execution.
        """
        try:
            result = await sandbox.run("readlink /proc/1/cwd", timeout=15)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to probe /proc/1/cwd; falling back to '/app'", exc_info=True)
            return "/app"

        candidate = (result.stdout or "").strip()
        if result.exit_code == 0 and candidate and candidate != "/":
            return candidate

        probe = await sandbox.run(
            "[ -d /app ] && echo /app || echo /",
            timeout=15,
        )
        fallback = (probe.stdout or "").strip() or "/"
        return fallback if fallback != "/" else "/app"

    async def stop(self, delete: bool) -> None:
        """Tear down the LangSmith sandbox and its dedicated template.

        Deletes the sandbox first so the template has no remaining
        dependents and can be safely removed.

        Args:
            delete: If True, delete the sandbox and template before
                closing the client.
        """
        if self._sandbox and self._client and delete:
            try:
                await self._client.delete_sandbox(self._sandbox.name)
                logger.info("Deleted LangSmith sandbox '%s'", self._sandbox.name)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Failed to delete sandbox '%s' -- resource may be leaked",
                    self._sandbox.name,
                    exc_info=True,
                )
        if self._template_name and self._client and delete:
            try:
                await self._client.delete_template(self._template_name)
                logger.info("Deleted LangSmith template '%s'", self._template_name)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Failed to delete template '%s' -- resource may be leaked",
                    self._template_name,
                    exc_info=True,
                )
        if self._client:
            await self._client.aclose()
        self._sandbox = None
        self._client = None
        self._template_name = None
        self._default_cwd = None

    # -- Command execution -----------------------------------------------------

    def _require_sandbox(self) -> AsyncSandbox:
        """Return the active sandbox or raise if not started.

        Raises:
            RuntimeError: If `start()` has not been called.
        """
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

        When ``cwd`` is not provided, defaults to the container's
        ``WORKDIR`` (resolved at ``start()``) rather than LangSmith's
        dataplane default of ``/``. This preserves the semantics that
        terminal-bench verifier scripts assume when they probe ``$PWD``.

        Args:
            command: Shell command string to execute.
            cwd: Working directory for command execution. Overrides the
                detected default when provided.
            env: Environment variables to set.
            timeout_sec: Timeout in seconds.

        Returns:
            Execution result containing stdout, stderr, and return code.
        """
        sandbox = self._require_sandbox()
        effective_timeout = timeout_sec if timeout_sec is not None else _DEFAULT_EXEC_TIMEOUT_SEC
        effective_cwd = cwd if cwd is not None else self._default_cwd

        result = await sandbox.run(
            command,
            timeout=effective_timeout,
            cwd=effective_cwd,
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
        content = await asyncio.to_thread(Path(source_path).read_bytes)

        parent = str(Path(target_path).parent)
        if parent != "/":
            await sandbox.run(f"mkdir -p {shlex.quote(parent)}", timeout=30)

        await sandbox.write(target_path, content)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        """Upload a local directory to the sandbox recursively.

        Args:
            source_dir: Local directory path.
            target_dir: Destination directory inside the sandbox.
        """
        self._require_sandbox()
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
        data = await sandbox.read(source_path)
        local = Path(target_path)
        local.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(local.write_bytes, data)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        """Download a directory from the sandbox to the local machine.

        Args:
            source_dir: Directory path inside the sandbox.
            target_dir: Local destination directory.
        """
        local_dir = Path(target_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        result = await self.exec(
            f"find {shlex.quote(source_dir)} -type f",
            timeout_sec=60,
        )
        if result.return_code != 0:
            logger.warning(
                "Failed to list files in '%s': exit_code=%d, stderr=%s",
                source_dir,
                result.return_code,
                result.stderr,
            )
            return

        if not result.stdout or not result.stdout.strip():
            logger.info("No files found in '%s'", source_dir)
            return

        files = [f for f in result.stdout.strip().split("\n") if f]
        failures: list[str] = []
        for file_path in files:
            relative = Path(file_path).relative_to(source_dir)
            local_file = local_dir / relative
            try:
                await self.download_file(file_path, local_file)
            except Exception:  # noqa: BLE001
                logger.warning("Failed to download '%s'", file_path, exc_info=True)
                failures.append(file_path)

        if failures:
            logger.warning(
                "download_dir('%s') completed with %d/%d file failures",
                source_dir,
                len(failures),
                len(files),
            )
