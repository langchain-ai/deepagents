"""Harbor environments for running Switchyard beside the evaluated agent."""

from __future__ import annotations

import asyncio
import json
import math
import os
import shlex
import tarfile
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, override

from harbor.environments.docker import COMPOSE_NO_NETWORK_PATH
from harbor.environments.docker.docker import DockerEnvironment
from harbor.environments.langsmith import LangSmithEnvironment

if TYPE_CHECKING:
    import logging
    from collections.abc import Mapping

    from harbor.environments.base import BaseEnvironment, ExecResult
    from harbor.models.trial.paths import TrialPaths

_COMPOSE_ENV_NAMES = (
    "ANTHROPIC_API_KEY",
    "BASETEN_API_KEY",
    "GOOGLE_API_KEY",
    "NVIDIA_API_KEY",
    "SWITCHYARD_IMAGE",
)
_REMOTE_CONFIG_PATH = "/harbor/compose/switchyard-routes.toml"
_SWITCHYARD_BASE_URL = "http://switchyard:4000"
_SWITCHYARD_DOCKER_BASE_URL = "http://127.0.0.1:4000"
_AGENT_PYTHON = "/opt/harbor-langgraph-venv/bin/python"
_HEALTH_TIMEOUT_SECONDS = 60
# The largest lite task allows 1,800 seconds for its full environment startup;
# reserve five minutes for sandbox boot, dockerd, and post-Compose health checks.
_COMPOSE_UP_TIMEOUT_SECONDS = 1500
_MIN_MEMORY_BYTES_PER_VCPU = 2 * 1024**3
_MAX_MEMORY_BYTES_PER_VCPU = 6 * 1024**3
_REMOTE_TMP_DIR = "/tmp"  # noqa: S108  # fixed path in the remote LangSmith sandbox
_DOCKERD_START_TIMEOUT_SECONDS = 15
_DOCKERD_STARTED_MARKER = "DOCKERD_STARTED"
_DOCKERD_START_COMMAND = (
    "mkdir -p /var/run /var/log && "
    "command -v setsid >/dev/null 2>&1 && "
    "setsid -f dockerd --registry-mirror=https://mirror.gcr.io "
    ">>/var/log/dockerd.log 2>&1 </dev/null && "
    f"echo {_DOCKERD_STARTED_MARKER}"
)


def _forwarded_compose_env(environ: Mapping[str, str]) -> dict[str, str]:
    """Select the exact host variables Switchyard's compose service may receive.

    Args:
        environ: Host process environment.

    Returns:
        Non-empty provider credentials and the pinned image reference.
    """
    return {name: environ[name] for name in _COMPOSE_ENV_NAMES if environ.get(name)}


def _switchyard_config_path(config: str | Path) -> Path:
    """Resolve an existing Switchyard route TOML.

    Args:
        config: Candidate route configuration path.

    Returns:
        Absolute path to the validated TOML.

    Raises:
        ValueError: If the path does not identify an existing TOML file.
    """
    path = Path(config).expanduser()
    if not path.is_file() or path.suffix != ".toml":
        msg = f"Switchyard route config must be an existing TOML file: {path}"
        raise ValueError(msg)
    return path.resolve()


def _python_http_command(
    path: str,
    *,
    parse_json: bool,
    base_url: str = _SWITCHYARD_BASE_URL,
) -> str:
    """Build a fixed-origin Python command that reads one HTTP response.

    Args:
        path: Absolute Switchyard HTTP path.
        parse_json: Whether to parse and print the response as JSON.
        base_url: Trusted Switchyard origin reachable from Harbor's main service.

    Returns:
        Shell-safe command for Harbor's main service.
    """
    read = "print(json.dumps(json.load(response)))" if parse_json else "response.read()"
    script = (
        "import json, urllib.request; "
        f"response = urllib.request.urlopen('{base_url}{path}', timeout=10); "
        f"{read}"
    )
    return shlex.join([_AGENT_PYTHON, "-c", script])


def _bash_health_command(host: str = "switchyard") -> str:
    """Build a dependency-free HTTP health probe for the main container.

    Harbor task images do not necessarily include Python, curl, wget, or netcat
    before the agent is installed. Bash is part of the task image contract, so
    its `/dev/tcp` support can verify both Compose DNS and the HTTP endpoint.

    Returns:
        Shell-safe command that succeeds only for an HTTP 200 response.
    """
    script = (
        f"exec 3<>/dev/tcp/{host}/4000; "
        f"printf 'GET /health HTTP/1.0\\r\\nHost: {host}\\r\\n\\r\\n' >&3; "
        "IFS= read -r status <&3; "
        "[[ $status == HTTP/*' 200 '* ]]"
    )
    return shlex.join(["bash", "-lc", script])


def _create_upload_archive(source: Path, archive: Path) -> None:
    """Archive every source entry exactly once for LangSmith directory uploads.

    Args:
        source: Directory whose children should be archived.
        archive: Destination `.tar.gz` path.
    """
    with tarfile.open(archive, "w:gz") as tar:
        for path in source.rglob("*"):
            tar.add(path, arcname=path.relative_to(source), recursive=False)


class DeepAgentsLangSmithEnvironment(LangSmithEnvironment):
    """Apply Deep Agents compatibility fixes to Harbor's LangSmith provider."""

    @override
    def _create_sandbox_payload(self, snapshot_name: str | None) -> dict[str, Any]:
        """Preserve requested resources while satisfying LangSmith's ratio limits."""
        payload = super()._create_sandbox_payload(snapshot_name)
        vcpus = payload.get("vcpus")
        memory = payload.get("mem_bytes")
        if type(vcpus) is not int or vcpus <= 0 or type(memory) is not int:
            return payload

        minimum = vcpus * _MIN_MEMORY_BYTES_PER_VCPU
        maximum = vcpus * _MAX_MEMORY_BYTES_PER_VCPU
        if memory < minimum:
            payload["mem_bytes"] = minimum
        elif memory > maximum:
            payload["vcpus"] = math.ceil(memory / _MAX_MEMORY_BYTES_PER_VCPU)
        return payload

    @override
    async def _upload_dir_to_sandbox(
        self,
        source_dir: Path | str,
        target_dir: str,
    ) -> None:
        """Upload directories without recursively re-adding nested entries."""
        source = Path(source_dir)
        with tempfile.TemporaryDirectory() as temp_dir:
            archive = Path(temp_dir) / "upload.tar.gz"
            await asyncio.to_thread(_create_upload_archive, source, archive)
            remote_archive = f"{_REMOTE_TMP_DIR}/harbor-upload-{uuid.uuid4().hex}.tar.gz"
            await self._upload_file_to_sandbox(archive, remote_archive)

        target = shlex.quote(target_dir)
        remote = shlex.quote(remote_archive)
        result = await self._exec_sandbox(
            f"mkdir -p {target} && tar -xzf {remote} -C {target} && rm -f {remote}",
            cwd=_REMOTE_TMP_DIR,
        )
        if result.return_code != 0:
            detail = result.stderr or result.stdout or ""
            msg = f"upload_dir extraction failed: {detail}"
            raise RuntimeError(msg)


class _SwitchyardStatsMixin:
    """Capture a sidecar's final counters into the Harbor trial artifacts."""

    _switchyard_config: Path
    if TYPE_CHECKING:
        logger: logging.Logger
        trial_paths: TrialPaths

    async def _snapshot_switchyard_stats(
        self,
        *,
        base_url: str = _SWITCHYARD_BASE_URL,
    ) -> bool:
        result = await cast("BaseEnvironment", self).exec(
            _python_http_command("/v1/stats", parse_json=True, base_url=base_url),
            cwd="/",
            timeout_sec=30,
        )
        if result.return_code != 0 or not result.stdout:
            self.logger.warning("Could not capture Switchyard stats before teardown")
            return False
        try:
            stats = json.loads(result.stdout)
        except json.JSONDecodeError:
            self.logger.warning("Switchyard returned invalid JSON from /v1/stats")
            return False
        if not isinstance(stats, dict):
            self.logger.warning("Switchyard returned a non-object /v1/stats payload")
            return False

        stats["switchyard_config"] = self._switchyard_config.name
        path = self.trial_paths.artifacts_dir / "switchyard-stats.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        return True


class SwitchyardDockerEnvironment(_SwitchyardStatsMixin, DockerEnvironment):
    """Run Switchyard in the task container's network namespace on Docker.

    The Compose overlay makes Switchyard share `main`'s network namespace, so
    the agent reaches it through loopback while Harbor's phase network policy
    still governs all outbound traffic. Provider keys enter only the sidecar at
    container start. The custom agent snapshots stats and stops the sidecar
    before verification begins.

    Args:
        switchyard_config: Rendered Switchyard route TOML mounted into the sidecar.
        args: Positional arguments accepted by `DockerEnvironment`.
        kwargs: Keyword arguments accepted by `DockerEnvironment`.
    """

    def __init__(
        self,
        *args: Any,
        switchyard_config: str | Path,
        **kwargs: Any,
    ) -> None:
        """Validate the route config before initializing Docker."""
        self._switchyard_config = _switchyard_config_path(switchyard_config)
        self._switchyard_stopped = False
        super().__init__(*args, **kwargs)

    @override
    def _compose_env_vars(self, include_os_env: bool = True) -> dict[str, str]:
        """Expose the non-secret route path needed by the Compose overlay."""
        env = super()._compose_env_vars(include_os_env=include_os_env)
        env["SWITCHYARD_CONFIG_PATH"] = str(self._switchyard_config)
        return env

    @override
    async def start(self, force_build: bool) -> None:
        """Start the native task Compose project and verify the router endpoint."""
        await super().start(force_build)
        await self._wait_for_switchyard()

    async def _wait_for_switchyard(self) -> None:
        command = _bash_health_command("127.0.0.1")
        last_failure = ""
        for _ in range(_HEALTH_TIMEOUT_SECONDS // 2):
            result = await self.exec(command, cwd="/", timeout_sec=15)
            if result.return_code == 0:
                return
            last_failure = (result.stderr or result.stdout or "").strip()[-500:]
            await asyncio.sleep(2)

        logs = await self._run_docker_compose_command(
            ["logs", "--no-color", "--tail", "50", "switchyard"],
            check=False,
            timeout_sec=15,
        )
        detail = (logs.stderr or logs.stdout or "")[-1000:]
        probe = f" Probe failure: {last_failure}." if last_failure else ""
        msg = f"Switchyard sidecar did not become healthy.{probe} Logs: {detail}"
        raise RuntimeError(msg)

    async def capture_and_stop_switchyard(self) -> None:
        """Persist router stats and remove provider access before verification."""
        if self._switchyard_stopped:
            return
        try:
            await self._snapshot_switchyard_stats(base_url=_SWITCHYARD_DOCKER_BASE_URL)
        finally:
            await self.stop_service("switchyard")
            self._switchyard_stopped = True

    @override
    async def stop(self, delete: bool) -> None:
        """Best-effort capture if agent execution ended before its cleanup hook."""
        try:
            try:
                await self.capture_and_stop_switchyard()
            except Exception as exc:  # noqa: BLE001  # teardown must still remove containers
                self.logger.warning("Switchyard cleanup before Docker teardown failed: %s", exc)
        finally:
            await super().stop(delete)


class SwitchyardLangSmithEnvironment(
    _SwitchyardStatsMixin,
    DeepAgentsLangSmithEnvironment,
):
    """Run Switchyard beside Harbor's main service in a LangSmith sandbox.

    Harbor's `--agent-env` reaches only the agent process, not Docker Compose.
    This adapter forwards a fixed allowlist of provider variables directly from
    the runner environment into Compose, without serializing their values into a
    job config or command argument. It also stages the selected route TOML and
    captures the router's final stats before the sandbox is torn down.

    Args:
        switchyard_config: Rendered Switchyard route TOML to stage for the sidecar.
        args: Positional arguments accepted by `LangSmithEnvironment`.
        kwargs: Keyword arguments accepted by `LangSmithEnvironment`.
    """

    def __init__(
        self,
        *args: Any,
        switchyard_config: str | Path,
        **kwargs: Any,
    ) -> None:
        """Validate the route config before initializing the sandbox provider."""
        self._switchyard_config = _switchyard_config_path(switchyard_config)
        super().__init__(*args, **kwargs)

    @override
    async def _ensure_docker_daemon(self) -> None:
        """Start Docker without holding LangSmith's command session open.

        Harbor 0.20 backgrounds `dockerd` with `&`, but LangSmith's sandbox
        command API waits for that process group and consumes Harbor's entire
        environment-start timeout. `setsid -f` forks it into a detached session
        so the command returns immediately; Harbor's inherited readiness loop
        still verifies the daemon and reports its log if startup fails.
        """
        probe = await self._exec_sandbox(
            "docker info >/dev/null 2>&1 && echo ready",
            cwd="/",
            timeout_sec=10,
        )
        if probe.return_code == 0 and "ready" in (probe.stdout or ""):
            self.logger.debug("Docker daemon already running in LangSmith sandbox")
            return

        self.logger.debug("Starting detached Docker daemon in LangSmith sandbox")
        result = await self._exec_sandbox(
            _DOCKERD_START_COMMAND,
            cwd="/",
            timeout_sec=_DOCKERD_START_TIMEOUT_SECONDS,
        )
        output = (result.stdout or "") + (result.stderr or "")
        if result.return_code != 0 or _DOCKERD_STARTED_MARKER not in output:
            msg = f"Failed to detach Docker daemon in LangSmith sandbox: {output[-500:]}"
            raise RuntimeError(msg)

    @override
    def _create_sandbox_payload(self, snapshot_name: str | None) -> dict[str, Any]:
        """Allow sidecar egress while Compose isolates the agent container.

        The stock LangSmith no-network policy applies to the whole sandbox,
        including Docker sidecars. Switchyard must reach model APIs, so its
        Compose egress network is the enforcement boundary instead: `main` is
        attached only to an internal network, while Switchyard is also attached
        to an egress network.
        """
        payload = super()._create_sandbox_payload(snapshot_name)
        payload.pop("proxy_config", None)
        return payload

    @override
    async def _compose_exec(
        self,
        subcommand: list[str],
        timeout_sec: int | None = None,
    ) -> ExecResult:
        """Give large task images enough time to pull during Compose startup."""
        if subcommand == ["up", "-d"]:
            timeout_sec = max(timeout_sec or 0, _COMPOSE_UP_TIMEOUT_SECONDS)
        return await super()._compose_exec(subcommand, timeout_sec=timeout_sec)

    @override
    def _compose_file_flags(self) -> list[str]:
        """Replace Harbor's `network_mode: none` with the split overlay network."""
        flags = super()._compose_file_flags()
        no_network = f"/harbor/compose/{COMPOSE_NO_NETWORK_PATH.name}"
        filtered: list[str] = []
        index = 0
        while index < len(flags):
            if flags[index : index + 2] == ["-f", no_network]:
                index += 2
                continue
            filtered.append(flags[index])
            index += 1
        return filtered

    @override
    def _compose_env_vars(self) -> dict[str, str]:
        env = super()._compose_env_vars()
        env.update(_forwarded_compose_env(os.environ))
        return env

    @override
    async def _stage_extra_compose_files(self) -> None:
        await super()._stage_extra_compose_files()
        await self._upload_file_to_sandbox(
            self._switchyard_config,
            _REMOTE_CONFIG_PATH,
        )

    @override
    async def _start_compose(self, force_build: bool) -> None:
        await super()._start_compose(force_build)
        await self._wait_for_switchyard()

    async def _wait_for_switchyard(self) -> None:
        command = _bash_health_command()
        last_failure = ""
        for _ in range(_HEALTH_TIMEOUT_SECONDS // 2):
            result = await self.exec(command, cwd="/", timeout_sec=15)
            if result.return_code == 0:
                return
            last_failure = (result.stderr or result.stdout or "").strip()[-500:]
            await asyncio.sleep(2)

        logs = await self._compose_exec(
            ["logs", "--no-color", "--tail", "50", "switchyard"],
            timeout_sec=15,
        )
        detail = (logs.stderr or logs.stdout or "")[-1000:]
        probe = f" Probe failure: {last_failure}." if last_failure else ""
        msg = f"Switchyard sidecar did not become healthy.{probe} Logs: {detail}"
        raise RuntimeError(msg)

    async def isolate_main_after_setup(self) -> None:
        """Remove setup egress before the benchmarked agent phase begins.

        Harbor installs the LangGraph runtime after the task container starts,
        so `main` initially shares Switchyard's egress network. The custom
        Switchyard agent calls this method in a `finally` block after setup.
        The topology check fails closed if Docker leaves any network other than
        the private agent-to-router network attached.

        Raises:
            RuntimeError: If `main` cannot be identified or fully isolated.
        """
        container = await self._compose_exec(["ps", "-q", "main"], timeout_sec=15)
        container_id = (container.stdout or "").strip()
        if container.return_code != 0 or not container_id:
            detail = (container.stderr or container.stdout or "")[-500:]
            msg = f"Could not identify the Harbor main container: {detail}"
            raise RuntimeError(msg)

        project = self._compose_project_name()
        egress_network = f"{project}_switchyard-egress"
        disconnect = await self._exec_sandbox(
            shlex.join(["docker", "network", "disconnect", egress_network, container_id]),
            cwd="/",
            timeout_sec=15,
        )
        if disconnect.return_code != 0:
            detail = (disconnect.stderr or disconnect.stdout or "")[-500:]
            msg = f"Could not disconnect Harbor main setup egress: {detail}"
            raise RuntimeError(msg)

        inspect = await self._exec_sandbox(
            shlex.join(
                [
                    "docker",
                    "inspect",
                    "--format",
                    "{{json .NetworkSettings.Networks}}",
                    container_id,
                ]
            ),
            cwd="/",
            timeout_sec=15,
        )
        if inspect.return_code != 0:
            detail = (inspect.stderr or inspect.stdout or "")[-500:]
            msg = f"Could not inspect Harbor main network isolation: {detail}"
            raise RuntimeError(msg)
        try:
            networks = json.loads(inspect.stdout or "")
        except json.JSONDecodeError as exc:
            msg = "Docker returned invalid main-container network metadata"
            raise RuntimeError(msg) from exc

        expected = {f"{project}_switchyard-internal"}
        if not isinstance(networks, dict) or set(networks) != expected:
            found = sorted(networks) if isinstance(networks, dict) else networks
            msg = (
                f"Harbor main network isolation failed: expected {sorted(expected)}, found {found}"
            )
            raise RuntimeError(msg)

        health = await self.exec(_bash_health_command(), cwd="/", timeout_sec=15)
        if health.return_code != 0:
            detail = (health.stderr or health.stdout or "")[-500:]
            msg = f"Harbor main lost Switchyard access after isolation: {detail}"
            raise RuntimeError(msg)

    @override
    async def stop(self, delete: bool) -> None:
        try:
            if self._compose_mode:
                await self._snapshot_switchyard_stats()
        finally:
            await super().stop(delete)
