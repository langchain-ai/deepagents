"""LangSmith Harbor environment for an in-sandbox Switchyard sidecar."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any, override

from harbor.environments.docker import COMPOSE_NO_NETWORK_PATH
from harbor.environments.langsmith import LangSmithEnvironment

if TYPE_CHECKING:
    from collections.abc import Mapping

_COMPOSE_ENV_NAMES = (
    "ANTHROPIC_API_KEY",
    "BASETEN_API_KEY",
    "GOOGLE_API_KEY",
    "NVIDIA_API_KEY",
    "SWITCHYARD_IMAGE",
)
_REMOTE_CONFIG_PATH = "/harbor/compose/switchyard-routes.toml"
_SWITCHYARD_BASE_URL = "http://switchyard:4000"
_HEALTH_TIMEOUT_SECONDS = 60
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


def _python_http_command(path: str, *, parse_json: bool) -> str:
    """Build a fixed-origin Python command that reads one HTTP response.

    Args:
        path: Absolute Switchyard HTTP path.
        parse_json: Whether to parse and print the response as JSON.

    Returns:
        Shell-safe command for Harbor's main service.
    """
    read = "print(json.dumps(json.load(response)))" if parse_json else "response.read()"
    script = (
        "import json, urllib.request; "
        f"response = urllib.request.urlopen('{_SWITCHYARD_BASE_URL}{path}', timeout=10); "
        f"{read}"
    )
    return shlex.join(["python", "-c", script])


class SwitchyardLangSmithEnvironment(LangSmithEnvironment):
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
        config = Path(switchyard_config).expanduser()
        if not config.is_file() or config.suffix != ".toml":
            msg = f"Switchyard route config must be an existing TOML file: {config}"
            raise ValueError(msg)
        self._switchyard_config = config.resolve()
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
        command = _python_http_command("/health", parse_json=False)
        for _ in range(_HEALTH_TIMEOUT_SECONDS // 2):
            result = await self.exec(command, cwd="/", timeout_sec=15)
            if result.return_code == 0:
                return
            await asyncio.sleep(2)

        logs = await self._compose_exec(
            ["logs", "--no-color", "--tail", "50", "switchyard"],
            timeout_sec=15,
        )
        detail = (logs.stderr or logs.stdout or "")[-1000:]
        msg = f"Switchyard sidecar did not become healthy: {detail}"
        raise RuntimeError(msg)

    async def _snapshot_switchyard_stats(self) -> None:
        if not self._compose_mode:
            return
        result = await self.exec(
            _python_http_command("/v1/stats", parse_json=True),
            cwd="/",
            timeout_sec=30,
        )
        if result.return_code != 0 or not result.stdout:
            self.logger.warning("Could not capture Switchyard stats before teardown")
            return
        try:
            stats = json.loads(result.stdout)
        except json.JSONDecodeError:
            self.logger.warning("Switchyard returned invalid JSON from /v1/stats")
            return
        if not isinstance(stats, dict):
            self.logger.warning("Switchyard returned a non-object /v1/stats payload")
            return

        stats["switchyard_config"] = self._switchyard_config.name
        path = self.trial_paths.artifacts_dir / "switchyard-stats.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    @override
    async def stop(self, delete: bool) -> None:
        try:
            await self._snapshot_switchyard_stats()
        finally:
            await super().stop(delete)
