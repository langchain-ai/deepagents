"""LangSmith Harbor environment for an in-sandbox Switchyard sidecar."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any, override

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
