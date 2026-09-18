"""Opt-in, workspace-owned native Steel lifecycle (macOS/Linux)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import signal
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deepagents_talon.config import TalonConfig

STEEL_REVISION = "2b41124d8e2953b0afe355c534e3c9aa71edae26"
_READY = b'{"event":"talon_steel_ready"}\n'
_SHUTDOWN_TIMEOUT = 30
_MAX_PORT = 65535
_BOOTSTRAP = Path(__file__).with_name("steel_runtime") / "bootstrap.mjs"


class SteelProcess:
    """Own a prepared Steel installation and an exclusive workspace profile.

    Args:
        config: Assistant home and operator-supplied browser configuration.
    """

    def __init__(self, config: TalonConfig) -> None:
        """Read configuration without starting processes or touching the profile."""
        self.root = config.home / "browser"
        self.source = (
            Path(config.env.get("TALON_BROWSER_STEEL_DIR", "~/.deepagents/steel"))
            .expanduser()
            .resolve()
        )
        self.chrome = config.env.get("TALON_BROWSER_CHROME", "")
        self.port = int(config.env.get("TALON_BROWSER_PORT", "3000"))
        self.timeout = float(config.env.get("TALON_BROWSER_START_TIMEOUT", "60"))
        if not 1 <= self.port <= _MAX_PORT or not math.isfinite(self.timeout) or self.timeout <= 0:
            msg = "Invalid TALON_BROWSER_PORT or TALON_BROWSER_START_TIMEOUT"
            raise ValueError(msg)
        self._process: asyncio.subprocess.Process | None = None
        self._reader: asyncio.Task[None] | None = None
        self._lock: int | None = None

    def _prepare(self) -> str:
        msg = "Steel is not prepared; run the documented one-time setup and configure Chrome"
        try:
            prepared = json.loads((self.source / ".talon-prepared.json").read_text())
            node = prepared["node"]
            valid = (
                prepared["revision"] == STEEL_REVISION
                and all(
                    Path(executable).is_absolute() and os.access(executable, os.X_OK)
                    for executable in (node, self.chrome)
                )
                and (self.source / "api/build/steel-browser-plugin.js").is_file()
            )
        except (OSError, ValueError, KeyError, TypeError):
            raise RuntimeError(msg) from None
        if not valid:
            raise RuntimeError(msg)
        return node

    def _acquire(self) -> None:
        import fcntl  # noqa: PLC0415  # Optional POSIX-only browser support.

        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(self.root / "profile.lock", os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(descriptor)
            msg = "Steel profile is already in use by another Talon process"
            raise RuntimeError(msg) from None
        self._lock = descriptor
        profile = self.root / "profile"
        profile.mkdir(mode=0o700, exist_ok=True)
        if (profile / ".talon-dirty").exists():
            msg = "Steel profile had an unclean shutdown; inspect it before removing .talon-dirty"
            raise RuntimeError(msg)

    def _environment(self) -> dict[str, str]:
        return {
            "PATH": os.defpath,
            "NODE_ENV": "development",
            "HOST": "127.0.0.1",
            "PORT": str(self.port),
            "CHROME_EXECUTABLE_PATH": self.chrome,
            "CHROME_USER_DATA_DIR": str((self.root / "profile").resolve()),
            "CHROME_HEADLESS": "true",
            "DISABLE_CHROME_SANDBOX": "false",
            "CHROME_ARGS": "--remote-debugging-port=0",
            "FILTER_CHROME_ARGS": "--remote-debugging-port=9222 --remote-allow-origins=*",
            "LOG_STORAGE_ENABLED": "false",
            "ENABLE_CDP_LOGGING": "false",
            "LOG_CUSTOM_EMIT_EVENTS": "false",
            "DEBUG_CHROME_PROCESS": "false",
            "ENABLE_VERBOSE_LOGGING": "false",
            "SKIP_FINGERPRINT_INJECTION": "true",
            "DEFAULT_TIMEZONE": "UTC",
        }

    async def start(self) -> None:
        """Start without downloads; unwind resources on failure or cancellation."""
        if self._process is not None:
            return
        node = self._prepare()
        try:
            self._acquire()
            spawn = asyncio.create_task(self._spawn(node))
            try:
                self._process = await asyncio.shield(spawn)
            except asyncio.CancelledError:
                self._process = await spawn
                raise
            await self._ready()
        except BaseException:
            await self.stop()
            raise

    async def _spawn(self, node: str) -> asyncio.subprocess.Process:
        return await asyncio.create_subprocess_exec(
            node,
            str(_BOOTSTRAP),
            str(self.source),
            cwd=self.root,
            env=self._environment(),
            start_new_session=True,
            pass_fds=(self._lock,) if self._lock is not None else (),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )

    async def _ready(self) -> None:
        if self._process is None or self._process.stdout is None:
            msg = "Steel did not start"
            raise RuntimeError(msg)
        try:
            async with asyncio.timeout(self.timeout):
                while line := await self._process.stdout.readline():
                    if line == _READY:
                        self._reader = asyncio.create_task(self._drain())
                        return
        except (TimeoutError, ValueError):
            msg = "Steel readiness timed out; check Chrome and the prepared installation"
            raise RuntimeError(msg) from None
        msg = "Steel exited before readiness; check Chrome, the configured port, and setup"
        raise RuntimeError(msg)

    async def _drain(self) -> None:
        if self._process is not None and self._process.stdout is not None:
            while await self._process.stdout.read(65536):
                pass

    async def wait(self) -> int:
        """Wait for the owned process to exit."""
        return await self._process.wait() if self._process is not None else 0

    async def stop(self) -> None:
        """Close Chrome gracefully, then kill remaining process-group members."""
        try:
            if self._process is not None:
                if self._process.returncode is None:
                    with contextlib.suppress(ProcessLookupError):
                        self._process.send_signal(signal.SIGTERM)
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._process.wait(), _SHUTDOWN_TIMEOUT)
                _signal_group(self._process.pid, signal.SIGKILL)
                await self._process.wait()
        finally:
            if self._reader is not None:
                self._reader.cancel()
                await asyncio.gather(self._reader, return_exceptions=True)
                self._reader = None
            self._process = None
            if self._lock is not None:
                os.close(self._lock)
                self._lock = None


def _signal_group(pid: int, signum: signal.Signals) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pid, signum)
