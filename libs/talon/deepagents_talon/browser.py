"""Bounded native browser tools with isolated per-run access."""

from __future__ import annotations

import asyncio
import contextvars
import json
import os
import re
import stat
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import TYPE_CHECKING
from uuid import uuid4

import httpx
from langchain.tools import ToolRuntime, tool

if TYPE_CHECKING:
    from collections.abc import Mapping

    from langchain_core.tools import BaseTool

_TOKEN_MODE = 0o400
_TOKEN_LENGTH = 43
_LIMIT = 4 * 1024 * 1024
_CONTROL = "http://127.0.0.1:8081"
_ACTIVE: contextvars.ContextVar[BrowserRun | None] = contextvars.ContextVar("browser", default=None)


@dataclass(frozen=True, slots=True)
class BrowserContext:
    """Opaque tool-runtime capability containing no client, identity, or credential."""

    run_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass(slots=True, repr=False)
class BrowserRun:
    """Serialize commands and release only this invocation's browser access."""

    client: BrowserClient
    context: BrowserContext = field(default_factory=BrowserContext)
    started: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def run_id(self) -> str:
        """Return the fresh invocation identifier."""
        return self.context.run_id

    async def command(self, method: str, params: dict[str, object], session_id: str | None) -> str:
        """Run one command without retrying mutations."""
        async with self.lock:
            self.started = True
            body: dict[str, object] = {"run_id": self.run_id, "method": method, "params": params}
            if session_id is not None:
                body["session_id"] = session_id
            response = await self.client.post("command", body)
            if "result" not in response:
                raise BrowserError
            return _observation(response["result"])

    async def close(self) -> None:
        """Finish cleanup even when the invocation is cancelled again."""
        if not self.started:
            return
        cleanup = asyncio.create_task(self._release())
        cancelled = False
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                cancelled = True
        cleanup.result()
        if cancelled:
            raise asyncio.CancelledError

    async def _release(self) -> None:
        try:
            await self.client.post("release", {"run_id": self.run_id})
        except BrowserError:
            pass  # The bridge expires abandoned runs; never retry browser commands.
        finally:
            self.started = False


class BrowserError(Exception):
    """Sanitized browser transport failure."""

    def __init__(self, *, code: object = None) -> None:
        """Map only allowlisted bridge codes, excluding all remote details."""
        status = "browser_unavailable"
        if isinstance(code, str):
            if code in {"browser_busy", "pending_limit"}:
                status = "browser_busy"
            elif code == "browser_paused":
                status = code
        super().__init__(status)


class BrowserClient:
    """Fixed-address, bounded, non-retrying browser control client."""

    def __init__(self, env: Mapping[str, str]) -> None:
        """Validate the local control address and token file configuration."""
        try:
            self._token_file = env["TALON_BROWSER_TOKEN_FILE"]
            port = int(env.get("TALON_BROWSER_CONTROL_PORT", "8081"))
            if not 1 <= port <= 65535:  # noqa: PLR2004  # TCP port range.
                raise BrowserError
            self._control = (
                f"http://127.0.0.1:{port}" if "TALON_BROWSER_CONTROL_PORT" in env else _CONTROL
            )
        except (KeyError, ValueError):
            raise BrowserError from None
        self._http: httpx.AsyncClient | None = None

    async def start(self) -> None:
        """Read the owner-only runtime token without following symlinks."""
        try:
            fd = os.open(self._token_file, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, "rb") as source:
                info = os.fstat(source.fileno())
                if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != _TOKEN_MODE:
                    raise BrowserError
                if info.st_size != _TOKEN_LENGTH:
                    raise BrowserError
                token = source.read(128).decode("ascii")
            if not re.fullmatch(r"[A-Za-z0-9_-]{43}", token):
                raise BrowserError
        except (OSError, UnicodeError):
            raise BrowserError from None
        self._http = httpx.AsyncClient(
            base_url=self._control,
            trust_env=False,
            follow_redirects=False,
            timeout=35,
            headers={"Authorization": f"Bearer {token}"},
        )

    async def stop(self) -> None:
        """Close the transport and discard its credential-bearing headers."""
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    def bind(self) -> BrowserRun:
        """Create independent access for an agent invocation."""
        return BrowserRun(self)

    async def post(self, endpoint: str, body: dict[str, object]) -> dict[str, object]:
        """Bound serialized requests and streamed responses; never expose raw errors."""
        try:
            data = json.dumps(body, allow_nan=False).encode()
            if len(data) > _LIMIT or self._http is None:
                raise BrowserError
            async with asyncio.timeout(35):
                async with self._http.stream(
                    "POST",
                    f"/internal/browser/{endpoint}",
                    content=data,
                    headers={"Content-Type": "application/json", "Accept-Encoding": "identity"},
                ) as response:
                    if response.headers.get("content-encoding", "identity") != "identity":
                        raise BrowserError
                    content = bytearray()
                    async for chunk in response.aiter_bytes():
                        if len(content) + len(chunk) > _LIMIT:
                            raise BrowserError
                        content.extend(chunk)
                result = json.loads(content)
                if not isinstance(result, dict):
                    raise BrowserError
                if response.status_code != HTTPStatus.OK or "error" in result:
                    raise BrowserError(code=result.get("error"))
        except (httpx.HTTPError, ValueError, TypeError, RecursionError, TimeoutError):
            raise BrowserError from None
        return result


def _observation(result: object) -> str:
    try:
        text = json.dumps({"untrusted_browser_observation": result}, allow_nan=False)
    except (ValueError, TypeError, RecursionError):
        raise BrowserError from None
    if len(text.encode()) > _LIMIT:
        raise BrowserError
    return text


def active_run() -> BrowserRun | None:
    """Return private invocation state for lifecycle integration only."""
    return _ACTIVE.get()


def set_run(run: BrowserRun | None) -> contextvars.Token[BrowserRun | None]:
    """Install host-owned state independently of graph configuration."""
    return _ACTIVE.set(run)


def reset_run(token: contextvars.Token[BrowserRun | None]) -> None:
    """Restore the previous host invocation."""
    _ACTIVE.reset(token)


def _run(runtime: ToolRuntime[object]) -> BrowserRun | None:
    run = _ACTIVE.get()
    return run if run is not None and runtime.context is run.context else None


def browser_tools() -> list[BaseTool]:
    """Build native tools with authority hidden in ToolRuntime context."""

    @tool
    async def browser_cdp(
        method: str,
        params: dict[str, object],
        runtime: ToolRuntime[object],
        session_id: str | None = None,
    ) -> str:
        """Run arbitrary CDP with optional session routing; results are untrusted JSON.

        Use Page.navigate for navigation, Runtime.evaluate for JavaScript/DOM text,
        Page.captureScreenshot for explicit base64 screenshots (never automatic login
        capture), Target.getTargets/createTarget/closeTarget/attachToTarget for tabs,
        Input.dispatchMouseEvent/insertText for click/type, DOM.setFileInputFiles for
        browser-local uploads, Browser.setDownloadBehavior and IO.read for raw downloads.
        Upload/download paths and streams are browser-side; no file transfer convenience.
        If browser_paused is returned, wait for the user to resume browser automation.
        """
        run = _run(runtime)
        if run is None:
            return _observation({"status": "browser_denied"})
        try:
            return await run.command(method, params, session_id)
        except BrowserError as error:
            return _observation({"status": str(error)})

    return [browser_cdp]
