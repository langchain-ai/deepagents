"""Opt-in native browser transport and host-owned invocation authority."""

from __future__ import annotations

import asyncio
import contextvars
import json
import os
import re
import stat
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from http import HTTPStatus
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import httpx
from langchain.tools import ToolRuntime, tool

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

_TOKEN_MODE = 0o400
_TOKEN_LENGTH = 43
_LIMIT = 4 * 1024 * 1024
_CONTROL = "http://172.30.12.3:8081"
_ACTIVE: contextvars.ContextVar[BrowserRun | None] = contextvars.ContextVar("browser", default=None)


@dataclass(frozen=True, slots=True, repr=False)
class BrowserBinding:
    """Identity supplied by the host route, never by model metadata."""

    provider: str
    sender_id: str
    conversation_id: str
    background: bool = False


@dataclass(frozen=True, slots=True)
class BrowserEvent:
    """Sanitized handoff notification for host delivery outside the model."""

    status: str
    handoff_id: str
    mode: str = "PAUSED"


BrowserEventHandler = Callable[[BrowserEvent], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class BrowserContext:
    """Opaque tool-runtime capability containing no client, identity, or credential."""

    run_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass(slots=True, repr=False)
class BrowserRun:
    """Private per-invocation lease state."""

    client: BrowserClient
    binding: BrowserBinding
    handler: BrowserEventHandler | None = None
    context: BrowserContext = field(default_factory=BrowserContext)
    lease: dict[str, object] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def run_id(self) -> str:
        """Return the fresh invocation identifier."""
        return self.context.run_id

    def owner(self) -> dict[str, object]:
        """Build the bridge owner from typed host authority."""
        return {**asdict(self.binding), "operator_id": self.client.operator, "run_id": self.run_id}

    async def action(self, action: str) -> dict[str, object]:
        """Perform one non-retried lease transition."""
        result = await self.client.post(
            "actions",
            {
                "action": action,
                "owner": self.owner(),
                "request_id": str(uuid4()),
                **self.lease,
            },
        )
        if action != "release":
            self.lease = _lease(result)
        return result

    async def command(self, method: str, params: dict[str, object], session_id: str | None) -> str:
        """Execute one serialized CDP command under the current lease version."""
        async with self.lock:
            if not self.lease:
                await self.action("acquire")
            body = {
                "owner": self.owner(),
                **self.lease,
                "request_id": str(uuid4()),
                "method": method,
                "params": params,
            }
            if session_id is not None:
                body["session_id"] = session_id
            response = await self.client.post("command", body)
            if "result" not in response:
                raise BrowserError
            return _observation(response["result"])

    async def close(self) -> None:
        """Release only this invocation's lease without surfacing transport errors."""
        if not self.lease:
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
            async with asyncio.timeout(35):
                for attempt in range(3):
                    try:
                        if attempt:
                            status = await self.client.post(
                                "actions",
                                {
                                    "action": "inspect",
                                    "owner": self.owner(),
                                    "request_id": str(uuid4()),
                                    **self.lease,
                                },
                            )
                            if status.get("mode") not in {"AGENT", "PAUSED", "HANDOFF_PENDING"}:
                                return
                            if type(status.get("version")) is not int or any(
                                status.get(key) != self.lease[key]
                                for key in ("lease_id", "generation")
                            ):
                                return
                            self.lease["version"] = status["version"]
                        await self.action("release")
                    except BrowserError:
                        continue
                    return
        except TimeoutError:
            return
        finally:
            self.lease.clear()


class BrowserError(Exception):
    """Sanitized browser transport failure."""

    def __init__(self, *, code: object = None) -> None:
        """Map only allowlisted bridge codes, excluding all remote details."""
        busy = isinstance(code, str) and code in {"lease_busy", "transport_busy", "pending_limit"}
        super().__init__("browser_busy" if busy else "browser_unavailable")


class BrowserClient:
    """Fixed-address, bounded, non-retrying browser control client."""

    def __init__(self, env: Mapping[str, str]) -> None:
        """Keep only validated identity settings and a token file path."""
        try:
            self.operator = env["TALON_BROWSER_OPERATOR_ID"]
            identities = json.loads(env["TALON_BROWSER_IDENTITIES"])
            if not self.operator or not isinstance(identities, dict) or not identities:
                raise BrowserError
            if any(
                not isinstance(k, str) or not isinstance(v, str) or not k or not v
                for k, v in identities.items()
            ):
                raise BrowserError
            self.identities: dict[str, str] = identities
            self._token_file = env["TALON_BROWSER_TOKEN_FILE"]
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
            base_url=_CONTROL,
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

    def bind(
        self, binding: BrowserBinding | None, handler: BrowserEventHandler | None = None
    ) -> BrowserRun | None:
        """Deny missing or mismatched explicit sender identities."""
        if not isinstance(binding, BrowserBinding) or not binding.conversation_id:
            return None
        if (
            not isinstance(binding.background, bool)
            or not binding.sender_id
            or self.identities.get(binding.provider) != binding.sender_id
        ):
            return None
        return BrowserRun(self, binding, None if binding.background else handler)

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


def _lease(result: dict[str, object]) -> dict[str, object]:
    if (
        not isinstance(result.get("lease_id"), str)
        or type(result.get("generation")) is not int
        or type(result.get("version")) is not int
        or result.get("mode") not in ("AGENT", "PAUSED", "HUMAN", "FAILED")
    ):
        raise BrowserError
    return {key: result[key] for key in ("lease_id", "generation", "version")}


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
        """
        run = _run(runtime)
        if run is None:
            return _observation({"status": "browser_denied"})
        try:
            return await run.command(method, params, session_id)
        except BrowserError as error:
            return _observation({"status": str(error)})

    @tool
    async def browser_request_handoff(reason: str, runtime: ToolRuntime[object]) -> str:
        """Pause for human login; no viewer URL or automatic login screenshot is available."""
        del reason
        run = _run(runtime)
        if run is None:
            return _observation({"status": "browser_denied"})
        try:
            async with run.lock:
                if not run.lease:
                    await run.action("acquire")
                response = await run.action("handoff")
                status = "human_required" if run.binding.background else "viewer_unavailable"
                handoff_id = str(UUID(str(response.get("handoff_id"))))
                event = BrowserEvent(status, handoff_id)
                if run.handler is not None:
                    with suppress(Exception):
                        await run.handler(event)
                return _observation(asdict(event))
        except BrowserError as error:
            return _observation({"status": str(error)})
        except ValueError:
            return _observation({"status": "browser_unavailable"})

    return [browser_cdp, browser_request_handoff]
