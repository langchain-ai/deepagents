"""A provider that runs Chrome inside the agent's sandbox and attaches over CDP.

Shows what the `BrowserProvider` protocol is for: nothing here is special-cased
in the middleware. The provider launches a browser through the agent's own
backend, reads back the CDP endpoint, and `local_browser.connect` does the rest.

    backend = LocalShellBackend()
    create_deep_agent(
        backend=backend,
        middleware=[BrowserMiddleware(SandboxChrome(backend))],
    )

Reachability is the catch. The tools run in the agent process, so the CDP URL
the sandbox prints has to be routable from there. That holds when the sandbox
shares a host with the agent -- `LocalShellBackend`, a dev container, a VM with
the port published. It does not hold for a Managed Deep Agent, whose per-thread
sandbox exposes shell and filesystem but no inbound ports.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
from dataclasses import dataclass, field
from typing import Any

from deepagents.middleware.browser import BrowserContext

# Without these the extension cannot be loaded or attached to, and
# `Stagehand.create` fails with "CDP websocket failed to open".
CHROME_FLAGS = (
    "--enable-unsafe-extension-debugging",
    "--remote-allow-origins=*",
    "--enable-features=WebMCPTesting,DevToolsWebMCPSupport",
    "--disable-component-extensions-with-background-pages",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-dev-shm-usage",
)


@dataclass
class SandboxChrome:
    """Chrome launched inside the agent's sandbox, driven over CDP.

    The backend is passed in rather than read off the runtime: `ToolRuntime`
    exposes `emit_output_delta`, `execution_info` and `server_info`, and no
    way to reach the agent's backend. You already build one to hand to
    `create_deep_agent`, so hand the same object here.
    """

    backend: Any
    port: int = 9222
    headless: bool = True
    chrome: str = os.environ.get("CHROME_PATH", "google-chrome")  # the sandbox image's Chrome
    _pids: dict[str, str] = field(default_factory=dict)

    async def open(self, ctx: BrowserContext) -> Any:
        from stagehand import local_browser

        backend = self.backend
        flags = " ".join(shlex.quote(flag) for flag in CHROME_FLAGS)
        headless = "--headless" if self.headless else ""
        # Detached, so the browser outlives the shell that started it.
        start = (
            f"{shlex.quote(self.chrome)} {headless} {flags} "
            f"--remote-debugging-port={self.port} "
            f"--user-data-dir=$(mktemp -d) about:blank "
            f">/tmp/chrome.log 2>&1 & echo $!"
        )
        self._pids[ctx.thread_id] = (await backend.aexecute(start)).output.strip()

        for _ in range(60):
            probe = (await backend.aexecute(f"curl -sf http://127.0.0.1:{self.port}/json/version || true")).output.strip()
            if probe.startswith("{"):
                url = json.loads(probe)["webSocketDebuggerUrl"]
                return await local_browser.connect(cdp_url=url)
            await asyncio.sleep(0.5)

        log = (await backend.aexecute("tail -5 /tmp/chrome.log || true")).output
        msg = f"Chrome did not open a CDP port in the sandbox. Last output:\n{log}"
        raise RuntimeError(msg)

    async def close(self, browser: Any, ctx: BrowserContext) -> None:
        """Stop the browser we started, and only that one."""
        pid = self._pids.pop(ctx.thread_id, None)
        if pid:
            await self.backend.aexecute(f"kill {shlex.quote(pid)} 2>/dev/null || true")
