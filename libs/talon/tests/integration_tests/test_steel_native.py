"""Opt-in native browser smoke; requires prepared Steel, Node 24, and Chrome."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import sys
from pathlib import Path

import httpx
import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from deepagents_talon.browser import BrowserBinding, BrowserClient, BrowserError
from deepagents_talon.config import TalonConfig
from deepagents_talon.host import TalonHost
from deepagents_talon.interfaces import AgentRequest
from deepagents_talon.runtime import DeepAgentRuntime, EchoAgentRuntime
from deepagents_talon.steel import SteelProcess
from tests.unit_tests.test_background_runtime import ToolModel

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]

pytestmark = [
    pytest.mark.skipif(not os.environ.get("TALON_TEST_STEEL_DIR"), reason="native smoke is opt-in"),
    pytest.mark.enable_socket,
    pytest.mark.timeout(120),
]


async def _login(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        request = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 5)
        login = request.startswith(b"GET /login ")
        authenticated = b"talon_synthetic=logged-in" in request
        body = (
            b"<title>Synthetic account</title>"
            if login or authenticated
            else b"<title>Login</title>"
        )
        cookie = (
            b"Set-Cookie: talon_synthetic=logged-in; Max-Age=3600; "
            b"Path=/; HttpOnly; SameSite=Strict\r\n"
            if login
            else b""
        )
        writer.write(b"HTTP/1.1 200 OK\r\nConnection: close\r\n" + cookie + b"\r\n" + body)
        await writer.drain()
    except (TimeoutError, asyncio.IncompleteReadError):
        pass
    finally:
        writer.close()
        await writer.wait_closed()


def _port() -> int:
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        return reservation.getsockname()[1]


async def _exercise(source: Path, origin: str, login: str, phase: str) -> None:
    node = json.loads((source / ".talon-prepared.json").read_text())["node"]
    process = await asyncio.create_subprocess_exec(
        node,
        str(Path(__file__).with_name("steel_smoke.mjs")),
        str(source),
        origin,
        login,
        phase,
    )
    try:
        assert await asyncio.wait_for(process.wait(), 40) == 0
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


async def test_native_navigation_streaming_and_login_restart(tmp_path: Path) -> None:
    source = Path(os.environ["TALON_TEST_STEEL_DIR"])
    port, control_port, viewer_port = _port(), _port(), _port()
    config = TalonConfig(
        "smoke",
        tmp_path / "workspace",
        env={
            "TALON_BROWSER_ENABLED": "true",
            "TALON_BROWSER_STEEL_DIR": str(source),
            "TALON_BROWSER_CHROME": os.environ["TALON_TEST_CHROME"],
            "TALON_BROWSER_PORT": str(port),
            "TALON_BROWSER_CONTROL_PORT": str(control_port),
            "TALON_BROWSER_VIEWER_PORT": str(viewer_port),
        },
    )
    origin = f"http://127.0.0.1:{port}"
    server = await asyncio.start_server(_login, "127.0.0.1", 0)
    async with server:
        login = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
        for phase in ("login", "restart"):
            host = TalonHost(config=config, agent=EchoAgentRuntime())
            try:
                await host.start()
                assert host._steel is not None
                assert host._steel._process is not None
                pid = host._steel._process.pid
                with pytest.raises(RuntimeError, match="already in use"):
                    await SteelProcess(config).start()
                async with httpx.AsyncClient(trust_env=False) as client:
                    assert (
                        await client.get(
                            origin + "/v1/sessions", headers={"Origin": "https://example.com"}
                        )
                    ).status_code == 403
                await _exercise(source, origin, login, phase)
            finally:
                await host.stop()
            assert not (config.home / "browser/profile/.talon-dirty").exists()
            with pytest.raises(ProcessLookupError):
                os.killpg(pid, 0)


async def test_cli_ctrl_c_cleans_up(tmp_path: Path) -> None:
    port, control_port, viewer_port = _port(), _port(), _port()
    environment = {
        "PATH": os.defpath,
        "HOME": str(tmp_path),
        "PYTHONPATH": str(_PACKAGE_ROOT),
        "DEEPAGENTS_TALON_HOME": str(tmp_path / "talon"),
        "TALON_BROWSER_ENABLED": "true",
        "TALON_BROWSER_STEEL_DIR": os.environ["TALON_TEST_STEEL_DIR"],
        "TALON_BROWSER_CHROME": os.environ["TALON_TEST_CHROME"],
        "TALON_BROWSER_PORT": str(port),
        "TALON_BROWSER_CONTROL_PORT": str(control_port),
        "TALON_BROWSER_VIEWER_PORT": str(viewer_port),
    }
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "deepagents_talon",
        env=environment,
        cwd=tmp_path,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        assert process.stderr is not None
        async with asyncio.timeout(30):
            while line := await process.stderr.readline():
                if b"Talon host started" in line:
                    break
            else:
                pytest.fail("Talon CLI exited before startup")
        process.send_signal(signal.SIGINT)
        assert await asyncio.wait_for(process.wait(), 35) == 0
        assert not (tmp_path / "talon/default/browser/profile/.talon-dirty").exists()
        for listener in (port, control_port, viewer_port):
            with socket.socket() as probe:
                assert probe.connect_ex(("127.0.0.1", listener)) != 0
    finally:
        if process.returncode is None:
            process.terminate()
            await asyncio.wait_for(process.wait(), 35)


async def test_occupied_port_fails_without_orphans(tmp_path: Path) -> None:
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        config = TalonConfig(
            "failed",
            tmp_path,
            env={
                "TALON_BROWSER_STEEL_DIR": os.environ["TALON_TEST_STEEL_DIR"],
                "TALON_BROWSER_CHROME": os.environ["TALON_TEST_CHROME"],
                "TALON_BROWSER_PORT": str(occupied.getsockname()[1]),
                "TALON_BROWSER_CONTROL_PORT": str(_port()),
                "TALON_BROWSER_VIEWER_PORT": str(_port()),
            },
        )
        browser = SteelProcess(config)
        with pytest.raises(RuntimeError, match="before readiness"):
            await browser.start()
        assert not (browser.root / "profile/.talon-dirty").exists()
    try:
        await browser.start()
    finally:
        await browser.stop()


async def test_native_talon_tools_and_contention(tmp_path: Path) -> None:
    class BrowserModel(ToolModel):
        session: str | None = None

        def _generate(self, messages, stop=None, run_manager=None, **kwargs: object):
            del stop, run_manager, kwargs
            observations = [
                json.loads(message.content)["untrusted_browser_observation"]
                for message in messages
                if message.type == "tool" and message.name == "browser_cdp"
            ]
            step = len(observations)
            if step == 2:
                self.session = observations[-1]["sessionId"]
            commands = [
                ("Target.createTarget", {"url": "about:blank"}),
                (
                    "Target.attachToTarget",
                    {
                        "targetId": observations[0]["targetId"] if observations else "",
                        "flatten": True,
                    },
                ),
                (
                    "Page.navigate",
                    {"url": "data:text/html,<input id=entry autofocus><title>Local</title>"},
                ),
                ("Runtime.evaluate", {"expression": "document.querySelector('#entry').focus()"}),
                ("Input.insertText", {"text": "talon works"}),
                (
                    "Runtime.evaluate",
                    {"expression": "document.querySelector('#entry').value", "returnByValue": True},
                ),
            ]
            if step == len(commands):
                assert observations[-1]["result"]["value"] == "talon works"
                message = AIMessage(content="done")
            else:
                method, params = commands[step]
                args = {"method": method, "params": params}
                if self.session:
                    args["session_id"] = self.session
                message = AIMessage(
                    content="", tool_calls=[{"name": "browser_cdp", "id": str(step), "args": args}]
                )
            return ChatResult(generations=[ChatGeneration(message=message)])

    config = TalonConfig(
        "native-tools",
        tmp_path,
        env={
            "TALON_BROWSER_ENABLED": "true",
            "TALON_BROWSER_STEEL_DIR": os.environ["TALON_TEST_STEEL_DIR"],
            "TALON_BROWSER_CHROME": os.environ["TALON_TEST_CHROME"],
            "TALON_BROWSER_PORT": str(_port()),
            "TALON_BROWSER_CONTROL_PORT": str(_port()),
            "TALON_BROWSER_VIEWER_PORT": str(_port()),
        },
    )
    client = BrowserClient(
        {**config.env, "TALON_BROWSER_TOKEN_FILE": str(tmp_path / "browser/control-token")}
    )
    runtime = DeepAgentRuntime(
        model=BrowserModel(responses=[]),
        browser=client,
        assistant_dir=tmp_path,
        include_web_tools=False,
        skills=(),
        memory=(),
        env={},
    )
    host = TalonHost(config=config, agent=runtime)
    try:
        await host.start()
        foreground = client.bind(BrowserBinding("telegram", "sender", "foreground"))
        background = client.bind(
            BrowserBinding("telegram", "sender", "background", background=True)
        )
        await foreground.action("acquire")
        with pytest.raises(BrowserError, match="browser_busy"):
            await background.command("Target.getTargets", {}, None)
        await foreground.close()
        await runtime.invoke(
            AgentRequest(
                "chat", "browse", browser_binding=BrowserBinding("telegram", "sender", "chat")
            )
        )
        await background.command("Target.getTargets", {}, None)
        await background.close()
    finally:
        await host.stop()
    assert not (tmp_path / "browser/control-token").exists()
    assert not (tmp_path / "browser/profile/.talon-dirty").exists()
