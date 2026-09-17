"""Real child-process coverage without Chrome, npm, or network access."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from typing import TYPE_CHECKING

import pytest

from deepagents_talon import steel
from deepagents_talon.config import TalonConfig
from deepagents_talon.host import TalonHost
from deepagents_talon.runtime import EchoAgentRuntime

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.skipif(os.name != "posix", reason="native Steel requires POSIX")


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TalonConfig:
    source = tmp_path / "steel"
    (source / "api/build").mkdir(parents=True)
    (source / "api/build/steel-browser-plugin.js").touch()
    (source / ".talon-prepared.json").write_text(
        json.dumps({"revision": steel.STEEL_REVISION, "node": sys.executable})
    )
    child = tmp_path / "child.py"
    child.write_text(
        "import os, signal, time\n"
        "from pathlib import Path\n"
        "Path('pid').write_text(str(os.getpid()))\n"
        "signal.signal(signal.SIGTERM, lambda *args: exit(0))\n"
        'print(\'{"event":"talon_steel_ready"}\', flush=True)\n'
        "while True: time.sleep(0.05)\n"
    )
    monkeypatch.setattr(steel, "_BOOTSTRAP", child)
    monkeypatch.setattr(steel, "_SHUTDOWN_TIMEOUT", 0.2)
    return TalonConfig(
        "test",
        tmp_path / "home",
        env={
            "TALON_BROWSER_ENABLED": "true",
            "TALON_BROWSER_STEEL_DIR": str(source),
            "TALON_BROWSER_CHROME": sys.executable,
            "TALON_BROWSER_START_TIMEOUT": "1",
        },
    )


def assert_gone(pid: int) -> None:
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


async def test_exclusive_profile_and_restart(config: TalonConfig) -> None:
    first, second = steel.SteelProcess(config), steel.SteelProcess(config)
    try:
        await first.start()
        pid = int((first.root / "pid").read_text())
        with pytest.raises(RuntimeError, match="already in use"):
            await second.start()
        await first.stop()
        assert_gone(pid)
        await second.start()
    finally:
        await first.stop()
        await second.stop()


@pytest.mark.parametrize("mode", ["exit", "timeout", "cancel"])
async def test_failed_start_releases_child_and_lock(config: TalonConfig, mode: str) -> None:
    steel._BOOTSTRAP.write_text(
        "import os, signal, time\n"
        "from pathlib import Path\n"
        "Path('pid').write_text(str(os.getpid()))\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        + ("raise SystemExit(9)\n" if mode == "exit" else "time.sleep(60)\n")
    )
    browser = steel.SteelProcess(config)
    task = asyncio.create_task(browser.start())
    try:
        if mode == "cancel":
            async with asyncio.timeout(2):
                while not (browser.root / "pid").exists():  # noqa: ASYNC110  # Cross-process file signal.
                    await asyncio.sleep(0.01)
            task.cancel()
        with pytest.raises(asyncio.CancelledError if mode == "cancel" else RuntimeError):
            await task
        assert_gone(int((browser.root / "pid").read_text()))
        other = steel.SteelProcess(config)
        try:
            other._acquire()
        finally:
            await other.stop()
    finally:
        await browser.stop()


async def test_dirty_profile_is_preserved(config: TalonConfig) -> None:
    browser = steel.SteelProcess(config)
    profile = browser.root / "profile"
    profile.mkdir(parents=True)
    (profile / ".talon-dirty").write_text("active")
    with pytest.raises(RuntimeError, match="unclean shutdown"):
        await browser.start()
    assert (profile / ".talon-dirty").read_text() == "active"
    assert browser._lock is None


async def test_missing_setup_does_not_spawn(config: TalonConfig) -> None:
    browser = steel.SteelProcess(config)
    (browser.source / ".talon-prepared.json").unlink()
    with pytest.raises(RuntimeError, match="not prepared"):
        await browser.start()
    assert not browser.root.exists()


async def test_host_start_failure_stops_browser(config: TalonConfig) -> None:
    class FailingAgent(EchoAgentRuntime):
        async def start(self) -> None:
            msg = "agent failed"
            raise RuntimeError(msg)

    host = TalonHost(config=config, agent=FailingAgent())
    with pytest.raises(RuntimeError, match="agent failed"):
        await host.start()
    assert_gone(int((config.home / "browser/pid").read_text()))


async def test_child_crash_stops_host(config: TalonConfig) -> None:
    host = TalonHost(config=config, agent=EchoAgentRuntime())
    task = asyncio.create_task(host.run_until_stopped())
    try:
        async with asyncio.timeout(2):
            while not host.running:  # noqa: ASYNC110  # Host has no public ready event.
                await asyncio.sleep(0.01)
        os.kill(int((config.home / "browser/pid").read_text()), signal.SIGKILL)
        await asyncio.wait_for(task, 2)
        assert not host.running
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_shutdown_during_host_start(config: TalonConfig) -> None:
    entered = asyncio.Event()

    class SlowAgent(EchoAgentRuntime):
        async def start(self) -> None:
            entered.set()
            await asyncio.Event().wait()

    host = TalonHost(config=config, agent=SlowAgent())
    task = asyncio.create_task(host.run_until_stopped())
    try:
        await asyncio.wait_for(entered.wait(), 2)
        host.request_shutdown()
        await asyncio.wait_for(task, 2)
        assert_gone(int((config.home / "browser/pid").read_text()))
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
