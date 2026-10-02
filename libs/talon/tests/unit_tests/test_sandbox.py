from __future__ import annotations

import asyncio
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING

import pytest
from deepagents.backends import StateBackend
from deepagents.backends.protocol import ExecuteResponse, FileDownloadResponse, FileUploadResponse
from deepagents.backends.sandbox import BaseSandbox

from deepagents_code.integrations import sandbox_factory
from deepagents_talon.config import TalonConfig
from deepagents_talon.runtime import DeepAgentRuntime
from deepagents_talon.sandbox import SandboxStartupError, open_sandbox, sandbox_backend

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


class _FakeSandbox(BaseSandbox):
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.files: dict[str, bytes] = {}

    @property
    def id(self) -> str:
        return "fake-sandbox"

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:  # noqa: ARG002
        self.commands.append(command)
        return ExecuteResponse(output="sandboxed", exit_code=0)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        self.files.update(files)
        return [FileUploadResponse(path=path) for path, _ in files]

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        return [FileDownloadResponse(path=path, error="file_not_found") for path in paths]


def _config(tmp_path: Path, **env: str) -> TalonConfig:
    return TalonConfig.from_env(env, base_home=tmp_path)


def test_sandbox_unset_by_default(tmp_path: Path) -> None:
    assert _config(tmp_path).sandbox is None
    assert _config(tmp_path, DEEPAGENTS_TALON_SANDBOX="  ").sandbox is None


def test_sandbox_settings_from_env(tmp_path: Path) -> None:
    settings = _config(
        tmp_path,
        DEEPAGENTS_TALON_SANDBOX=" langsmith ",
        DEEPAGENTS_TALON_SANDBOX_ID="sbx-1",
        DEEPAGENTS_TALON_SANDBOX_SETUP="/opt/setup.sh",
    ).sandbox

    assert settings is not None
    assert settings.provider == "langsmith"
    assert settings.sandbox_id == "sbx-1"
    assert settings.snapshot is None
    assert settings.setup_script == "/opt/setup.sh"


@pytest.mark.parametrize(
    ("talon_env", "process_env", "expected"),
    [
        ({}, {}, "talon-bot"),
        ({"DEEPAGENTS_TALON_SANDBOX_SNAPSHOT": "mine"}, {}, "mine"),
        ({}, {"LANGSMITH_SANDBOX_SNAPSHOT_NAME": "shared"}, None),
        ({}, {"DEEPAGENTS_CODE_LANGSMITH_SANDBOX_SNAPSHOT_NAME": "shared"}, None),
        ({"DEEPAGENTS_TALON_SANDBOX_ID": "sbx"}, {}, None),
        ({"DEEPAGENTS_TALON_SANDBOX": "daytona"}, {}, None),
    ],
)
async def test_langsmith_snapshot_defaults_to_assistant_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    talon_env: dict[str, str],
    process_env: dict[str, str],
    expected: str | None,
) -> None:
    for key in (
        "LANGSMITH_SANDBOX_SNAPSHOT_NAME",
        "DEEPAGENTS_CODE_LANGSMITH_SANDBOX_SNAPSHOT_NAME",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in process_env.items():
        monkeypatch.setenv(key, value)
    requested: list[object] = []

    @contextmanager
    def create_sandbox(_provider: str, **kwargs: object) -> Iterator[_FakeSandbox]:
        requested.append(kwargs["snapshot_name"])
        yield _FakeSandbox()

    _patch_factory(monkeypatch, create_sandbox)
    env = {"DEEPAGENTS_TALON_SANDBOX": "langsmith", **talon_env, **process_env}
    config = _config(tmp_path, DEEPAGENTS_TALON_ASSISTANT_ID="bot", **env)

    async with open_sandbox(config):
        pass

    assert requested == [expected]


def test_backend_keeps_approvals_out_of_host_routes(tmp_path: Path) -> None:
    fake = _FakeSandbox()
    backend = sandbox_backend(fake, tmp_path)

    backend.write(f"{tmp_path}/memory/AGENTS.md", "remember")
    backend.write(f"{tmp_path}/tools.json", "{}")
    with pytest.raises(ValueError, match="traversal"):
        backend.write(f"{tmp_path}/memory/../tools.json", "{}")

    assert (tmp_path / "memory" / "AGENTS.md").read_text() == "remember"
    assert not (tmp_path / "tools.json").exists()
    assert f"{tmp_path}/tools.json" in fake.files
    assert backend.execute("uname").output == "sandboxed"
    assert fake.commands[-1] == "uname"


async def test_open_sandbox_yields_none_when_unset(tmp_path: Path) -> None:
    async with open_sandbox(_config(tmp_path)) as session:
        assert session is None


async def test_open_sandbox_cleans_up_on_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeSandbox()
    events: list[str] = []

    @contextmanager
    def create_sandbox(provider: str, **kwargs: object) -> Iterator[_FakeSandbox]:
        events.append(f"create:{provider}:{kwargs['sandbox_id']}")
        yield fake
        events.append("delete")

    _patch_factory(monkeypatch, create_sandbox)
    config = _config(tmp_path, DEEPAGENTS_TALON_SANDBOX="langsmith")

    async with open_sandbox(config) as session:
        assert session is not None
        assert session.working_dir == "/work"
        assert session.backend.execute("ls").output == "sandboxed"
        assert events == ["create:langsmith:None"]

    assert events == ["create:langsmith:None", "delete"]


async def test_cancelled_startup_deletes_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started, release, deleted = threading.Event(), threading.Event(), threading.Event()

    @contextmanager
    def create_sandbox(_provider: str, **_: object) -> Iterator[_FakeSandbox]:
        started.set()
        release.wait(5)
        yield _FakeSandbox()
        deleted.set()

    _patch_factory(monkeypatch, create_sandbox)
    config = _config(tmp_path, DEEPAGENTS_TALON_SANDBOX="langsmith")

    async def start() -> None:
        async with open_sandbox(config):
            pytest.fail("cancelled startup should not open a session")

    task = asyncio.create_task(start())
    await asyncio.to_thread(started.wait, 5)
    task.cancel()
    await asyncio.sleep(0)
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert await asyncio.to_thread(deleted.wait, 5)


async def test_open_sandbox_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    @contextmanager
    def create_sandbox(provider: str, **_: object) -> Iterator[_FakeSandbox]:
        msg = f"no credentials for {provider}"
        raise RuntimeError(msg)
        yield _FakeSandbox()

    _patch_factory(monkeypatch, create_sandbox)
    config = _config(tmp_path, DEEPAGENTS_TALON_SANDBOX="daytona")

    with pytest.raises(SandboxStartupError, match="no credentials for daytona"):
        async with open_sandbox(config):
            pytest.fail("sandbox session should not open")


def test_system_prompt_notes_sandbox(tmp_path: Path) -> None:
    runtime = DeepAgentRuntime(
        model="fake",
        system_prompt="Base prompt",
        backend=StateBackend(),
        assistant_dir=tmp_path,
        env={},
        sandbox_working_dir="/work",
    )

    prompt = runtime._resolve_system_prompt()

    assert prompt is not None
    assert prompt.startswith("Base prompt")
    assert "`/work`" in prompt


def _patch_factory(monkeypatch: pytest.MonkeyPatch, create_sandbox: object) -> None:
    monkeypatch.setattr(sandbox_factory, "create_sandbox", create_sandbox)
    monkeypatch.setattr(sandbox_factory, "get_default_working_dir", lambda _provider: "/work")


def test_sandbox_mode_keeps_outside_memory_paths_off_the_host(tmp_path: Path) -> None:
    outside = tmp_path / "outside" / "notes.md"
    inside = tmp_path / "memory" / "notes.md"
    runtime = DeepAgentRuntime(
        model="fake",
        backend=StateBackend(),
        assistant_dir=tmp_path,
        env={
            "DEEPAGENTS_TALON_MEMORY_PATHS": f"{outside}:{tmp_path}/memory/../tools.json:{inside}"
        },
        sandbox_working_dir="/work",
    )

    assert runtime._resolve_memory() == [str(inside)]
    assert not outside.exists()
    assert not (tmp_path / "tools.json").exists()
