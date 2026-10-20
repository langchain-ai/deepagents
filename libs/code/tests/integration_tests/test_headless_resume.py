"""End-to-end coverage for `dcode -r <id> -n ...` against a real local server.

Every dcode invocation is a fresh CLI subprocess with its own `HOME` and
`DEEPAGENTS_HOME`: the profile paths and sessions database are captured at
import, so an in-process run would leak into the developer's profile.
"""

from __future__ import annotations

import json
import os
import shlex
import sqlite3
import subprocess
import sys
import textwrap
from contextlib import closing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pathlib import Path

_CONFIG = """
[models]
default = "itest:fake"

[models.providers.itest]
class_path = "deepagents_code._testing_models:DeterministicIntegrationChatModel"
models = ["fake", "fake-alt"]
"""

# Reads (and optionally seeds) a thread through the real client. Registering
# the thread and updating its state do not bind the workspace, so this helper
# cannot disturb the binding the CLI runs check.
_STATE_HELPER = textwrap.dedent(
    """
    import asyncio
    import json
    import sys

    from langchain_core.messages import AIMessage


    async def main(thread_id: str, seed: str) -> None:
        from deepagents_code.client.launch.server_manager import server_session
        from deepagents_code.config import _ensure_bootstrap

        _ensure_bootstrap()

        async with server_session(
            assistant_id="agent",
            model_name="itest:fake",
            no_mcp=True,
            enable_shell=False,
            interactive=False,
            sandbox_type="none",
        ) as (agent, _proc):
            config = {"configurable": {"thread_id": thread_id}}
            await agent.aensure_thread(config)
            update = json.loads(seed)
            if tool_call := update.pop("tool_call", None):
                update["messages"] = [AIMessage(content="", tool_calls=[tool_call])]
            if update:
                await agent.aupdate_state(config, update, as_node="model")
            state = await agent.aget_state(config)
            values = dict(state.values) if state and state.values else {}
            messages = values.get("messages") or []
            print(json.dumps({
                "messages": [
                    {
                        "type": m.get("type"),
                        "content": m.get("content"),
                        "tool_calls": [c.get("id") for c in m.get("tool_calls") or []],
                        "tool_call_id": m.get("tool_call_id"),
                    }
                    for m in messages
                    if isinstance(m, dict)
                ],
                "model_spec": values.get("_model_spec"),
                "rubric": values.get("rubric"),
                "rubric_iterations": values.get("_rubric_iterations"),
                "rubric_evaluations": len(values.get("_rubric_evaluations") or []),
            }))


    asyncio.run(main(sys.argv[1], sys.argv[2]))
    """
)


@dataclass(frozen=True)
class _Run:
    code: int
    stdout: str
    stderr: str

    @property
    def thread_id(self) -> str:
        """The thread ID from the `Thread: <id>` line scripts parse."""
        ids = [
            line.removeprefix("Thread: ")
            for line in self.stderr.splitlines()
            if line.startswith("Thread: ")
        ]
        assert len(ids) == 1, self.stderr
        return ids[0]


class _Profile:
    """An isolated dcode profile plus a project directory."""

    def __init__(self, root: Path) -> None:
        self.home = root / "home"
        self.profile = root / "profile"
        self.project = root / "project"
        for path in (self.home, self.profile, self.project):
            path.mkdir()
        (self.profile / "config.toml").write_text(_CONFIG.lstrip())
        self.helper = root / "state_helper.py"
        self.helper.write_text(_STATE_HELPER)

    @property
    def db(self) -> Path:
        return self.profile / ".state" / "sessions.db"

    def _env(self) -> dict[str, str]:
        return {
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(self.home),
            "DEEPAGENTS_HOME": str(self.profile),
            "DEEPAGENTS_CODE_NO_UPDATE_CHECK": "1",
        }

    def dcode(self, *args: str, cwd: Path | None = None) -> _Run:
        result = subprocess.run(  # fixed argv built by the test
            [sys.executable, "-m", "deepagents_code", *args, "-q"],
            cwd=cwd or self.project,
            env=self._env(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        return _Run(result.returncode, result.stdout, result.stderr)

    def state(self, thread_id: str, **seed: Any) -> dict[str, Any]:
        result = subprocess.run(  # fixed argv built by the test
            [sys.executable, str(self.helper), thread_id, json.dumps(seed)],
            cwd=self.project,
            env=self._env(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
        return json.loads(result.stdout.strip().splitlines()[-1])

    def thread_count(self) -> int:
        with closing(sqlite3.connect(self.db)) as conn:
            row = conn.execute(
                "SELECT COUNT(DISTINCT thread_id) FROM checkpoints"
            ).fetchone()
        return int(row[0])


@pytest.fixture
def profile(tmp_path: Path) -> _Profile:
    return _Profile(tmp_path)


def _user_messages(state: dict[str, Any]) -> list[str]:
    return [m["content"] for m in state["messages"] if m["type"] == "human"]


@pytest.mark.timeout(300)
def test_script_resumes_thread_from_stderr_id(profile: _Profile) -> None:
    first = profile.dcode("-n", "first message")
    assert first.code == 0, first.stderr
    assert profile.db.is_file()
    thread_id = first.thread_id

    second = profile.dcode("-r", thread_id, "-n", "second message")

    assert second.code == 0, second.stderr
    assert second.thread_id == thread_id
    assert "first message" in second.stdout
    assert _user_messages(profile.state(thread_id)) == [
        "first message",
        "second message",
    ]
    assert profile.thread_count() == 1


@pytest.mark.timeout(300)
def test_resume_restores_model_unless_overridden(profile: _Profile) -> None:
    first = profile.dcode("-n", "first", "-M", "itest:fake-alt")
    assert first.code == 0, first.stderr
    thread_id = first.thread_id

    adopted = profile.dcode("-r", thread_id, "-n", "second")
    assert adopted.code == 0, adopted.stderr
    # `_model_spec` is rewritten after every successful call, so a failed
    # restore would record the launch default `itest:fake` here.
    assert profile.state(thread_id)["model_spec"] == "itest:fake-alt"

    overridden = profile.dcode("-r", thread_id, "-n", "third", "-M", "itest:fake")
    assert overridden.code == 0, overridden.stderr
    assert profile.state(thread_id)["model_spec"] == "itest:fake"


@pytest.mark.timeout(300)
def test_resume_clears_rubric_left_by_earlier_run(profile: _Profile) -> None:
    first = profile.dcode("-n", "first")
    assert first.code == 0, first.stderr
    thread_id = first.thread_id
    seeded = profile.state(thread_id, rubric="every answer cites a source")
    assert seeded["rubric"] == "every answer cites a source"

    resumed = profile.dcode("-r", thread_id, "-n", "second")

    assert resumed.code == 0, resumed.stderr
    after = profile.state(thread_id)
    assert not after["rubric"]
    assert after["rubric_evaluations"] == seeded["rubric_evaluations"]
    assert after["rubric_iterations"] == seeded["rubric_iterations"]


@pytest.mark.timeout(300)
def test_resume_fails_closed(profile: _Profile) -> None:
    first = profile.dcode("-n", "first")
    assert first.code == 0, first.stderr
    thread_id = first.thread_id

    unknown = profile.dcode("-r", "not-a-thread", "-n", "x")
    assert unknown.code == 1
    assert "'not-a-thread' not found" in unknown.stderr

    elsewhere = profile.dcode("-r", thread_id, "-n", "x", cwd=profile.home)
    assert elsewhere.code == 1
    assert "Run from that directory" in elsewhere.stderr

    other_flags = profile.dcode("-r", thread_id, "-n", "x", "--shell-allow-list", "ls")
    assert other_flags.code == 1
    assert "created with different settings" in other_flags.stderr

    assert profile.thread_count() == 1
    assert _user_messages(profile.state(thread_id)) == ["first"]


@pytest.mark.timeout(300)
def test_resume_refuses_thread_open_elsewhere(profile: _Profile) -> None:
    from deepagents_code.thread_ownership import try_acquire

    first = profile.dcode("-n", "first")
    assert first.code == 0, first.stderr
    thread_id = first.thread_id

    lease = try_acquire(thread_id, db_path=profile.db)
    assert lease is not None
    try:
        blocked = profile.dcode("-r", thread_id, "-n", "second")
    finally:
        lease.release()
    assert blocked.code == 1
    assert "open elsewhere" in blocked.stderr

    resumed = profile.dcode("-r", thread_id, "-n", "second")
    assert resumed.code == 0, resumed.stderr


@pytest.mark.timeout(300)
def test_resume_abandons_unfinished_tool_call(profile: _Profile) -> None:
    first = profile.dcode("-n", "first")
    assert first.code == 0, first.stderr
    thread_id = first.thread_id
    target = profile.project / "abandoned.txt"
    profile.state(
        thread_id,
        tool_call={
            "name": "write_file",
            "args": {"file_path": str(target), "content": "should not exist"},
            "id": "call-abandoned",
        },
    )

    resumed = profile.dcode("-r", thread_id, "-n", "second")

    assert resumed.code == 0, resumed.stderr
    assert not target.exists()
    messages = profile.state(thread_id)["messages"]
    answered = {m["tool_call_id"] for m in messages if m["type"] == "tool"}
    requested = {call for m in messages for call in m["tool_calls"]}
    assert "call-abandoned" in requested
    assert requested <= answered
    assert _user_messages({"messages": messages})[-1] == "second"


@pytest.mark.timeout(300)
def test_resume_refuses_thread_deleted_during_startup(profile: _Profile) -> None:
    first = profile.dcode("-n", "first")
    assert first.code == 0, first.stderr
    thread_id = first.thread_id
    delete = shlex.join(
        [sys.executable, "-m", "deepagents_code", "threads", "delete", thread_id]
    )

    resumed = profile.dcode("-r", thread_id, "-n", "second", "--startup-cmd", delete)

    assert resumed.code == 1
    assert "was deleted before it could be resumed" in resumed.stderr
    assert profile.thread_count() == 0
