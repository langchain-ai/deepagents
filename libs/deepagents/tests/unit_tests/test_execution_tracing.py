"""Script tracing and local execution context propagation."""

import asyncio
import json
import shlex
import sys
from pathlib import Path
from typing import Literal
from unittest.mock import Mock
from uuid import uuid4

import pytest
from langchain.tools import ToolRuntime
from langchain_core.tracers.langchain import LangChainTracer
from langsmith import Client, traceable, tracing_context
from langsmith.run_helpers import get_current_run_tree
from langsmith.run_trees import RunTree

from deepagents.backends import LocalShellBackend
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.tracing import execution_tracing

_CONTEXT_ENV = "DEEPAGENTS_TRACE_CONTEXT"


@pytest.fixture
def client() -> Mock:
    return Mock(spec=Client)


def _context(parent: RunTree, *, enabled: bool | Literal["local"] = True) -> str:
    return json.dumps({"parent": parent.dotted_order, "project": parent.session_name, "enabled": enabled})


@pytest.mark.parametrize("fail", [False, True])
def test_nested_operations_and_flush(monkeypatch: pytest.MonkeyPatch, client: Mock, *, fail: bool) -> None:
    parent = RunTree(name="execute", project_name="script-demo", ls_client=client)
    monkeypatch.setenv(_CONTEXT_ENV, _context(parent))
    spans = []

    @traceable
    def inner() -> None:
        spans.append(get_current_run_tree())
        if fail:
            msg = "script failure"
            raise RuntimeError(msg)

    @traceable
    def outer() -> None:
        spans.append(get_current_run_tree())
        inner()

    with tracing_context(enabled=False):
        if fail:
            with pytest.raises(RuntimeError, match="script failure"), execution_tracing(client=client):
                outer()
        else:
            with execution_tracing(client=client):
                outer()
        assert get_current_run_tree() is None
    assert spans[0].parent_run_id == parent.id
    assert spans[1].parent_run_id == spans[0].id
    assert all(span.trace_id == parent.trace_id and span.session_name == "script-demo" for span in spans)
    assert bool(spans[1].error) is fail
    client.flush.assert_called_once_with(timeout=5)


@pytest.mark.parametrize("enabled", [False, "local"])
def test_disabled_and_local(monkeypatch: pytest.MonkeyPatch, client: Mock, *, enabled: bool | Literal["local"]) -> None:
    parent = RunTree(name="execute", ls_client=client)
    payload = _context(parent, enabled=enabled) if enabled else json.dumps({"enabled": False})
    monkeypatch.setenv(_CONTEXT_ENV, payload)

    @traceable
    def operation() -> RunTree | None:
        return get_current_run_tree()

    with tracing_context(enabled=True), execution_tracing(client=client):
        run = operation()
    assert (run is not None) is bool(enabled)
    client.create_run.assert_not_called()
    client.flush.assert_not_called()


@pytest.mark.parametrize("payload", ["secret-invalid-json", "[]", '{"enabled": 1}', '{"parent":"secret","project":"p","enabled":true}', "x" * 17000])
def test_malformed_context(monkeypatch: pytest.MonkeyPatch, client: Mock, payload: str) -> None:
    monkeypatch.setenv(_CONTEXT_ENV, payload)
    with pytest.raises(ValueError, match=r"^Invalid Deep Agents execution trace context\.$"), execution_tracing(client=client):
        pytest.fail("Invalid context must not run the block")
    assert not client.mock_calls


def test_missing_context_is_noop(monkeypatch: pytest.MonkeyPatch, client: Mock) -> None:
    monkeypatch.delenv(_CONTEXT_ENV, raising=False)
    with execution_tracing(client=client):
        pass
    assert not client.mock_calls


def test_environment_is_per_call_and_allowlisted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, client: Mock) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "not-for-the-child")
    env = {"EXPLICIT": "kept"}
    backend = LocalShellBackend(root_dir=tmp_path, env=env, propagate_trace_context=True)
    command = f"{shlex.quote(sys.executable)} -c 'import os,json; print(json.dumps(dict(os.environ)))'"
    parents = [RunTree(name="execute", ls_client=client), RunTree(name="execute", ls_client=client)]
    for parent in parents:
        with tracing_context(parent=parent, enabled=True, metadata={"secret": "not-for-the-child"}):
            result = backend.execute(command)
        assert result.exit_code == 0
        received = json.loads(result.output)
        payload = json.loads(received[_CONTEXT_ENV])
        assert payload == {"parent": parent.dotted_order, "project": parent.session_name, "enabled": True}
        assert "LANGSMITH_API_KEY" not in received
        assert received["EXPLICIT"] == "kept"
    assert env == {"EXPLICIT": "kept"}
    assert _CONTEXT_ENV not in backend._env


@pytest.mark.parametrize("enabled", [False, "local"])
def test_backend_tracing_mode(tmp_path: Path, client: Mock, *, enabled: bool | Literal["local"]) -> None:
    backend = LocalShellBackend(root_dir=tmp_path, propagate_trace_context=True)
    command = f"{shlex.quote(sys.executable)} -c " + shlex.quote(f"import os; print(os.environ[{_CONTEXT_ENV!r}])")
    parent = RunTree(name="execute", ls_client=client)
    with tracing_context(parent=parent, enabled=enabled):
        result = backend.execute(command)
    assert json.loads(result.output)["enabled"] == enabled


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_real_execute_parent(tmp_path: Path, client: Mock, *, asynchronous: bool) -> None:
    script = tmp_path / "script.py"
    script.write_text(
        "import json\n"
        "from unittest.mock import Mock\n"
        "from langsmith import Client, traceable\n"
        "from langsmith.run_helpers import get_current_run_tree\n"
        "from deepagents.tracing import execution_tracing\n"
        "@traceable\n"
        "def operation():\n"
        "    run = get_current_run_tree()\n"
        "    return {'parent': str(run.parent_run_id), 'trace': str(run.trace_id)}\n"
        "with execution_tracing(client=Mock(spec=Client)):\n"
        "    print(json.dumps(operation()))\n"
    )
    backend = LocalShellBackend(root_dir=tmp_path, propagate_trace_context=True)
    tool = next(tool for tool in FilesystemMiddleware(backend=backend).tools if tool.name == "execute")
    tracer = LangChainTracer(client=client, project_name="script-demo")
    runtime = ToolRuntime(state={}, context=None, tool_call_id="call", store=None, stream_writer=lambda _: None, config={})
    parent = RunTree(name="agent", ls_client=client)
    tool_id = uuid4()
    args = {"command": f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}", "runtime": runtime}
    with tracing_context(parent=parent, enabled=True, client=client):
        if asynchronous:
            result = await tool.ainvoke(args, config={"callbacks": [tracer], "run_id": tool_id})
        else:
            result = tool.invoke(args, config={"callbacks": [tracer], "run_id": tool_id})
    assert result.artifact["exit_code"] == 0, result.content
    child = json.loads(result.content.splitlines()[0])
    assert child["parent"] == str(tool_id)
    assert child["trace"] == str(parent.trace_id)


def test_default_backend_does_not_propagate(tmp_path: Path, client: Mock) -> None:
    backend = LocalShellBackend(root_dir=tmp_path)
    command = f"{shlex.quote(sys.executable)} -c " + shlex.quote(f"import os; print({_CONTEXT_ENV!r} in os.environ)")
    with tracing_context(parent=RunTree(name="execute", ls_client=client), enabled=True):
        result = backend.execute(command)
    assert result.output.strip() == "False"


def test_flush_failure_preserves_script_error(monkeypatch: pytest.MonkeyPatch, client: Mock) -> None:
    monkeypatch.setenv(_CONTEXT_ENV, _context(RunTree(name="execute", ls_client=client)))
    client.flush.side_effect = RuntimeError("telemetry failure")
    msg = "application failure"
    with pytest.raises(ValueError, match="application failure"), execution_tracing(client=client):
        raise ValueError(msg)


async def test_concurrent_execution_contexts(tmp_path: Path, client: Mock) -> None:
    backend = LocalShellBackend(root_dir=tmp_path, propagate_trace_context=True)
    command = f"{shlex.quote(sys.executable)} -c " + shlex.quote(f"import os; print(os.environ[{_CONTEXT_ENV!r}])")
    parents = [RunTree(name="execute", ls_client=client), RunTree(name="execute", ls_client=client)]

    async def run(parent: RunTree) -> str:
        with tracing_context(parent=parent, enabled=True):
            response = await backend.aexecute(command)
            return json.loads(response.output)["parent"]

    assert await asyncio.gather(*(run(parent) for parent in parents)) == [parent.dotted_order for parent in parents]
