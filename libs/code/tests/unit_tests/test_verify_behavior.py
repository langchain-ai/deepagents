"""Unit tests for the `verify_behavior` tool."""

import asyncio

from langchain_core.messages import HumanMessage

from deepagents_code.verify_behavior_tool import make_verify_behavior_tool

_TASK = "Save /app/dist.npy whose KL divergence is 10.0 (tolerance 0.001)."


class _ExecResp:
    def __init__(self, output: str, exit_code: int = 0) -> None:
        self.output = output
        self.exit_code = exit_code
        self.truncated = False


class _GlobResp:
    def __init__(self, matches: list[dict]) -> None:
        self.matches = matches


class _ReadResp:
    def __init__(self, content: str) -> None:
        self.error = None
        self.file_data = {"content": content, "encoding": "utf-8"}


class FakeBackend:
    """Backend stub exposing the glob/read/execute surface the tool uses."""

    def __init__(
        self,
        files: dict[str, str] | None = None,
        exec_output: str = "",
        execute_raises: bool = False,
    ) -> None:
        self._files = files or {"/app/solution.py": "print('hi')"}
        self._exec_output = exec_output
        self._execute_raises = execute_raises
        self.executed: list[str] = []

    def glob(self, pattern: str, path: str | None = None) -> _GlobResp:
        return _GlobResp([{"path": p, "is_dir": False} for p in self._files])

    async def aglob(self, pattern: str, path: str | None = None) -> _GlobResp:
        return self.glob(pattern, path)

    def read(self, path: str) -> _ReadResp:
        return _ReadResp(self._files.get(path, ""))

    async def aread(self, path: str) -> _ReadResp:
        return self.read(path)

    def execute(self, command: str, *, timeout: int | None = None) -> _ExecResp:
        if self._execute_raises:
            raise RuntimeError("boom")
        self.executed.append(command)
        return _ExecResp(self._exec_output)

    async def aexecute(self, command: str, *, timeout: int | None = None) -> _ExecResp:
        return self.execute(command, timeout=timeout)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeJudge:
    """Judge-model stub returning canned content and recording its prompts."""

    def __init__(self, content: str) -> None:
        self._content = content
        self.calls: list[list] = []

    def invoke(self, messages: list) -> _FakeResponse:
        self.calls.append(messages)
        return _FakeResponse(self._content)

    async def ainvoke(self, messages: list) -> _FakeResponse:
        return self.invoke(messages)


class FakeRuntime:
    def __init__(self, task: str = _TASK) -> None:
        self.state = {"messages": [HumanMessage(content=task)]}


def _make(judge_content: str, **backend_kwargs):
    backend = FakeBackend(**backend_kwargs)
    judge = FakeJudge(judge_content)
    tool = make_verify_behavior_tool(judge, backend, cwd="/app")
    return tool, backend, judge


def test_no_oracle_returns_incomplete() -> None:
    tool, backend, _ = _make("NO_ORACLE", exec_output="VERIFY: PASS")
    result = tool.func(FakeRuntime())
    assert "INCOMPLETE" in result
    assert "VERIFY: PASS" not in result
    # No test should have been executed when there is no oracle.
    assert backend.executed == []


def test_pass_verdict() -> None:
    tool, _, _ = _make(
        "cat > t.py <<'EOF'\n...\nEOF\npython3 t.py",
        exec_output="checked KL=10.0\nVERIFY: PASS",
    )
    result = tool.func(FakeRuntime())
    assert "VERIFY: PASS" in result
    assert "INCOMPLETE" not in result


def test_fail_verdict_includes_mismatch() -> None:
    tool, _, _ = _make(
        "python3 check.py",
        exec_output="expected 10.0, got 7.72\nVERIFY: FAIL",
    )
    result = tool.func(FakeRuntime())
    assert "VERIFY: FAIL" in result
    assert "7.72" in result


def test_no_sentinel_is_incomplete() -> None:
    tool, _, _ = _make("python3 check.py", exec_output="Traceback: ImportError")
    result = tool.func(FakeRuntime())
    assert "INCOMPLETE" in result


def test_execution_error_is_failsoft() -> None:
    tool, _, _ = _make("python3 check.py", execute_raises=True)
    result = tool.func(FakeRuntime())
    assert "INCOMPLETE" in result  # must not raise


def test_empty_task_is_incomplete() -> None:
    tool, _, _ = _make("python3 check.py", exec_output="VERIFY: PASS")
    result = tool.func(FakeRuntime(task=""))
    assert "INCOMPLETE" in result


def test_task_text_passed_to_judge() -> None:
    tool, _, judge = _make("NO_ORACLE")
    tool.func(FakeRuntime())
    # The original task must reach the judge verbatim, not be paraphrased away.
    sent = "\n".join(str(m.content) for m in judge.calls[0])
    assert "dist.npy" in sent


def test_fenced_script_is_stripped_and_run() -> None:
    tool, backend, _ = _make(
        "```bash\npython3 check.py\n```",
        exec_output="VERIFY: PASS",
    )
    result = tool.func(FakeRuntime())
    assert "VERIFY: PASS" in result
    assert backend.executed == ["python3 check.py"]


def test_async_path_pass() -> None:
    tool, _, _ = _make("python3 check.py", exec_output="VERIFY: PASS")
    result = asyncio.run(tool.coroutine(FakeRuntime()))
    assert "VERIFY: PASS" in result


def test_runtime_injected_through_ainvoke() -> None:
    """Regression: invoking via the real tool-call path must inject `runtime`.

    The other tests call `tool.coroutine(...)` directly, bypassing the schema-driven
    argument binding. A fieldless `args_schema` made langchain treat the tool as
    taking no args, dropping the framework-injected `ToolRuntime` and crashing with a
    missing-arg `TypeError`. Going through `ainvoke` exercises that binding.
    """
    from langchain.tools import ToolRuntime

    tool, _, _ = _make("python3 check.py", exec_output="VERIFY: PASS")
    runtime = ToolRuntime(
        state={"messages": [HumanMessage(content=_TASK)]},
        context=None,
        tool_call_id="verify-1",
        store=None,
        stream_writer=lambda _: None,
        config={},
    )
    result = asyncio.run(tool.ainvoke({"runtime": runtime}))
    assert "VERIFY: PASS" in result
    # The injected runtime must stay hidden from the model-facing schema.
    assert "runtime" not in tool.args
