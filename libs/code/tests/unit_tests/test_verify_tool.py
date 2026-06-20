"""Unit tests for the `verify_implementation` tool helpers and factory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from deepagents_code import verify_tool as vt

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol
    from langchain.tools import ToolRuntime
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import StructuredTool


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
@dataclass
class _FakeRead:
    file_data: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class _FakeGlob:
    matches: list[dict[str, Any]] | None = None


class _FakeBackend:
    """Minimal backend over an in-memory tree (glob/read/aglob/aread)."""

    def __init__(
        self, files: dict[str, dict[str, Any]], dirs: set[str] | None = None
    ) -> None:
        self._files = files  # path -> file_data dict ({"content", "encoding"})
        self._dirs = dirs or set()

    def glob(self, pattern: str, path: str | None = None) -> _FakeGlob:  # noqa: ARG002
        matches: list[dict[str, Any]] = [
            {"path": d, "is_dir": True} for d in self._dirs
        ]
        matches += [{"path": p, "is_dir": False} for p in self._files]
        return _FakeGlob(matches=matches)

    async def aglob(self, pattern: str, path: str | None = None) -> _FakeGlob:
        return self.glob(pattern, path)

    def read(self, file_path: str) -> _FakeRead:
        if file_path not in self._files:
            return _FakeRead(error="file_not_found")
        return _FakeRead(file_data=self._files[file_path])

    async def aread(self, file_path: str) -> _FakeRead:
        return self.read(file_path)


class _FakeRuntime:
    def __init__(self, messages: list[Any]) -> None:
        self.state = {"messages": messages}
        self.tool_call_id = "call_1"
        self.config: dict[str, Any] = {}


class _FakeModel:
    """Records the messages it was invoked with and returns a canned reply."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.last_messages: list[Any] | None = None

    def invoke(self, messages: list[Any]) -> AIMessage:
        self.last_messages = messages
        return AIMessage(content=self.reply)

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        return self.invoke(messages)


_PROVIDER_ERROR = RuntimeError("provider exploded")


class _RaisingModel:
    def invoke(self, _messages: list[Any]) -> AIMessage:
        raise _PROVIDER_ERROR

    async def ainvoke(self, _messages: list[Any]) -> AIMessage:
        raise _PROVIDER_ERROR


# --------------------------------------------------------------------------- #
# Helpers (cast fakes to the concrete types the production code expects)
# --------------------------------------------------------------------------- #
def _runtime(*messages: Any) -> ToolRuntime:
    return cast("ToolRuntime", _FakeRuntime(list(messages)))


def _human(text: str) -> ToolRuntime:
    return _runtime(HumanMessage(content=text))


def _make(model: object, backend: object) -> StructuredTool:
    tool = vt.make_verify_tool(
        cast("BaseChatModel", model), cast("BackendProtocol", backend), cwd="/app"
    )
    return cast("StructuredTool", tool)


def _call(tool: StructuredTool, runtime: ToolRuntime) -> str:
    assert tool.func is not None
    return cast("str", tool.func(runtime))


async def _acall(tool: StructuredTool, runtime: ToolRuntime) -> str:
    assert tool.coroutine is not None
    return cast("str", await tool.coroutine(runtime))


def _kv_proto(field: str) -> dict[str, dict[str, Any]]:
    content = f"message SetValRequest {{ string key = 1; int32 {field} = 2; }}"
    return {"/app/kv.proto": {"content": content, "encoding": "utf-8"}}


# --------------------------------------------------------------------------- #
# _content_to_text
# --------------------------------------------------------------------------- #
def test_content_to_text_plain_str() -> None:
    assert vt._content_to_text("hello") == "hello"


def test_content_to_text_list_of_blocks() -> None:
    content = [
        {"type": "text", "text": "first"},
        {"type": "image", "url": "..."},
        {"type": "text", "text": "second"},
    ]
    assert vt._content_to_text(content) == "first\nsecond"


def test_content_to_text_other() -> None:
    assert vt._content_to_text(123) == "123"


# --------------------------------------------------------------------------- #
# _original_task
# --------------------------------------------------------------------------- #
def test_original_task_first_human() -> None:
    runtime = _human("build a KV store with a value field")
    assert "value field" in vt._original_task(runtime)


def test_original_task_skips_leading_system() -> None:
    runtime = _runtime(
        SystemMessage(content="sys"), HumanMessage(content="the real task")
    )
    assert vt._original_task(runtime) == "the real task"


def test_original_task_handles_block_content() -> None:
    runtime = _runtime(HumanMessage(content=[{"type": "text", "text": "blocky task"}]))
    assert vt._original_task(runtime) == "blocky task"


def test_original_task_empty() -> None:
    assert vt._original_task(_runtime()) == ""


# --------------------------------------------------------------------------- #
# eligibility / filtering
# --------------------------------------------------------------------------- #
def test_eligible_paths_filters_and_orders() -> None:
    matches = [
        {"path": "/app/kv.proto", "is_dir": False},
        {"path": "/app/server.py", "is_dir": False},
        {"path": "/app/__pycache__/x.pyc", "is_dir": False},
        {"path": "/app/sub/deep.py", "is_dir": False},
        {"path": "/app", "is_dir": True},
        {"path": "/large_tool_results/blob.txt", "is_dir": False},
        {"path": "/app/data.bin", "is_dir": False},  # non-text ext
    ]
    paths = vt._eligible_paths(matches, focus=None)
    assert paths == ["/app/kv.proto", "/app/server.py", "/app/sub/deep.py"]


def test_eligible_paths_focus_first() -> None:
    matches = [
        {"path": "/app/server.py", "is_dir": False},
        {"path": "/app/kv.proto", "is_dir": False},
    ]
    paths = vt._eligible_paths(matches, focus="proto")
    assert paths[0] == "/app/kv.proto"


# --------------------------------------------------------------------------- #
# _file_chunk
# --------------------------------------------------------------------------- #
def test_file_chunk_skips_binary() -> None:
    chunk = vt._file_chunk("/a.png", {"content": "x", "encoding": "base64"}, 1000)
    assert chunk is None


def test_file_chunk_truncates() -> None:
    chunk = vt._file_chunk("/a.txt", {"content": "x" * 50, "encoding": "utf-8"}, 10)
    assert chunk is not None
    assert "[truncated]" in chunk
    assert chunk.startswith("### FILE: /a.txt")


def test_file_chunk_legacy_list_content() -> None:
    chunk = vt._file_chunk("/a.txt", {"content": ["l1", "l2"]}, 1000)
    assert chunk is not None
    assert "l1\nl2" in chunk


# --------------------------------------------------------------------------- #
# _preflight / _format_report
# --------------------------------------------------------------------------- #
def test_preflight_empty_spec() -> None:
    out = vt._preflight("", 3)
    assert out is not None
    assert "INCOMPLETE" in out


def test_preflight_no_files() -> None:
    out = vt._preflight("spec", 0)
    assert out is not None
    assert "VERDICT: FAIL" in out


def test_preflight_ok() -> None:
    assert vt._preflight("spec", 2) is None


def test_format_report_injects_missing_verdict() -> None:
    out = vt._format_report("some checker text without a verdict")
    assert "VERDICT: FAIL" in out


def test_format_report_keeps_existing_verdict() -> None:
    out = vt._format_report("MATCH x\nVERDICT: PASS")
    assert out == "MATCH x\nVERDICT: PASS"


# --------------------------------------------------------------------------- #
# end-to-end tool behaviour (call func/coroutine directly with a fake runtime,
# since ToolRuntime is framework-injected in production)
# --------------------------------------------------------------------------- #
def test_tool_factory_returns_named_tool() -> None:
    tool = _make(_FakeModel("VERDICT: PASS"), _FakeBackend({}))
    assert tool.name == "verify_implementation"
    assert "WHEN TO CALL" in tool.description


def test_verify_passes_verbatim_spec_and_files_to_judge() -> None:
    # The buggy proto uses `val`; the judge must receive the verbatim spec
    # (which says `value`) AND the actual file content (which says `val`).
    model = _FakeModel("MISMATCH: expected `value`, found `val`\nVERDICT: FAIL")
    tool = _make(model, _FakeBackend(_kv_proto("val")))

    runtime = _human("Create SetValRequest with a key (string) and a value (int).")
    report = _call(tool, runtime)

    assert model.last_messages is not None
    human = model.last_messages[1].content
    assert "a value (int)" in human  # verbatim spec word, not paraphrased
    assert "int32 val = 2" in human  # actual file content
    assert "VERDICT: FAIL" in report


def test_verify_no_files_fails_without_calling_judge() -> None:
    model = _FakeModel("VERDICT: PASS")  # would wrongly pass if it were called
    tool = _make(model, _FakeBackend({}))

    report = _call(tool, _human("spec naming /app/result.txt"))

    assert "VERDICT: FAIL" in report
    assert model.last_messages is None  # judge never invoked


def test_verify_judge_error_never_auto_passes() -> None:
    files = {"/app/x.py": {"content": "print(1)", "encoding": "utf-8"}}
    tool = _make(_RaisingModel(), _FakeBackend(files))

    report = _call(tool, _human("do a thing"))

    assert "INCOMPLETE" in report
    assert "PASS" not in report


def test_verify_empty_spec_skips_judge() -> None:
    files = {"/app/x.py": {"content": "print(1)", "encoding": "utf-8"}}
    model = _FakeModel("VERDICT: PASS")
    tool = _make(model, _FakeBackend(files))

    report = _call(tool, _runtime())

    assert "INCOMPLETE" in report
    assert model.last_messages is None


async def test_averify_matches_sync_behaviour() -> None:
    model = _FakeModel("MATCH all\nVERDICT: PASS")
    tool = _make(model, _FakeBackend(_kv_proto("value")))

    report = await _acall(tool, _human("need int32 value"))

    assert model.last_messages is not None
    assert "VERDICT: PASS" in report
    assert "int32 value = 2" in model.last_messages[1].content
