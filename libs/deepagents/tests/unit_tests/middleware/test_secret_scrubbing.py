"""Tests for the secret-scrubbing helpers in `filesystem` middleware.

These guard against API-key values leaking into `execute` tool results.
Synthetic, clearly non-real key strings are used throughout — do NOT put any
real key bytes in tests.
"""

from langchain.agents.middleware.types import ToolCallRequest
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage

from deepagents.backends import StateBackend
from deepagents.middleware.filesystem import (
    _SECRET_PLACEHOLDER,
    FilesystemMiddleware,
    _scrub_secrets,
    _scrub_tool_message,
)

# Synthetic non-real samples — clearly fake but matching each pattern shape.
_FAKE_SK_PROJ = "sk-proj-" + "A" * 30
_FAKE_SK_ANT = "sk-ant-" + "B" * 30
_FAKE_LSV2_PT = "lsv2_pt_" + "C" * 30
_FAKE_SK_GENERIC = "sk-" + "D" * 45


def test_scrub_sk_proj_key() -> None:
    text = f"leaked={_FAKE_SK_PROJ} trailing"
    out = _scrub_secrets(text)
    assert _FAKE_SK_PROJ not in out
    assert _SECRET_PLACEHOLDER in out


def test_scrub_sk_ant_key() -> None:
    text = f"key={_FAKE_SK_ANT}"
    out = _scrub_secrets(text)
    assert _FAKE_SK_ANT not in out
    assert _SECRET_PLACEHOLDER in out


def test_scrub_lsv2_pt_key() -> None:
    text = f"token={_FAKE_LSV2_PT}"
    out = _scrub_secrets(text)
    assert _FAKE_LSV2_PT not in out
    assert _SECRET_PLACEHOLDER in out


def test_scrub_generic_sk_key() -> None:
    text = f"OPENAI_API_KEY={_FAKE_SK_GENERIC}"
    out = _scrub_secrets(text)
    assert _FAKE_SK_GENERIC not in out
    assert _SECRET_PLACEHOLDER in out


def test_scrub_benign_passthrough() -> None:
    text = "no secrets here, just a regular log line with sk- but too short"
    assert _scrub_secrets(text) == text


def test_scrub_handles_empty_and_non_string() -> None:
    assert _scrub_secrets("") == ""
    # _scrub_secrets is typed for str but defensively returns non-strings unchanged.
    assert _scrub_secrets(None) is None  # type: ignore[arg-type]


def test_scrub_tool_message_string_content() -> None:
    msg = ToolMessage(content=f"export OPENAI_API_KEY={_FAKE_SK_PROJ}", tool_call_id="t1")
    scrubbed = _scrub_tool_message(msg)
    assert isinstance(scrubbed.content, str)
    assert _FAKE_SK_PROJ not in scrubbed.content
    assert _SECRET_PLACEHOLDER in scrubbed.content


def test_scrub_tool_message_list_content() -> None:
    msg = ToolMessage(
        content=[{"type": "text", "text": f"value={_FAKE_SK_ANT}"}],
        tool_call_id="t2",
    )
    scrubbed = _scrub_tool_message(msg)
    assert isinstance(scrubbed.content, list)
    block = scrubbed.content[0]
    assert isinstance(block, dict)
    assert _FAKE_SK_ANT not in block["text"]
    assert _SECRET_PLACEHOLDER in block["text"]


def _runtime(tool_call_id: str = "") -> ToolRuntime:
    return ToolRuntime(
        state={},
        context=None,
        tool_call_id=tool_call_id,
        store=None,
        stream_writer=lambda _: None,
        config={},
    )


def test_wrap_tool_call_scrubs_execute_result() -> None:
    middleware = FilesystemMiddleware(backend=StateBackend())
    leaked = f"OPENAI_API_KEY={_FAKE_SK_PROJ}\n[Command succeeded with exit code 0]"
    leaked_msg = ToolMessage(content=leaked, tool_call_id="t3", name="execute")

    def handler(request: ToolCallRequest) -> ToolMessage:  # noqa: ARG001
        return leaked_msg

    request = ToolCallRequest(
        runtime=_runtime("t3"),
        tool_call={"id": "t3", "name": "execute", "args": {"command": "env"}},
        state={},
        tool=None,
    )

    result = middleware.wrap_tool_call(request, handler)
    assert isinstance(result, ToolMessage)
    assert _FAKE_SK_PROJ not in result.content
    assert _SECRET_PLACEHOLDER in result.content


def test_wrap_tool_call_skips_scrub_for_non_execute() -> None:
    """Non-execute tools are not scrubbed (scrubber is scoped to `execute`)."""
    middleware = FilesystemMiddleware(backend=StateBackend())
    content = f"file contents include {_FAKE_SK_PROJ}"
    msg = ToolMessage(content=content, tool_call_id="t4", name="read_file")

    def handler(request: ToolCallRequest) -> ToolMessage:  # noqa: ARG001
        return msg

    request = ToolCallRequest(
        runtime=_runtime("t4"),
        tool_call={"id": "t4", "name": "read_file", "args": {"file_path": "/x"}},
        state={},
        tool=None,
    )

    result = middleware.wrap_tool_call(request, handler)
    assert isinstance(result, ToolMessage)
    assert _FAKE_SK_PROJ in result.content
