"""Tests for the GLM-5.2 Deep Agents Code harness profile."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, cast

import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from deepagents_code import _glm_5p2_profile as glm_profile
from deepagents_code._glm_5p2_profile import _GlmReadFileMediaGuard

if TYPE_CHECKING:
    from deepagents.profiles import HarnessProfile


def _request(name: str = "read_file") -> ToolCallRequest:
    return ToolCallRequest(
        runtime=cast("Any", None),
        tool_call={
            "id": "call-media",
            "name": name,
            "args": {"file_path": "/private/input/secret.png"},
        },
        state={},
        tool=None,
    )


def _media_message(
    *, block_type: str = "image", mime: str = "image/png"
) -> ToolMessage:
    return ToolMessage(
        content=[
            {"type": "text", "text": "confidential caption"},
            {
                "type": block_type,
                "base64": "c2VjcmV0LW1lZGlhLXBheWxvYWQ=",
                "mime_type": mime,
            },
        ],
        name="read_file",
        tool_call_id="call-media",
        additional_kwargs={"read_file_path": "/private/input/secret.png"},
        status="success",
    )


def _assert_generic_media_error(result: ToolMessage | Command[Any]) -> None:
    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert result.name == "read_file"
    assert result.tool_call_id == "call-media"
    assert result.additional_kwargs == {}

    text = result.text.lower()
    assert "shell" in text
    assert "script" in text
    for sensitive in (
        "/private/input/secret.png",
        "image/png",
        "application/pdf",
        "c2VjcmV0LW1lZGlhLXBheWxvYWQ=",
        "confidential caption",
    ):
        assert sensitive not in str(result)


def test_registration_is_exact_and_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, HarnessProfile]] = []

    def register(key: str, profile: HarnessProfile) -> None:
        calls.append((key, profile))

    monkeypatch.setattr(glm_profile, "_glm_5p2_profile_registered", False)
    monkeypatch.setattr(glm_profile, "register_harness_profile", register)

    glm_profile._ensure_glm_5p2_profile_registered()
    glm_profile._ensure_glm_5p2_profile_registered()

    assert tuple(key for key, _ in calls) == (
        "fireworks:accounts/fireworks/models/glm-5p2",
        "openrouter:z-ai/glm-5.2",
        "baseten:zai-org/GLM-5.2",
    )
    assert all(profile is glm_profile._GLM_5P2_PROFILE for _, profile in calls)


def test_prompt_is_concise_and_execution_focused() -> None:
    suffix = glm_profile._SYSTEM_PROMPT_SUFFIX

    assert suffix.startswith("<glm_5p2_execution>\n")
    assert suffix.endswith("\n</glm_5p2_execution>")
    assert 150 <= len(suffix.split()) <= 260

    required_phrases = (
        "Do not call `read_file` on images, PDFs, audio, or video",
        "Do not reopen generated media for visual inspection",
        "Create the requested artifact first",
        "Treat supplied inputs as immutable",
        "preserve their fidelity",
        "one retry",
        "pivot",
        "Run task-named checks",
        "Stop immediately",
    )
    for phrase in required_phrases:
        assert phrase in suffix

    for omitted in (
        "write_todos",
        "<todo_rules>",
        "<tool_preferences>",
        "Prefer specialized tools",
    ):
        assert omitted not in suffix


@pytest.mark.parametrize(
    "content",
    [
        "1\tplain text",
        ["plain text"],
        [{"type": "text", "text": "plain text"}],
    ],
)
def test_media_guard_passes_text_read_unchanged(
    content: str | list[str] | list[dict[str, str]],
) -> None:
    middleware = _GlmReadFileMediaGuard()
    # LangChain's mutable-list overload is invariant even though these dicts
    # are valid text content blocks at runtime.
    tool_content = cast("str | list[str | dict[Any, Any]]", content)
    original = ToolMessage(
        content=tool_content,
        name="read_file",
        tool_call_id="call-media",
        status="success",
    )

    result = middleware.wrap_tool_call(_request(), lambda _request: original)

    assert result is original


def test_media_guard_passes_error_read_unchanged() -> None:
    middleware = _GlmReadFileMediaGuard()
    original = ToolMessage(
        content="Error: path validation rejected /private/input/secret.png",
        name="read_file",
        tool_call_id="call-media",
        status="error",
    )

    result = middleware.wrap_tool_call(_request(), lambda _request: original)

    assert result is original


def test_media_guard_replaces_error_media_without_payload() -> None:
    middleware = _GlmReadFileMediaGuard()
    original = _media_message().model_copy(update={"status": "error"})

    result = middleware.wrap_tool_call(_request(), lambda _request: original)

    _assert_generic_media_error(result)


@pytest.mark.parametrize(
    ("block_type", "mime"),
    [("image", "image/png"), ("file", "application/pdf")],
)
def test_media_guard_replaces_media_tool_message_without_payload(
    block_type: str,
    mime: str,
) -> None:
    middleware = _GlmReadFileMediaGuard()
    original = _media_message(block_type=block_type, mime=mime)

    result = middleware.wrap_tool_call(_request(), lambda _request: original)

    _assert_generic_media_error(result)


def test_media_guard_fails_closed_on_malformed_content_blocks() -> None:
    middleware = _GlmReadFileMediaGuard()
    malformed = ToolMessage(
        content=[{"type": "text"}],
        name="read_file",
        tool_call_id="call-media",
        status="success",
    )

    result = middleware.wrap_tool_call(_request(), lambda _request: malformed)

    _assert_generic_media_error(result)


def test_media_guard_replaces_video_command() -> None:
    middleware = _GlmReadFileMediaGuard()
    video = Command(
        update={
            "messages": [
                ToolMessage(
                    content="Read video /private/input/secret.png",
                    name="read_file",
                    tool_call_id="call-media",
                    status="success",
                ),
                HumanMessage(
                    content=[
                        {"type": "text", "text": "confidential caption"},
                        {
                            "type": "image",
                            "base64": "c2VjcmV0LW1lZGlhLXBheWxvYWQ=",
                            "mime_type": "image/png",
                        },
                    ]
                ),
            ]
        }
    )

    result = middleware.wrap_tool_call(_request(), lambda _request: video)

    _assert_generic_media_error(result)


async def test_media_guard_async_matches_sync_behavior() -> None:
    middleware = _GlmReadFileMediaGuard()
    original = _media_message()

    async def handler(_request: ToolCallRequest) -> ToolMessage:
        await asyncio.sleep(0)
        return original

    sync_result = middleware.wrap_tool_call(_request(), lambda _request: original)
    async_result = await middleware.awrap_tool_call(_request(), handler)

    assert async_result == sync_result
    _assert_generic_media_error(async_result)


def test_media_guard_ignores_other_tools() -> None:
    middleware = _GlmReadFileMediaGuard()
    original = _media_message()

    result = middleware.wrap_tool_call(
        _request("custom_media_tool"),
        lambda _request: original,
    )

    assert result is original


def test_profile_materializes_fresh_guard_instances() -> None:
    first = glm_profile._GLM_5P2_PROFILE.materialize_extra_middleware()
    second = glm_profile._GLM_5P2_PROFILE.materialize_extra_middleware()

    assert len(first) == len(second) == 1
    assert isinstance(first[0], _GlmReadFileMediaGuard)
    assert isinstance(second[0], _GlmReadFileMediaGuard)
    assert first[0] is not second[0]
