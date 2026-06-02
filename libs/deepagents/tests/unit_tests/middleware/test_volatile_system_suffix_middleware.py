"""Unit tests for `VolatileSystemSuffixMiddleware`.

The middleware appends a trailing system content block that is intentionally
*not* tagged with `cache_control`, so volatile per-turn content (user identity,
current date, ...) can live in the system prompt without invalidating the
Anthropic prompt-cache prefix. It is wired innermost by `create_deep_agent`
(after `AnthropicPromptCachingMiddleware` and `MemoryMiddleware`); the
end-to-end placement tests live in `test_volatile_system_suffix_graph.py`.
"""

from langchain.agents.middleware.types import ModelRequest
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import deepagents
from deepagents import middleware as deepagents_middleware
from deepagents.middleware.volatile_system_suffix import VolatileSystemSuffixMiddleware
from tests.unit_tests.chat_model import GenericFakeChatModel


def _build_model_request(model: BaseChatModel, system_message: SystemMessage | None) -> ModelRequest:
    """Construct a minimal `ModelRequest` for direct `modify_request` tests."""
    return ModelRequest(
        model=model,
        messages=[HumanMessage(content="hi")],
        system_message=system_message,
        state={"messages": []},  # type: ignore[typeddict-unknown-key]
    )


def _fake_anthropic() -> ChatAnthropic:
    """Build a `ChatAnthropic` instance with a dummy key."""
    return ChatAnthropic(model_name="claude-sonnet-4-6", anthropic_api_key="fake")  # type: ignore[call-arg]


def _fake_other() -> GenericFakeChatModel:
    """Build a non-Anthropic fake chat model."""
    return GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))


def _system_blocks(message: SystemMessage | None) -> list:
    """Extract content blocks from a required system message; fails test if absent."""
    assert message is not None, "modify_request must return a SystemMessage"
    return list(message.content_blocks)


def test_str_suffix_appended_as_trailing_text_block() -> None:
    """A string suffix becomes the final text block of the system message."""
    middleware = VolatileSystemSuffixMiddleware("Current date: 2026-06-01")
    request = _build_model_request(_fake_anthropic(), SystemMessage(content="base prompt"))

    result = middleware.modify_request(request)
    blocks = _system_blocks(result.system_message)

    assert blocks[-1].get("type") == "text"
    assert "Current date: 2026-06-01" in blocks[-1].get("text", "")
    # The stable base prompt is preserved as an earlier block.
    assert "base prompt" in blocks[0].get("text", "")


def test_str_suffix_never_carries_cache_control() -> None:
    """The volatile block must not be tagged with `cache_control`, even for Anthropic."""
    middleware = VolatileSystemSuffixMiddleware("volatile content")
    request = _build_model_request(_fake_anthropic(), SystemMessage(content="base prompt"))

    result = middleware.modify_request(request)
    blocks = _system_blocks(result.system_message)

    assert "cache_control" not in blocks[-1]


def test_str_suffix_separated_by_blank_line() -> None:
    """A string suffix is appended as a distinct block, not merged into the prior one."""
    middleware = VolatileSystemSuffixMiddleware("SUFFIX")
    request = _build_model_request(_fake_anthropic(), SystemMessage(content="BASE"))

    result = middleware.modify_request(request)
    blocks = _system_blocks(result.system_message)

    assert len(blocks) == 2
    assert blocks[0].get("text") == "BASE"
    assert blocks[1].get("text", "").endswith("SUFFIX")


def test_list_suffix_appended_verbatim_in_order() -> None:
    """A list suffix is appended block-for-block, preserving order, as the trailing blocks."""
    suffix = [
        {"type": "text", "text": "first volatile"},
        {"type": "text", "text": "second volatile"},
    ]
    middleware = VolatileSystemSuffixMiddleware(suffix)
    request = _build_model_request(_fake_anthropic(), SystemMessage(content="base prompt"))

    result = middleware.modify_request(request)
    blocks = _system_blocks(result.system_message)

    assert blocks[-2].get("text") == "first volatile"
    assert blocks[-1].get("text") == "second volatile"
    assert "cache_control" not in blocks[-1]
    assert "cache_control" not in blocks[-2]


def test_empty_str_suffix_is_noop() -> None:
    """An empty string leaves the request untouched."""
    middleware = VolatileSystemSuffixMiddleware("")
    base = SystemMessage(content="base prompt")
    request = _build_model_request(_fake_anthropic(), base)

    result = middleware.modify_request(request)

    assert result is request
    assert result.system_message is base


def test_empty_list_suffix_is_noop() -> None:
    """An empty list leaves the request untouched."""
    middleware = VolatileSystemSuffixMiddleware([])
    base = SystemMessage(content="base prompt")
    request = _build_model_request(_fake_anthropic(), base)

    result = middleware.modify_request(request)

    assert result is request
    assert result.system_message is base


def test_none_system_message_creates_message_with_suffix() -> None:
    """With no existing system message, the suffix becomes the sole system block."""
    middleware = VolatileSystemSuffixMiddleware("only volatile")
    request = _build_model_request(_fake_anthropic(), None)

    result = middleware.modify_request(request)
    blocks = _system_blocks(result.system_message)

    assert len(blocks) == 1
    assert "only volatile" in blocks[0].get("text", "")
    assert "cache_control" not in blocks[0]


def test_preserves_cache_control_on_existing_last_block() -> None:
    """Appending the suffix must not strip `cache_control` from the (now earlier) stable block."""
    cached_system = SystemMessage(content_blocks=[{"type": "text", "text": "stable prompt", "cache_control": {"type": "ephemeral"}}])
    middleware = VolatileSystemSuffixMiddleware("volatile content")
    request = _build_model_request(_fake_anthropic(), cached_system)

    result = middleware.modify_request(request)
    blocks = _system_blocks(result.system_message)

    # Breakpoint stays on the stable block...
    assert blocks[-2].get("cache_control") == {"type": "ephemeral"}
    # ...and the volatile suffix trails it uncached.
    assert "cache_control" not in blocks[-1]
    assert "volatile content" in blocks[-1].get("text", "")


def test_non_anthropic_model_appends_suffix_without_cache_control() -> None:
    """Non-Anthropic models get the trailing block too, and no `cache_control` anywhere."""
    middleware = VolatileSystemSuffixMiddleware("volatile content")
    request = _build_model_request(_fake_other(), SystemMessage(content="base prompt"))

    result = middleware.modify_request(request)
    blocks = _system_blocks(result.system_message)

    assert "volatile content" in blocks[-1].get("text", "")
    assert all("cache_control" not in block for block in blocks)


async def test_awrap_model_call_appends_suffix() -> None:
    """The async hook applies the same modification as the sync path."""
    middleware = VolatileSystemSuffixMiddleware("async volatile")
    request = _build_model_request(_fake_other(), SystemMessage(content="base prompt"))

    captured: dict[str, ModelRequest] = {}

    async def handler(req: ModelRequest) -> AIMessage:
        captured["request"] = req
        return AIMessage(content="ok")

    await middleware.awrap_model_call(request, handler)  # type: ignore[arg-type]

    blocks = _system_blocks(captured["request"].system_message)
    assert "async volatile" in blocks[-1].get("text", "")
    assert "cache_control" not in blocks[-1]


def test_wrap_model_call_appends_suffix() -> None:
    """The sync hook passes the modified request to the handler."""
    middleware = VolatileSystemSuffixMiddleware("sync volatile")
    request = _build_model_request(_fake_other(), SystemMessage(content="base prompt"))

    captured: dict[str, ModelRequest] = {}

    def handler(req: ModelRequest) -> AIMessage:
        captured["request"] = req
        return AIMessage(content="ok")

    middleware.wrap_model_call(request, handler)  # type: ignore[arg-type]

    blocks = _system_blocks(captured["request"].system_message)
    assert "sync volatile" in blocks[-1].get("text", "")


def test_public_exports() -> None:
    """The middleware is exported from both `deepagents` and `deepagents.middleware`."""
    assert deepagents.VolatileSystemSuffixMiddleware is VolatileSystemSuffixMiddleware
    assert deepagents_middleware.VolatileSystemSuffixMiddleware is VolatileSystemSuffixMiddleware
    assert "VolatileSystemSuffixMiddleware" in deepagents.__all__
    assert "VolatileSystemSuffixMiddleware" in deepagents_middleware.__all__
