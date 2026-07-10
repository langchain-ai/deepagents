"""Unit tests for the summarization middleware factory."""

from collections.abc import Iterable
from inspect import Parameter, signature
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, MessageLikeRepresentation

from deepagents.middleware.summarization import create_summarization_middleware
from tests.unit_tests.chat_model import GenericFakeChatModel


def _make_model(*, with_profile_limit: int | None) -> GenericFakeChatModel:
    """Create a fake model optionally configured with a max input token limit."""
    model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
    if with_profile_limit is None:
        model.profile = None
    else:
        model.profile = {"max_input_tokens": with_profile_limit}
    return model


def test_factory_uses_profile_based_defaults() -> None:
    """Uses fraction-based defaults when model profile has `max_input_tokens`."""
    model = _make_model(with_profile_limit=120_000)
    middleware = create_summarization_middleware(model, cast("Any", MagicMock()))

    assert middleware._lc_helper.trigger == ("fraction", 0.85)
    assert middleware._lc_helper.keep == ("fraction", 0.10)
    assert middleware._truncate_args_trigger == ("fraction", 0.85)
    assert middleware._truncate_args_keep == ("fraction", 0.10)


def test_factory_uses_fallback_defaults_without_profile() -> None:
    """Uses fixed token/message defaults when no model profile is available."""
    model = _make_model(with_profile_limit=None)
    middleware = create_summarization_middleware(model, cast("Any", MagicMock()))

    assert middleware._lc_helper.trigger == ("tokens", 170000)
    assert middleware._lc_helper.keep == ("messages", 6)
    assert middleware._truncate_args_trigger == ("messages", 20)
    assert middleware._truncate_args_keep == ("messages", 20)


def test_factory_uses_explicit_max_input_tokens_without_profile() -> None:
    """Sizes a fraction-of-ceiling token trigger when profile lacks the limit."""
    model = _make_model(with_profile_limit=None)
    middleware = create_summarization_middleware(
        model,
        cast("Any", MagicMock()),
        max_input_tokens=272_000,
    )

    # 0.75 * 272_000 -- fires before the provider ceiling despite a missing profile.
    assert middleware._lc_helper.trigger == ("tokens", 204000)
    assert middleware._lc_helper.keep == ("messages", 6)


def test_factory_default_prompt_explains_media_references() -> None:
    """Explains preserved media tags in the default summary prompt."""
    model = _make_model(with_profile_limit=None)
    middleware = create_summarization_middleware(model, cast("Any", MagicMock()))

    # The prompt is consumed via str.format(messages=...), so the example's
    # braces must be escaped in the template and survive formatting. Assert
    # against the rendered result -- this also guards against the literal
    # `{hash}` regression that made format() raise KeyError.
    rendered = middleware._lc_helper.summary_prompt.format(messages="<conversation>")
    assert '<image url="/conversation_history/media/{hash}.png" />' in rendered
    assert "preserve the media reference in your summary" in rendered
    assert "call `read_file` on the referenced path" in rendered


def test_factory_surfaces_summarization_knobs() -> None:
    """Passes explicit summarization settings through to the middleware."""
    model = _make_model(with_profile_limit=120_000)

    def token_counter(messages: Iterable[MessageLikeRepresentation]) -> int:
        return len(list(messages))

    middleware = create_summarization_middleware(
        model,
        cast("Any", MagicMock()),
        summary_prompt="custom summary prompt: {messages}",
        trim_tokens_to_summarize=123,
        token_counter=token_counter,
    )

    assert middleware._lc_helper.summary_prompt == "custom summary prompt: {messages}"
    assert middleware._lc_helper.trim_tokens_to_summarize == 123
    assert middleware._lc_helper.token_counter is token_counter


def test_factory_summarization_knobs_are_keyword_only() -> None:
    """Requires optional factory controls to be passed by name."""
    params = signature(create_summarization_middleware).parameters

    assert params["summary_prompt"].kind is Parameter.KEYWORD_ONLY
    assert params["trim_tokens_to_summarize"].kind is Parameter.KEYWORD_ONLY
    assert params["token_counter"].kind is Parameter.KEYWORD_ONLY


def test_factory_rejects_string_model() -> None:
    """Raises `TypeError` when called with a string model name."""
    with pytest.raises(TypeError, match="BaseChatModel"):
        create_summarization_middleware("openai:gpt-5", cast("Any", MagicMock()))  # type: ignore[arg-type]
