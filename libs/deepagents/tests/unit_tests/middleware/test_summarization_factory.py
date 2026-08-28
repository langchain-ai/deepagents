"""Unit tests for the summarization middleware factory."""

from inspect import Parameter, signature
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

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
