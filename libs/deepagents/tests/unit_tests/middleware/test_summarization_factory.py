"""Unit tests for the summarization middleware factory."""

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


def test_factory_accepts_custom_history_path_prefix() -> None:
    """Stores conversation history under a custom path prefix when configured."""
    model = _make_model(with_profile_limit=120_000)
    middleware = create_summarization_middleware(
        model,
        cast("Any", MagicMock()),
        history_path_prefix="/artifacts/conversation_history",
    )

    assert middleware._history_path_prefix == "/artifacts/conversation_history"


def test_factory_rejects_string_model() -> None:
    """Raises `TypeError` when called with a string model name."""
    with pytest.raises(TypeError, match="BaseChatModel"):
        create_summarization_middleware("openai:gpt-5", cast("Any", MagicMock()))  # type: ignore[arg-type]
