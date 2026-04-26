"""Unit tests for main.py pure-function utilities.

All tests are offline — no LLM, Browserbase, or network calls.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Make the example root importable when running pytest from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import (  # noqa: E402
    _final_text,
    _normalize_chat_model_name,
    _stringify_content,
    build_model,
)


# ---------------------------------------------------------------------------
# _normalize_chat_model_name
# ---------------------------------------------------------------------------


def test_normalize_strips_openai_prefix():
    assert _normalize_chat_model_name("openai:gpt-4o") == "gpt-4o"


def test_normalize_strips_openai_prefix_with_version():
    assert _normalize_chat_model_name("openai:gpt-4-turbo-2024-04-09") == "gpt-4-turbo-2024-04-09"


def test_normalize_passthrough_non_openai_prefix():
    assert _normalize_chat_model_name("anthropic:claude-opus-4") == "anthropic:claude-opus-4"


def test_normalize_passthrough_no_prefix():
    assert _normalize_chat_model_name("gpt-4o") == "gpt-4o"


def test_normalize_passthrough_empty():
    assert _normalize_chat_model_name("") == ""


# ---------------------------------------------------------------------------
# _stringify_content
# ---------------------------------------------------------------------------


def test_stringify_plain_string():
    assert _stringify_content("hello world") == "hello world"


def test_stringify_list_of_strings():
    assert _stringify_content(["foo", "bar"]) == "foo\nbar"


def test_stringify_list_with_text_dicts():
    content = [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]
    assert _stringify_content(content) == "hello\nworld"


def test_stringify_list_with_non_text_dict():
    content = [{"type": "image", "url": "http://example.com/img.png"}]
    result = _stringify_content(content)
    parsed = json.loads(result)
    assert parsed["type"] == "image"


def test_stringify_mixed_list():
    content = ["plain", {"type": "text", "text": "structured"}, {"type": "tool_use", "id": "x"}]
    result = _stringify_content(content)
    assert "plain" in result
    assert "structured" in result


def test_stringify_filters_empty_parts():
    # Empty-string items should be dropped from the join.
    content = ["", "real", ""]
    assert _stringify_content(content) == "real"


def test_stringify_non_string_non_list():
    result = _stringify_content({"key": "val"})
    parsed = json.loads(result)
    assert parsed["key"] == "val"


def test_stringify_none_item_in_list():
    content = [None]
    result = _stringify_content(content)
    # None stringified is "None"; it's non-empty so it gets included.
    assert "None" in result


# ---------------------------------------------------------------------------
# _final_text
# ---------------------------------------------------------------------------


def _make_ai_message(content: str) -> MagicMock:
    msg = MagicMock()
    msg.type = "ai"
    msg.content = content
    return msg


def _make_human_message(content: str) -> MagicMock:
    msg = MagicMock()
    msg.type = "human"
    msg.content = content
    return msg


def test_final_text_picks_last_ai_message():
    human = _make_human_message("hi")
    ai1 = _make_ai_message("first response")
    ai2 = _make_ai_message("second response")
    result = {"messages": [human, ai1, ai2]}
    assert _final_text(result) == "second response"


def test_final_text_skips_human_messages():
    human = _make_human_message("hi")
    ai = _make_ai_message("only ai")
    human2 = _make_human_message("follow-up")
    result = {"messages": [human, ai, human2]}
    assert _final_text(result) == "only ai"


def test_final_text_falls_back_to_json_when_no_ai_message():
    human = _make_human_message("hi")
    result = {"messages": [human]}
    output = _final_text(result)
    # Should be a JSON dump of the state, not raise.
    assert isinstance(output, str)
    assert len(output) > 0


def test_final_text_handles_dict_messages():
    result = {
        "messages": [
            {"type": "human", "content": "q"},
            {"type": "ai", "content": "answer"},
        ]
    }
    assert _final_text(result) == "answer"


def test_final_text_handles_assistant_role_dict():
    result = {
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "assistant answer"},
        ]
    }
    assert _final_text(result) == "assistant answer"


def test_final_text_handles_result_with_value_attr():
    ai = _make_ai_message("via value attr")
    result_obj = MagicMock()
    result_obj.value = {"messages": [ai]}
    assert _final_text(result_obj) == "via value attr"


def test_final_text_non_dict_falls_back_to_str():
    assert _final_text("raw string") == "raw string"


# ---------------------------------------------------------------------------
# build_model
# ---------------------------------------------------------------------------


def test_build_model_uses_openai_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    monkeypatch.delenv("BROWSERBASE_API_KEY", raising=False)
    monkeypatch.delenv("DEEPAGENT_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    with patch("main.ChatOpenAI") as mock_chat:
        build_model("gpt-4o")
        mock_chat.assert_called_once()
        call_kwargs = mock_chat.call_args.kwargs
        assert call_kwargs["api_key"] == "sk-test-123"
        assert call_kwargs["model"] == "gpt-4o"
        assert "base_url" not in call_kwargs


def test_build_model_uses_browserbase_key_with_base_url(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("BROWSERBASE_API_KEY", "bb-test-key")
    monkeypatch.setenv("DEEPAGENT_BASE_URL", "https://gateway.example.com/v1")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    with patch("main.ChatOpenAI") as mock_chat:
        build_model("gpt-4o")
        call_kwargs = mock_chat.call_args.kwargs
        assert call_kwargs["api_key"] == "bb-test-key"
        assert call_kwargs["base_url"] == "https://gateway.example.com/v1"


def test_build_model_normalizes_openai_prefix(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    monkeypatch.delenv("BROWSERBASE_API_KEY", raising=False)
    monkeypatch.delenv("DEEPAGENT_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    with patch("main.ChatOpenAI") as mock_chat:
        build_model("openai:gpt-4o-mini")
        call_kwargs = mock_chat.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o-mini"


def test_build_model_raises_without_any_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("BROWSERBASE_API_KEY", raising=False)
    monkeypatch.delenv("DEEPAGENT_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    with pytest.raises(ValueError, match="Missing Deep Agent model configuration"):
        build_model("gpt-4o")


def test_build_model_raises_bb_key_without_base_url(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("BROWSERBASE_API_KEY", "bb-key")
    monkeypatch.delenv("DEEPAGENT_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    with pytest.raises(ValueError):
        build_model("gpt-4o")
