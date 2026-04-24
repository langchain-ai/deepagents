from __future__ import annotations

from types import SimpleNamespace

import pytest

from browser_tools import (
    _extract_result_payload,
    _html_to_text,
    _json,
    _normalize,
    _session_id,
)
from main import _final_text, _get_interrupts, _normalize_chat_model_name, parse_args


class Dumpable:
    def model_dump(self) -> dict[str, object]:
        return {"answer": 42}


def test_html_to_text_strips_non_content_tags() -> None:
    title, text = _html_to_text(
        """
        <html>
          <head>
            <title> Example Page </title>
            <style>.hidden { display: none; }</style>
          </head>
          <body>
            <script>window.secret = true;</script>
            <h1>Heading</h1>
            <p>Useful text</p>
          </body>
        </html>
        """,
        max_chars=100,
    )

    assert title == "Example Page"
    assert "Heading" in text
    assert "Useful text" in text
    assert "window.secret" not in text
    assert "display: none" not in text


def test_html_to_text_applies_character_limit() -> None:
    _, text = _html_to_text("<body>abcdef</body>", max_chars=3)

    assert text == "abc"


def test_normalize_handles_sdk_objects() -> None:
    value = SimpleNamespace(
        visible="ok",
        _hidden="skip",
        nested=[Dumpable(), {"items": {1, 2}}],
    )

    normalized = _normalize(value)

    assert normalized["visible"] == "ok"
    assert normalized["nested"][0] == {"answer": 42}
    assert set(normalized["nested"][1]["items"]) == {1, 2}


def test_json_normalizes_before_serializing() -> None:
    payload = _json(SimpleNamespace(count=1))

    assert '"count": 1' in payload


def test_session_id_supports_nested_or_flat_responses() -> None:
    assert _session_id(SimpleNamespace(data=SimpleNamespace(session_id="nested"))) == "nested"
    assert _session_id(SimpleNamespace(session_id="flat")) == "flat"


def test_session_id_raises_for_unknown_response_shape() -> None:
    with pytest.raises(RuntimeError, match="Could not extract session id"):
        _session_id(SimpleNamespace(data=SimpleNamespace()))


def test_extract_result_payload_prefers_data_result() -> None:
    result = SimpleNamespace(data=SimpleNamespace(result=Dumpable()))

    assert _extract_result_payload(result) == {"answer": 42}


def test_normalize_openai_prefixed_model_name() -> None:
    assert _normalize_chat_model_name("openai:gpt-5.5") == "gpt-5.5"
    assert _normalize_chat_model_name("anthropic:claude-sonnet-4-6") == (
        "anthropic:claude-sonnet-4-6"
    )


def test_final_text_reads_latest_assistant_message() -> None:
    result = {
        "messages": [
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "answer"}, {"type": "image", "url": "x"}],
            },
        ]
    }

    text = _final_text(result)

    assert "answer" in text
    assert '"type": "image"' in text


def test_get_interrupts_supports_langgraph_state_and_result_objects() -> None:
    interrupt = SimpleNamespace(value={"action_requests": []})

    assert _get_interrupts({"__interrupt__": [interrupt]}) == [interrupt]
    assert _get_interrupts(SimpleNamespace(interrupts=[interrupt])) == [interrupt]


def test_parse_args_uses_default_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPAGENT_MODEL", raising=False)

    args = parse_args([])

    assert args.model == "gpt-5.5"
