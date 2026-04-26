"""Unit tests for browser_tools.py pure-function utilities.

All tests are offline — no Browserbase, Stagehand, or network calls.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from browser_tools import _html_to_text, _json, _normalize  # noqa: E402


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------


def test_normalize_none():
    assert _normalize(None) is None


def test_normalize_primitives():
    assert _normalize(42) == 42
    assert _normalize(3.14) == 3.14
    assert _normalize(True) is True
    assert _normalize("hello") == "hello"


def test_normalize_dict():
    assert _normalize({"a": 1, "b": "x"}) == {"a": 1, "b": "x"}


def test_normalize_dict_with_non_string_keys():
    result = _normalize({1: "one", 2: "two"})
    assert result == {"1": "one", "2": "two"}


def test_normalize_list():
    assert _normalize([1, "two", None]) == [1, "two", None]


def test_normalize_tuple():
    assert _normalize((1, 2)) == [1, 2]


def test_normalize_set_produces_list():
    result = _normalize({42})
    assert result == [42]


def test_normalize_nested():
    data = {"key": [1, {"inner": True}]}
    assert _normalize(data) == {"key": [1, {"inner": True}]}


def test_normalize_pydantic_like_model_dump():
    obj = MagicMock(spec=[])
    obj.model_dump = MagicMock(return_value={"field": "value"})
    result = _normalize(obj)
    assert result == {"field": "value"}


def test_normalize_pydantic_v1_dict():
    obj = MagicMock(spec=[])
    # model_dump not present, but .dict() is
    del obj.model_dump
    obj.dict = MagicMock(return_value={"field": "v1"})
    result = _normalize(obj)
    assert result == {"field": "v1"}


def test_normalize_arbitrary_object_with_dunder_dict():
    class Foo:
        def __init__(self):
            self.x = 1
            self.y = "hello"
            self._private = "skip"

    result = _normalize(Foo())
    assert result == {"x": 1, "y": "hello"}


def test_normalize_arbitrary_object_no_public_attrs():
    class Empty:
        pass

    result = _normalize(Empty())
    # Falls through to str()
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _json
# ---------------------------------------------------------------------------


def test_json_serializes_dict():
    payload = {"a": 1, "b": [2, 3]}
    text = _json(payload)
    parsed = json.loads(text)
    assert parsed == {"a": 1, "b": [2, 3]}


def test_json_pretty_prints():
    text = _json({"x": 1})
    assert "\n" in text  # indent=2 produces newlines


def test_json_handles_non_serializable_via_default():
    class Weird:
        def __str__(self):
            return "weird-repr"

    text = _json({"obj": Weird()})
    parsed = json.loads(text)
    assert parsed["obj"] == "weird-repr"


# ---------------------------------------------------------------------------
# _html_to_text
# ---------------------------------------------------------------------------


_SAMPLE_HTML = """<!DOCTYPE html>
<html>
<head>
  <title>Test Page</title>
  <style>body { font-family: sans-serif; }</style>
</head>
<body>
  <script>alert('x');</script>
  <h1>Hello</h1>
  <p>World paragraph.</p>
  <noscript>no js</noscript>
</body>
</html>"""


def test_html_to_text_extracts_title():
    title, _ = _html_to_text(_SAMPLE_HTML, max_chars=10000)
    assert title == "Test Page"


def test_html_to_text_removes_script():
    _, text = _html_to_text(_SAMPLE_HTML, max_chars=10000)
    assert "alert" not in text


def test_html_to_text_removes_style():
    _, text = _html_to_text(_SAMPLE_HTML, max_chars=10000)
    assert "font-family" not in text


def test_html_to_text_removes_noscript():
    _, text = _html_to_text(_SAMPLE_HTML, max_chars=10000)
    assert "no js" not in text


def test_html_to_text_keeps_body_content():
    _, text = _html_to_text(_SAMPLE_HTML, max_chars=10000)
    assert "Hello" in text
    assert "World paragraph" in text


def test_html_to_text_respects_max_chars():
    _, text = _html_to_text(_SAMPLE_HTML, max_chars=5)
    assert len(text) <= 5


def test_html_to_text_collapses_excess_newlines():
    html = "<body>" + "\n\n\n\n".join(["line"] * 10) + "</body>"
    _, text = _html_to_text(html, max_chars=10000)
    assert "\n\n\n" not in text


def test_html_to_text_empty_html():
    title, text = _html_to_text("", max_chars=10000)
    assert title == ""
    assert text == ""


def test_html_to_text_no_title_tag():
    html = "<html><body><p>No title here</p></body></html>"
    title, text = _html_to_text(html, max_chars=10000)
    assert title == ""
    assert "No title here" in text


def test_html_to_text_max_chars_zero():
    _, text = _html_to_text(_SAMPLE_HTML, max_chars=0)
    assert text == ""
