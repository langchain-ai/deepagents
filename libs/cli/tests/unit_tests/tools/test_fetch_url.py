"""Tests for tools module."""

import requests
import responses

import deepagents_cli.tools as tools_module
from deepagents_cli.tools import fetch_url


@responses.activate
def test_fetch_url_success() -> None:
    """Test successful URL fetch and HTML to markdown conversion."""
    responses.add(
        responses.GET,
        "http://example.com",
        body="<html><body><h1>Test</h1><p>Content</p></body></html>",
        status=200,
    )

    result = fetch_url("http://example.com")

    assert result["status_code"] == 200
    assert "Test" in result["markdown_content"]
    assert result["url"].startswith("http://example.com")
    assert result["content_length"] > 0


@responses.activate
def test_fetch_url_http_error() -> None:
    """Test handling of HTTP errors."""
    responses.add(
        responses.GET,
        "http://example.com/notfound",
        status=404,
    )

    result = fetch_url("http://example.com/notfound")

    assert "error" in result
    assert "Fetch URL error" in result["error"]
    assert result["url"] == "http://example.com/notfound"


@responses.activate
def test_fetch_url_timeout() -> None:
    """Test handling of request timeout."""
    responses.add(
        responses.GET,
        "http://example.com/slow",
        body=requests.exceptions.Timeout(),
    )

    result = fetch_url("http://example.com/slow", timeout=1)

    assert "error" in result
    assert "Fetch URL error" in result["error"]
    assert result["url"] == "http://example.com/slow"


@responses.activate
def test_fetch_url_connection_error() -> None:
    """Test handling of connection errors."""
    responses.add(
        responses.GET,
        "http://example.com/error",
        body=requests.exceptions.ConnectionError(),
    )

    result = fetch_url("http://example.com/error")

    assert "error" in result
    assert "Fetch URL error" in result["error"]
    assert result["url"] == "http://example.com/error"


@responses.activate
def test_fetch_url_markdownify_recursion_falls_back(monkeypatch) -> None:
    """Test fallback behavior when markdownify hits recursion limits."""
    responses.add(
        responses.GET,
        "http://example.com/recursive",
        body="<html><body><h1>Hello</h1><p>World</p></body></html>",
        status=200,
    )

    def _raise_recursion(_html: str) -> str:
        msg = "maximum recursion depth exceeded"
        raise RecursionError(msg)

    monkeypatch.setattr(tools_module, "markdownify", _raise_recursion)

    result = fetch_url("http://example.com/recursive")

    assert result["status_code"] == 200
    assert "conversion_warning" in result
    assert "RecursionError" in result["conversion_warning"]
    assert "Hello" in result["markdown_content"]
    assert "World" in result["markdown_content"]


@responses.activate
def test_fetch_url_markdownify_value_error_falls_back(monkeypatch) -> None:
    """Test fallback behavior when markdownify raises ValueError."""
    responses.add(
        responses.GET,
        "http://example.com/invalid",
        body="<html><body><div>Safe fallback</div></body></html>",
        status=200,
    )

    def _raise_value_error(_html: str) -> str:
        msg = "invalid character reference"
        raise ValueError(msg)

    monkeypatch.setattr(tools_module, "markdownify", _raise_value_error)

    result = fetch_url("http://example.com/invalid")

    assert result["status_code"] == 200
    assert "conversion_warning" in result
    assert "ValueError" in result["conversion_warning"]
    assert "Safe fallback" in result["markdown_content"]
