"""Tests for tools module."""

from unittest.mock import patch

import requests
import responses

from deepagents_cli.tools import fetch_url, web_search


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
def test_web_search_delegates_to_fetch_url_when_query_contains_url() -> None:
    """web_search should directly fetch a concrete URL instead of searching."""
    responses.add(
        responses.GET,
        "https://example.com/docs",
        body="<html><body><h1>Doc</h1><p>Detail</p></body></html>",
        status=200,
    )

    with patch("deepagents_cli.tools.tavily_client") as mock_tavily:
        result = web_search("Please summarize https://example.com/docs")

    assert result["direct_fetch"] is True
    assert result["source_tool"] == "fetch_url"
    assert result["url"].startswith("https://example.com/docs")
    assert "Doc" in result["markdown_content"]
    mock_tavily.search.assert_not_called()


def test_web_search_uses_tavily_when_query_has_no_url() -> None:
    """web_search should use Tavily for normal keyword queries."""
    expected = {"results": [{"title": "Result", "url": "https://x", "content": "c"}]}

    with patch("deepagents_cli.tools.tavily_client") as mock_tavily:
        mock_tavily.search.return_value = expected
        result = web_search("langgraph state checkpointing")

    assert result == expected
    mock_tavily.search.assert_called_once()
