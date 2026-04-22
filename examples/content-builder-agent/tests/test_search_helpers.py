import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Allow importing from the parent directory without a package install
sys.path.insert(0, str(Path(__file__).parent.parent))

from content_writer import (
    _auto_select_providers,
    _expand_query,
    _search_ddg,
    _search_exa,
    _search_tavily,
    web_search_auto,
    web_search_multi,
)


# ---------------------------------------------------------------------------
# _auto_select_providers
# ---------------------------------------------------------------------------


def test_auto_select_providers_tavily_only(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "fake-tavily-key")
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    assert _auto_select_providers() == ["tavily", "duckduckgo"]


def test_auto_select_providers_exa_only(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("EXA_API_KEY", "fake-exa-key")
    assert _auto_select_providers() == ["exa", "duckduckgo"]


def test_auto_select_providers_both(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "fake-tavily-key")
    monkeypatch.setenv("EXA_API_KEY", "fake-exa-key")
    assert _auto_select_providers() == ["tavily", "exa"]


def test_auto_select_providers_none(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    assert _auto_select_providers() == ["duckduckgo"]


# ---------------------------------------------------------------------------
# _expand_query
# ---------------------------------------------------------------------------


async def test_expand_query_fallback():
    with patch("langchain_openai.ChatOpenAI", side_effect=Exception("no key")):
        result = await _expand_query("test")
    assert result == ["test"]


async def test_expand_query_returns_list():
    mock_response = MagicMock()
    mock_response.content = '["q1","q2","q3"]'

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    mock_chat_class = MagicMock(return_value=mock_llm)

    with patch("langchain_openai.ChatOpenAI", mock_chat_class):
        result = await _expand_query("my topic")

    assert result == ["q1", "q2", "q3"]


# ---------------------------------------------------------------------------
# _search_tavily
# ---------------------------------------------------------------------------


async def test_search_tavily_no_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    result = await _search_tavily("test", 3, "general")
    assert result == []


# ---------------------------------------------------------------------------
# _search_exa
# ---------------------------------------------------------------------------


async def test_search_exa_no_key(monkeypatch):
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    result = await _search_exa("test", 3)
    assert result == []


# ---------------------------------------------------------------------------
# _search_ddg
# ---------------------------------------------------------------------------


async def test_search_ddg_returns_normalised_shape():
    raw_result = [{"href": "http://example.com", "title": "T", "body": "B"}]

    mock_ddgs_instance = MagicMock()
    mock_ddgs_instance.text = MagicMock(return_value=raw_result)

    with patch("duckduckgo_search.DDGS", return_value=mock_ddgs_instance):
        result = await _search_ddg("test query", 3)

    assert result == [
        {"url": "http://example.com", "title": "T", "content": "B", "source": "duckduckgo"}
    ]


# ---------------------------------------------------------------------------
# web_search_multi
# ---------------------------------------------------------------------------


async def test_web_search_multi_deduplicates_by_url():
    """Results with the same URL from different providers appear only once."""
    dup = {"url": "http://example.com", "title": "T", "content": "C", "source": "tavily"}
    with patch("content_writer._auto_select_providers", return_value=["tavily", "duckduckgo"]):
        with patch("content_writer._search_tavily", new=AsyncMock(return_value=[dup])):
            with patch("content_writer._search_ddg", new=AsyncMock(return_value=[dup])):
                result = await web_search_multi.ainvoke({"queries": ["test"]})
    assert len(result["results"]) == 1
    assert result["query_count"] == 1
    assert result["provider_count"] == 2


async def test_web_search_multi_provider_failure_returns_partial():
    """A failing provider does not prevent results from other providers."""
    good = {"url": "http://good.com", "title": "G", "content": "C", "source": "duckduckgo"}
    with patch("content_writer._auto_select_providers", return_value=["tavily", "duckduckgo"]):
        with patch("content_writer._search_tavily", new=AsyncMock(side_effect=RuntimeError("fail"))):
            with patch("content_writer._search_ddg", new=AsyncMock(return_value=[good])):
                result = await web_search_multi.ainvoke({"queries": ["test"]})
    assert len(result["results"]) == 1
    assert result["results"][0]["source"] == "duckduckgo"


# ---------------------------------------------------------------------------
# web_search_auto
# ---------------------------------------------------------------------------


async def test_web_search_auto_includes_expanded_queries():
    """web_search_auto attaches original_query and expanded_queries to result."""
    with patch("content_writer._expand_query", new=AsyncMock(return_value=["q1", "q2", "q3"])):
        with patch("content_writer.web_search_multi") as mock_multi:
            mock_multi.ainvoke = AsyncMock(return_value={
                "results": [], "query_count": 4, "provider_count": 2
            })
            result = await web_search_auto.ainvoke({"query": "original"})
    assert result["original_query"] == "original"
    assert result["expanded_queries"] == ["q1", "q2", "q3"]
    called_queries = mock_multi.ainvoke.call_args[0][0]["queries"]
    assert called_queries[0] == "original"
    assert len(called_queries) == 4
