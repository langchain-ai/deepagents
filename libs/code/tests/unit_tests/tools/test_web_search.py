"""The built-in and workspace-bound web search tools must stay interchangeable."""

from __future__ import annotations

import json
import sys
from typing import Any
from unittest import mock
from unittest.mock import patch

import responses

from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool

from deepagents_code.tools import (
    create_web_search_tool,
    is_web_search_tool,
    web_search,
)


class TestSchemaParity:
    """`is_web_search_tool` treats both variants as one, so they must match."""

    def test_workspace_variant_presents_the_same_arguments(self) -> None:
        """A `**kwargs` closure must not erode the advertised schema."""
        builtin = convert_to_openai_tool(tool("web_search")(web_search))
        workspace = convert_to_openai_tool(create_web_search_tool("key"))

        assert workspace == builtin


class TestToolIdentity:
    """Both variants are recognized; unrelated tools are not."""

    def test_recognizes_both_variants(self) -> None:
        assert is_web_search_tool(web_search)
        assert is_web_search_tool(create_web_search_tool("key"))

    def test_rejects_an_unrelated_tool(self) -> None:
        from deepagents_code.tools import fetch_url

        assert not is_web_search_tool(fetch_url)
        assert not is_web_search_tool(object())


class TestWorkspaceErrorTranslation:
    """Failures return a payload the model can act on, never an exception."""

    def test_missing_package_is_translated(self) -> None:
        """An uninstalled `tavily` must not raise out of the tool."""
        search = create_web_search_tool("key")

        with patch.dict(sys.modules, {"tavily": None}):
            result = search.invoke({"query": "anything"})

        assert result["error"].startswith("Required package not installed")

    def test_empty_key_reports_configuration_not_a_tavily_error(self) -> None:
        """`_build_tools` can pass `""`; that must read as unconfigured."""
        search = create_web_search_tool("")

        result = search.invoke({"query": "anything"})

        assert "Tavily API key not configured" in result["error"]
        assert result["query"] == "anything"

    def test_search_errors_are_translated_like_the_builtin(self) -> None:
        """A request failure returns the shared translated payload."""
        import requests

        search = create_web_search_tool("key")

        class _Client:
            def __init__(self, **_kwargs: Any) -> None:
                pass

            def search(self, *_args: Any, **_kwargs: Any) -> object:
                raise requests.exceptions.ConnectionError

        with patch("tavily.TavilyClient", _Client):
            result = search.invoke({"query": "anything"})

        assert "error" in result


class TestOllamaProviderSelection:
    """Tavily keeps priority; Ollama Cloud is the fallback."""

    @staticmethod
    def _credentials(has_tavily: bool, has_ollama: bool) -> Any:
        from types import SimpleNamespace

        return SimpleNamespace(
            has_tavily=has_tavily,
            has_ollama=has_ollama,
            ollama_api_key="k",
        )

    def test_tavily_wins_when_both_configured(self) -> None:
        from deepagents_code.tools import _active_web_provider

        with mock.patch(
            "deepagents_code.config.credentials", self._credentials(True, True)
        ):
            assert _active_web_provider() == "tavily"

    def test_ollama_when_only_ollama_configured(self) -> None:
        from deepagents_code.tools import _active_web_provider

        with mock.patch(
            "deepagents_code.config.credentials", self._credentials(False, True)
        ):
            assert _active_web_provider() == "ollama"

    def test_none_without_any_key(self) -> None:
        from deepagents_code.tools import _active_web_provider

        with mock.patch(
            "deepagents_code.config.credentials", self._credentials(False, False)
        ):
            assert _active_web_provider() is None


class TestOllamaSearch:
    """The Ollama Cloud backend posts to `ollama.com` and normalizes hits."""

    def test_posts_to_ollama_with_bearer_key_and_result_cap(self) -> None:
        from types import SimpleNamespace

        from deepagents_code.tools import _OLLAMA_WEB_SEARCH_URL

        stub = SimpleNamespace(has_tavily=False, has_ollama=True, ollama_api_key="k")

        with responses.RequestsMock() as rsps, mock.patch(
            "deepagents_code.config.credentials", stub
        ):
            rsps.add(
                "POST",
                _OLLAMA_WEB_SEARCH_URL,
                json={
                    "results": [
                        {"title": "T", "url": "https://x", "content": "c"},
                    ]
                },
            )

            result = web_search(query="q", max_results=25)
            request = rsps.calls[0][0]
        assert request.headers["Authorization"] == "Bearer k"
        assert json.loads(request.body) == {"query": "q", "max_results": 10}
        assert result == {"query": "q", "results": [{"title": "T", "url": "https://x", "content": "c"}]}

    def test_request_errors_are_translated(self) -> None:
        from types import SimpleNamespace

        from deepagents_code.tools import _OLLAMA_WEB_SEARCH_URL

        stub = SimpleNamespace(has_tavily=False, has_ollama=True, ollama_api_key="k")

        with responses.RequestsMock() as rsps, mock.patch(
            "deepagents_code.config.credentials", stub
        ):
            rsps.add("POST", _OLLAMA_WEB_SEARCH_URL, status=401)

            result = web_search(query="q")

        assert "Web search error" in result["error"]
        assert result["query"] == "q"

    def test_empty_ollama_key_reports_configuration(self) -> None:
        from types import SimpleNamespace

        stub = SimpleNamespace(has_tavily=False, has_ollama=True, ollama_api_key="")

        with mock.patch("deepagents_code.config.credentials", stub):
            result = web_search(query="q")

        assert "Ollama API key not configured" in result["error"]
        assert result["query"] == "q"

    def test_workspace_variant_ollama_message_for_empty_key(self) -> None:
        search = create_web_search_tool("", provider="ollama")

        result = search.invoke({"query": "anything"})

        assert "Ollama API key not configured" in result["error"]
