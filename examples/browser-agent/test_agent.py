"""Network-free tests for the browser agent example."""

from __future__ import annotations

import socket

import pytest

import agent as agent_module
from agent import UnsafeURLError, _browser_tools, browser_agent, ensure_navigable


def test_private_navigation_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", lambda *_args: [(None, None, None, None, ("127.0.0.1", 0))])

    with pytest.raises(UnsafeURLError, match="disallowed"):
        ensure_navigable("http://localhost/admin")


def test_public_navigation_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", lambda *_args: [(None, None, None, None, ("93.184.216.34", 0))])

    assert ensure_navigable("https://example.com/path") == "https://example.com/path"


def test_tools_delegate_to_browser() -> None:
    calls: list[tuple[object, ...]] = []

    class FakeBrowser:
        def observe(self) -> dict[str, str]:
            calls.append(("observe",))
            return {"title": "Example"}

        def navigate(self, url: str) -> str:
            calls.append(("navigate", url))
            return url

        def act(self, ref: int, action: str, value: str | None = None) -> str:
            calls.append(("act", ref, action, value))
            return "done"

        def scroll(self, direction: str) -> str:
            calls.append(("scroll", direction))
            return "done"

    tools = {tool.name: tool for tool in _browser_tools(FakeBrowser())}  # type: ignore[arg-type]
    assert tools["observe_browser"].invoke({}) == {"title": "Example"}
    assert tools["click"].invoke({"ref": 7}) == "done"
    assert tools["type_text"].invoke({"ref": 8, "text": "hello"}) == "done"
    assert tools["select_option"].invoke({"ref": 9, "value": "one"}) == "done"
    assert tools["navigate"].invoke({"url": "https://example.com"}) == "https://example.com"
    assert tools["scroll_page"].invoke({"direction": "down"}) == "done"
    assert calls == [
        ("observe",),
        ("act", 7, "click", None),
        ("act", 8, "type", "hello"),
        ("act", 9, "select", "one"),
        ("navigate", "https://example.com"),
        ("scroll", "down"),
    ]


def test_browser_agent_closes_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    closed = False

    class FakeBrowser:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def close(self) -> None:
            nonlocal closed
            closed = True

    monkeypatch.setattr(agent_module, "Browser", FakeBrowser)
    monkeypatch.setattr(agent_module, "_browser_tools", lambda _browser: [])
    monkeypatch.setattr(agent_module, "create_agent", lambda **_kwargs: "agent")

    with browser_agent("https://example.com") as graph:
        assert graph == "agent"
    assert closed
