from __future__ import annotations

from deepagents_code.tools import fetch_url, web_search
from deepagents_talon.tools import build_web_tools


def test_build_web_tools_returns_shared_code_tools() -> None:
    tools = build_web_tools()

    assert tools == [fetch_url, web_search]
