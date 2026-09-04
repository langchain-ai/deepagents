from __future__ import annotations

import pytest

from deepagents.mcp import MCPServerInfo, MCPToolInfo


def test_server_info_accepts_loaded_tools() -> None:
    tool = MCPToolInfo(
        name="search",
        description="Search documents",
        input_schema={"type": "object"},
    )

    server = MCPServerInfo(name="docs", transport="http", tools=(tool,), uses_oauth=True)

    assert server.tools == (tool,)
    assert server.status == "ok"
    assert server.needs_attention() is False


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"error": "failed"}, "status='ok' cannot carry an error"),
        ({"status": "error"}, "requires an error message"),
        (
            {
                "status": "unauthenticated",
                "error": "login",
                "tools": (MCPToolInfo(name="search", description=""),),
            },
            "cannot carry tools",
        ),
        (
            {"pending_reconnect": True},
            "pending_reconnect requires status='disabled'",
        ),
    ],
)
def test_server_info_rejects_inconsistent_state(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        MCPServerInfo(name="docs", transport="http", **kwargs)


def test_server_info_reports_authentication_attention() -> None:
    server = MCPServerInfo(
        name="docs",
        transport="http",
        status="unauthenticated",
        error="login required",
    )

    assert server.needs_attention() is True
