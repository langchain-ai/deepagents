"""Tests for MCP tool loading, caching, and config resolution."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Generator

    from langchain_mcp_adapters.client import Connection

from deepagents_code.mcp_auth import FileTokenStorage, MCPReauthRequiredError
from deepagents_code.mcp_tools import (
    MCPServerInfo,
    MCPSessionManager,
    MCPToolInfo,
    _apply_tool_filter,
    _check_remote_server,
    _check_stdio_server,
    _gather_bounded,
    _json_error_snippet,
    _load_tools_from_config,
    _normalize_mcp_arguments,
    classify_discovered_configs,
    discover_mcp_configs,
    extract_project_server_summaries,
    extract_stdio_server_commands,
    get_mcp_tools,
    load_mcp_config,
    load_mcp_config_lenient,
    merge_mcp_configs,
    resolve_and_load_mcp_tools,
)
from deepagents_code.project_utils import ProjectContext


def _make_mcp_tool(
    name: str,
    description: str = "",
    input_schema: dict | None = None,
) -> MagicMock:
    """Build a mock MCP `Tool` object suitable for conversion."""
    tool = MagicMock(spec=["name", "description", "inputSchema", "annotations", "meta"])
    tool.name = name
    tool.description = description
    tool.inputSchema = input_schema or {"type": "object", "properties": {}}
    tool.annotations = None
    tool.meta = None
    return tool


def _make_tool_page(
    tools: list[MagicMock],
    next_cursor: str | None = None,
) -> MagicMock:
    """Build a mock `list_tools` page result."""
    page = MagicMock(spec=["tools", "nextCursor"])
    page.tools = tools
    page.nextCursor = next_cursor
    return page


@pytest.fixture
def valid_config_data() -> dict:
    """Fixture providing a valid stdio server configuration."""
    return {
        "mcpServers": {
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                "env": {},
            }
        }
    }


@pytest.fixture
def write_config(tmp_path: Path) -> Callable[..., str]:
    """Write a JSON config dict to a temp file and return the path."""

    def _write(config_data: dict, filename: str = "mcp-config.json") -> str:
        config_file = tmp_path / filename
        config_file.write_text(json.dumps(config_data))
        return str(config_file)

    return _write


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect `Path.home()` and `DEFAULT_STATE_DIR` into a temp directory.

    `Path.home` is patched for code that resolves it at call time;
    `DEFAULT_STATE_DIR` is patched for code (like `mcp_auth._tokens_dir`)
    that pulls from the import-time-frozen constant in `model_config`.
    Without the second patch, `FileTokenStorage` reads/writes the real
    `~/.deepagents/.state/mcp-tokens/` directory, which leaks token state
    across tests and causes flakes (e.g. one test's `set_tokens` makes a
    later test's `get_tokens` return non-`None`).
    """
    fake = tmp_path / "home"
    fake.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake))
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_STATE_DIR",
        fake / ".deepagents" / ".state",
    )
    return fake


@pytest.fixture
def fake_create_session() -> Generator[tuple[AsyncMock, list[dict[str, Any]]]]:
    """Patch `create_session` and record passed connection configs."""
    session = AsyncMock()
    session.initialize = AsyncMock()
    session.list_tools = AsyncMock(return_value=_make_tool_page([]))

    recorded: list[dict[str, Any]] = []

    @asynccontextmanager
    async def _fake(
        connection: dict[str, Any],
        *,
        _mcp_callbacks: object | None = None,
    ) -> AsyncIterator[AsyncMock]:
        await asyncio.sleep(0)
        recorded.append(connection)
        yield session

    with patch("langchain_mcp_adapters.sessions.create_session", _fake):
        yield session, recorded


@pytest.fixture
def fake_tool_result() -> Any:  # noqa: ANN401
    """Build a valid `CallToolResult` for runtime tool tests."""
    from mcp.types import CallToolResult, TextContent

    return CallToolResult(content=[TextContent(type="text", text="ok")])


class TestLoadMCPConfig:
    """Test MCP configuration loading and validation."""

    def test_load_valid_config(
        self,
        write_config: Callable[..., str],
        valid_config_data: dict,
    ) -> None:
        """A valid config loads unchanged."""
        path = write_config(valid_config_data)
        assert load_mcp_config(path) == valid_config_data

    def test_load_config_auth_oauth_http_ok(
        self,
        write_config: Callable[..., str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """`auth: oauth` is valid on HTTP servers."""
        monkeypatch.setenv("MCP_TOKEN", "tok")
        path = write_config(
            {
                "mcpServers": {
                    "notion": {
                        "transport": "http",
                        "url": "https://mcp.notion.com/mcp",
                        "auth": "oauth",
                        "headers": {"X-Token": "${MCP_TOKEN}"},
                    }
                }
            }
        )

        config = load_mcp_config(path)
        assert config["mcpServers"]["notion"]["auth"] == "oauth"

    def test_load_config_auth_oauth_on_stdio_rejected(
        self,
        write_config: Callable[..., str],
    ) -> None:
        """`auth: oauth` is rejected on stdio servers."""
        path = write_config(
            {
                "mcpServers": {
                    "filesystem": {
                        "command": "npx",
                        "args": [],
                        "auth": "oauth",
                    }
                }
            }
        )

        with pytest.raises(ValueError, match=r"stdio.*oauth|oauth.*stdio"):
            load_mcp_config(path)

    def test_load_config_auth_oauth_with_authorization_header_rejected(
        self,
        write_config: Callable[..., str],
    ) -> None:
        """OAuth servers cannot also define a static `Authorization` header."""
        path = write_config(
            {
                "mcpServers": {
                    "notion": {
                        "transport": "http",
                        "url": "https://mcp.notion.com/mcp",
                        "auth": "oauth",
                        "headers": {"Authorization": "Bearer token"},
                    }
                }
            }
        )

        with pytest.raises(ValueError, match="Authorization"):
            load_mcp_config(path)

    @pytest.mark.parametrize(
        "oauth",
        [
            {"clientId": "id", "clientSecret": "secret"},
            {"clientId": "id", "clientSecret": "secret", "redirectUri": 1},
            {"clientId": "id", "clientSecret": "secret", "redirectUri": "https://example.com/callback"},
        ],
    )
    def test_load_config_rejects_invalid_configured_oauth_client(
        self,
        write_config: Callable[..., str],
        oauth: dict[str, object],
    ) -> None:
        """Configured OAuth clients must be complete and use a local callback."""
        path = write_config(
            {
                "mcpServers": {
                    "gmail": {
                        "type": "http",
                        "url": "https://mcp.example.com/mcp",
                        "oauth": oauth,
                    }
                }
            }
        )

        with pytest.raises((TypeError, ValueError), match=r"oauth|redirectUri"):
            load_mcp_config(path)

    def test_load_config_defers_configured_oauth_interpolation(
        self,
        write_config: Callable[..., str],
    ) -> None:
        """Unset configured-client variables do not break unrelated config loading."""
        path = write_config(
            {
                "mcpServers": {
                    "gmail": {
                        "type": "http",
                        "url": "https://mcp.example.com/mcp",
                        "oauth": {
                            "clientId": "${GMAIL_CLIENT_ID}",
                            "clientSecret": "${GMAIL_CLIENT_SECRET}",
                            "redirectUri": "${GMAIL_REDIRECT_URI}",
                        },
                    }
                }
            }
        )

        assert "gmail" in load_mcp_config(path)["mcpServers"]

    def test_load_config_rejects_configured_oauth_for_stdio_or_auth_header(
        self,
        write_config: Callable[..., str],
    ) -> None:
        """Configured credentials cannot be sent to local or header-auth servers."""
        client = {
            "clientId": "id",
            "clientSecret": "secret",
            "redirectUri": "http://localhost:8765/callback",
        }
        for server in (
            {"command": "local", "oauth": client},
            {
                "type": "http",
                "url": "https://mcp.example.com/mcp",
                "headers": {"Authorization": "Bearer static"},
                "oauth": client,
            },
        ):
            path = write_config({"mcpServers": {"gmail": server}})
            with pytest.raises(ValueError, match=r"OAuth|Authorization|stdio"):
                load_mcp_config(path)

    def test_load_config_unset_header_env_var_defers_to_activation(
        self,
        write_config: Callable[..., str],
    ) -> None:
        """Load succeeds on unset `${VAR}` — resolution is deferred per-server.

        This lets one bad reference surface as a single errored server
        rather than hiding every other entry in the same config file.
        """
        path = write_config(
            {
                "mcpServers": {
                    "linear": {
                        "transport": "http",
                        "url": "https://mcp.linear.app/mcp",
                        "headers": {"Authorization": "Bearer ${NO_SUCH_ENV_VAR}"},
                    }
                }
            }
        )

        config = load_mcp_config(path)
        assert "linear" in config["mcpServers"]

    def test_invalid_server_name_rejected(
        self,
        write_config: Callable[..., str],
    ) -> None:
        """Server names must remain path-safe."""
        path = write_config(
            {
                "mcpServers": {
                    "../evil": {
                        "transport": "http",
                        "url": "https://example.com/mcp",
                    }
                }
            }
        )

        with pytest.raises(ValueError, match="Invalid server name"):
            load_mcp_config(path)

    @pytest.mark.parametrize(
        "bad_name",
        ["../evil", "", "a/b", "a b", "slåck", "name.with.dot"],
    )
    def test_invalid_server_name_variants_rejected(
        self,
        write_config: Callable[..., str],
        bad_name: str,
    ) -> None:
        """Server names containing path-unsafe characters are rejected."""
        path = write_config(
            {
                "mcpServers": {
                    bad_name: {
                        "transport": "http",
                        "url": "https://example.com/mcp",
                    }
                }
            }
        )
        with pytest.raises(ValueError, match=r"Invalid server name|empty"):
            load_mcp_config(path)

    @pytest.mark.parametrize("good_name", ["slack-bot_1", "A", "z9", "_under"])
    def test_valid_server_names_accepted(
        self,
        write_config: Callable[..., str],
        good_name: str,
    ) -> None:
        """Alphanumeric, hyphen, and underscore server names pass validation."""
        path = write_config(
            {
                "mcpServers": {
                    good_name: {
                        "transport": "http",
                        "url": "https://example.com/mcp",
                    }
                }
            }
        )
        assert good_name in load_mcp_config(path)["mcpServers"]

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """Missing config path raises `FileNotFoundError`."""
        with pytest.raises(FileNotFoundError):
            load_mcp_config(str(tmp_path / "nope.json"))

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        """Malformed JSON raises `JSONDecodeError` with message context."""
        path = tmp_path / "bad.json"
        path.write_text("{not json")
        with pytest.raises(json.JSONDecodeError):
            load_mcp_config(str(path))

    def test_trailing_comma_error_has_hint_and_snippet(self, tmp_path: Path) -> None:
        """A trailing comma surfaces an actionable hint plus a caret snippet."""
        path = tmp_path / "bad.json"
        path.write_text(
            '{\n  "mcpServers": {\n    "fs": {\n      "command": "x",\n    },\n  }\n}'
        )
        with pytest.raises(json.JSONDecodeError) as exc_info:
            load_mcp_config(str(path))
        message = str(exc_info.value)
        assert "Invalid JSON in MCP config file" in message
        assert "trailing commas" in message
        assert "line" in message
        assert "column" in message
        # The caret must point at the offending comma, not merely be present:
        # find the caret line and the source line above it (both share the
        # same indent), then confirm the character under the `^` is the comma.
        msg_lines = message.splitlines()
        caret_idx = next(
            i for i, line in enumerate(msg_lines) if line.lstrip().startswith("^")
        )
        source_line = msg_lines[caret_idx - 1]
        caret_col = msg_lines[caret_idx].index("^")
        assert source_line[caret_col] == ","

    def test_missing_value_error_keeps_decoder_caret(self, tmp_path: Path) -> None:
        """A missing value keeps the caret at the decoder-reported token."""
        path = tmp_path / "bad.json"
        path.write_text('{"mcpServers": {"fs": {"command": }, "other": {}}}')
        with pytest.raises(json.JSONDecodeError) as exc_info:
            load_mcp_config(str(path))
        message = str(exc_info.value)
        assert "missing value" in message
        msg_lines = message.splitlines()
        caret_idx = next(
            i for i, line in enumerate(msg_lines) if line.lstrip().startswith("^")
        )
        source_line = msg_lines[caret_idx - 1]
        caret_col = msg_lines[caret_idx].index("^")
        assert source_line[caret_col] == "}"

    def test_comment_error_has_hint(self, tmp_path: Path) -> None:
        """A JSON file with a comment surfaces a comment-specific hint.

        The underlying decoder message is "Expecting property name...", so a
        passing assertion proves the comment heuristic fired and won the
        ordering rather than the generic property-name branch.
        """
        path = tmp_path / "bad.json"
        path.write_text('{\n  // not allowed\n  "mcpServers": {}\n}')
        with pytest.raises(json.JSONDecodeError) as exc_info:
            load_mcp_config(str(path))
        message = str(exc_info.value)
        assert "comments" in message
        # The comment hint must win over the generic property-name hint;
        # "missing key" is unique to that hint and absent from both the raw
        # decoder message and the comment hint.
        assert "missing key" not in message

    @pytest.mark.parametrize(
        ("content", "expected_hint_fragment"),
        [
            # "Expecting value" -> missing-value hint.
            ('{"mcpServers": }', "missing value"),
            # "Expecting property name..." (unquoted key) -> property-name hint.
            ("{\n  mcpServers: {}\n}", "property name"),
            # "Expecting ',' delimiter" -> missing-comma hint.
            ('{"a": 1 "b": 2}', "missing comma"),
        ],
    )
    def test_json_error_hint_branches(
        self, tmp_path: Path, content: str, expected_hint_fragment: str
    ) -> None:
        """Each recognized decoder message yields its specific hint."""
        path = tmp_path / "bad.json"
        path.write_text(content)
        with pytest.raises(json.JSONDecodeError) as exc_info:
            load_mcp_config(str(path))
        assert expected_hint_fragment in str(exc_info.value)

    def test_url_scheme_does_not_trigger_comment_hint(self, tmp_path: Path) -> None:
        """A `://` URL on the failing line is not mistaken for a comment.

        This is the entire reason `_looks_like_comment` checks `startswith`
        rather than substring containment; the guard must stay covered so a
        refactor cannot silently emit a bogus comment hint on URL configs.
        """
        path = tmp_path / "bad.json"
        # Missing comma after the URL value, so the error lands on the URL line.
        path.write_text(
            '{\n  "mcpServers": {\n    "remote": {\n'
            '      "url": "https://example.com" "type": "http"\n'
            "    }\n  }\n}"
        )
        with pytest.raises(json.JSONDecodeError) as exc_info:
            load_mcp_config(str(path))
        message = str(exc_info.value)
        assert "comments" not in message
        # The URL line is rendered in the snippet and treated as a delimiter
        # error, not a comment.
        assert "https://" in message
        assert "missing comma" in message

    def test_unrecognized_error_has_no_hint(self, tmp_path: Path) -> None:
        """An error matching no known pattern omits the hint line entirely."""
        path = tmp_path / "bad.json"
        path.write_text('{"mcpServers": {"fs": {"command": "x}}')
        with pytest.raises(json.JSONDecodeError) as exc_info:
            load_mcp_config(str(path))
        message = str(exc_info.value)
        assert "Invalid JSON in MCP config file" in message
        assert "Hint:" not in message

    def test_json_error_snippet_blank_line_returns_none(self) -> None:
        """A blank failing line yields no snippet (avoids a bare caret)."""
        assert _json_error_snippet("{\n\n}", 2, 1) is None

    def test_json_error_snippet_out_of_range_returns_none(self) -> None:
        """A line number past the source (e.g. truncated input) yields None."""
        assert _json_error_snippet("{}", 5, 1) is None

    def test_json_error_snippet_clamps_caret_to_line_end(self) -> None:
        """A column past the line length pins the caret to the line end."""
        source = '  "abc"'
        snippet = _json_error_snippet(source, 1, 999)
        assert snippet is not None
        caret_line = snippet.splitlines()[1]
        # Snippet lines carry a 4-space indent; the caret offset within the
        # source text must not exceed its length.
        assert caret_line.index("^") - 4 == len(source)

    def test_missing_mcpservers_field(self, write_config: Callable[..., str]) -> None:
        """Config without `mcpServers` field is rejected."""
        path = write_config({"other": {}})
        with pytest.raises(ValueError, match="mcpServers"):
            load_mcp_config(path)

    def test_mcpservers_wrong_type(self, write_config: Callable[..., str]) -> None:
        """`mcpServers` must be a dict."""
        path = write_config({"mcpServers": []})
        with pytest.raises(TypeError, match="dictionary"):
            load_mcp_config(path)

    def test_empty_mcpservers_rejected(self, write_config: Callable[..., str]) -> None:
        """Empty `mcpServers` is treated as a misconfiguration."""
        path = write_config({"mcpServers": {}})
        with pytest.raises(ValueError, match="empty"):
            load_mcp_config(path)

    def test_stdio_missing_command(self, write_config: Callable[..., str]) -> None:
        """Stdio servers must declare a `command`."""
        path = write_config({"mcpServers": {"fs": {"args": []}}})
        with pytest.raises(ValueError, match="command"):
            load_mcp_config(path)

    def test_stdio_args_wrong_type(self, write_config: Callable[..., str]) -> None:
        """Stdio `args` must be a list."""
        path = write_config({"mcpServers": {"fs": {"command": "x", "args": "oops"}}})
        with pytest.raises(TypeError, match="args"):
            load_mcp_config(path)

    def test_stdio_env_wrong_type(self, write_config: Callable[..., str]) -> None:
        """Stdio `env` must be a dict."""
        path = write_config({"mcpServers": {"fs": {"command": "x", "env": []}}})
        with pytest.raises(TypeError, match="env"):
            load_mcp_config(path)

    def test_remote_missing_url(self, write_config: Callable[..., str]) -> None:
        """Remote servers must declare a `url`."""
        path = write_config({"mcpServers": {"api": {"transport": "http"}}})
        with pytest.raises(ValueError, match="url"):
            load_mcp_config(path)

    def test_remote_headers_wrong_type(self, write_config: Callable[..., str]) -> None:
        """Remote `headers` must be a dict."""
        path = write_config(
            {
                "mcpServers": {
                    "api": {
                        "transport": "http",
                        "url": "https://example.com",
                        "headers": ["X-Bad", "value"],
                    }
                }
            }
        )
        with pytest.raises(TypeError, match="headers"):
            load_mcp_config(path)

    def test_unknown_transport_rejected(self, write_config: Callable[..., str]) -> None:
        """Unknown transport strings fail with a helpful message."""
        path = write_config({"mcpServers": {"s": {"transport": "ipc", "command": "x"}}})
        with pytest.raises(ValueError, match="unsupported transport"):
            load_mcp_config(path)

    def test_type_alias_for_transport(self, write_config: Callable[..., str]) -> None:
        """`type` is accepted as an alias for `transport`."""
        path = write_config(
            {"mcpServers": {"api": {"type": "sse", "url": "https://example.com"}}}
        )
        assert load_mcp_config(path)["mcpServers"]["api"]["type"] == "sse"

    def test_url_only_server_defaults_to_http_transport(
        self, write_config: Callable[..., str]
    ) -> None:
        """`url`-only entries are treated as HTTP remote servers.

        Matches Claude Code's `.mcp.json` convention: `{"url": "..."}` alone
        implies a remote server rather than stdio missing a `command`.
        """
        path = write_config(
            {"mcpServers": {"notion": {"url": "https://mcp.notion.com/mcp"}}}
        )
        # Should not raise; load_mcp_config validates by calling _resolve_server_type.
        assert "notion" in load_mcp_config(path)["mcpServers"]

    def test_url_only_inference_does_not_override_explicit_type(
        self, write_config: Callable[..., str]
    ) -> None:
        """Explicit `type` always wins over url-based inference."""
        path = write_config(
            {"mcpServers": {"api": {"type": "sse", "url": "https://example.com/mcp"}}}
        )
        loaded = load_mcp_config(path)["mcpServers"]["api"]
        assert loaded["type"] == "sse"

    def test_resolve_server_type_direct(self) -> None:
        """Direct unit test for `_resolve_server_type` inference rules."""
        from deepagents_code.mcp_tools import _resolve_server_type

        assert _resolve_server_type({"command": "x"}) == "stdio"
        assert _resolve_server_type({"url": "https://x"}) == "http"
        assert _resolve_server_type({"type": "sse", "url": "https://x"}) == "sse"
        assert _resolve_server_type({"transport": "http"}) == "http"
        assert _resolve_server_type({}) == "stdio"

    def test_streamable_http_alias_accepted(
        self, write_config: Callable[..., str]
    ) -> None:
        """`streamable_http` and `streamable-http` normalize to `http`."""
        from deepagents_code.mcp_tools import _resolve_server_type

        assert (
            _resolve_server_type({"transport": "streamable_http", "url": "https://x"})
            == "http"
        )
        assert (
            _resolve_server_type({"type": "streamable-http", "url": "https://x"})
            == "http"
        )
        path = write_config(
            {
                "mcpServers": {
                    "slack": {
                        "transport": "streamable_http",
                        "url": "https://slack.com/mcp",
                        "auth": "oauth",
                    }
                }
            }
        )
        assert "slack" in load_mcp_config(path)["mcpServers"]

    def test_stdio_with_url_rejected(self, write_config: Callable[..., str]) -> None:
        """Stdio + url is contradictory — url would be silently dropped."""
        path = write_config(
            {
                "mcpServers": {
                    "weird": {
                        "type": "stdio",
                        "command": "cat",
                        "url": "https://example.com/mcp",
                    }
                }
            }
        )
        with pytest.raises(ValueError, match=r"stdio.*url|url.*stdio"):
            load_mcp_config(path)

    def test_remote_with_command_rejected(
        self, write_config: Callable[..., str]
    ) -> None:
        """Remote type + command is contradictory — command silently dropped."""
        path = write_config(
            {
                "mcpServers": {
                    "weird": {
                        "type": "http",
                        "url": "https://example.com/mcp",
                        "command": "cat",
                    }
                }
            }
        )
        with pytest.raises(ValueError, match=r"remote.*command|command"):
            load_mcp_config(path)

    def test_mcp_config_error_is_value_error(self) -> None:
        """`MCPConfigError` subclasses `ValueError` for backward-compatible catching."""
        from deepagents_code.mcp_tools import MCPConfigError

        assert issubclass(MCPConfigError, ValueError)
        msg = "boom"
        with pytest.raises(ValueError, match="boom"):
            raise MCPConfigError(msg)


class TestDiscoverMcpConfigs:
    """Tests for file-system discovery of MCP config files."""

    def test_discovers_user_project_and_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """All three config locations are returned when present."""
        home = tmp_path / "home"
        project = tmp_path / "proj"
        (home / ".deepagents").mkdir(parents=True)
        (home / ".deepagents" / ".mcp.json").write_text("{}")
        (project / ".deepagents").mkdir(parents=True)
        (project / ".deepagents" / ".mcp.json").write_text("{}")
        (project / ".mcp.json").write_text("{}")
        monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
        monkeypatch.setattr(
            "deepagents_code.project_utils.find_project_root",
            lambda: project,
        )

        paths = discover_mcp_configs()
        assert len(paths) == 3
        assert any(str(p).endswith(".mcp.json") for p in paths)

    def test_no_configs_returns_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No discovered files yields an empty list without error."""
        home = tmp_path / "h"
        home.mkdir()
        monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
        monkeypatch.setattr(
            "deepagents_code.project_utils.find_project_root",
            lambda: None,
        )
        assert discover_mcp_configs() == []

    def test_explicit_project_context_overrides_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`project_context` overrides the fallback project root."""
        home = tmp_path / "home"
        home.mkdir()
        project = tmp_path / "p"
        (project / ".deepagents").mkdir(parents=True)
        (project / ".deepagents" / ".mcp.json").write_text("{}")
        monkeypatch.setattr(Path, "home", staticmethod(lambda: home))

        ctx = ProjectContext(user_cwd=project, project_root=project)
        paths = discover_mcp_configs(project_context=ctx)
        assert any(".deepagents" in str(p) for p in paths)


class TestLoadMcpConfigLenient:
    """Tests for `load_mcp_config_lenient` / `load_mcp_config_with_error`."""

    def test_missing_file_returns_none_without_error(self, tmp_path: Path) -> None:
        """Missing files are silent — not worth surfacing as errors."""
        from deepagents_code.mcp_tools import load_mcp_config_with_error

        cfg, err = load_mcp_config_with_error(tmp_path / "nope.json")
        assert cfg is None
        assert err is None

    def test_malformed_json_reports_error(self, tmp_path: Path) -> None:
        """Malformed JSON yields a populated error alongside `None`."""
        from deepagents_code.mcp_tools import load_mcp_config_with_error

        path = tmp_path / "bad.json"
        path.write_text("{not json")
        cfg, err = load_mcp_config_with_error(path)
        assert cfg is None
        assert err is not None

    def test_lenient_returns_none_for_invalid(
        self, write_config: Callable[..., str]
    ) -> None:
        """Legacy lenient API preserves the `None` return contract."""
        path = write_config({"mcpServers": {"fs": {"args": []}}})
        assert load_mcp_config_lenient(Path(path)) is None


class TestMCPServerInfoInvariants:
    """Tests for `MCPServerInfo.__post_init__` invariants."""

    def test_status_ok_rejects_error(self) -> None:
        """`status='ok'` cannot carry an error message."""
        with pytest.raises(ValueError, match="cannot carry an error"):
            MCPServerInfo(name="srv", transport="http", status="ok", error="oops")

    def test_status_error_requires_message(self) -> None:
        """Non-`ok` statuses require a non-`None` error."""
        with pytest.raises(ValueError, match="requires an error"):
            MCPServerInfo(name="srv", transport="http", status="error")

    def test_status_unauth_rejects_tools(self) -> None:
        """Failed servers can't also carry tools."""
        with pytest.raises(ValueError, match="cannot carry tools"):
            MCPServerInfo(
                name="srv",
                transport="http",
                status="unauthenticated",
                error="login",
                tools=(MCPToolInfo(name="t", description=""),),
            )


class TestMCPSessionManager:
    """Tests for lazy runtime session caching."""

    @patch("langchain_mcp_adapters.sessions.create_session")
    async def test_reuses_single_session_for_concurrent_first_use(
        self,
        mock_create_session: MagicMock,
    ) -> None:
        """Concurrent first-use only creates one live session."""
        session = AsyncMock()
        session.initialize = AsyncMock()

        @asynccontextmanager
        async def _fake_create_session(
            _connection: dict[str, Any],
            *,
            _mcp_callbacks: object | None = None,
        ) -> AsyncIterator[AsyncMock]:
            await asyncio.sleep(0.01)
            yield session

        mock_create_session.side_effect = _fake_create_session

        manager = MCPSessionManager(
            connections={
                "filesystem": {
                    "transport": "stdio",
                    "command": "npx",
                    "args": [],
                }
            }
        )

        first, second = await asyncio.gather(
            manager.get_session("filesystem"),
            manager.get_session("filesystem"),
        )

        assert first is session
        assert second is session
        mock_create_session.assert_called_once()

    @patch("langchain_mcp_adapters.sessions.create_session")
    async def test_cleanup_closes_cached_sessions_and_blocks_future_creation(
        self,
        mock_create_session: MagicMock,
    ) -> None:
        """Cleanup closes live sessions and rejects future creation."""
        session = AsyncMock()
        session.initialize = AsyncMock()
        exit_mock = AsyncMock(return_value=None)

        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=session)
        cm.__aexit__ = exit_mock
        mock_create_session.return_value = cm

        manager = MCPSessionManager(
            connections={
                "filesystem": {
                    "transport": "stdio",
                    "command": "npx",
                    "args": [],
                }
            }
        )

        await manager.get_session("filesystem")
        await manager.cleanup()

        exit_mock.assert_awaited_once()
        with pytest.raises(RuntimeError, match="after cleanup"):
            await manager.get_session("filesystem")

    async def test_configure_noop_when_connections_match(self) -> None:
        """`configure` is a no-op if the same connection dict is re-applied."""
        conn = {"filesystem": {"transport": "stdio", "command": "npx", "args": []}}
        manager = MCPSessionManager(connections=conn)  # ty: ignore
        # Should not raise even without any sessions yet.
        manager.configure(dict(conn))  # ty: ignore

    @pytest.mark.usefixtures("fake_home")
    async def test_configure_accepts_equivalent_oauth_connections(self) -> None:
        """Fresh OAuth provider instances do not count as reconfiguration."""
        from deepagents_code.mcp_auth import FileTokenStorage, build_oauth_provider

        session = AsyncMock()
        session.initialize = AsyncMock()
        url = "https://mcp.notion.com/mcp"

        @asynccontextmanager
        async def _fake(
            _conn: dict[str, Any],
            *,
            _mcp_callbacks: object | None = None,
        ) -> AsyncIterator[AsyncMock]:
            await asyncio.sleep(0)
            yield session

        def _connection() -> Connection:
            return cast(
                "Connection",
                {
                    "transport": "streamable_http",
                    "url": url,
                    "auth": build_oauth_provider(
                        server_name="notion",
                        server_url=url,
                        storage=FileTokenStorage("notion", server_url=url),
                        interactive=False,
                    ),
                },
            )

        manager = MCPSessionManager(connections={"notion": _connection()})
        with patch("langchain_mcp_adapters.sessions.create_session", _fake):
            await manager.get_session("notion")

        manager.configure({"notion": _connection()})
        await manager.cleanup()

    async def test_configure_after_sessions_rejects_changes(self) -> None:
        """Changing connections after sessions exist raises `RuntimeError`."""
        session = AsyncMock()
        session.initialize = AsyncMock()

        @asynccontextmanager
        async def _fake(
            _conn: dict[str, Any],
            *,
            _mcp_callbacks: object | None = None,
        ) -> AsyncIterator[AsyncMock]:
            await asyncio.sleep(0)
            yield session

        conn = {"filesystem": {"transport": "stdio", "command": "npx", "args": []}}
        manager = MCPSessionManager(connections=conn)  # ty: ignore
        with patch("langchain_mcp_adapters.sessions.create_session", _fake):
            await manager.get_session("filesystem")

        with pytest.raises(RuntimeError, match="Cannot reconfigure"):
            manager.configure(
                {"other": {"transport": "stdio", "command": "x", "args": []}}
            )
        await manager.cleanup()

    async def test_configure_on_closed_manager_raises(self) -> None:
        """Reconfiguring a closed manager raises `RuntimeError`."""
        manager = MCPSessionManager()
        await manager.cleanup()
        with pytest.raises(RuntimeError, match="closed MCP session manager"):
            manager.configure({})

    async def test_invalidate_with_mismatched_identity_skips(self) -> None:
        """`expected_session` identity check prevents racing evictions."""
        session_a = AsyncMock()
        session_a.initialize = AsyncMock()
        exit_mock = AsyncMock(return_value=None)

        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=session_a)
        cm.__aexit__ = exit_mock

        manager = MCPSessionManager(
            connections={
                "filesystem": {"transport": "stdio", "command": "x", "args": []}
            }
        )
        with patch("langchain_mcp_adapters.sessions.create_session", return_value=cm):
            cached = await manager.get_session("filesystem")
        assert cached is session_a

        stale = AsyncMock()
        await manager.invalidate("filesystem", expected_session=stale)
        # Cached session is still live — identity mismatch short-circuited.
        exit_mock.assert_not_awaited()
        await manager.cleanup()


class TestTransientErrorDetection:
    """Tests for `_is_transient_session_error` classification."""

    @pytest.mark.parametrize(
        "exc",
        [
            BrokenPipeError("pipe"),
            ConnectionAbortedError("abort"),
            ConnectionResetError("reset"),
            EOFError("eof"),
            asyncio.IncompleteReadError(b"", 1),
        ],
    )
    def test_stdlib_exceptions_are_transient(self, exc: BaseException) -> None:
        """Standard-library transport errors always classify as transient."""
        from deepagents_code.mcp_tools import _is_transient_session_error

        assert _is_transient_session_error(exc)

    def test_anyio_closed_resource_is_transient(self) -> None:
        """Anyio's `ClosedResourceError` also classifies as transient."""
        import anyio

        from deepagents_code.mcp_tools import _is_transient_session_error

        assert _is_transient_session_error(anyio.ClosedResourceError())

    def test_unrelated_exception_is_not_transient(self) -> None:
        """Non-transport errors do not trigger retry."""
        from deepagents_code.mcp_tools import _is_transient_session_error

        assert not _is_transient_session_error(RuntimeError("boom"))
        assert not _is_transient_session_error(ValueError("bad"))


class TestGetMCPTools:
    """Test MCP tool loading from configuration."""

    @pytest.fixture(autouse=True)
    def _bypass_health_checks(self) -> Generator[None]:
        """Bypass pre-flight health checks for tests in this class."""
        with (
            patch("deepagents_code.mcp_tools._check_stdio_server"),
            patch(
                "deepagents_code.mcp_tools._check_remote_server",
                new_callable=AsyncMock,
            ),
        ):
            yield

    async def test_get_mcp_tools_success(
        self,
        write_config: Callable[..., str],
        fake_create_session: tuple[AsyncMock, list[dict[str, Any]]],
    ) -> None:
        """Discovery returns tools and metadata without opening runtime sessions."""
        path = write_config(
            {"mcpServers": {"srv": {"command": "node", "args": ["server.js"]}}}
        )
        session, recorded = fake_create_session
        session.list_tools = AsyncMock(
            return_value=_make_tool_page(
                [
                    _make_mcp_tool("read_file", "Read a file"),
                    _make_mcp_tool("write_file", "Write a file"),
                ]
            )
        )

        tools, manager, server_infos = await get_mcp_tools(path)

        assert isinstance(manager, MCPSessionManager)
        assert [tool.name for tool in tools] == ["srv_read_file", "srv_write_file"]
        assert recorded == [
            {
                "command": "node",
                "args": ["server.js"],
                "env": None,
                "transport": "stdio",
            }
        ]
        empty_schema: dict[str, Any] = {"type": "object", "properties": {}}
        assert server_infos == [
            MCPServerInfo(
                name="srv",
                transport="stdio",
                tools=(
                    MCPToolInfo(
                        name="srv_read_file",
                        description="Read a file",
                        input_schema=empty_schema,
                    ),
                    MCPToolInfo(
                        name="srv_write_file",
                        description="Write a file",
                        input_schema=empty_schema,
                    ),
                ),
            )
        ]
        await manager.cleanup()

    async def test_discovery_failure_marks_server_error(
        self,
        write_config: Callable[..., str],
    ) -> None:
        """Discovery failures are reported per-server instead of aborting load."""
        path = write_config(
            {"mcpServers": {"srv": {"command": "node", "args": ["server.js"]}}}
        )

        @asynccontextmanager
        async def _fake(
            _connection: dict[str, Any],
            *,
            _mcp_callbacks: object | None = None,
        ) -> AsyncIterator[None]:
            await asyncio.sleep(0)
            msg = "boom"
            raise RuntimeError(msg)
            yield

        with patch("langchain_mcp_adapters.sessions.create_session", _fake):
            tools, manager, server_infos = await get_mcp_tools(path)

        assert tools == []
        assert isinstance(manager, MCPSessionManager)
        assert server_infos[0].status == "error"
        assert "boom" in (server_infos[0].error or "")
        await manager.cleanup()

    async def test_stdio_health_check_failure_is_non_fatal(
        self,
        write_config: Callable[..., str],
    ) -> None:
        """A failing stdio pre-flight becomes server status, not a hard error."""
        path = write_config({"mcpServers": {"srv": {"command": "missing", "args": []}}})

        with patch(
            "deepagents_code.mcp_tools._check_stdio_server",
            side_effect=RuntimeError("command missing"),
        ):
            tools, manager, server_infos = await get_mcp_tools(path)

        assert tools == []
        assert server_infos[0].status == "error"
        assert "command missing" in (server_infos[0].error or "")
        assert manager is not None
        await manager.cleanup()

    async def test_remote_headers_are_resolved_and_passed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_create_session: tuple[AsyncMock, list[dict[str, Any]]],
    ) -> None:
        """Resolved static headers are attached to remote connections."""
        monkeypatch.setenv("DA_TOKEN", "tok-123")
        _session, recorded = fake_create_session
        config = {
            "mcpServers": {
                "linear": {
                    "transport": "http",
                    "url": "https://mcp.linear.app/mcp",
                    "headers": {"Authorization": "Bearer ${DA_TOKEN}"},
                }
            }
        }

        await _load_tools_from_config(config)
        assert recorded[0]["headers"] == {"Authorization": "Bearer tok-123"}

    async def test_empty_env_is_coerced_to_none(
        self,
        fake_create_session: tuple[AsyncMock, list[dict[str, Any]]],
    ) -> None:
        """Empty stdio env dicts are normalized to `None`."""
        _session, recorded = fake_create_session
        config = {
            "mcpServers": {
                "srv": {
                    "command": "node",
                    "args": ["server.js"],
                    "env": {},
                }
            }
        }

        await _load_tools_from_config(config)
        assert recorded[0]["env"] is None

    async def test_input_schema_is_carried_into_mcp_tool_info(
        self,
        write_config: Callable[..., str],
        fake_create_session: tuple[AsyncMock, list[dict[str, Any]]],
    ) -> None:
        """Per-tool `inputSchema` lands on `MCPToolInfo.input_schema`."""
        path = write_config(
            {"mcpServers": {"srv": {"command": "node", "args": ["server.js"]}}}
        )
        session, _recorded = fake_create_session
        rich_schema = {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "depth": {"type": "integer"},
            },
            "required": ["path"],
        }
        session.list_tools = AsyncMock(
            return_value=_make_tool_page(
                [
                    _make_mcp_tool(
                        "read_file", "Read a file", input_schema=rich_schema
                    ),
                ]
            )
        )

        _tools, manager, server_infos = await get_mcp_tools(path)

        assert server_infos[0].tools[0].input_schema == rich_schema
        await manager.cleanup()  # ty: ignore

    async def test_input_schema_extraction_survives_attribute_error(
        self,
        write_config: Callable[..., str],
        fake_create_session: tuple[AsyncMock, list[dict[str, Any]]],
    ) -> None:
        """If `mcp_tool.inputSchema` access raises, schema falls back to `None`.

        The downstream LangChain conversion still needs `inputSchema`, so we
        give the tool a minimal valid schema there but use a custom property
        to make `getattr` raise during the schema-extraction path.
        """
        path = write_config(
            {"mcpServers": {"srv": {"command": "node", "args": ["server.js"]}}}
        )
        session, _recorded = fake_create_session

        class _ExplodingSchemaTool:
            name = "read_file"
            description = "Read a file"
            annotations = None
            meta = None
            _access_count = 0

            @property
            def inputSchema(self) -> dict[str, Any]:  # noqa: N802 - matches MCP spec
                self._access_count += 1
                if self._access_count == 1:
                    # Allow LangChain conversion to succeed first.
                    return {"type": "object", "properties": {}}
                msg = "metadata access failed"
                raise AttributeError(msg)

        session.list_tools = AsyncMock(
            return_value=_make_tool_page(
                [_ExplodingSchemaTool()]  # ty: ignore
            )
        )

        _tools, manager, server_infos = await get_mcp_tools(path)

        assert server_infos[0].tools[0].input_schema is None
        await manager.cleanup()  # ty: ignore

    async def test_input_schema_pairs_when_tool_name_starts_with_server_prefix(
        self,
        write_config: Callable[..., str],
        fake_create_session: tuple[AsyncMock, list[dict[str, Any]]],
    ) -> None:
        """A bare tool name that itself starts with the server prefix still pairs.

        Server `srv` exposing tool `srv_read` produces LangChain name
        `srv_srv_read`. The schema dict is keyed by the LC name directly, so
        the lookup is unambiguous regardless of how the bare name happens
        to look. (Regression guard for the prior `removeprefix`-based path.)
        """
        path = write_config(
            {"mcpServers": {"srv": {"command": "node", "args": ["server.js"]}}}
        )
        session, _recorded = fake_create_session
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        session.list_tools = AsyncMock(
            return_value=_make_tool_page(
                [_make_mcp_tool("srv_read", "Read", input_schema=schema)]
            )
        )

        _tools, manager, server_infos = await get_mcp_tools(path)

        info = server_infos[0]
        assert [t.name for t in info.tools] == ["srv_srv_read"]
        assert info.tools[0].input_schema == schema
        await manager.cleanup()  # ty: ignore

    async def test_input_schema_paired_to_post_filter_tools(
        self,
        write_config: Callable[..., str],
        fake_create_session: tuple[AsyncMock, list[dict[str, Any]]],
    ) -> None:
        """When `disabledTools` filters out a tool, surviving tools keep schemas."""
        path = write_config(
            {
                "mcpServers": {
                    "srv": {
                        "command": "node",
                        "args": ["server.js"],
                        "disabledTools": ["write_file"],
                    }
                }
            }
        )
        session, _recorded = fake_create_session
        read_schema = {"type": "object", "properties": {"path": {"type": "string"}}}
        write_schema = {"type": "object", "properties": {"path": {"type": "string"}}}
        session.list_tools = AsyncMock(
            return_value=_make_tool_page(
                [
                    _make_mcp_tool("read_file", "Read", input_schema=read_schema),
                    _make_mcp_tool("write_file", "Write", input_schema=write_schema),
                ]
            )
        )

        _tools, manager, server_infos = await get_mcp_tools(path)

        names = [t.name for t in server_infos[0].tools]
        assert names == ["srv_read_file"]
        assert server_infos[0].tools[0].input_schema == read_schema
        await manager.cleanup()  # ty: ignore


@pytest.mark.usefixtures("fake_home")
class TestLoadToolsFromConfigOAuth:
    """OAuth-specific MCP loading behavior."""

    @pytest.fixture(autouse=True)
    def _bypass_health_checks(self) -> Generator[None]:
        """Bypass remote health checks for tests in this class."""
        with patch(
            "deepagents_code.mcp_tools._check_remote_server",
            new_callable=AsyncMock,
        ):
            yield

    async def test_missing_tokens_skip_server_with_login_hint(
        self,
    ) -> None:
        """An OAuth server without tokens is marked unauthenticated."""
        config = {
            "mcpServers": {
                "notion": {
                    "transport": "http",
                    "url": "https://mcp.notion.com/mcp",
                    "auth": "oauth",
                }
            }
        }

        tools, manager, server_infos = await _load_tools_from_config(config)

        assert tools == []
        assert isinstance(manager, MCPSessionManager)
        assert server_infos[0].status == "unauthenticated"
        assert "re-authentication" in (server_infos[0].error or "")
        await manager.cleanup()

    async def test_existing_tokens_attach_oauth_provider(
        self,
    ) -> None:
        """Stored tokens attach an OAuth provider to the runtime connection."""
        from mcp.client.auth import OAuthClientProvider
        from mcp.shared.auth import OAuthToken

        storage = FileTokenStorage(
            "notion",
            server_url="https://mcp.notion.com/mcp",
        )
        await storage.set_tokens(OAuthToken(access_token="at", token_type="Bearer"))

        recorded: list[dict[str, Any]] = []
        session = AsyncMock()
        session.initialize = AsyncMock()
        session.list_tools = AsyncMock(return_value=_make_tool_page([]))

        @asynccontextmanager
        async def _fake(
            connection: dict[str, Any],
            *,
            _mcp_callbacks: object | None = None,
        ) -> AsyncIterator[AsyncMock]:
            await asyncio.sleep(0)
            recorded.append(connection)
            yield session

        with patch("langchain_mcp_adapters.sessions.create_session", _fake):
            config = {
                "mcpServers": {
                    "notion": {
                        "transport": "http",
                        "url": "https://mcp.notion.com/mcp",
                        "auth": "oauth",
                    }
                }
            }
            tools, manager, _ = await _load_tools_from_config(config)

        assert tools == []
        assert isinstance(manager, MCPSessionManager)
        assert isinstance(recorded[0].get("auth"), OAuthClientProvider)
        await manager.cleanup()

    async def test_configured_client_attaches_provider_without_saved_tokens(
        self,
    ) -> None:
        """Configured OAuth clients load at runtime before the first login."""
        from mcp.client.auth import OAuthClientProvider

        recorded: list[dict[str, Any]] = []
        session = AsyncMock()
        session.initialize = AsyncMock()
        session.list_tools = AsyncMock(return_value=_make_tool_page([]))

        @asynccontextmanager
        async def _fake(
            connection: dict[str, Any],
            *,
            _mcp_callbacks: object | None = None,
        ) -> AsyncIterator[AsyncMock]:
            await asyncio.sleep(0)
            recorded.append(connection)
            yield session

        with patch("langchain_mcp_adapters.sessions.create_session", _fake):
            config = {
                "mcpServers": {
                    "gmail": {
                        "transport": "http",
                        "url": "https://mcp.example.com/mcp",
                        "oauth": {
                            "clientId": "configured-client",
                            "clientSecret": "configured-secret",
                            "redirectUri": "http://localhost:8765/callback",
                        },
                    }
                }
            }
            tools, manager, _ = await _load_tools_from_config(config)

        assert tools == []
        assert isinstance(manager, MCPSessionManager)
        assert isinstance(recorded[0].get("auth"), OAuthClientProvider)
        await manager.cleanup()

    async def test_discovery_reauth_marks_server_unauthenticated(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """OAuth re-auth during discovery is surfaced without warning tracebacks."""
        from mcp.shared.auth import OAuthToken

        storage = FileTokenStorage(
            "notion",
            server_url="https://mcp.notion.com/mcp",
        )
        await storage.set_tokens(OAuthToken(access_token="at", token_type="Bearer"))

        @asynccontextmanager
        async def _fake(
            _connection: dict[str, Any],
            *,
            _mcp_callbacks: object | None = None,
        ) -> AsyncIterator[None]:
            await asyncio.sleep(0)
            msg = "discovery failed"
            raise ExceptionGroup(msg, [MCPReauthRequiredError("notion")])
            yield

        caplog.set_level(logging.WARNING, logger="deepagents_code.mcp_tools")

        with patch("langchain_mcp_adapters.sessions.create_session", _fake):
            config = {
                "mcpServers": {
                    "notion": {
                        "transport": "http",
                        "url": "https://mcp.notion.com/mcp",
                        "auth": "oauth",
                    }
                }
            }
            tools, manager, server_infos = await _load_tools_from_config(config)

        assert tools == []
        assert isinstance(manager, MCPSessionManager)
        assert server_infos[0].status == "unauthenticated"
        assert "re-authentication" in (server_infos[0].error or "")
        warning_records = [
            record
            for record in caplog.records
            if record.name == "deepagents_code.mcp_tools"
            and record.levelno == logging.WARNING
        ]
        assert warning_records
        assert all(record.exc_info is None for record in warning_records)
        assert "Exception Group Traceback" not in caplog.text
        await manager.cleanup()

    async def test_stored_tokens_attach_provider_without_explicit_oauth(
        self,
    ) -> None:
        """Stored tokens attach a provider even when `auth: oauth` is absent."""
        from mcp.client.auth import OAuthClientProvider
        from mcp.shared.auth import OAuthToken

        storage = FileTokenStorage("notion", server_url="https://mcp.notion.com/mcp")
        await storage.set_tokens(OAuthToken(access_token="at", token_type="Bearer"))

        recorded: list[dict[str, Any]] = []
        session = AsyncMock()
        session.initialize = AsyncMock()
        session.list_tools = AsyncMock(return_value=_make_tool_page([]))

        @asynccontextmanager
        async def _fake(
            connection: dict[str, Any],
            *,
            _mcp_callbacks: object | None = None,
        ) -> AsyncIterator[AsyncMock]:
            await asyncio.sleep(0)
            recorded.append(connection)
            yield session

        with patch("langchain_mcp_adapters.sessions.create_session", _fake):
            config = {
                "mcpServers": {
                    "notion": {
                        "transport": "http",
                        "url": "https://mcp.notion.com/mcp",
                    }
                }
            }
            tools, manager, _ = await _load_tools_from_config(config)

        assert tools == []
        assert isinstance(manager, MCPSessionManager)
        assert isinstance(recorded[0].get("auth"), OAuthClientProvider)
        await manager.cleanup()

    async def test_authorization_header_skips_stored_oauth_without_explicit_oauth(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Static `Authorization` headers take precedence over stored OAuth."""
        from mcp.shared.auth import OAuthToken

        monkeypatch.setenv("DA_TOKEN", "tok-123")
        storage = FileTokenStorage("notion", server_url="https://mcp.notion.com/mcp")
        await storage.set_tokens(OAuthToken(access_token="at", token_type="Bearer"))

        recorded: list[dict[str, Any]] = []
        session = AsyncMock()
        session.initialize = AsyncMock()
        session.list_tools = AsyncMock(return_value=_make_tool_page([]))

        @asynccontextmanager
        async def _fake(
            connection: dict[str, Any],
            *,
            _mcp_callbacks: object | None = None,
        ) -> AsyncIterator[AsyncMock]:
            await asyncio.sleep(0)
            recorded.append(connection)
            yield session

        with patch("langchain_mcp_adapters.sessions.create_session", _fake):
            config = {
                "mcpServers": {
                    "notion": {
                        "transport": "http",
                        "url": "https://mcp.notion.com/mcp",
                        "headers": {"Authorization": "Bearer ${DA_TOKEN}"},
                    }
                }
            }
            tools, manager, _ = await _load_tools_from_config(config)

        assert tools == []
        assert isinstance(manager, MCPSessionManager)
        assert recorded[0]["headers"] == {"Authorization": "Bearer tok-123"}
        assert "auth" not in recorded[0]
        await manager.cleanup()

    async def test_discovery_401_challenge_marks_unauthenticated(self) -> None:
        """A 401 OAuth challenge during discovery is surfaced as unauthenticated."""
        request = httpx.Request("GET", "https://mcp.notion.com/mcp")
        response = httpx.Response(
            401,
            headers={
                "WWW-Authenticate": (
                    'Bearer resource_metadata="https://mcp.notion.com/.well-known/'
                    'oauth-protected-resource"'
                )
            },
            request=request,
        )
        challenge = httpx.HTTPStatusError("boom", request=request, response=response)

        @asynccontextmanager
        async def _fake(
            _connection: dict[str, Any],
            *,
            _mcp_callbacks: object | None = None,
        ) -> AsyncIterator[None]:
            await asyncio.sleep(0)
            raise challenge
            yield

        with patch("langchain_mcp_adapters.sessions.create_session", _fake):
            config = {
                "mcpServers": {
                    "notion": {
                        "transport": "http",
                        "url": "https://mcp.notion.com/mcp",
                    }
                }
            }
            tools, manager, server_infos = await _load_tools_from_config(config)

        assert tools == []
        assert isinstance(manager, MCPSessionManager)
        assert server_infos[0].status == "unauthenticated"
        assert "mcp login notion" in (server_infos[0].error or "")
        await manager.cleanup()

    async def test_discovery_401_without_challenge_stays_error(self) -> None:
        """A 401 lacking `WWW-Authenticate` is not treated as an OAuth challenge."""
        request = httpx.Request("GET", "https://mcp.notion.com/mcp")
        response = httpx.Response(401, request=request)
        error = httpx.HTTPStatusError("boom", request=request, response=response)

        @asynccontextmanager
        async def _fake(
            _connection: dict[str, Any],
            *,
            _mcp_callbacks: object | None = None,
        ) -> AsyncIterator[None]:
            await asyncio.sleep(0)
            raise error
            yield

        with patch("langchain_mcp_adapters.sessions.create_session", _fake):
            config = {
                "mcpServers": {
                    "notion": {
                        "transport": "http",
                        "url": "https://mcp.notion.com/mcp",
                    }
                }
            }
            tools, manager, server_infos = await _load_tools_from_config(config)

        assert tools == []
        assert isinstance(manager, MCPSessionManager)
        assert server_infos[0].status == "error"
        await manager.cleanup()

    async def test_discovery_401_basic_challenge_stays_error(self) -> None:
        """A non-OAuth auth challenge is not treated as an MCP login prompt."""
        request = httpx.Request("GET", "https://mcp.notion.com/mcp")
        response = httpx.Response(
            401,
            headers={"WWW-Authenticate": 'Basic realm="mcp"'},
            request=request,
        )
        error = httpx.HTTPStatusError("boom", request=request, response=response)

        @asynccontextmanager
        async def _fake(
            _connection: dict[str, Any],
            *,
            _mcp_callbacks: object | None = None,
        ) -> AsyncIterator[None]:
            await asyncio.sleep(0)
            raise error
            yield

        with patch("langchain_mcp_adapters.sessions.create_session", _fake):
            config = {
                "mcpServers": {
                    "notion": {
                        "transport": "http",
                        "url": "https://mcp.notion.com/mcp",
                    }
                }
            }
            tools, manager, server_infos = await _load_tools_from_config(config)

        assert tools == []
        assert isinstance(manager, MCPSessionManager)
        assert server_infos[0].status == "error"
        await manager.cleanup()

    async def test_discovery_401_challenge_marks_unauthenticated_sse(self) -> None:
        """The 401 challenge classification also applies to SSE transports."""
        request = httpx.Request("GET", "https://mcp.notion.com/sse")
        response = httpx.Response(
            401,
            headers={
                "WWW-Authenticate": (
                    'Bearer resource_metadata="https://mcp.notion.com/.well-known/'
                    'oauth-protected-resource"'
                )
            },
            request=request,
        )
        challenge = httpx.HTTPStatusError("boom", request=request, response=response)

        @asynccontextmanager
        async def _fake(
            _connection: dict[str, Any],
            *,
            _mcp_callbacks: object | None = None,
        ) -> AsyncIterator[None]:
            await asyncio.sleep(0)
            raise challenge
            yield

        with patch("langchain_mcp_adapters.sessions.create_session", _fake):
            config = {
                "mcpServers": {
                    "notion": {
                        "transport": "sse",
                        "url": "https://mcp.notion.com/sse",
                    }
                }
            }
            tools, manager, server_infos = await _load_tools_from_config(config)

        assert tools == []
        assert isinstance(manager, MCPSessionManager)
        assert server_infos[0].transport == "sse"
        assert server_infos[0].status == "unauthenticated"
        assert "mcp login notion" in (server_infos[0].error or "")
        await manager.cleanup()


class TestResolveAndLoadMcpTools:
    """Test the unified resolve-and-load entrypoint."""

    async def test_no_mcp_returns_empty(self) -> None:
        """`no_mcp=True` returns immediately."""
        tools, manager, infos = await resolve_and_load_mcp_tools(no_mcp=True)
        assert tools == []
        assert manager is None
        assert infos == []

    @patch("deepagents_code.mcp_tools._warm_mcp_adapter_imports")
    @patch("deepagents_code.mcp_tools.discover_mcp_configs")
    async def test_no_adapter_warmup_when_no_active_servers(
        self,
        mock_discover: MagicMock,
        mock_warm: MagicMock,
    ) -> None:
        """With no configured servers, MCP adapters are never imported.

        `_warm_mcp_adapter_imports` (and the adapter imports that follow it)
        live inside `_load_tools_from_config`, which the resolver never reaches
        when discovery yields no servers — so the warmup must not run.
        """
        mock_discover.return_value = []

        tools, manager, infos = await resolve_and_load_mcp_tools(no_mcp=False)

        assert tools == []
        assert manager is None
        assert infos == []
        mock_warm.assert_not_called()

    @patch("deepagents_code.mcp_tools._load_tools_from_config")
    @patch("deepagents_code.mcp_tools.discover_mcp_configs")
    async def test_explicit_path_merges_with_discovery(
        self,
        mock_discover: MagicMock,
        mock_load: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """Explicit config is merged on top of auto-discovered configs."""
        discovered = tmp_path / "discovered.json"
        discovered.write_text(
            json.dumps({"mcpServers": {"fs": {"command": "npx", "args": []}}})
        )
        explicit = tmp_path / "explicit.json"
        explicit.write_text(
            json.dumps({"mcpServers": {"search": {"command": "brave", "args": []}}})
        )
        mock_discover.return_value = [discovered]
        mock_load.return_value = ([], MCPSessionManager(), [])

        await resolve_and_load_mcp_tools(
            explicit_config_path=str(explicit),
            trust_project_mcp=True,
        )

        merged = mock_load.call_args.args[0]
        assert "fs" in merged["mcpServers"]
        assert "search" in merged["mcpServers"]

    @patch("deepagents_code.mcp_tools._load_tools_from_config")
    @patch("deepagents_code.mcp_tools.discover_mcp_configs")
    async def test_stateless_and_manager_forwarded(
        self,
        mock_discover: MagicMock,
        mock_load: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """Server-mode kwargs are forwarded into the shared loader."""
        cfg = tmp_path / "mcp.json"
        cfg.write_text(
            json.dumps({"mcpServers": {"fs": {"command": "npx", "args": []}}})
        )
        manager = MCPSessionManager()
        mock_discover.return_value = [cfg]
        mock_load.return_value = ([], None, [])

        await resolve_and_load_mcp_tools(
            trust_project_mcp=True,
            stateless=True,
            session_manager=manager,
        )

        assert mock_load.call_args.kwargs["stateless"] is True
        assert mock_load.call_args.kwargs["session_manager"] is manager

    async def test_explicit_missing_path_raises(self, tmp_path: Path) -> None:
        """Missing explicit config remains fatal."""
        with pytest.raises(FileNotFoundError):
            await resolve_and_load_mcp_tools(
                explicit_config_path=str(tmp_path / "missing.json")
            )

    async def test_invalid_explicit_config_raises(self, tmp_path: Path) -> None:
        """Invalid explicit config remains fatal."""
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")

        with pytest.raises(json.JSONDecodeError):
            await resolve_and_load_mcp_tools(explicit_config_path=str(bad))

    @patch("deepagents_code.mcp_tools._load_tools_from_config")
    @patch("deepagents_code.mcp_tools.classify_discovered_configs")
    @patch("deepagents_code.mcp_tools.discover_mcp_configs")
    async def test_malformed_project_config_without_summaries_is_nonfatal(
        self,
        mock_discover: MagicMock,
        mock_classify: MagicMock,
        mock_load: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """Malformed-only project configs are reported instead of crashing."""
        project_cfg = tmp_path / ".mcp.json"
        project_cfg.write_text(
            json.dumps({"mcpServers": {"bad": ["not", "a", "dict"]}})
        )
        mock_discover.return_value = [project_cfg]
        mock_classify.return_value = ([], [project_cfg])
        mock_load.return_value = ([], None, [])

        tools, manager, infos = await resolve_and_load_mcp_tools(
            trust_project_mcp=True,
        )

        assert tools == []
        assert manager is None
        assert mock_load.call_count == 0
        assert len(infos) == 1
        assert infos[0].name == "<config:.mcp.json>"
        assert infos[0].status == "error"
        assert "must be a dictionary" in (infos[0].error or "")

    @patch("deepagents_code.mcp_tools._load_tools_from_config")
    @patch("deepagents_code.mcp_tools.classify_discovered_configs")
    @patch("deepagents_code.mcp_tools.discover_mcp_configs")
    async def test_untrusted_project_remote_dropped_when_flag_false(
        self,
        mock_discover: MagicMock,
        mock_classify: MagicMock,
        mock_load: AsyncMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Project remote MCP entries do not reach the loader without trust.

        Guards against SSRF and `${VAR}` header exfiltration via attacker
        URLs in `.mcp.json` (Corridor findings c419138c, 337d33ee).
        """
        project_cfg = tmp_path / ".mcp.json"
        project_cfg.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "evil": {
                            "transport": "http",
                            "url": "http://169.254.169.254",
                            "headers": {"X-Token": "${OPENAI_API_KEY}"},
                        },
                        "docs-langchain": {
                            "transport": "http",
                            "url": "https://docs.langchain.com/mcp",
                        },
                    }
                }
            )
        )
        mock_discover.return_value = [project_cfg]
        mock_classify.return_value = ([], [project_cfg])
        mock_load.return_value = ([], None, [])
        monkeypatch.setattr(
            "deepagents_code.model_config.DEFAULT_CONFIG_PATH",
            tmp_path / "config.toml",
        )
        monkeypatch.delenv("DEEPAGENTS_CODE_ENABLED_PROJECT_MCP_SERVERS", raising=False)
        monkeypatch.delenv(
            "DEEPAGENTS_CODE_DISABLED_PROJECT_MCP_SERVERS", raising=False
        )
        caplog.set_level(logging.WARNING, logger="deepagents_code.mcp_tools")

        tools, _manager, _infos = await resolve_and_load_mcp_tools(
            trust_project_mcp=False,
        )

        assert tools == []
        assert mock_load.call_count == 0
        assert "Skipped untrusted project MCP servers:\n" in caplog.text
        assert "- evil [http]: http://169.254.169.254" in caplog.text
        assert "- docs-langchain [http]: https://docs.langchain.com/mcp" in caplog.text
        assert "; docs-langchain" not in caplog.text

    @patch("deepagents_code.mcp_tools._load_tools_from_config")
    @patch("deepagents_code.mcp_tools.classify_discovered_configs")
    @patch("deepagents_code.mcp_tools.discover_mcp_configs")
    async def test_untrusted_project_remote_dropped_when_store_unknown(
        self,
        mock_discover: MagicMock,
        mock_classify: MagicMock,
        mock_load: AsyncMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Trust-store miss drops project remote entries (no preflight HEAD)."""
        project_cfg = tmp_path / ".mcp.json"
        project_cfg.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "evil": {
                            "transport": "http",
                            "url": "http://127.0.0.1",
                        }
                    }
                }
            )
        )
        mock_discover.return_value = [project_cfg]
        mock_classify.return_value = ([], [project_cfg])
        mock_load.return_value = ([], None, [])

        monkeypatch.setattr(
            "deepagents_code.mcp_trust.is_project_mcp_trusted",
            lambda *_args, **_kwargs: False,
        )

        await resolve_and_load_mcp_tools(trust_project_mcp=None)

        assert mock_load.call_count == 0

    @patch("deepagents_code.mcp_tools._load_tools_from_config")
    @patch("deepagents_code.mcp_tools.classify_discovered_configs")
    @patch("deepagents_code.mcp_tools.discover_mcp_configs")
    async def test_trusted_project_remote_passes_through(
        self,
        mock_discover: MagicMock,
        mock_classify: MagicMock,
        mock_load: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """Explicit `trust_project_mcp=True` keeps project remote entries."""
        project_cfg = tmp_path / ".mcp.json"
        project_cfg.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "remote": {
                            "transport": "http",
                            "url": "https://example.com",
                        }
                    }
                }
            )
        )
        mock_discover.return_value = [project_cfg]
        mock_classify.return_value = ([], [project_cfg])
        mock_load.return_value = ([], None, [])

        await resolve_and_load_mcp_tools(trust_project_mcp=True)

        merged = mock_load.call_args.args[0]
        assert "remote" in merged["mcpServers"]

    @patch("deepagents_code.mcp_tools._load_tools_from_config")
    @patch("deepagents_code.mcp_tools.discover_mcp_configs")
    async def test_disabled_server_is_split_off(
        self,
        mock_discover: MagicMock,
        mock_load: AsyncMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A disabled server is removed from the loader payload and surfaced as info."""
        cfg = tmp_path / "mcp.json"
        cfg.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "fs": {"command": "npx", "args": []},
                        "off": {"command": "node", "args": []},
                    },
                },
            ),
        )
        mock_discover.return_value = [cfg]
        mock_load.return_value = ([], None, [])
        monkeypatch.setattr(
            "deepagents_code.mcp_disabled.get_disabled_servers",
            lambda *_a, **_k: {"off"},
        )

        _tools, _manager, infos = await resolve_and_load_mcp_tools(
            trust_project_mcp=True,
        )

        merged = mock_load.call_args.args[0]
        assert "fs" in merged["mcpServers"]
        assert "off" not in merged["mcpServers"]
        disabled = [i for i in infos if i.status == "disabled"]
        assert len(disabled) == 1
        assert disabled[0].name == "off"
        assert disabled[0].transport == "stdio"

    @patch("deepagents_code.mcp_tools._load_tools_from_config")
    @patch("deepagents_code.mcp_tools.discover_mcp_configs")
    async def test_all_servers_disabled_short_circuits_loader(
        self,
        mock_discover: MagicMock,
        mock_load: AsyncMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When every configured server is disabled, the loader is never called."""
        cfg = tmp_path / "mcp.json"
        cfg.write_text(
            json.dumps(
                {"mcpServers": {"fs": {"command": "npx", "args": []}}},
            ),
        )
        mock_discover.return_value = [cfg]
        mock_load.return_value = ([], None, [])
        monkeypatch.setattr(
            "deepagents_code.mcp_disabled.get_disabled_servers",
            lambda *_a, **_k: {"fs"},
        )

        tools, manager, infos = await resolve_and_load_mcp_tools(
            trust_project_mcp=True,
        )

        assert tools == []
        assert manager is None
        assert mock_load.call_count == 0
        assert [i.name for i in infos if i.status == "disabled"] == ["fs"]

    @patch("deepagents_code.mcp_tools._load_tools_from_config")
    @patch("deepagents_code.mcp_tools.discover_mcp_configs")
    async def test_disabled_non_dict_config_gets_unknown_transport(
        self,
        mock_discover: MagicMock,
        mock_load: AsyncMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Non-dict server config in the disabled set surfaces `transport=unknown`."""
        cfg = tmp_path / "mcp.json"
        # Force a non-dict server entry into the merged config. JSON does
        # not preserve type fidelity across all loaders, so we monkeypatch
        # merge_mcp_configs to return a known-shape payload.
        cfg.write_text(
            json.dumps({"mcpServers": {"weird": {"command": "x"}}}),
        )
        mock_discover.return_value = [cfg]
        mock_load.return_value = ([], None, [])
        monkeypatch.setattr(
            "deepagents_code.mcp_disabled.get_disabled_servers",
            lambda *_a, **_k: {"weird"},
        )
        monkeypatch.setattr(
            "deepagents_code.mcp_tools.merge_mcp_configs",
            lambda _configs: {"mcpServers": {"weird": "not-a-dict"}},
        )

        _tools, _manager, infos = await resolve_and_load_mcp_tools(
            trust_project_mcp=True,
        )

        disabled = [i for i in infos if i.status == "disabled"]
        assert len(disabled) == 1
        assert disabled[0].name == "weird"
        assert disabled[0].transport == "unknown"


class TestDiscoveryHelpers:
    """Test config discovery and merge helpers."""

    def test_discover_mcp_configs_finds_standard_paths(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Discovery checks user and project config locations in order."""
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))
        monkeypatch.setattr(
            "deepagents_code.project_utils.find_project_root",
            lambda: tmp_path / "repo",
        )

        user_cfg = fake_home / ".deepagents" / ".mcp.json"
        user_cfg.parent.mkdir(parents=True)
        user_cfg.write_text("{}")

        project_cfg = tmp_path / "repo" / ".mcp.json"
        project_cfg.parent.mkdir(parents=True)
        project_cfg.write_text("{}")

        assert discover_mcp_configs() == [user_cfg, project_cfg]

    def test_classify_discovered_configs_splits_user_and_project(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Configs under `~/.deepagents` are user-level."""
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))

        user_cfg = fake_home / ".deepagents" / ".mcp.json"
        project_cfg = tmp_path / "repo" / ".mcp.json"
        user, project = classify_discovered_configs([user_cfg, project_cfg])

        assert user == [user_cfg]
        assert project == [project_cfg]

    def test_extract_stdio_server_commands(self) -> None:
        """Only stdio entries are extracted."""
        config = {
            "mcpServers": {
                "fs": {"command": "npx", "args": ["a"]},
                "remote": {"transport": "http", "url": "https://example.com"},
            }
        }

        assert extract_stdio_server_commands(config) == [("fs", "npx", ["a"])]

    def test_extract_project_server_summaries_covers_remote(self) -> None:
        """Remote and stdio entries surface so trust gating can list both."""
        config = {
            "mcpServers": {
                "fs": {"command": "npx", "args": ["a", "b"]},
                "remote": {"transport": "http", "url": "https://example.com"},
                "sse_srv": {"type": "sse", "url": "https://sse.example"},
            }
        }

        assert sorted(extract_project_server_summaries(config)) == [
            ("fs", "stdio", "npx a b"),
            ("remote", "http", "https://example.com"),
            ("sse_srv", "sse", "https://sse.example"),
        ]

    def test_merge_mcp_configs_last_wins(self) -> None:
        """Later configs override earlier ones by server name."""
        merged = merge_mcp_configs(
            [
                {"mcpServers": {"srv": {"command": "a"}}},
                {"mcpServers": {"srv": {"command": "b"}, "other": {"command": "c"}}},
            ]
        )

        assert merged == {
            "mcpServers": {
                "srv": {"command": "b"},
                "other": {"command": "c"},
            }
        }

    def test_load_mcp_config_lenient_returns_none_for_invalid(
        self, tmp_path: Path
    ) -> None:
        """Lenient loader returns `None` for invalid config files."""
        bad = tmp_path / "bad.json"
        bad.write_text('{"other": true}')
        assert load_mcp_config_lenient(bad) is None


class TestHealthChecks:
    """Direct tests for health-check helpers."""

    def test_check_stdio_server_command_missing(self) -> None:
        """Missing stdio commands are rejected."""
        with (
            patch("deepagents_code.mcp_tools.shutil.which", return_value=None),
            pytest.raises(RuntimeError, match="not found on PATH"),
        ):
            _check_stdio_server("srv", {"command": "missing"})

    async def test_check_stdio_server_runs_off_event_loop(
        self,
        write_config: Callable[..., str],
    ) -> None:
        """The stdio pre-flight's `shutil.which` runs off the event loop."""
        path = write_config({"mcpServers": {"srv": {"command": "missing"}}})
        event_loop_thread = threading.current_thread()
        which_threads: list[threading.Thread] = []

        def _record_missing_command(_command: str) -> str | None:
            which_threads.append(threading.current_thread())
            return None

        with patch(
            "deepagents_code.mcp_tools.shutil.which",
            side_effect=_record_missing_command,
        ):
            tools, manager, server_infos = await get_mcp_tools(path)

        try:
            assert which_threads
            assert which_threads[0] is not event_loop_thread
            assert tools == []
            assert server_infos[0].name == "srv"
            assert server_infos[0].status == "error"
            error = server_infos[0].error or ""
            assert "command 'missing' not found on PATH" in error
            assert manager is not None
        finally:
            if manager is not None:
                await manager.cleanup()

    async def test_check_remote_server_transport_error(self) -> None:
        """Transport errors are wrapped as `RuntimeError`."""
        import httpx

        client = AsyncMock()
        client.head.side_effect = httpx.TransportError("refused")
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("httpx.AsyncClient", return_value=client),
            pytest.raises(RuntimeError, match="unreachable"),
        ):
            await _check_remote_server("srv", {"url": "http://down:9999"})


class TestToolOrdering:
    """Tools are sorted deterministically by final name."""

    @pytest.fixture(autouse=True)
    def _bypass_health_checks(self) -> Generator[None]:
        """Bypass health checks for ordering tests."""
        with (
            patch("deepagents_code.mcp_tools._check_stdio_server"),
            patch(
                "deepagents_code.mcp_tools._check_remote_server",
                new_callable=AsyncMock,
            ),
        ):
            yield

    async def test_tools_sorted_alphabetically(
        self,
        write_config: Callable[..., str],
    ) -> None:
        """Tools are sorted alphabetically across discovery order."""
        path = write_config(
            {"mcpServers": {"srv": {"command": "node", "args": ["server.js"]}}}
        )

        session = AsyncMock()
        session.initialize = AsyncMock()
        session.list_tools = AsyncMock(
            return_value=_make_tool_page(
                [
                    _make_mcp_tool("zeta", "z"),
                    _make_mcp_tool("alpha", "a"),
                    _make_mcp_tool("mu", "m"),
                ]
            )
        )

        @asynccontextmanager
        async def _fake(
            _connection: dict[str, Any],
            *,
            _mcp_callbacks: object | None = None,
        ) -> AsyncIterator[AsyncMock]:
            await asyncio.sleep(0)
            yield session

        with patch("langchain_mcp_adapters.sessions.create_session", _fake):
            tools, manager, _ = await get_mcp_tools(path)

        assert [tool.name for tool in tools] == ["srv_alpha", "srv_mu", "srv_zeta"]
        assert manager is not None
        await manager.cleanup()


class TestLoadToolsConcurrency:
    """`_load_tools_from_config` probes independent servers concurrently.

    These tests pin the new behavior: bounded-concurrency preflight and
    discovery, with per-server error isolation and cancellation semantics
    matching the previous sequential loader. The load-bearing ordering
    guarantee is that `server_infos` follows config order regardless of
    completion order; the returned tool list is always sorted by tool name
    (via the terminal sort in the loader), so tool-name assertions here are
    content checks rather than ordering proofs.
    """

    @pytest.fixture(autouse=True)
    def _bypass_stdio_health_check(self) -> Generator[None]:
        """Bypass stdio pre-flight so tests focus on discovery concurrency."""
        with patch("deepagents_code.mcp_tools._check_stdio_server"):
            yield

    @staticmethod
    def _config(*names: str) -> dict[str, Any]:
        return {
            "mcpServers": {
                name: {"command": "node", "args": [f"{name}.js"]} for name in names
            }
        }

    def _tracking_session_factory(
        self,
        *,
        tool_by_server: dict[str, str],
        hold: asyncio.Event | None = None,
        sleep_s: float = 0.05,
    ) -> tuple[Any, dict[str, int]]:
        """Build a `create_session` fake that records concurrency.

        Returns the async context-manager fake plus a mutable stats dict with
        `max_inflight` (peak simultaneous open discovery sessions).
        """
        stats = {"inflight": 0, "max_inflight": 0}

        @asynccontextmanager
        async def _fake(
            connection: dict[str, Any],
            *,
            _mcp_callbacks: object | None = None,
        ) -> AsyncIterator[AsyncMock]:
            stats["inflight"] += 1
            stats["max_inflight"] = max(stats["max_inflight"], stats["inflight"])
            try:
                # Derive the server name from the recorded command arg so each
                # session yields that server's tool.
                args = connection.get("args") or []
                server = args[0].removesuffix(".js") if args else "srv"
                session = AsyncMock()
                session.initialize = AsyncMock()
                session.list_tools = AsyncMock(
                    return_value=_make_tool_page(
                        [_make_mcp_tool(tool_by_server[server])]
                    )
                )
                if hold is not None:
                    await hold.wait()
                else:
                    await asyncio.sleep(sleep_s)
                yield session
            finally:
                stats["inflight"] -= 1

        return _fake, stats

    async def test_discovery_runs_concurrently(self) -> None:
        """All servers' discovery sessions are open at the same time."""
        names = ["a", "b", "c", "d"]
        tool_by_server = {n: f"tool_{n}" for n in names}
        hold = asyncio.Event()
        fake, stats = self._tracking_session_factory(
            tool_by_server=tool_by_server, hold=hold
        )

        async def _release_when_all_open() -> None:
            for _ in range(200):
                if stats["inflight"] >= len(names):
                    break
                await asyncio.sleep(0.005)
            hold.set()

        with patch("langchain_mcp_adapters.sessions.create_session", fake):
            releaser = asyncio.create_task(_release_when_all_open())
            tools, manager, infos = await _load_tools_from_config(self._config(*names))
            await releaser

        assert stats["max_inflight"] == len(names)
        assert [t.name for t in tools] == [
            "a_tool_a",
            "b_tool_b",
            "c_tool_c",
            "d_tool_d",
        ]
        assert [i.name for i in infos] == names
        assert manager is not None
        await manager.cleanup()

    async def test_discovery_concurrency_is_bounded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No more than `_MCP_LOAD_CONCURRENCY` sessions are open at once."""
        monkeypatch.setattr(
            "deepagents_code.mcp_tools._MCP_LOAD_CONCURRENCY", 2, raising=True
        )
        names = ["s1", "s2", "s3", "s4", "s5"]
        tool_by_server = {n: f"t_{n}" for n in names}
        fake, stats = self._tracking_session_factory(
            tool_by_server=tool_by_server, sleep_s=0.03
        )

        with patch("langchain_mcp_adapters.sessions.create_session", fake):
            tools, manager, infos = await _load_tools_from_config(self._config(*names))

        assert stats["max_inflight"] == 2
        assert [i.name for i in infos] == names
        assert len(tools) == len(names)
        assert manager is not None
        await manager.cleanup()

    async def test_order_preserved_when_later_servers_finish_first(self) -> None:
        """`server_infos` follows config order regardless of completion order."""
        names = ["first", "second", "third"]
        # Later servers sleep less, so they finish discovery before earlier ones.
        delays = {"first": 0.09, "second": 0.05, "third": 0.01}
        finish_order: list[str] = []

        @asynccontextmanager
        async def _fake(
            connection: dict[str, Any],
            *,
            _mcp_callbacks: object | None = None,
        ) -> AsyncIterator[AsyncMock]:
            server = (connection.get("args") or ["x"])[0].removesuffix(".js")
            session = AsyncMock()
            session.initialize = AsyncMock()
            session.list_tools = AsyncMock(
                return_value=_make_tool_page([_make_mcp_tool(f"tool_{server}")])
            )
            await asyncio.sleep(delays[server])
            finish_order.append(server)
            yield session

        with patch("langchain_mcp_adapters.sessions.create_session", _fake):
            tools, manager, infos = await _load_tools_from_config(self._config(*names))

        assert finish_order == ["third", "second", "first"]
        assert [i.name for i in infos] == names
        assert [t.name for t in tools] == [
            "first_tool_first",
            "second_tool_second",
            "third_tool_third",
        ]
        assert manager is not None
        await manager.cleanup()

    async def test_one_server_failure_isolated_from_others(self) -> None:
        """A single discovery failure does not abort the other servers."""
        names = ["ok1", "boom", "ok2"]

        @asynccontextmanager
        async def _fake(
            connection: dict[str, Any],
            *,
            _mcp_callbacks: object | None = None,
        ) -> AsyncIterator[AsyncMock]:
            server = (connection.get("args") or ["x"])[0].removesuffix(".js")
            await asyncio.sleep(0.01)
            if server == "boom":
                msg = "discovery exploded"
                raise RuntimeError(msg)
            session = AsyncMock()
            session.initialize = AsyncMock()
            session.list_tools = AsyncMock(
                return_value=_make_tool_page([_make_mcp_tool(f"tool_{server}")])
            )
            yield session

        with patch("langchain_mcp_adapters.sessions.create_session", _fake):
            tools, manager, infos = await _load_tools_from_config(self._config(*names))

        by_name = {i.name: i for i in infos}
        assert [i.name for i in infos] == names
        assert by_name["ok1"].status == "ok"
        assert by_name["ok2"].status == "ok"
        assert by_name["boom"].status == "error"
        assert "discovery exploded" in (by_name["boom"].error or "")
        assert [t.name for t in tools] == ["ok1_tool_ok1", "ok2_tool_ok2"]
        assert manager is not None
        await manager.cleanup()

    async def test_preflight_failure_isolated_and_order_preserved(self) -> None:
        """A mid-config preflight failure is skipped; survivors keep order.

        Exercises the preflight-error path and the fold-in loop that
        interleaves a skipped server *between* discovered ones in config order,
        with discovery finishing out of order so the ordering cannot be an
        accident of completion timing.
        """
        names = ["ok_a", "pf_fail", "ok_b"]
        # `ok_b` discovers faster than `ok_a`, so completion order is reversed.
        delays = {"ok_a": 0.06, "ok_b": 0.01}

        def _check(name: str, _cfg: dict[str, Any]) -> None:
            if name == "pf_fail":
                msg = "preflight boom"
                raise RuntimeError(msg)

        @asynccontextmanager
        async def _fake(
            connection: dict[str, Any],
            *,
            _mcp_callbacks: object | None = None,
        ) -> AsyncIterator[AsyncMock]:
            server = (connection.get("args") or ["x"])[0].removesuffix(".js")
            session = AsyncMock()
            session.initialize = AsyncMock()
            session.list_tools = AsyncMock(
                return_value=_make_tool_page([_make_mcp_tool(f"tool_{server}")])
            )
            await asyncio.sleep(delays[server])
            yield session

        with (
            patch("deepagents_code.mcp_tools._check_stdio_server", _check),
            patch("langchain_mcp_adapters.sessions.create_session", _fake),
        ):
            tools, manager, infos = await _load_tools_from_config(self._config(*names))

        by_name = {i.name: i for i in infos}
        # server_infos follows config order, with the skipped server in place.
        assert [i.name for i in infos] == names
        assert by_name["ok_a"].status == "ok"
        assert by_name["ok_b"].status == "ok"
        assert by_name["pf_fail"].status == "error"
        assert "preflight boom" in (by_name["pf_fail"].error or "")
        assert [t.name for t in tools] == ["ok_a_tool_ok_a", "ok_b_tool_ok_b"]
        assert manager is not None
        await manager.cleanup()

    async def test_all_servers_fail_preflight_yields_empty(self) -> None:
        """Every server failing preflight yields no tools, infos in order.

        Drives the empty-discovery `_gather_bounded([], ...)` path and asserts a
        non-`None` (empty) session manager is still returned.
        """
        names = ["x1", "x2", "x3"]

        def _check(_name: str, _cfg: dict[str, Any]) -> None:
            msg = "nope"
            raise RuntimeError(msg)

        with patch("deepagents_code.mcp_tools._check_stdio_server", _check):
            tools, manager, infos = await _load_tools_from_config(self._config(*names))

        assert tools == []
        assert [i.name for i in infos] == names
        assert all(i.status == "error" for i in infos)
        assert manager is not None
        await manager.cleanup()

    async def test_tool_construction_failure_isolated(self) -> None:
        """A post-discovery construction failure degrades one server only.

        Discovery succeeds for every server, but tool filtering raises for one.
        That server must become an `error` info while its siblings load
        normally — proving isolation now covers the post-discovery construction
        block, not just the discovery session. Without that guard the failure
        would abort the entire concurrent load.
        """
        names = ["good", "bad_build"]
        fake, _ = self._tracking_session_factory(
            tool_by_server={n: f"t_{n}" for n in names}, sleep_s=0.0
        )

        def _filter(
            server_tools: list[Any], server_name: str, _server_config: dict[str, Any]
        ) -> list[Any]:
            if server_name == "bad_build":
                msg = "build boom"
                raise RuntimeError(msg)
            return server_tools

        with (
            patch("langchain_mcp_adapters.sessions.create_session", fake),
            patch("deepagents_code.mcp_tools._apply_tool_filter", _filter),
        ):
            tools, manager, infos = await _load_tools_from_config(self._config(*names))

        by_name = {i.name: i for i in infos}
        assert [i.name for i in infos] == names
        assert by_name["good"].status == "ok"
        assert by_name["bad_build"].status == "error"
        assert "build boom" in (by_name["bad_build"].error or "")
        assert [t.name for t in tools] == ["good_t_good"]
        assert manager is not None
        await manager.cleanup()

    async def test_cancellation_propagates_and_cancels_siblings(self) -> None:
        """A cancelled worker propagates and tears down its siblings."""
        names = ["cancel", "sibling"]
        sibling_cancelled = asyncio.Event()

        @asynccontextmanager
        async def _fake(
            connection: dict[str, Any],
            *,
            _mcp_callbacks: object | None = None,
        ) -> AsyncIterator[AsyncMock]:
            server = (connection.get("args") or ["x"])[0].removesuffix(".js")
            if server == "cancel":
                await asyncio.sleep(0.01)
                raise asyncio.CancelledError
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                sibling_cancelled.set()
                raise
            yield AsyncMock()  # pragma: no cover - never reached

        with (
            patch("langchain_mcp_adapters.sessions.create_session", _fake),
            pytest.raises(asyncio.CancelledError),
        ):
            await _load_tools_from_config(self._config(*names))

        assert sibling_cancelled.is_set()

    async def test_preflight_runs_concurrently(self) -> None:
        """Stdio pre-flight checks run concurrently across servers."""
        names = ["p1", "p2", "p3"]
        tool_by_server = {n: f"pt_{n}" for n in names}
        stats = {"inflight": 0, "max_inflight": 0}
        # `_slow_check` runs in `asyncio.to_thread` worker threads, so the shared
        # counters must be guarded with a real lock (`+=` is not atomic across
        # threads) and the barrier must be a thread-safe primitive.
        stats_lock = threading.Lock()
        barrier = threading.Event()

        def _slow_check(_name: str, _cfg: dict[str, Any]) -> None:
            # `_check_stdio_server` is sync and invoked via asyncio.to_thread,
            # so bump the counter and block until every worker is in-flight.
            with stats_lock:
                stats["inflight"] += 1
                stats["max_inflight"] = max(stats["max_inflight"], stats["inflight"])
            barrier.wait()
            with stats_lock:
                stats["inflight"] -= 1

        async def _release() -> None:
            for _ in range(200):
                with stats_lock:
                    peak = stats["max_inflight"]
                if peak >= len(names):
                    break
                await asyncio.sleep(0.005)
            barrier.set()

        fake, _ = self._tracking_session_factory(
            tool_by_server=tool_by_server, sleep_s=0.0
        )
        with (
            patch("deepagents_code.mcp_tools._check_stdio_server", _slow_check),
            patch("langchain_mcp_adapters.sessions.create_session", fake),
        ):
            releaser = asyncio.create_task(_release())
            _tools, manager, infos = await _load_tools_from_config(self._config(*names))
            await releaser

        assert stats["max_inflight"] == len(names)
        assert [i.name for i in infos] == names
        assert manager is not None
        await manager.cleanup()

    async def test_warmup_runs_off_loop_before_discovery(self) -> None:
        """Adapter warmup runs, off the event loop, before any discovery."""
        loop_thread_id = threading.get_ident()
        events: list[tuple[str, int]] = []

        def _warm() -> None:
            events.append(("warm", threading.get_ident()))

        @asynccontextmanager
        async def _fake(
            connection: dict[str, Any],
            *,
            _mcp_callbacks: object | None = None,
        ) -> AsyncIterator[AsyncMock]:
            events.append(("discover", threading.get_ident()))
            session = AsyncMock()
            session.initialize = AsyncMock()
            session.list_tools = AsyncMock(return_value=_make_tool_page([]))
            yield session

        with (
            patch("deepagents_code.mcp_tools._warm_mcp_adapter_imports", _warm),
            patch("langchain_mcp_adapters.sessions.create_session", _fake),
        ):
            _tools, manager, _infos = await _load_tools_from_config(
                self._config("only")
            )

        assert events[0][0] == "warm"
        assert events[0][1] != loop_thread_id
        assert any(kind == "discover" for kind, _ in events)
        assert manager is not None
        await manager.cleanup()


class TestGatherBounded:
    """Direct tests for the `_gather_bounded` concurrency helper.

    These pin the helper's contract independently of MCP loading: submission
    (not completion) ordering, the empty and clamped-limit edge cases, and the
    failure path that cancels + awaits siblings and never silently drops a
    concurrent failure.
    """

    async def test_results_follow_submission_order(self) -> None:
        """Results zip back to submission order even when completion differs."""
        completed: list[int] = []

        def _factory(idx: int, delay: float) -> Callable[[], Any]:
            async def _run() -> int:
                await asyncio.sleep(delay)
                completed.append(idx)
                return idx

            return _run

        # Index 0 finishes last, index 2 finishes first.
        factories = [_factory(0, 0.03), _factory(1, 0.02), _factory(2, 0.001)]
        results = await _gather_bounded(factories, limit=8)

        assert results == [0, 1, 2]
        assert completed == [2, 1, 0]

    async def test_empty_returns_empty(self) -> None:
        """Zero factories return an empty list without touching the loop."""
        assert await _gather_bounded([], limit=8) == []

    async def test_limit_below_one_is_clamped_to_serial(self) -> None:
        """A limit < 1 is clamped to 1, so factories run strictly serially."""
        active = {"n": 0, "max": 0}

        def _factory() -> Callable[[], Any]:
            async def _run() -> None:
                active["n"] += 1
                active["max"] = max(active["max"], active["n"])
                await asyncio.sleep(0.01)
                active["n"] -= 1

            return _run

        await _gather_bounded([_factory(), _factory(), _factory()], limit=0)
        assert active["max"] == 1

    async def test_failure_cancels_and_awaits_siblings(self) -> None:
        """A raising factory cancels the rest and awaits them before raising."""
        sibling = {"cancelled": False, "completed": False}
        started = asyncio.Event()

        def _failing() -> Callable[[], Any]:
            async def _run() -> None:
                await started.wait()
                msg = "boom"
                raise RuntimeError(msg)

            return _run

        def _sibling() -> Callable[[], Any]:
            async def _run() -> None:
                started.set()
                try:
                    await asyncio.sleep(10)
                except asyncio.CancelledError:
                    sibling["cancelled"] = True
                    raise
                sibling["completed"] = True  # pragma: no cover - never reached

            return _run

        with pytest.raises(RuntimeError, match="boom"):
            await _gather_bounded([_failing(), _sibling()], limit=8)

        assert sibling["cancelled"] is True
        assert sibling["completed"] is False

    async def test_concurrent_failures_are_logged_not_lost(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When several factories fail, no failure vanishes silently.

        `asyncio.gather` propagates only the first exception; the others are
        logged at debug so a concurrent failure is never dropped without trace.
        """

        def _failing(message: str) -> Callable[[], Any]:
            async def _run() -> None:
                raise RuntimeError(message)

            return _run

        with (
            caplog.at_level(logging.DEBUG, logger="deepagents_code.mcp_tools"),
            pytest.raises(RuntimeError),
        ):
            await _gather_bounded(
                [_failing("first_failure"), _failing("second_failure")], limit=8
            )

        assert "sibling task failed" in caplog.text
        # Both failures are represented in the captured logs (the propagated one
        # plus the logged sibling), so neither is lost.
        assert "first_failure" in caplog.text
        assert "second_failure" in caplog.text


class TestCachedSessionProxy:
    """Runtime tool wrappers use lazy cached sessions with retry semantics."""

    @pytest.fixture(autouse=True)
    def _bypass_health_checks(self) -> Generator[None]:
        """Bypass health checks for runtime tool tests."""
        with (
            patch("deepagents_code.mcp_tools._check_stdio_server"),
            patch(
                "deepagents_code.mcp_tools._check_remote_server",
                new_callable=AsyncMock,
            ),
        ):
            yield

    async def test_first_call_opens_runtime_session_after_discovery(
        self,
        write_config: Callable[..., str],
        fake_tool_result: Any,  # noqa: ANN401
    ) -> None:
        """The first tool call opens one cached runtime session."""
        path = write_config(
            {"mcpServers": {"srv": {"command": "node", "args": ["s.js"]}}}
        )
        sessions: list[AsyncMock] = []

        def _new_session() -> AsyncMock:
            session = AsyncMock()
            session.initialize = AsyncMock()
            session.list_tools = AsyncMock(
                return_value=_make_tool_page([_make_mcp_tool("echo")])
            )
            session.call_tool = AsyncMock(return_value=fake_tool_result)
            sessions.append(session)
            return session

        @asynccontextmanager
        async def _fake(
            _connection: dict[str, Any],
            *,
            _mcp_callbacks: object | None = None,
        ) -> AsyncIterator[AsyncMock]:
            await asyncio.sleep(0)
            yield _new_session()

        with patch("langchain_mcp_adapters.sessions.create_session", _fake):
            tools, manager, _ = await get_mcp_tools(path)
            result = await tools[0].ainvoke({})  # ty: ignore

        assert len(sessions) == 2
        sessions[1].call_tool.assert_awaited_once_with("echo", {})
        assert "ok" in str(result)
        assert manager is not None
        await manager.cleanup()

    async def test_second_call_reuses_cached_runtime_session(
        self,
        write_config: Callable[..., str],
        fake_tool_result: Any,  # noqa: ANN401
    ) -> None:
        """Back-to-back tool calls reuse the same runtime session."""
        path = write_config(
            {"mcpServers": {"srv": {"command": "node", "args": ["s.js"]}}}
        )
        sessions: list[AsyncMock] = []

        def _new_session() -> AsyncMock:
            session = AsyncMock()
            session.initialize = AsyncMock()
            session.list_tools = AsyncMock(
                return_value=_make_tool_page([_make_mcp_tool("echo")])
            )
            session.call_tool = AsyncMock(return_value=fake_tool_result)
            sessions.append(session)
            return session

        @asynccontextmanager
        async def _fake(
            _connection: dict[str, Any],
            *,
            _mcp_callbacks: object | None = None,
        ) -> AsyncIterator[AsyncMock]:
            await asyncio.sleep(0)
            yield _new_session()

        with patch("langchain_mcp_adapters.sessions.create_session", _fake):
            tools, manager, _ = await get_mcp_tools(path)
            await tools[0].ainvoke({})  # ty: ignore
            await tools[0].ainvoke({})  # ty: ignore

        # Reuse is the observable: the runtime session services both
        # calls. Counting sessions is implementation detail — await_count
        # on sessions[1] captures what matters.
        assert sessions[1].call_tool.await_count == 2
        assert manager is not None
        await manager.cleanup()

    async def test_transient_error_invalidates_and_retries(
        self,
        write_config: Callable[..., str],
        fake_tool_result: Any,  # noqa: ANN401
    ) -> None:
        """A transient transport error triggers invalidate and retry-once."""
        from anyio import ClosedResourceError

        path = write_config(
            {"mcpServers": {"srv": {"command": "node", "args": ["s.js"]}}}
        )

        call_counter = {"n": 0}
        sessions: list[AsyncMock] = []

        def _new_session(*, dead: bool = False) -> AsyncMock:
            session = AsyncMock()
            session.initialize = AsyncMock()
            session.list_tools = AsyncMock(
                return_value=_make_tool_page([_make_mcp_tool("echo")])
            )
            if dead:
                session.call_tool = AsyncMock(side_effect=ClosedResourceError())
            else:
                session.call_tool = AsyncMock(return_value=fake_tool_result)
            sessions.append(session)
            return session

        @asynccontextmanager
        async def _fake(
            _connection: dict[str, Any],
            *,
            _mcp_callbacks: object | None = None,
        ) -> AsyncIterator[AsyncMock]:
            await asyncio.sleep(0)
            call_counter["n"] += 1
            yield _new_session(dead=(call_counter["n"] == 2))

        with patch("langchain_mcp_adapters.sessions.create_session", _fake):
            tools, manager, _ = await get_mcp_tools(path)
            await tools[0].ainvoke({})  # ty: ignore

        assert call_counter["n"] == 3
        sessions[2].call_tool.assert_awaited_once()
        await manager.cleanup()

    async def test_repeated_transient_error_surfaces_tool_message(
        self,
        write_config: Callable[..., str],
    ) -> None:
        """A second transient failure becomes a tool-local error message."""
        from anyio import ClosedResourceError
        from langchain_core.messages import ToolMessage

        path = write_config(
            {"mcpServers": {"srv": {"command": "node", "args": ["s.js"]}}}
        )
        call_counter = {"n": 0}

        def _new_session(*, dead: bool) -> AsyncMock:
            session = AsyncMock()
            session.initialize = AsyncMock()
            session.list_tools = AsyncMock(
                return_value=_make_tool_page([_make_mcp_tool("echo")])
            )
            session.call_tool = AsyncMock(
                side_effect=ClosedResourceError() if dead else None
            )
            return session

        @asynccontextmanager
        async def _fake(
            _connection: dict[str, Any],
            *,
            _mcp_callbacks: object | None = None,
        ) -> AsyncIterator[AsyncMock]:
            await asyncio.sleep(0)
            call_counter["n"] += 1
            yield _new_session(dead=(call_counter["n"] >= 2))

        with patch("langchain_mcp_adapters.sessions.create_session", _fake):
            tools, manager, _ = await get_mcp_tools(path)
            result = await tools[0].ainvoke(
                {"args": {}, "id": "call-1", "type": "tool_call"}
            )  # ty: ignore

        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "failed after one retry" in result.content
        assert call_counter["n"] == 3
        await manager.cleanup()

    async def test_generic_oserror_is_not_retried(
        self,
        write_config: Callable[..., str],
        fake_tool_result: Any,  # noqa: ANN401
    ) -> None:
        """Generic `OSError`s do not trigger session invalidation and retry."""
        from langchain_core.messages import ToolMessage

        path = write_config(
            {"mcpServers": {"srv": {"command": "node", "args": ["s.js"]}}}
        )
        call_counter = {"n": 0}

        def _new_session(*, fail: bool) -> AsyncMock:
            session = AsyncMock()
            session.initialize = AsyncMock()
            session.list_tools = AsyncMock(
                return_value=_make_tool_page([_make_mcp_tool("echo")])
            )
            if fail:
                msg = "socket glitch"
                session.call_tool = AsyncMock(side_effect=OSError(msg))
            else:
                session.call_tool = AsyncMock(return_value=fake_tool_result)
            return session

        @asynccontextmanager
        async def _fake(
            _connection: dict[str, Any],
            *,
            _mcp_callbacks: object | None = None,
        ) -> AsyncIterator[AsyncMock]:
            await asyncio.sleep(0)
            call_counter["n"] += 1
            yield _new_session(fail=(call_counter["n"] >= 2))

        with patch("langchain_mcp_adapters.sessions.create_session", _fake):
            tools, manager, _ = await get_mcp_tools(path)
            result = await tools[0].ainvoke(
                {"args": {}, "id": "call-1", "type": "tool_call"}
            )  # ty: ignore

        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "socket glitch" in result.content
        assert call_counter["n"] == 2
        await manager.cleanup()

    async def test_logical_tool_exception_is_not_retried(
        self,
        write_config: Callable[..., str],
    ) -> None:
        """MCP `isError=True` returns a failed `ToolMessage` without retrying."""
        from langchain_core.messages import ToolMessage
        from mcp.types import CallToolResult, TextContent

        path = write_config(
            {"mcpServers": {"srv": {"command": "node", "args": ["s.js"]}}}
        )
        call_counter = {"n": 0}
        runtime_session: AsyncMock | None = None

        def _new_session() -> AsyncMock:
            nonlocal runtime_session
            session = AsyncMock()
            session.initialize = AsyncMock()
            session.list_tools = AsyncMock(
                return_value=_make_tool_page([_make_mcp_tool("echo")])
            )
            session.call_tool = AsyncMock(
                return_value=CallToolResult(
                    content=[TextContent(type="text", text="boom")],
                    isError=True,
                )
            )
            if call_counter["n"] >= 1:
                runtime_session = session
            return session

        @asynccontextmanager
        async def _fake(
            _connection: dict[str, Any],
            *,
            _mcp_callbacks: object | None = None,
        ) -> AsyncIterator[AsyncMock]:
            await asyncio.sleep(0)
            call_counter["n"] += 1
            yield _new_session()

        with patch("langchain_mcp_adapters.sessions.create_session", _fake):
            tools, manager, _ = await get_mcp_tools(path)
            result = await tools[0].ainvoke(
                {"args": {}, "id": "call-1", "type": "tool_call"}
            )  # ty: ignore

        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert result.tool_call_id == "call-1"
        assert result.content_blocks[0]["type"] == "text"
        assert result.content_blocks[0]["text"] == "boom"
        assert call_counter["n"] == 2
        assert runtime_session is not None
        assert runtime_session.call_tool.await_count == 1
        await manager.cleanup()

    async def test_empty_mcp_error_content_uses_placeholder_tool_message(
        self,
        write_config: Callable[..., str],
    ) -> None:
        """Empty MCP error content gets the adapter placeholder text block."""
        from langchain_core.messages import ToolMessage
        from mcp.types import CallToolResult

        path = write_config(
            {"mcpServers": {"srv": {"command": "node", "args": ["s.js"]}}}
        )
        runtime_session: AsyncMock | None = None

        def _new_session() -> AsyncMock:
            nonlocal runtime_session
            session = AsyncMock()
            session.initialize = AsyncMock()
            session.list_tools = AsyncMock(
                return_value=_make_tool_page([_make_mcp_tool("echo")])
            )
            session.call_tool = AsyncMock(
                return_value=CallToolResult(content=[], isError=True)
            )
            runtime_session = session
            return session

        @asynccontextmanager
        async def _fake(
            _connection: dict[str, Any],
            *,
            _mcp_callbacks: object | None = None,
        ) -> AsyncIterator[AsyncMock]:
            await asyncio.sleep(0)
            yield _new_session()

        with patch("langchain_mcp_adapters.sessions.create_session", _fake):
            tools, manager, _ = await get_mcp_tools(path)
            result = await tools[0].ainvoke(
                {"args": {}, "id": "call-1", "type": "tool_call"}
            )  # ty: ignore

        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert result.content_blocks[0]["type"] == "text"
        assert result.content_blocks[0]["text"] == (
            "MCP tool returned an error with empty content."
        )
        assert runtime_session is not None
        assert runtime_session.call_tool.await_count == 1
        assert manager is not None
        await manager.cleanup()

    async def test_reauth_signal_surfaces_tool_message_without_retry(
        self,
        write_config: Callable[..., str],
    ) -> None:
        """Runtime re-auth signals surface as actionable tool messages."""
        from langchain_core.messages import ToolMessage

        path = write_config(
            {
                "mcpServers": {
                    "srv": {
                        "command": "node",
                        "args": ["s.js"],
                    }
                }
            }
        )
        sessions: list[AsyncMock] = []

        def _new_session(*, reauth: bool = False) -> AsyncMock:
            session = AsyncMock()
            session.initialize = AsyncMock()
            session.list_tools = AsyncMock(
                return_value=_make_tool_page([_make_mcp_tool("echo")])
            )
            if reauth:
                session.call_tool = AsyncMock(side_effect=MCPReauthRequiredError("srv"))
            else:
                session.call_tool = AsyncMock(return_value=None)
            sessions.append(session)
            return session

        call_counter = {"n": 0}

        @asynccontextmanager
        async def _fake(
            _connection: dict[str, Any],
            *,
            _mcp_callbacks: object | None = None,
        ) -> AsyncIterator[AsyncMock]:
            await asyncio.sleep(0)
            call_counter["n"] += 1
            yield _new_session(reauth=(call_counter["n"] == 2))

        with patch("langchain_mcp_adapters.sessions.create_session", _fake):
            tools, manager, _ = await get_mcp_tools(path)
            result = await tools[0].ainvoke(
                {"args": {}, "id": "call-1", "type": "tool_call"}
            )  # ty: ignore

        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "re-authentication" in result.content
        assert call_counter["n"] == 2
        await manager.cleanup()

    def test_discovery_and_runtime_use_different_event_loops(
        self,
        write_config: Callable[..., str],
        fake_tool_result: Any,  # noqa: ANN401
    ) -> None:
        """Discovery sessions created in one loop are not reused in another."""
        path = write_config(
            {"mcpServers": {"srv": {"command": "node", "args": ["s.js"]}}}
        )

        class LoopBoundSession:
            def __init__(self) -> None:
                self.loop = asyncio.get_running_loop()
                self.initialize = AsyncMock()
                self.list_tools = AsyncMock(
                    return_value=_make_tool_page([_make_mcp_tool("echo")])
                )
                self.call_tool = AsyncMock(side_effect=self._call_tool)

            async def _call_tool(
                self,
                name: str,
                arguments: dict[str, Any],
            ) -> object:
                if asyncio.get_running_loop() is not self.loop:
                    msg = "session bound to a different event loop"
                    raise RuntimeError(msg)
                assert name == "echo"
                assert arguments == {}
                return fake_tool_result

        sessions: list[LoopBoundSession] = []

        @asynccontextmanager
        async def _fake(
            _connection: dict[str, Any],
            *,
            _mcp_callbacks: object | None = None,
        ) -> AsyncIterator[LoopBoundSession]:
            await asyncio.sleep(0)
            session = LoopBoundSession()
            sessions.append(session)
            yield session

        with patch("langchain_mcp_adapters.sessions.create_session", _fake):
            tools, manager, _ = asyncio.run(get_mcp_tools(path))
            result = asyncio.run(tools[0].ainvoke({}))  # ty: ignore
            assert manager is not None
            asyncio.run(manager.cleanup())

        assert len(sessions) == 2
        assert sessions[0].loop is not sessions[1].loop
        sessions[0].call_tool.assert_not_called()
        sessions[1].call_tool.assert_awaited_once_with("echo", {})
        assert "ok" in str(result)


def _make_prefixed_tool(name: str, description: str = "") -> MagicMock:
    """Build a mock tool as the adapter produces with `tool_name_prefix=True`."""
    tool = MagicMock()
    tool.name = name
    tool.description = description
    return tool


class TestToolFilterValidation:
    """Validation of `allowedTools` / `disabledTools` server fields."""

    def test_allowed_tools_accepted(self, write_config: Callable[..., str]) -> None:
        """`allowedTools` with a list of strings is accepted."""
        path = write_config(
            {
                "mcpServers": {
                    "fs": {
                        "command": "node",
                        "allowedTools": ["read_file", "list_dir"],
                    }
                }
            }
        )
        assert load_mcp_config(path)["mcpServers"]["fs"]["allowedTools"] == [
            "read_file",
            "list_dir",
        ]

    def test_disabled_tools_accepted(self, write_config: Callable[..., str]) -> None:
        """`disabledTools` with a list of strings is accepted."""
        path = write_config(
            {"mcpServers": {"fs": {"command": "node", "disabledTools": ["write_file"]}}}
        )
        assert load_mcp_config(path)["mcpServers"]["fs"]["disabledTools"] == [
            "write_file"
        ]

    def test_accepted_on_remote_server(self, write_config: Callable[..., str]) -> None:
        """Filter fields also apply to http/sse servers."""
        path = write_config(
            {
                "mcpServers": {
                    "api": {
                        "type": "http",
                        "url": "https://example.com/mcp",
                        "allowedTools": ["search"],
                    }
                }
            }
        )
        assert load_mcp_config(path)["mcpServers"]["api"]["allowedTools"] == ["search"]

    @pytest.mark.parametrize("field", ["allowedTools", "disabledTools"])
    def test_rejects_non_list(
        self, write_config: Callable[..., str], field: str
    ) -> None:
        """Non-list filter field raises TypeError."""
        path = write_config(
            {"mcpServers": {"fs": {"command": "node", field: "read_file"}}}
        )
        with pytest.raises(TypeError, match=rf"'{field}' must be a list of strings"):
            load_mcp_config(path)

    @pytest.mark.parametrize("field", ["allowedTools", "disabledTools"])
    def test_rejects_non_string_items(
        self, write_config: Callable[..., str], field: str
    ) -> None:
        """Filter list with non-string items raises TypeError."""
        path = write_config(
            {"mcpServers": {"fs": {"command": "node", field: ["ok", 42]}}}
        )
        with pytest.raises(TypeError, match=rf"'{field}' must be a list of strings"):
            load_mcp_config(path)

    def test_rejects_both_set(self, write_config: Callable[..., str]) -> None:
        """Setting both `allowedTools` and `disabledTools` on one server errors."""
        path = write_config(
            {
                "mcpServers": {
                    "fs": {
                        "command": "node",
                        "allowedTools": ["a"],
                        "disabledTools": ["b"],
                    }
                }
            }
        )
        with pytest.raises(
            ValueError, match=r"cannot set both 'allowedTools' and 'disabledTools'"
        ):
            load_mcp_config(path)

    @pytest.mark.parametrize("field", ["allowedTools", "disabledTools"])
    def test_rejects_empty_list(
        self, write_config: Callable[..., str], field: str
    ) -> None:
        """An empty filter list is a footgun and is rejected at load time."""
        path = write_config({"mcpServers": {"fs": {"command": "node", field: []}}})
        with pytest.raises(ValueError, match=rf"'{field}' must be non-empty"):
            load_mcp_config(path)


class TestApplyToolFilter:
    """Behavior of the `_apply_tool_filter` helper."""

    def test_no_filter_returns_input_unchanged(self) -> None:
        """Absent filter fields pass tools through."""
        tools = [
            _make_prefixed_tool("fs_read"),
            _make_prefixed_tool("fs_write"),
        ]
        assert _apply_tool_filter(tools, "fs", {"command": "node"}) is tools

    def test_allowed_keeps_only_listed(self) -> None:
        """`allowedTools` keeps only matching tools."""
        tools = [
            _make_prefixed_tool("fs_read"),
            _make_prefixed_tool("fs_write"),
            _make_prefixed_tool("fs_stat"),
        ]
        result = _apply_tool_filter(
            tools, "fs", {"command": "node", "allowedTools": ["read", "stat"]}
        )
        assert [t.name for t in result] == ["fs_read", "fs_stat"]

    def test_allowed_matches_prefixed_name(self) -> None:
        """`allowedTools` entries may include the server prefix."""
        tools = [_make_prefixed_tool("fs_read"), _make_prefixed_tool("fs_write")]
        result = _apply_tool_filter(
            tools, "fs", {"command": "node", "allowedTools": ["fs_read"]}
        )
        assert [t.name for t in result] == ["fs_read"]

    def test_allowed_unknown_name_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Names in `allowedTools` that don't match any tool produce a warning."""
        tools = [_make_prefixed_tool("fs_read")]
        with caplog.at_level("WARNING", logger="deepagents_code.mcp_tools"):
            result = _apply_tool_filter(
                tools, "fs", {"command": "node", "allowedTools": ["read", "gone"]}
            )
        assert [t.name for t in result] == ["fs_read"]
        assert "allowedTools entries matched no tools: gone" in caplog.text

    def test_allowed_glob_against_bare_name(self) -> None:
        """Glob entries match against the bare (unprefixed) tool name."""
        tools = [
            _make_prefixed_tool("fs_read_file"),
            _make_prefixed_tool("fs_read_dir"),
            _make_prefixed_tool("fs_write_file"),
        ]
        result = _apply_tool_filter(
            tools, "fs", {"command": "node", "allowedTools": ["read_*"]}
        )
        assert [t.name for t in result] == ["fs_read_file", "fs_read_dir"]

    def test_allowed_glob_against_prefixed_name(self) -> None:
        """Glob entries may include the server prefix."""
        tools = [
            _make_prefixed_tool("fs_read_file"),
            _make_prefixed_tool("fs_write_file"),
        ]
        result = _apply_tool_filter(
            tools, "fs", {"command": "node", "allowedTools": ["fs_read_*"]}
        )
        assert [t.name for t in result] == ["fs_read_file"]

    def test_disabled_glob_drops_matching(self) -> None:
        """Glob entries in `disabledTools` drop all matching tools."""
        tools = [
            _make_prefixed_tool("fs_read_file"),
            _make_prefixed_tool("fs_write_file"),
            _make_prefixed_tool("fs_write_dir"),
        ]
        result = _apply_tool_filter(
            tools, "fs", {"command": "node", "disabledTools": ["write_*"]}
        )
        assert [t.name for t in result] == ["fs_read_file"]

    def test_glob_with_no_matches_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Glob patterns that match zero tools also produce a warning."""
        tools = [_make_prefixed_tool("fs_read_file")]
        with caplog.at_level("WARNING", logger="deepagents_code.mcp_tools"):
            result = _apply_tool_filter(
                tools,
                "fs",
                {"command": "node", "allowedTools": ["read_*", "search_*"]},
            )
        assert [t.name for t in result] == ["fs_read_file"]
        assert "allowedTools entries matched no tools: search_*" in caplog.text

    def test_glob_question_mark_and_charclass(self) -> None:
        """`?` and `[...]` metachars are honored."""
        tools = [
            _make_prefixed_tool("srv_t1"),
            _make_prefixed_tool("srv_t2"),
            _make_prefixed_tool("srv_tx"),
        ]
        result = _apply_tool_filter(
            tools, "srv", {"command": "node", "allowedTools": ["t[12]"]}
        )
        assert [t.name for t in result] == ["srv_t1", "srv_t2"]

    def test_disabled_drops_listed(self) -> None:
        """`disabledTools` drops matching tools, keeps the rest."""
        tools = [
            _make_prefixed_tool("fs_read"),
            _make_prefixed_tool("fs_write"),
            _make_prefixed_tool("fs_stat"),
        ]
        result = _apply_tool_filter(
            tools, "fs", {"command": "node", "disabledTools": ["write"]}
        )
        assert [t.name for t in result] == ["fs_read", "fs_stat"]

    def test_disabled_matches_prefixed_name(self) -> None:
        """`disabledTools` entries may include the server prefix."""
        tools = [_make_prefixed_tool("fs_read"), _make_prefixed_tool("fs_write")]
        result = _apply_tool_filter(
            tools, "fs", {"command": "node", "disabledTools": ["fs_write"]}
        )
        assert [t.name for t in result] == ["fs_read"]

    def test_disabled_unknown_name_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A `disabledTools` typo should be visible.

        Otherwise the user thinks a tool was disabled when it's still active.
        """
        tools = [_make_prefixed_tool("fs_read"), _make_prefixed_tool("fs_write")]
        with caplog.at_level("WARNING", logger="deepagents_code.mcp_tools"):
            result = _apply_tool_filter(
                tools,
                "fs",
                {"command": "node", "disabledTools": ["write", "tpyo"]},
            )
        assert [t.name for t in result] == ["fs_read"]
        assert "disabledTools entries matched no tools: tpyo" in caplog.text


class TestToolFilterEndToEnd:
    """`get_mcp_tools` applies filtering after loading."""

    @pytest.fixture(autouse=True)
    def _bypass_health_checks(self) -> Generator[None]:
        with (
            patch("deepagents_code.mcp_tools._check_stdio_server"),
            patch("deepagents_code.mcp_tools._check_remote_server"),
        ):
            yield

    async def test_allowed_tools_filters_loaded_tools(
        self,
        write_config: Callable[..., str],
    ) -> None:
        """Only tools listed in `allowedTools` end up in the returned list."""
        path = write_config(
            {
                "mcpServers": {
                    "fs": {
                        "command": "node",
                        "args": ["server.js"],
                        "allowedTools": ["read_file"],
                    }
                }
            }
        )

        session = AsyncMock()
        session.initialize = AsyncMock()
        session.list_tools = AsyncMock(
            return_value=_make_tool_page(
                [_make_mcp_tool("read_file", "r"), _make_mcp_tool("write_file", "w")]
            )
        )

        @asynccontextmanager
        async def _fake(
            _connection: dict[str, Any],
            *,
            _mcp_callbacks: object | None = None,
        ) -> AsyncIterator[AsyncMock]:
            await asyncio.sleep(0)
            yield session

        with patch("langchain_mcp_adapters.sessions.create_session", _fake):
            tools, manager, server_infos = await get_mcp_tools(path)

        assert [t.name for t in tools] == ["fs_read_file"]
        assert [t.name for t in server_infos[0].tools] == ["fs_read_file"]
        assert manager is not None
        await manager.cleanup()

    async def test_disabled_tools_removes_loaded_tools(
        self,
        write_config: Callable[..., str],
    ) -> None:
        """Tools listed in `disabledTools` are dropped from the returned list."""
        path = write_config(
            {
                "mcpServers": {
                    "fs": {
                        "command": "node",
                        "args": ["server.js"],
                        "disabledTools": ["write_file"],
                    }
                }
            }
        )

        session = AsyncMock()
        session.initialize = AsyncMock()
        session.list_tools = AsyncMock(
            return_value=_make_tool_page(
                [_make_mcp_tool("read_file", "r"), _make_mcp_tool("write_file", "w")]
            )
        )

        @asynccontextmanager
        async def _fake(
            _connection: dict[str, Any],
            *,
            _mcp_callbacks: object | None = None,
        ) -> AsyncIterator[AsyncMock]:
            await asyncio.sleep(0)
            yield session

        with patch("langchain_mcp_adapters.sessions.create_session", _fake):
            tools, manager, _ = await get_mcp_tools(path)

        assert [t.name for t in tools] == ["fs_read_file"]
        assert manager is not None
        await manager.cleanup()

    async def test_filter_applies_to_http_server(
        self,
        write_config: Callable[..., str],
    ) -> None:
        """`allowedTools` is honored for http (remote) servers, not just stdio."""
        path = write_config(
            {
                "mcpServers": {
                    "api": {
                        "type": "http",
                        "url": "https://example.com/mcp",
                        "allowedTools": ["search"],
                    }
                }
            }
        )

        session = AsyncMock()
        session.initialize = AsyncMock()
        session.list_tools = AsyncMock(
            return_value=_make_tool_page(
                [_make_mcp_tool("search", "s"), _make_mcp_tool("delete", "d")]
            )
        )

        @asynccontextmanager
        async def _fake(
            _connection: dict[str, Any],
            *,
            _mcp_callbacks: object | None = None,
        ) -> AsyncIterator[AsyncMock]:
            await asyncio.sleep(0)
            yield session

        with patch("langchain_mcp_adapters.sessions.create_session", _fake):
            tools, manager, _ = await get_mcp_tools(path)

        assert [t.name for t in tools] == ["api_search"]
        assert manager is not None
        await manager.cleanup()

    async def test_filters_are_per_server(
        self,
        write_config: Callable[..., str],
    ) -> None:
        """Each server's filter applies only to its own tools, never the union."""
        path = write_config(
            {
                "mcpServers": {
                    "fs": {
                        "command": "node",
                        "args": ["server.js"],
                        "allowedTools": ["read_file"],
                    },
                    "api": {
                        "type": "http",
                        "url": "https://example.com/mcp",
                        "disabledTools": ["delete"],
                    },
                }
            }
        )

        fs_session = AsyncMock()
        fs_session.initialize = AsyncMock()
        fs_session.list_tools = AsyncMock(
            return_value=_make_tool_page(
                [_make_mcp_tool("read_file", "r"), _make_mcp_tool("write_file", "w")]
            )
        )
        api_session = AsyncMock()
        api_session.initialize = AsyncMock()
        api_session.list_tools = AsyncMock(
            return_value=_make_tool_page(
                [_make_mcp_tool("search", "s"), _make_mcp_tool("delete", "d")]
            )
        )

        sessions_by_url = {
            "https://example.com/mcp": api_session,
        }

        @asynccontextmanager
        async def _fake(
            connection: dict[str, Any],
            *,
            _mcp_callbacks: object | None = None,
        ) -> AsyncIterator[AsyncMock]:
            await asyncio.sleep(0)
            url = connection.get("url")
            if isinstance(url, str):
                session = sessions_by_url.get(url)
                yield session if session is not None else fs_session
            else:
                yield fs_session

        with patch("langchain_mcp_adapters.sessions.create_session", _fake):
            tools, manager, _ = await get_mcp_tools(path)

        names = sorted(t.name for t in tools)
        assert names == ["api_search", "fs_read_file"]
        assert manager is not None
        await manager.cleanup()


class TestNormalizeMCPArguments:
    """Cover the empty-string-stripping at the MCP tool boundary."""

    def _schema(
        self,
        properties: dict[str, dict],
        required: list[str] | None = None,
    ) -> dict:
        return {
            "type": "object",
            "properties": properties,
            "required": required or [],
        }

    def test_drops_empty_optional_string(self) -> None:
        schema = self._schema(
            {
                "query": {"type": "string"},
                "context_channel_id": {"type": "string"},
            },
            required=["query"],
        )
        out = _normalize_mcp_arguments(
            {"query": "hello", "context_channel_id": ""}, schema
        )
        assert out == {"query": "hello"}

    def test_keeps_empty_required_string(self) -> None:
        schema = self._schema({"query": {"type": "string"}}, required=["query"])
        out = _normalize_mcp_arguments({"query": ""}, schema)
        assert out == {"query": ""}

    def test_keeps_nonempty_strings(self) -> None:
        schema = self._schema({"q": {"type": "string"}})
        out = _normalize_mcp_arguments({"q": "x"}, schema)
        assert out == {"q": "x"}

    def test_keeps_non_string_values(self) -> None:
        schema = self._schema(
            {
                "limit": {"type": "integer"},
                "include_bots": {"type": "boolean"},
            }
        )
        out = _normalize_mcp_arguments({"limit": 0, "include_bots": False}, schema)
        assert out == {"limit": 0, "include_bots": False}

    def test_drops_empty_when_property_missing_type(self) -> None:
        schema = self._schema({"hint": {"description": "free-form"}})
        out = _normalize_mcp_arguments({"hint": ""}, schema)
        assert out == {}

    def test_drops_empty_for_unknown_property(self) -> None:
        # Tool calls sometimes carry extra fields not listed in `properties`.
        # Without a schema entry we can't prove the field is string-typed,
        # but treating empty as "omitted" is the safer default.
        schema = self._schema({"known": {"type": "string"}})
        out = _normalize_mcp_arguments({"known": "a", "extra": ""}, schema)
        assert out == {"known": "a"}

    def test_passes_through_when_schema_is_not_dict(self) -> None:
        out = _normalize_mcp_arguments({"a": "", "b": 1}, None)
        assert out == {"a": "", "b": 1}

    def test_handles_union_string_type(self) -> None:
        schema = self._schema({"v": {"type": ["string", "null"]}})
        out = _normalize_mcp_arguments({"v": ""}, schema)
        assert out == {}

    def test_drops_empty_for_oneof_schema(self) -> None:
        """`oneOf` props have no top-level `type` → conservative drop."""
        schema = self._schema({"v": {"oneOf": [{"type": "string"}, {"type": "null"}]}})
        out = _normalize_mcp_arguments({"v": ""}, schema)
        assert out == {}

    def test_drops_empty_for_anyof_schema(self) -> None:
        """`anyOf` props share the same no-top-level-`type` shape."""
        schema = self._schema({"v": {"anyOf": [{"type": "string"}]}})
        out = _normalize_mcp_arguments({"v": ""}, schema)
        assert out == {}

    def test_drops_empty_for_ref_schema(self) -> None:
        """`$ref` props look like `{"$ref": "#/..."}` — no `type` either."""
        schema = self._schema({"v": {"$ref": "#/definitions/ChannelId"}})
        out = _normalize_mcp_arguments({"v": ""}, schema)
        assert out == {}

    def test_drops_empty_when_property_is_boolean_schema(self) -> None:
        """JSON Schema allows `{"properties": {"k": true}}` — `prop` non-dict.

        `isinstance(prop, dict)` guards the `.get("type")` call so we don't
        crash, and the field is treated as ambiguous (drop).
        """
        schema = {"type": "object", "properties": {"k": True}, "required": []}
        out = _normalize_mcp_arguments({"k": ""}, schema)
        assert out == {}

    def test_passes_through_falsy_non_string_values(self) -> None:
        """Guards against a `if not value` refactor — `0`/`False`/`[]`/`{}` survive."""
        schema = self._schema(
            {
                "i": {"type": "integer"},
                "b": {"type": "boolean"},
                "a": {"type": "array"},
                "o": {"type": "object"},
            }
        )
        out = _normalize_mcp_arguments({"i": 0, "b": False, "a": [], "o": {}}, schema)
        assert out == {"i": 0, "b": False, "a": [], "o": {}}

    def test_required_takes_precedence_over_string_schema(self) -> None:
        """`required` wins even when properties confirm the field is string-typed."""
        schema = self._schema({"query": {"type": "string"}}, required=["query"])
        out = _normalize_mcp_arguments({"query": ""}, schema)
        assert out == {"query": ""}

    def test_logs_dropped_keys(self, caplog: pytest.LogCaptureFixture) -> None:
        """Diagnostic log fires when at least one key is stripped."""
        import logging

        schema = self._schema(
            {"q": {"type": "string"}, "ctx": {"type": "string"}},
            required=["q"],
        )
        with caplog.at_level(logging.DEBUG, logger="deepagents_code.mcp_tools"):
            _normalize_mcp_arguments({"q": "x", "ctx": ""}, schema)
        assert any(
            "dropped empty-string keys" in r.message and "ctx" in r.message
            for r in caplog.records
        )


class TestSelectiveProjectMcpTrust:
    """Per-server allow/deny filtering of project MCP servers.

    The user-level allow/deny lists are honored only from the user's own
    `config.toml` (via `DEFAULT_CONFIG_PATH`), never from a repo-committed
    file, and only allowlisted (or fully trusted) names reach the loader — so
    the SSRF/exfiltration gate on untrusted remote entries stays intact.
    """

    @staticmethod
    def _write_project_config(project_root: Path, servers: dict[str, Any]) -> None:
        (project_root / ".mcp.json").write_text(json.dumps({"mcpServers": servers}))

    @staticmethod
    def _stdio(command: str = "echo") -> dict[str, Any]:
        return {"command": command, "args": []}

    @staticmethod
    def _remote(url: str = "https://example.test/mcp") -> dict[str, Any]:
        return {"type": "sse", "url": url}

    async def _resolve_merged(
        self,
        project_root: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        user_config: Path,
        trust_project_mcp: bool | None,
    ) -> dict[str, Any] | None:
        """Run resolution and return the merged config passed to the loader.

        Returns `None` when the loader is never reached (i.e. every project
        server was dropped and no other config remained).
        """
        # Isolate discovery and the trust store from the developer's real home.
        home = project_root.parent / "home"
        (home / ".deepagents").mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setattr(
            "deepagents_code.model_config.DEFAULT_CONFIG_PATH", user_config
        )

        loader = AsyncMock(return_value=([], None, []))
        monkeypatch.setattr("deepagents_code.mcp_tools._load_tools_from_config", loader)

        ctx = ProjectContext(user_cwd=project_root, project_root=project_root)
        await resolve_and_load_mcp_tools(
            project_context=ctx,
            trust_project_mcp=trust_project_mcp,
        )
        if not loader.called:
            return None
        return loader.call_args.args[0]

    async def test_allowlisted_loads_and_sibling_dropped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An allowlisted server loads while a non-listed sibling is dropped."""
        project = tmp_path / "project"
        project.mkdir()
        self._write_project_config(
            project, {"docs": self._stdio(), "other": self._stdio()}
        )
        user_config = tmp_path / "config.toml"
        user_config.write_text('[mcp]\nenabled_project_servers = ["docs"]\n')

        merged = await self._resolve_merged(
            project, monkeypatch, user_config=user_config, trust_project_mcp=False
        )

        assert merged is not None
        assert set(merged["mcpServers"]) == {"docs"}

    async def test_allowlisted_loads_with_invalid_unlisted_sibling(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An invalid unlisted server cannot block an allowlisted sibling."""
        project = tmp_path / "project"
        project.mkdir()
        self._write_project_config(
            project,
            {
                "docs": self._stdio(),
                "broken": ["not", "a", "server"],
            },
        )
        user_config = tmp_path / "config.toml"
        user_config.write_text('[mcp]\nenabled_project_servers = ["docs"]\n')

        merged = await self._resolve_merged(
            project, monkeypatch, user_config=user_config, trust_project_mcp=False
        )

        assert merged is not None
        assert set(merged["mcpServers"]) == {"docs"}

    async def test_disabled_dropped_even_when_trusted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An explicitly disabled server is dropped even from a trusted config."""
        project = tmp_path / "project"
        project.mkdir()
        self._write_project_config(
            project, {"docs": self._stdio(), "blocked": self._stdio()}
        )
        user_config = tmp_path / "config.toml"
        user_config.write_text('[mcp]\ndisabled_project_servers = ["blocked"]\n')

        merged = await self._resolve_merged(
            project, monkeypatch, user_config=user_config, trust_project_mcp=True
        )

        assert merged is not None
        assert set(merged["mcpServers"]) == {"docs"}

    async def test_disabled_invalid_server_dropped_before_validation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An invalid disabled server cannot block a trusted sibling."""
        project = tmp_path / "project"
        project.mkdir()
        self._write_project_config(
            project,
            {
                "docs": self._stdio(),
                "blocked": ["not", "a", "server"],
            },
        )
        user_config = tmp_path / "config.toml"
        user_config.write_text('[mcp]\ndisabled_project_servers = ["blocked"]\n')

        merged = await self._resolve_merged(
            project, monkeypatch, user_config=user_config, trust_project_mcp=True
        )

        assert merged is not None
        assert set(merged["mcpServers"]) == {"docs"}

    async def test_repo_committed_allowlist_is_ignored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An allowlist committed in the repo does not self-approve servers."""
        project = tmp_path / "project"
        project.mkdir()
        self._write_project_config(project, {"evil": self._stdio()})
        # Attacker-committed project config that tries to approve its own server.
        (project / "config.toml").write_text(
            '[mcp]\nenabled_project_servers = ["evil"]\n'
        )
        (project / ".deepagents").mkdir()
        (project / ".deepagents" / "config.toml").write_text(
            '[mcp]\nenabled_project_servers = ["evil"]\n'
        )
        # User has no allowlist of their own.
        user_config = tmp_path / "config.toml"

        merged = await self._resolve_merged(
            project, monkeypatch, user_config=user_config, trust_project_mcp=False
        )

        # The repo allowlist is never read, so the server stays dropped.
        assert merged is None

    async def test_name_in_both_lists_is_disabled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A server named in both lists is disabled (reject precedence)."""
        project = tmp_path / "project"
        project.mkdir()
        self._write_project_config(project, {"both": self._stdio()})
        user_config = tmp_path / "config.toml"
        user_config.write_text(
            "[mcp]\n"
            'enabled_project_servers = ["both"]\n'
            'disabled_project_servers = ["both"]\n'
        )

        merged = await self._resolve_merged(
            project, monkeypatch, user_config=user_config, trust_project_mcp=False
        )

        assert merged is None

    async def test_malformed_table_falls_back_to_full_drop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A wrong-typed enabled value drops all untrusted servers, no crash.

        (A bare string is coerced to a single name, so use a genuinely wrong
        type — an integer — which degrades to an empty allowlist and full drop.)
        """
        project = tmp_path / "project"
        project.mkdir()
        self._write_project_config(project, {"docs": self._stdio()})
        user_config = tmp_path / "config.toml"
        user_config.write_text(
            "[mcp]\nenabled_project_servers = 123\n"
        )  # not a list or string

        merged = await self._resolve_merged(
            project, monkeypatch, user_config=user_config, trust_project_mcp=False
        )

        assert merged is None

    async def test_env_allowlist_honored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The env allowlist approves a server with no TOML allowlist set."""
        project = tmp_path / "project"
        project.mkdir()
        self._write_project_config(
            project, {"docs": self._stdio(), "other": self._stdio()}
        )
        user_config = tmp_path / "config.toml"  # no [mcp] table
        monkeypatch.setenv("DEEPAGENTS_CODE_ENABLED_PROJECT_MCP_SERVERS", "docs")

        merged = await self._resolve_merged(
            project, monkeypatch, user_config=user_config, trust_project_mcp=False
        )

        assert merged is not None
        assert set(merged["mcpServers"]) == {"docs"}

    async def test_allowlisted_remote_kept_and_sibling_dropped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Remote entries are gated by name too: only the allowlisted one loads.

        This is the SSRF/exfiltration case the design exists for — a
        non-allowlisted remote entry must never survive into the merged config
        (and so never reach the preflight probe or `${VAR}` header resolution).
        """
        project = tmp_path / "project"
        project.mkdir()
        self._write_project_config(
            project, {"docs": self._remote(), "evil": self._remote()}
        )
        user_config = tmp_path / "config.toml"
        user_config.write_text('[mcp]\nenabled_project_servers = ["docs"]\n')

        merged = await self._resolve_merged(
            project, monkeypatch, user_config=user_config, trust_project_mcp=False
        )

        assert merged is not None
        assert set(merged["mcpServers"]) == {"docs"}
        assert merged["mcpServers"]["docs"]["type"] == "sse"

    async def test_allow_and_deny_combined_in_untrusted_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One untrusted config: allowed kept, denied and unlisted both dropped."""
        project = tmp_path / "project"
        project.mkdir()
        self._write_project_config(
            project,
            {
                "keep": self._stdio(),
                "block": self._stdio(),
                "other": self._stdio(),
            },
        )
        user_config = tmp_path / "config.toml"
        user_config.write_text(
            "[mcp]\n"
            'enabled_project_servers = ["keep"]\n'
            'disabled_project_servers = ["block"]\n'
        )

        merged = await self._resolve_merged(
            project, monkeypatch, user_config=user_config, trust_project_mcp=False
        )

        assert merged is not None
        assert set(merged["mcpServers"]) == {"keep"}

    async def test_disabled_dropped_when_fingerprint_trusted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deny wins over fingerprint trust, not just the explicit flag."""
        project = tmp_path / "project"
        project.mkdir()
        self._write_project_config(
            project, {"docs": self._stdio(), "blocked": self._stdio()}
        )
        user_config = tmp_path / "config.toml"
        user_config.write_text('[mcp]\ndisabled_project_servers = ["blocked"]\n')
        # Fingerprint store reports the whole config as trusted.
        monkeypatch.setattr(
            "deepagents_code.mcp_trust.is_project_mcp_trusted", lambda *_a, **_k: True
        )

        merged = await self._resolve_merged(
            project, monkeypatch, user_config=user_config, trust_project_mcp=None
        )

        assert merged is not None
        assert set(merged["mcpServers"]) == {"docs"}

    async def test_allowlisted_loads_when_fingerprint_untrusted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On the `None` path with untrusted fingerprint, allowlist still loads."""
        project = tmp_path / "project"
        project.mkdir()
        self._write_project_config(
            project, {"docs": self._stdio(), "other": self._stdio()}
        )
        user_config = tmp_path / "config.toml"
        user_config.write_text('[mcp]\nenabled_project_servers = ["docs"]\n')
        monkeypatch.setattr(
            "deepagents_code.mcp_trust.is_project_mcp_trusted", lambda *_a, **_k: False
        )

        merged = await self._resolve_merged(
            project, monkeypatch, user_config=user_config, trust_project_mcp=None
        )

        assert merged is not None
        assert set(merged["mcpServers"]) == {"docs"}

    async def test_allowlisted_but_invalid_server_is_nonfatal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An allowlisted server that is itself invalid is dropped, not fatal.

        Exercises the deferred per-server validation branch: the kept subset is
        validated after trust filtering, and a validation failure drops the
        config rather than crashing resolution.
        """
        project = tmp_path / "project"
        project.mkdir()
        # A dict (so it yields a summary and skips the empty-summaries fast path),
        # but setting both tool filters at once — a documented per-server error.
        self._write_project_config(
            project,
            {
                "docs": {
                    "command": "echo",
                    "args": [],
                    "allowedTools": ["a"],
                    "disabledTools": ["b"],
                }
            },
        )
        user_config = tmp_path / "config.toml"
        user_config.write_text('[mcp]\nenabled_project_servers = ["docs"]\n')

        merged = await self._resolve_merged(
            project, monkeypatch, user_config=user_config, trust_project_mcp=False
        )

        # Invalid kept server -> whole filtered config dropped -> loader unreached.
        assert merged is None

    async def test_unreadable_user_config_fails_closed_and_surfaces_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A corrupt user config.toml drops project servers even under --trust.

        The allow/deny policy could not be read, so the loader records a
        `read_error`; resolution then treats the project config as untrusted
        (fail closed) and surfaces the error as an MCP config error rather than
        loading a server the user might have meant to deny.
        """
        project = tmp_path / "project"
        project.mkdir()
        self._write_project_config(project, {"docs": self._stdio()})
        home = tmp_path / "home"
        (home / ".deepagents").mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("HOME", str(home))
        user_config = tmp_path / "config.toml"
        user_config.write_text("[[not valid toml")
        monkeypatch.setattr(
            "deepagents_code.model_config.DEFAULT_CONFIG_PATH", user_config
        )
        loader = AsyncMock(return_value=([], None, []))
        monkeypatch.setattr("deepagents_code.mcp_tools._load_tools_from_config", loader)

        ctx = ProjectContext(user_cwd=project, project_root=project)
        _tools, _manager, infos = await resolve_and_load_mcp_tools(
            project_context=ctx, trust_project_mcp=True
        )

        # Fail closed: even with trust_project_mcp=True, nothing loads.
        assert loader.call_count == 0
        # The read failure is surfaced (not just a debug-only warning).
        assert any(
            info.status == "error" and "config.toml" in (info.error or "")
            for info in infos
        )

    async def test_env_enabled_survives_unreadable_user_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A read_error fails closed, but an env-enabled name still loads.

        On a corrupt config.toml the loader forces the config untrusted, yet the
        allowlist read from a still-readable source (the env var) applies: the
        env-enabled server loads while a non-listed sibling is dropped. Pins the
        "readable source (shell env) still survives" branch so a future hardening
        that also empties `enabled` on read_error doesn't silently drop a server
        the user explicitly allowlisted.
        """
        project = tmp_path / "project"
        project.mkdir()
        self._write_project_config(
            project, {"docs": self._stdio(), "other": self._stdio()}
        )
        user_config = tmp_path / "config.toml"
        user_config.write_text("[[not valid toml")
        monkeypatch.setenv("DEEPAGENTS_CODE_ENABLED_PROJECT_MCP_SERVERS", "docs")

        merged = await self._resolve_merged(
            project, monkeypatch, user_config=user_config, trust_project_mcp=True
        )

        assert merged is not None
        assert set(merged["mcpServers"]) == {"docs"}
