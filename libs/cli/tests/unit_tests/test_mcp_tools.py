"""Tests for MCP tools configuration loading and validation."""

import json
from collections.abc import Callable, Generator
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepagents_cli.mcp_tools import (
    MCPServerInfo,
    MCPSessionManager,
    MCPToolInfo,
    _check_remote_server,
    _check_stdio_server,
    _filter_project_stdio_servers,
    classify_discovered_configs,
    discover_mcp_configs,
    extract_stdio_server_commands,
    get_mcp_tools,
    load_mcp_config,
    load_mcp_config_lenient,
    merge_mcp_configs,
    reset_default_session_manager_for_testing,
    resolve_and_load_mcp_tools,
)
from deepagents_cli.project_utils import ProjectContext


def _make_mcp_tool(
    name: str, description: str = "", input_schema: dict | None = None
) -> MagicMock:
    """Build a mock MCP `Tool` object suitable for proxy-tool construction."""
    mock = MagicMock(
        spec=["name", "description", "inputSchema", "annotations", "meta"]
    )
    mock.name = name
    mock.description = description
    mock.inputSchema = input_schema or {"type": "object", "properties": {}}
    mock.annotations = None
    mock.meta = None
    return mock


def _make_tool_page(tools: list[MagicMock], next_cursor: str | None = None) -> MagicMock:
    """Build a mock `list_tools` page result."""
    page = MagicMock(spec=["tools", "nextCursor"])
    page.tools = tools
    page.nextCursor = next_cursor
    return page


@pytest.fixture(autouse=True)
def _reset_mcp_singleton() -> Generator[None]:
    """Clear the process-wide MCP session cache between tests.

    The singleton accumulates cached sessions across calls; tests that
    exercise the cache must start from a clean slate so they see exactly
    the `create_session` invocations they expect.
    """
    reset_default_session_manager_for_testing()
    yield
    reset_default_session_manager_for_testing()

# Test Fixtures


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
    """Fixture that writes a JSON config dict to a temp file and returns the path."""

    def _write(config_data: dict, filename: str = "mcp-config.json") -> str:
        config_file = tmp_path / filename
        config_file.write_text(json.dumps(config_data))
        return str(config_file)

    return _write


@pytest.fixture
def fake_create_session() -> Generator[tuple[AsyncMock, list]]:
    """Patch `langchain_mcp_adapters.sessions.create_session`.

    Yields `(mock_session, recorded_connections)` where `mock_session` is the
    shared `ClientSession` mock returned by every `async with create_session`
    block, and `recorded_connections` is an append-only list of the
    `Connection` dict passed on each invocation.

    Tests that need per-server behavior can inspect or replace
    `mock_session.list_tools.side_effect`.
    """
    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    mock_session.list_tools = AsyncMock(return_value=_make_tool_page([]))

    recorded: list = []

    @asynccontextmanager
    async def _fake(connection, *, mcp_callbacks=None):  # noqa: ANN001, ARG001
        recorded.append(connection)
        yield mock_session

    with patch("langchain_mcp_adapters.sessions.create_session", _fake):
        yield mock_session, recorded


@pytest.fixture
def mock_tools() -> list[MagicMock]:
    """Fixture providing mock MCP tool metadata objects."""
    return [
        _make_mcp_tool("read_file", "Read a file"),
        _make_mcp_tool("write_file", "Write a file"),
    ]


class TestLoadMCPConfig:
    """Test MCP configuration file loading and validation."""

    def test_load_valid_config(
        self, write_config: Callable[..., str], valid_config_data: dict
    ) -> None:
        """Test loading a valid MCP configuration file."""
        path = write_config(valid_config_data)
        config = load_mcp_config(path)
        assert config == valid_config_data

    def test_load_config_file_not_found(self, tmp_path: Path) -> None:
        """Test that FileNotFoundError is raised for missing config file."""
        nonexistent_file = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError, match="MCP config file not found"):
            load_mcp_config(str(nonexistent_file))

    def test_load_config_invalid_json(self, tmp_path: Path) -> None:
        """Test that JSONDecodeError is raised for invalid JSON."""
        config_file = tmp_path / "invalid.json"
        config_file.write_text("{invalid json")

        with pytest.raises(
            json.JSONDecodeError, match="Invalid JSON in MCP config file"
        ):
            load_mcp_config(str(config_file))

    def test_load_config_missing_mcpservers_field(
        self, write_config: Callable[..., str]
    ) -> None:
        """Test that ValueError is raised when mcpServers field is missing."""
        path = write_config({"someOtherField": "value"})

        with pytest.raises(ValueError, match="must contain 'mcpServers' field"):
            load_mcp_config(path)

    @pytest.mark.parametrize(
        ("config_data", "expected_error", "exception_type"),
        [
            (
                {"mcpServers": ["not", "a", "dict"]},
                "'mcpServers' field must be a dictionary",
                TypeError,
            ),
            ({"mcpServers": {}}, "'mcpServers' field is empty", ValueError),
        ],
    )
    def test_load_config_invalid_mcpservers(
        self,
        write_config: Callable[..., str],
        config_data: dict,
        expected_error: str,
        exception_type: type[Exception],
    ) -> None:
        """Test that appropriate exception is raised for invalid mcpServers field."""
        path = write_config(config_data)

        with pytest.raises(exception_type, match=expected_error):
            load_mcp_config(path)

    def test_load_config_server_missing_command(
        self, write_config: Callable[..., str]
    ) -> None:
        """Test that ValueError is raised when server config is missing command."""
        path = write_config(
            {
                "mcpServers": {
                    "filesystem": {
                        "args": ["/tmp"],
                        # Missing "command" field
                    }
                }
            }
        )

        with pytest.raises(
            ValueError, match=r"filesystem.*missing required 'command' field"
        ):
            load_mcp_config(path)

    @pytest.mark.parametrize(
        ("server_config", "expected_error"),
        [
            ("not a dict", "filesystem.*config must be a dictionary"),
            (
                {"command": "npx", "args": "not a list"},
                "filesystem.*'args' must be a list",
            ),
            (
                {"command": "npx", "args": ["/tmp"], "env": ["not", "a", "dict"]},
                "filesystem.*'env' must be a dictionary",
            ),
        ],
    )
    def test_load_config_invalid_field_types(
        self,
        write_config: Callable[..., str],
        server_config: dict | str,
        expected_error: str,
    ) -> None:
        """Test that TypeError is raised for invalid server config field types."""
        path = write_config({"mcpServers": {"filesystem": server_config}})

        with pytest.raises(TypeError, match=expected_error):
            load_mcp_config(path)

    def test_load_config_optional_fields(
        self, write_config: Callable[..., str]
    ) -> None:
        """Test that args and env are optional fields."""
        config_data = {
            "mcpServers": {
                "simple": {
                    "command": "simple-server",
                    # No args or env - should be valid
                }
            }
        }
        path = write_config(config_data)

        config = load_mcp_config(path)

        assert config == config_data
        assert "simple" in config["mcpServers"]

    def test_load_config_multiple_servers(
        self, write_config: Callable[..., str]
    ) -> None:
        """Test loading config with multiple MCP servers."""
        config_data = {
            "mcpServers": {
                "filesystem": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                    "env": {},
                },
                "brave-search": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-brave-search"],
                    "env": {"BRAVE_API_KEY": "test-key"},
                },
                "github": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-github"],
                    "env": {"GITHUB_TOKEN": "test-token"},
                },
            }
        }
        path = write_config(config_data)

        config = load_mcp_config(path)

        assert len(config["mcpServers"]) == 3
        assert "filesystem" in config["mcpServers"]
        assert "brave-search" in config["mcpServers"]
        assert "github" in config["mcpServers"]

    def test_load_config_sse_server(self, write_config: Callable[..., str]) -> None:
        """Test loading config with SSE server type."""
        config_data = {
            "mcpServers": {
                "remote-api": {
                    "type": "sse",
                    "url": "https://api.example.com/mcp",
                }
            }
        }
        path = write_config(config_data)

        config = load_mcp_config(path)

        assert config == config_data
        assert config["mcpServers"]["remote-api"]["type"] == "sse"
        assert (
            config["mcpServers"]["remote-api"]["url"] == "https://api.example.com/mcp"
        )

    def test_load_config_http_server(self, write_config: Callable[..., str]) -> None:
        """Test loading config with HTTP server type."""
        config_data = {
            "mcpServers": {
                "web-api": {
                    "type": "http",
                    "url": "https://api.example.com/mcp",
                }
            }
        }
        path = write_config(config_data)

        config = load_mcp_config(path)

        assert config == config_data
        assert config["mcpServers"]["web-api"]["type"] == "http"

    @pytest.mark.parametrize(
        ("server_name", "server_type"),
        [
            ("remote-api", "sse"),
            ("web-api", "http"),
        ],
    )
    def test_load_config_remote_server_missing_url(
        self,
        write_config: Callable[..., str],
        server_name: str,
        server_type: str,
    ) -> None:
        """Test that ValueError is raised when SSE/HTTP server is missing url field."""
        path = write_config({"mcpServers": {server_name: {"type": server_type}}})

        with pytest.raises(
            ValueError, match=f"{server_name}.*missing required 'url' field"
        ):
            load_mcp_config(path)

    def test_load_config_mixed_server_types(
        self, write_config: Callable[..., str]
    ) -> None:
        """Test loading config with mixed stdio, SSE, and HTTP servers."""
        config_data = {
            "mcpServers": {
                "filesystem": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                },
                "remote-sse": {
                    "type": "sse",
                    "url": "https://api.example.com/sse",
                },
                "remote-http": {
                    "type": "http",
                    "url": "https://api.example.com/http",
                },
            }
        }
        path = write_config(config_data)

        config = load_mcp_config(path)

        assert len(config["mcpServers"]) == 3
        assert "command" in config["mcpServers"]["filesystem"]
        assert config["mcpServers"]["remote-sse"]["type"] == "sse"
        assert config["mcpServers"]["remote-http"]["type"] == "http"

    def test_load_config_sse_with_headers(
        self, write_config: Callable[..., str]
    ) -> None:
        """Test loading SSE server config with custom headers."""
        config_data = {
            "mcpServers": {
                "authenticated-api": {
                    "type": "sse",
                    "url": "https://api.example.com/mcp",
                    "headers": {
                        "Authorization": "Bearer token123",
                        "X-Custom-Header": "value",
                    },
                }
            }
        }
        path = write_config(config_data)

        config = load_mcp_config(path)

        headers = config["mcpServers"]["authenticated-api"]["headers"]
        assert headers["Authorization"] == "Bearer token123"

    def test_load_config_http_with_headers(
        self, write_config: Callable[..., str]
    ) -> None:
        """Test loading HTTP server config with custom headers."""
        config_data = {
            "mcpServers": {
                "authenticated-api": {
                    "type": "http",
                    "url": "https://api.example.com/mcp",
                    "headers": {"Authorization": "Bearer secret"},
                }
            }
        }
        path = write_config(config_data)

        config = load_mcp_config(path)

        headers = config["mcpServers"]["authenticated-api"]["headers"]
        assert headers["Authorization"] == "Bearer secret"

    @pytest.mark.parametrize(
        ("transport_field", "transport_value"),
        [
            ("transport", "http"),
            ("transport", "sse"),
            ("type", "http"),
            ("type", "sse"),
        ],
    )
    def test_load_config_transport_field_alias(
        self,
        write_config: Callable[..., str],
        transport_field: str,
        transport_value: str,
    ) -> None:
        """Test that both 'type' and 'transport' fields are accepted for server type."""
        config_data = {
            "mcpServers": {
                "remote": {
                    transport_field: transport_value,
                    "url": "https://api.example.com/mcp",
                }
            }
        }
        path = write_config(config_data)

        config = load_mcp_config(path)

        assert config == config_data

    def test_load_config_invalid_headers_type(
        self, write_config: Callable[..., str]
    ) -> None:
        """Test that TypeError is raised when headers is not a dictionary."""
        path = write_config(
            {
                "mcpServers": {
                    "api": {
                        "type": "sse",
                        "url": "https://api.example.com/mcp",
                        "headers": ["not", "a", "dict"],
                    }
                }
            }
        )

        with pytest.raises(TypeError, match=r"api.*'headers' must be a dictionary"):
            load_mcp_config(path)

    def test_load_config_unknown_server_type(
        self, write_config: Callable[..., str]
    ) -> None:
        """Test that ValueError is raised for unsupported transport types."""
        path = write_config(
            {
                "mcpServers": {
                    "ws-server": {
                        "type": "websocket",
                        "url": "ws://example.com/mcp",
                    }
                }
            }
        )

        with pytest.raises(
            ValueError, match=r"ws-server.*unsupported transport type 'websocket'"
        ):
            load_mcp_config(path)

    @pytest.mark.parametrize(
        "bad_name",
        [
            "../evil",
            "../../etc/passwd",
            "a/b",
            "a\\b",
            "",
            ".",
            "..",
            "foo bar",
            "foo\x00bar",
            "foo.json",
        ],
    )
    def test_load_config_rejects_unsafe_server_name(
        self, write_config: Callable[..., str], bad_name: str
    ) -> None:
        """Server names must be path-safe — they're used in token file paths."""
        config_data = {
            "mcpServers": {
                bad_name: {"command": "echo"},
            }
        }
        path = write_config(config_data)

        with pytest.raises(ValueError, match="server name"):
            load_mcp_config(path)

    @pytest.mark.parametrize(
        "ok_name",
        ["filesystem", "brave-search", "linear_v2", "Notion", "gh-123"],
    )
    def test_load_config_accepts_safe_server_name(
        self, write_config: Callable[..., str], ok_name: str
    ) -> None:
        """Alphanumeric, hyphens, and underscores are allowed server names."""
        config_data = {
            "mcpServers": {
                ok_name: {"command": "echo"},
            }
        }
        path = write_config(config_data)

        config = load_mcp_config(path)

        assert ok_name in config["mcpServers"]

    def test_load_config_header_with_missing_env_var_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
        write_config: Callable[..., str],
    ) -> None:
        monkeypatch.delenv("DA_MISSING", raising=False)
        config_data = {
            "mcpServers": {
                "linear": {
                    "transport": "http",
                    "url": "https://mcp.linear.app/mcp",
                    "headers": {"Authorization": "Bearer ${DA_MISSING}"},
                }
            }
        }
        path = write_config(config_data)
        with pytest.raises(RuntimeError, match="DA_MISSING"):
            load_mcp_config(path)

    def test_load_config_header_with_present_env_var_ok(
        self,
        monkeypatch: pytest.MonkeyPatch,
        write_config: Callable[..., str],
    ) -> None:
        monkeypatch.setenv("DA_PRESENT", "tok")
        config_data = {
            "mcpServers": {
                "linear": {
                    "transport": "http",
                    "url": "https://mcp.linear.app/mcp",
                    "headers": {"Authorization": "Bearer ${DA_PRESENT}"},
                }
            }
        }
        path = write_config(config_data)
        config = load_mcp_config(path)
        # The raw config is preserved; substitution happens at connection time.
        assert (
            config["mcpServers"]["linear"]["headers"]["Authorization"]
            == "Bearer ${DA_PRESENT}"
        )

    def test_load_config_auth_oauth_http_ok(
        self, write_config: Callable[..., str]
    ) -> None:
        config_data = {
            "mcpServers": {
                "notion": {
                    "transport": "http",
                    "url": "https://mcp.notion.com/mcp",
                    "auth": "oauth",
                }
            }
        }
        path = write_config(config_data)
        config = load_mcp_config(path)
        assert config["mcpServers"]["notion"]["auth"] == "oauth"

    def test_load_config_auth_unknown_value_rejected(
        self, write_config: Callable[..., str]
    ) -> None:
        config_data = {
            "mcpServers": {
                "notion": {
                    "transport": "http",
                    "url": "https://mcp.notion.com/mcp",
                    "auth": "saml",
                }
            }
        }
        path = write_config(config_data)
        with pytest.raises(ValueError, match="auth"):
            load_mcp_config(path)

    def test_load_config_auth_oauth_on_stdio_rejected(
        self, write_config: Callable[..., str]
    ) -> None:
        config_data = {
            "mcpServers": {
                "local": {
                    "command": "npx",
                    "args": [],
                    "auth": "oauth",
                }
            }
        }
        path = write_config(config_data)
        with pytest.raises(ValueError, match="stdio.*oauth|oauth.*stdio"):
            load_mcp_config(path)

    def test_load_config_auth_oauth_with_authorization_header_rejected(
        self, write_config: Callable[..., str]
    ) -> None:
        config_data = {
            "mcpServers": {
                "notion": {
                    "transport": "http",
                    "url": "https://mcp.notion.com/mcp",
                    "auth": "oauth",
                    "headers": {"Authorization": "Bearer x"},
                }
            }
        }
        path = write_config(config_data)
        with pytest.raises(ValueError, match="Authorization"):
            load_mcp_config(path)


class TestGetMCPTools:
    """Test MCP tools loading from configuration."""

    @pytest.fixture(autouse=True)
    def _bypass_health_checks(self) -> Generator[None]:
        """Bypass pre-flight health checks for all tests in this class."""
        with (
            patch("deepagents_cli.mcp_tools._check_stdio_server"),
            patch(
                "deepagents_cli.mcp_tools._check_remote_server",
                new_callable=AsyncMock,
            ),
        ):
            yield

    async def test_get_mcp_tools_success(
        self,
        write_config: Callable[..., str],
        valid_config_data: dict,
        fake_create_session: tuple[AsyncMock, list],
        mock_tools: list,
    ) -> None:
        """Test successful loading of MCP tools."""
        path = write_config(valid_config_data)
        mock_session, recorded = fake_create_session
        mock_session.list_tools = AsyncMock(return_value=_make_tool_page(mock_tools))

        tools, manager, server_infos = await get_mcp_tools(path)

        # Verify connection passed to create_session
        assert len(recorded) == 1
        conn = recorded[0]
        assert conn["command"] == "npx"
        assert conn["transport"] == "stdio"
        assert conn["args"] == [
            "-y",
            "@modelcontextprotocol/server-filesystem",
            "/tmp",
        ]

        # Discovery session lifecycle
        mock_session.initialize.assert_awaited_once()
        mock_session.list_tools.assert_awaited_once()

        # Proxy tools prefixed with server name and sorted by name
        assert len(tools) == 2
        assert tools[0].name == "filesystem_read_file"
        assert tools[1].name == "filesystem_write_file"
        assert isinstance(manager, MCPSessionManager)

        # server_infos reflects proxy tool names/descriptions
        assert len(server_infos) == 1
        assert server_infos[0].name == "filesystem"
        assert server_infos[0].transport == "stdio"
        assert server_infos[0].tools == [
            MCPToolInfo(name="filesystem_read_file", description="Read a file"),
            MCPToolInfo(name="filesystem_write_file", description="Write a file"),
        ]

    async def test_get_mcp_tools_discovery_failure(
        self,
        write_config: Callable[..., str],
        valid_config_data: dict,
        fake_create_session: tuple[AsyncMock, list],
    ) -> None:
        """Discovery errors surface as RuntimeError wrapping the original cause."""
        path = write_config(valid_config_data)
        mock_session, _ = fake_create_session
        mock_session.initialize = AsyncMock(side_effect=Exception("handshake failed"))

        with pytest.raises(
            RuntimeError,
            match=r"Failed to load tools from MCP server.*handshake failed",
        ):
            await get_mcp_tools(path)

    async def test_get_mcp_tools_list_tools_failure(
        self,
        write_config: Callable[..., str],
        valid_config_data: dict,
        fake_create_session: tuple[AsyncMock, list],
    ) -> None:
        """`list_tools` errors surface as RuntimeError."""
        path = write_config(valid_config_data)
        mock_session, _ = fake_create_session
        mock_session.list_tools = AsyncMock(side_effect=Exception("protocol error"))

        with pytest.raises(
            RuntimeError,
            match=r"Failed to load tools from MCP server.*protocol error",
        ):
            await get_mcp_tools(path)

    async def test_get_mcp_tools_empty_env_dict_coerced_to_none(
        self,
        write_config: Callable[..., str],
        fake_create_session: tuple[AsyncMock, list],
    ) -> None:
        """`"env": {}` is coerced to None (inherit parent env)."""
        path = write_config(
            {"mcpServers": {"fs": {"command": "npx", "args": [], "env": {}}}}
        )
        _, recorded = fake_create_session

        await get_mcp_tools(path)

        assert recorded[0]["env"] is None

    async def test_get_mcp_tools_multiple_servers(
        self,
        write_config: Callable[..., str],
    ) -> None:
        """Discovery opens one session per server; proxy tools collate sorted."""
        path = write_config(
            {
                "mcpServers": {
                    "filesystem": {
                        "command": "npx",
                        "args": [
                            "-y",
                            "@modelcontextprotocol/server-filesystem",
                            "/tmp",
                        ],
                        "env": {},
                    },
                    "brave-search": {
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-brave-search"],
                        "env": {"BRAVE_API_KEY": "test-key"},
                    },
                }
            }
        )

        # Build per-server tool pages that the shared mock session returns in
        # order on each discovery pass.
        fs_tools = [_make_mcp_tool("read_file"), _make_mcp_tool("write_file")]
        search_tools = [_make_mcp_tool("web_search")]
        page_fs = _make_tool_page(fs_tools)
        page_search = _make_tool_page(search_tools)

        recorded: list = []
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        # side_effect yields pages in call order — servers are processed in
        # insertion order, so fs first, then brave-search.
        mock_session.list_tools = AsyncMock(side_effect=[page_fs, page_search])

        @asynccontextmanager
        async def _fake(connection, *, mcp_callbacks=None):  # noqa: ANN001, ARG001
            recorded.append(connection)
            yield mock_session

        with patch("langchain_mcp_adapters.sessions.create_session", _fake):
            tools, manager, server_infos = await get_mcp_tools(path)

        # Two discovery sessions opened, one per server
        assert len(recorded) == 2
        assert {c["command"] for c in recorded} == {"npx"}
        brave_connection = next(
            c for c in recorded if (c.get("env") or {}).get("BRAVE_API_KEY")
        )
        assert brave_connection["env"]["BRAVE_API_KEY"] == "test-key"

        # All three proxy tools present, sorted by prefixed name
        assert [t.name for t in tools] == [
            "brave-search_web_search",
            "filesystem_read_file",
            "filesystem_write_file",
        ]
        assert isinstance(manager, MCPSessionManager)

        assert len(server_infos) == 2
        infos_by_name = {info.name: info for info in server_infos}
        assert set(infos_by_name) == {"filesystem", "brave-search"}
        assert len(infos_by_name["filesystem"].tools) == 2
        assert len(infos_by_name["brave-search"].tools) == 1

    async def test_get_mcp_tools_invalid_config(
        self, write_config: Callable[..., str]
    ) -> None:
        """Test that config validation errors are propagated."""
        path = write_config(
            {
                "mcpServers": {
                    "filesystem": {
                        "args": ["/tmp"],
                        # Missing command field
                    }
                }
            }
        )

        with pytest.raises(ValueError, match="missing required 'command' field"):
            await get_mcp_tools(path)

    async def test_get_mcp_tools_env_variables_passed(
        self,
        write_config: Callable[..., str],
        fake_create_session: tuple[AsyncMock, list],
    ) -> None:
        """Env variables flow through to the stdio `Connection`."""
        path = write_config(
            {
                "mcpServers": {
                    "github": {
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-github"],
                        "env": {
                            "GITHUB_TOKEN": "ghp_test123",
                            "GITHUB_API_URL": "https://api.github.com",
                        },
                    }
                }
            }
        )
        _, recorded = fake_create_session

        await get_mcp_tools(path)

        assert recorded[0]["env"] == {
            "GITHUB_TOKEN": "ghp_test123",
            "GITHUB_API_URL": "https://api.github.com",
        }

    async def test_get_mcp_tools_env_none_when_not_provided(
        self,
        write_config: Callable[..., str],
        fake_create_session: tuple[AsyncMock, list],
    ) -> None:
        """`env` defaults to None (inherit parent env) when not in config."""
        path = write_config(
            {"mcpServers": {"simple": {"command": "simple-server"}}}
        )
        _, recorded = fake_create_session

        await get_mcp_tools(path)

        assert recorded[0]["env"] is None

    async def test_get_mcp_tools_headers_passed_for_sse(
        self,
        write_config: Callable[..., str],
        fake_create_session: tuple[AsyncMock, list],
    ) -> None:
        """SSE server headers flow through to the `Connection`."""
        path = write_config(
            {
                "mcpServers": {
                    "api": {
                        "type": "sse",
                        "url": "https://api.example.com/mcp",
                        "headers": {
                            "Authorization": "Bearer token123",
                            "X-API-Key": "key456",
                        },
                    }
                }
            }
        )
        _, recorded = fake_create_session

        await get_mcp_tools(path)

        conn = recorded[0]
        assert conn["transport"] == "sse"
        assert conn["headers"] == {
            "Authorization": "Bearer token123",
            "X-API-Key": "key456",
        }

    async def test_get_mcp_tools_headers_passed_for_http(
        self,
        write_config: Callable[..., str],
        fake_create_session: tuple[AsyncMock, list],
    ) -> None:
        """HTTP server headers flow through to the `Connection`."""
        path = write_config(
            {
                "mcpServers": {
                    "api": {
                        "type": "http",
                        "url": "https://api.example.com/mcp",
                        "headers": {"Authorization": "Bearer secret"},
                    }
                }
            }
        )
        _, recorded = fake_create_session

        await get_mcp_tools(path)

        conn = recorded[0]
        assert conn["transport"] == "streamable_http"
        assert conn["headers"] == {"Authorization": "Bearer secret"}

    async def test_get_mcp_tools_no_headers_when_not_provided(
        self,
        write_config: Callable[..., str],
        fake_create_session: tuple[AsyncMock, list],
    ) -> None:
        """`headers` key is omitted when not provided in config."""
        path = write_config(
            {
                "mcpServers": {
                    "api": {
                        "type": "sse",
                        "url": "https://api.example.com/mcp",
                    }
                }
            }
        )
        _, recorded = fake_create_session

        await get_mcp_tools(path)

        assert "headers" not in recorded[0]


class TestDiscoverMcpConfigs:
    """Test auto-discovery of MCP config files."""

    def test_project_context_overrides_process_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit project context should drive discovery instead of cwd."""
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: home)

        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / ".git").mkdir()
        user_cwd = project_root / "src"
        user_cwd.mkdir()
        project_cfg = project_root / ".mcp.json"
        project_cfg.write_text('{"mcpServers": {}}')

        other_cwd = tmp_path / "elsewhere"
        other_cwd.mkdir()
        monkeypatch.chdir(other_cwd)

        project_context = ProjectContext.from_user_cwd(user_cwd)
        result = discover_mcp_configs(project_context=project_context)

        assert result == [project_cfg]

    def test_no_configs_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Empty list when no config files exist."""
        project = tmp_path / "project"
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
        monkeypatch.setattr(
            "deepagents_cli.project_utils.find_project_root",
            lambda _start_path=None: project,
        )
        (tmp_path / "home").mkdir()
        project.mkdir()
        assert discover_mcp_configs() == []

    def test_user_level_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only ~/.deepagents/.mcp.json exists."""
        home = tmp_path / "home"
        user_dir = home / ".deepagents"
        user_dir.mkdir(parents=True)
        cfg = user_dir / ".mcp.json"
        cfg.write_text('{"mcpServers": {}}')

        project = tmp_path / "project"
        project.mkdir()

        monkeypatch.setattr(Path, "home", lambda: home)
        monkeypatch.setattr(
            "deepagents_cli.project_utils.find_project_root",
            lambda _start_path=None: project,
        )
        result = discover_mcp_configs()
        assert result == [cfg]

    def test_project_root_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only <project>/.mcp.json exists."""
        home = tmp_path / "home"
        home.mkdir()
        project = tmp_path / "project"
        project.mkdir()
        cfg = project / ".mcp.json"
        cfg.write_text('{"mcpServers": {}}')

        monkeypatch.setattr(Path, "home", lambda: home)
        monkeypatch.setattr(
            "deepagents_cli.project_utils.find_project_root",
            lambda _start_path=None: project,
        )
        result = discover_mcp_configs()
        assert result == [cfg]

    def test_project_deepagents_subdir_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only <project>/.deepagents/.mcp.json exists."""
        home = tmp_path / "home"
        home.mkdir()
        project = tmp_path / "project"
        subdir = project / ".deepagents"
        subdir.mkdir(parents=True)
        cfg = subdir / ".mcp.json"
        cfg.write_text('{"mcpServers": {}}')

        monkeypatch.setattr(Path, "home", lambda: home)
        monkeypatch.setattr(
            "deepagents_cli.project_utils.find_project_root",
            lambda _start_path=None: project,
        )
        result = discover_mcp_configs()
        assert result == [cfg]

    def test_all_three_locations(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """All three config locations exist, returned in precedence order."""
        home = tmp_path / "home"
        user_dir = home / ".deepagents"
        user_dir.mkdir(parents=True)
        user_cfg = user_dir / ".mcp.json"
        user_cfg.write_text('{"mcpServers": {}}')

        project = tmp_path / "project"
        proj_subdir = project / ".deepagents"
        proj_subdir.mkdir(parents=True)
        proj_sub_cfg = proj_subdir / ".mcp.json"
        proj_sub_cfg.write_text('{"mcpServers": {}}')

        proj_root_cfg = project / ".mcp.json"
        proj_root_cfg.write_text('{"mcpServers": {}}')

        monkeypatch.setattr(Path, "home", lambda: home)
        monkeypatch.setattr(
            "deepagents_cli.project_utils.find_project_root",
            lambda _start_path=None: project,
        )
        result = discover_mcp_configs()
        assert result == [user_cfg, proj_sub_cfg, proj_root_cfg]

    def test_falls_back_to_cwd_when_no_git(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Falls back to CWD when find_project_root returns None."""
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: home)
        monkeypatch.setattr(
            "deepagents_cli.project_utils.find_project_root",
            lambda _start_path=None: None,
        )
        monkeypatch.chdir(tmp_path)
        cfg = tmp_path / ".mcp.json"
        cfg.write_text('{"mcpServers": {}}')

        result = discover_mcp_configs()
        assert cfg in result


class TestMergeMcpConfigs:
    """Test merging multiple MCP config dicts."""

    def test_single_config(self) -> None:
        """Single config passes through."""
        cfg = {"mcpServers": {"fs": {"command": "npx"}}}
        result = merge_mcp_configs([cfg])
        assert result == cfg

    def test_disjoint_servers(self) -> None:
        """Disjoint server names are all present."""
        c1 = {"mcpServers": {"fs": {"command": "npx"}}}
        c2 = {"mcpServers": {"search": {"command": "brave"}}}
        result = merge_mcp_configs([c1, c2])
        assert "fs" in result["mcpServers"]
        assert "search" in result["mcpServers"]

    def test_duplicate_server_name_last_wins(self) -> None:
        """Later config overrides earlier for same server name."""
        c1 = {"mcpServers": {"fs": {"command": "old"}}}
        c2 = {"mcpServers": {"fs": {"command": "new"}}}
        result = merge_mcp_configs([c1, c2])
        assert result["mcpServers"]["fs"]["command"] == "new"

    def test_empty_list(self) -> None:
        """Empty input returns empty mcpServers."""
        result = merge_mcp_configs([])
        assert result == {"mcpServers": {}}


class TestLoadMcpConfigLenient:
    """Test lenient config loading for auto-discovery."""

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        """Missing file returns None without raising."""
        result = load_mcp_config_lenient(tmp_path / "nonexistent.json")
        assert result is None

    def test_invalid_json_returns_none(self, tmp_path: Path) -> None:
        """Invalid JSON returns None and logs warning."""
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")
        result = load_mcp_config_lenient(bad)
        assert result is None

    def test_validation_error_returns_none(self, tmp_path: Path) -> None:
        """Config missing mcpServers returns None."""
        bad = tmp_path / "bad.json"
        bad.write_text('{"other": true}')
        result = load_mcp_config_lenient(bad)
        assert result is None

    def test_valid_file_returns_config(self, tmp_path: Path) -> None:
        """Valid config file returns parsed dict."""
        good = tmp_path / "good.json"
        good.write_text(
            json.dumps({"mcpServers": {"fs": {"command": "npx", "args": []}}})
        )
        result = load_mcp_config_lenient(good)
        assert result is not None
        assert "fs" in result["mcpServers"]


class TestResolveAndLoadMcpTools:
    """Test the unified resolve_and_load_mcp_tools entry point."""

    async def test_no_mcp_returns_empty(self) -> None:
        """no_mcp=True returns empty tuple immediately."""
        tools, manager, infos = await resolve_and_load_mcp_tools(no_mcp=True)
        assert tools == []
        assert manager is None
        assert infos == []

    @patch("deepagents_cli.mcp_tools._load_tools_from_config")
    @patch("deepagents_cli.mcp_tools.discover_mcp_configs")
    async def test_explicit_path_merges_with_discovery(
        self,
        mock_discover: MagicMock,
        mock_load: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """Explicit path is merged on top of auto-discovered configs."""
        # Auto-discovered config
        discovered = tmp_path / "discovered.json"
        discovered.write_text(
            json.dumps({"mcpServers": {"fs": {"command": "npx", "args": []}}})
        )
        mock_discover.return_value = [discovered]

        # Explicit config
        explicit = tmp_path / "explicit.json"
        explicit.write_text(
            json.dumps({"mcpServers": {"search": {"command": "brave", "args": []}}})
        )
        mock_load.return_value = ([], MCPSessionManager(), [])

        await resolve_and_load_mcp_tools(
            explicit_config_path=str(explicit), trust_project_mcp=True
        )

        mock_discover.assert_called_once()
        mock_load.assert_awaited_once()
        merged = mock_load.call_args.args[0]
        assert "fs" in merged["mcpServers"]
        assert "search" in merged["mcpServers"]

    @patch("deepagents_cli.mcp_tools._load_tools_from_config")
    @patch("deepagents_cli.mcp_tools.discover_mcp_configs")
    async def test_auto_discovery_merges_and_loads(
        self, mock_discover: MagicMock, mock_load: AsyncMock, tmp_path: Path
    ) -> None:
        """Auto-discovery finds configs, merges, and loads tools."""
        # Write two config files
        c1 = tmp_path / "user.json"
        c1.write_text(
            json.dumps({"mcpServers": {"fs": {"command": "npx", "args": []}}})
        )
        c2 = tmp_path / "project.json"
        c2.write_text(
            json.dumps({"mcpServers": {"search": {"command": "brave", "args": []}}})
        )
        mock_discover.return_value = [c1, c2]
        mock_load.return_value = ([], MCPSessionManager(), [])

        await resolve_and_load_mcp_tools(trust_project_mcp=True)

        mock_load.assert_awaited_once()
        merged = mock_load.call_args.args[0]
        assert "fs" in merged["mcpServers"]
        assert "search" in merged["mcpServers"]

    @patch("deepagents_cli.mcp_tools.discover_mcp_configs")
    async def test_auto_discovery_no_configs_returns_empty(
        self, mock_discover: MagicMock
    ) -> None:
        """No discovered configs returns empty tuple."""
        mock_discover.return_value = []
        tools, manager, infos = await resolve_and_load_mcp_tools()
        assert tools == []
        assert manager is None
        assert infos == []

    async def test_explicit_path_missing_raises(self, tmp_path: Path) -> None:
        """FileNotFoundError propagates for missing explicit config."""
        with pytest.raises(FileNotFoundError):
            await resolve_and_load_mcp_tools(
                explicit_config_path=str(tmp_path / "nope.json")
            )

    async def test_explicit_path_invalid_json_raises(self, tmp_path: Path) -> None:
        """JSONDecodeError propagates for invalid explicit config."""
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")
        with pytest.raises(json.JSONDecodeError):
            await resolve_and_load_mcp_tools(explicit_config_path=str(bad))

    @patch("deepagents_cli.mcp_tools.discover_mcp_configs")
    async def test_no_mcp_skips_discovery(self, mock_discover: MagicMock) -> None:
        """no_mcp=True should not call discover_mcp_configs."""
        await resolve_and_load_mcp_tools(no_mcp=True)
        mock_discover.assert_not_called()

    @patch("deepagents_cli.mcp_tools._load_tools_from_config")
    @patch("deepagents_cli.mcp_trust.is_project_mcp_trusted")
    async def test_project_context_drives_trust_root(
        self,
        mock_is_trusted: MagicMock,
        mock_load: AsyncMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Trust lookup should use explicit project context, not process cwd."""
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: home)

        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / ".git").mkdir()
        user_cwd = project_root / "src"
        user_cwd.mkdir()
        project_cfg = project_root / ".mcp.json"
        project_cfg.write_text(
            json.dumps({"mcpServers": {"local": {"command": "npx", "args": []}}})
        )

        other_cwd = tmp_path / "elsewhere"
        other_cwd.mkdir()
        monkeypatch.chdir(other_cwd)

        mock_is_trusted.return_value = True
        mock_load.return_value = ([], MCPSessionManager(), [])

        project_context = ProjectContext.from_user_cwd(user_cwd)
        await resolve_and_load_mcp_tools(project_context=project_context)

        mock_is_trusted.assert_called_once()
        assert mock_is_trusted.call_args.args[0] == str(project_root.resolve())
        mock_load.assert_awaited_once()

    @patch("deepagents_cli.mcp_tools._load_tools_from_config")
    async def test_project_context_normalizes_relative_explicit_path(
        self,
        mock_load: AsyncMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Explicit config paths should resolve relative to project context cwd."""
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: home)

        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / ".git").mkdir()
        user_cwd = project_root / "src"
        user_cwd.mkdir()
        explicit = user_cwd / "configs" / "mcp.json"
        explicit.parent.mkdir(parents=True)
        explicit.write_text(
            json.dumps({"mcpServers": {"fs": {"command": "npx", "args": []}}})
        )

        mock_load.return_value = ([], MCPSessionManager(), [])

        project_context = ProjectContext.from_user_cwd(user_cwd)
        await resolve_and_load_mcp_tools(
            explicit_config_path="configs/mcp.json",
            trust_project_mcp=True,
            project_context=project_context,
        )

        merged = mock_load.call_args.args[0]
        assert "fs" in merged["mcpServers"]

    @patch("deepagents_cli.mcp_tools._load_tools_from_config")
    @patch("deepagents_cli.mcp_tools.discover_mcp_configs")
    async def test_trust_false_filters_project_stdio(
        self,
        mock_discover: MagicMock,
        mock_load: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """trust_project_mcp=False filters out project-level stdio servers."""
        project_cfg = tmp_path / ".mcp.json"
        project_cfg.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "pwn": {"command": "bash", "args": ["-c", "evil"]},
                        "remote": {"type": "sse", "url": "http://ok"},
                    }
                }
            )
        )
        mock_discover.return_value = [project_cfg]
        mock_load.return_value = ([], MCPSessionManager(), [])

        await resolve_and_load_mcp_tools(trust_project_mcp=False)

        mock_load.assert_awaited_once()
        merged = mock_load.call_args.args[0]
        assert "pwn" not in merged["mcpServers"]
        assert "remote" in merged["mcpServers"]

    @patch("deepagents_cli.mcp_tools._load_tools_from_config")
    @patch("deepagents_cli.mcp_tools.discover_mcp_configs")
    async def test_trust_true_allows_project_stdio(
        self,
        mock_discover: MagicMock,
        mock_load: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """trust_project_mcp=True allows project-level stdio servers."""
        project_cfg = tmp_path / ".mcp.json"
        project_cfg.write_text(
            json.dumps({"mcpServers": {"local": {"command": "npx", "args": []}}})
        )
        mock_discover.return_value = [project_cfg]
        mock_load.return_value = ([], MCPSessionManager(), [])

        await resolve_and_load_mcp_tools(trust_project_mcp=True)

        mock_load.assert_awaited_once()
        merged = mock_load.call_args.args[0]
        assert "local" in merged["mcpServers"]


class TestClassifyDiscoveredConfigs:
    """Tests for classify_discovered_configs."""

    def test_user_config_classified(self) -> None:
        """Paths under ~/.deepagents/ are classified as user."""
        user_path = Path.home() / ".deepagents" / ".mcp.json"
        user, project = classify_discovered_configs([user_path])
        assert user == [user_path]
        assert project == []

    def test_project_config_classified(self, tmp_path: Path) -> None:
        """Paths outside ~/.deepagents/ are classified as project."""
        project_path = tmp_path / ".mcp.json"
        project_path.touch()
        user, project = classify_discovered_configs([project_path])
        assert user == []
        assert project == [project_path]

    def test_mixed_classification(self, tmp_path: Path) -> None:
        """Mixed paths are split correctly."""
        user_path = Path.home() / ".deepagents" / ".mcp.json"
        project_path = tmp_path / ".mcp.json"
        project_path.touch()
        user, project = classify_discovered_configs([user_path, project_path])
        assert user == [user_path]
        assert project == [project_path]


class TestExtractStdioServerCommands:
    """Tests for extract_stdio_server_commands."""

    def test_extracts_stdio(self) -> None:
        """Extracts name, command, args from stdio servers."""
        config = {
            "mcpServers": {
                "fs": {"command": "npx", "args": ["-y", "fs-server"]},
            }
        }
        result = extract_stdio_server_commands(config)
        assert result == [("fs", "npx", ["-y", "fs-server"])]

    def test_skips_remote(self) -> None:
        """SSE/HTTP servers are not extracted."""
        config = {
            "mcpServers": {
                "remote": {"type": "sse", "url": "http://example.com"},
            }
        }
        assert extract_stdio_server_commands(config) == []

    def test_mixed(self) -> None:
        """Only stdio servers are returned from mixed configs."""
        config = {
            "mcpServers": {
                "local": {"command": "bash", "args": []},
                "remote": {"type": "http", "url": "http://x"},
            }
        }
        result = extract_stdio_server_commands(config)
        assert len(result) == 1
        assert result[0][0] == "local"

    def test_empty_servers(self) -> None:
        """Empty mcpServers returns empty list."""
        assert extract_stdio_server_commands({"mcpServers": {}}) == []

    def test_no_servers_key(self) -> None:
        """Missing mcpServers returns empty list."""
        assert extract_stdio_server_commands({}) == []


class TestFilterProjectStdioServers:
    """Tests for _filter_project_stdio_servers."""

    def test_removes_stdio_keeps_remote(self) -> None:
        """Stdio servers are removed, remote servers are kept."""
        config = {
            "mcpServers": {
                "local": {"command": "bash", "args": []},
                "remote": {"type": "sse", "url": "http://x"},
            }
        }
        result = _filter_project_stdio_servers(config)
        assert "local" not in result["mcpServers"]
        assert "remote" in result["mcpServers"]

    def test_all_stdio_returns_empty(self) -> None:
        """Config with only stdio servers returns empty mcpServers."""
        config = {"mcpServers": {"a": {"command": "x", "args": []}}}
        result = _filter_project_stdio_servers(config)
        assert result["mcpServers"] == {}


class TestCheckStdioServer:
    """Tests for _check_stdio_server pre-flight validation."""

    def test_command_not_found(self) -> None:
        """Raises RuntimeError with server name and command when missing."""
        with (
            patch("deepagents_cli.mcp_tools.shutil.which", return_value=None),
            pytest.raises(
                RuntimeError,
                match="MCP server 'test-server': command 'nonexistent' not found",
            ),
        ):
            _check_stdio_server("test-server", {"command": "nonexistent"})

    def test_command_exists(self) -> None:
        """No error when command exists on PATH."""
        with patch(
            "deepagents_cli.mcp_tools.shutil.which", return_value="/usr/bin/npx"
        ):
            _check_stdio_server("test-server", {"command": "npx"})

    def test_missing_command_key(self) -> None:
        """Raises RuntimeError when config lacks `command` key."""
        with pytest.raises(RuntimeError, match="missing 'command' in config"):
            _check_stdio_server("test-server", {})


class TestCheckRemoteServer:
    """Tests for _check_remote_server pre-flight validation."""

    async def test_unreachable(self) -> None:
        """Raises RuntimeError when URL is unreachable."""
        import httpx

        mock_client = AsyncMock()
        mock_client.head.side_effect = httpx.TransportError("connection refused")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            pytest.raises(RuntimeError, match="unreachable"),
        ):
            await _check_remote_server("test-server", {"url": "http://localhost:9999"})

    async def test_reachable(self) -> None:
        """No error when URL responds."""
        mock_client = AsyncMock()
        mock_client.head = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            await _check_remote_server("test-server", {"url": "http://localhost:8080"})

        mock_client.head.assert_awaited_once_with("http://localhost:8080", timeout=2)

    async def test_unreachable_os_error(self) -> None:
        """Raises RuntimeError on OS-level socket errors."""
        mock_client = AsyncMock()
        mock_client.head.side_effect = OSError("Network is unreachable")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            pytest.raises(RuntimeError, match="unreachable"),
        ):
            await _check_remote_server(
                "test-server", {"url": "http://10.255.255.1:9999"}
            )

    async def test_invalid_url(self) -> None:
        """Raises RuntimeError on malformed URLs."""
        import httpx

        mock_client = AsyncMock()
        mock_client.head.side_effect = httpx.InvalidURL("Invalid URL")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            pytest.raises(RuntimeError, match="unreachable"),
        ):
            await _check_remote_server("test-server", {"url": "http://"})

    async def test_missing_url_key(self) -> None:
        """Raises RuntimeError when config lacks `url` key."""
        with pytest.raises(RuntimeError, match="missing 'url' in config"):
            await _check_remote_server("test-server", {})

    async def test_timeout(self) -> None:
        """Raises RuntimeError on connection timeout."""
        import httpx

        mock_client = AsyncMock()
        mock_client.head.side_effect = httpx.TimeoutException("timed out")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            pytest.raises(RuntimeError, match="unreachable"),
        ):
            await _check_remote_server("test-server", {"url": "http://slow:8080"})


class TestHealthCheckIntegration:
    """Tests for health check integration in _load_tools_from_config."""

    async def test_stdio_health_check_failure_skips_session(
        self,
        write_config: Callable[..., str],
    ) -> None:
        """Health check failure raises before any session is opened."""
        path = write_config(
            {"mcpServers": {"fs": {"command": "missing-cmd", "args": []}}}
        )
        fake_session = AsyncMock()

        @asynccontextmanager
        async def _fake(connection, *, mcp_callbacks=None):  # noqa: ANN001, ARG001
            yield fake_session

        with (
            patch("deepagents_cli.mcp_tools.shutil.which", return_value=None),
            patch(
                "langchain_mcp_adapters.sessions.create_session", _fake
            ) as patched,
            pytest.raises(RuntimeError, match="Pre-flight health check"),
        ):
            await get_mcp_tools(path)

        # create_session is the ContextDecorator itself; just assert no call
        # reached `initialize` / `list_tools`.
        fake_session.initialize.assert_not_called()
        fake_session.list_tools.assert_not_called()
        # patched is the raw asynccontextmanager function, not a Mock; the
        # underlying mock assertions above already cover the invariant.
        del patched

    async def test_remote_health_check_failure_skips_session(
        self,
        write_config: Callable[..., str],
    ) -> None:
        """Remote health check failure prevents session creation."""
        import httpx

        path = write_config(
            {"mcpServers": {"api": {"type": "sse", "url": "http://down:9999"}}}
        )
        fake_session = AsyncMock()

        @asynccontextmanager
        async def _fake(connection, *, mcp_callbacks=None):  # noqa: ANN001, ARG001
            yield fake_session

        mock_http = AsyncMock()
        mock_http.head.side_effect = httpx.TransportError("refused")
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("httpx.AsyncClient", return_value=mock_http),
            patch("langchain_mcp_adapters.sessions.create_session", _fake),
            pytest.raises(RuntimeError, match="Pre-flight health check"),
        ):
            await get_mcp_tools(path)

        fake_session.initialize.assert_not_called()

    async def test_multi_server_collects_all_failures(
        self,
        write_config: Callable[..., str],
    ) -> None:
        """All server failures are reported in a single error."""
        path = write_config(
            {
                "mcpServers": {
                    "a": {"command": "missing-a", "args": []},
                    "b": {"command": "missing-b", "args": []},
                }
            }
        )
        fake_session = AsyncMock()

        @asynccontextmanager
        async def _fake(connection, *, mcp_callbacks=None):  # noqa: ANN001, ARG001
            yield fake_session

        with (
            patch("deepagents_cli.mcp_tools.shutil.which", return_value=None),
            patch("langchain_mcp_adapters.sessions.create_session", _fake),
            pytest.raises(RuntimeError) as exc_info,
        ):
            await get_mcp_tools(path)

        error_msg = str(exc_info.value)
        assert "missing-a" in error_msg
        assert "missing-b" in error_msg
        fake_session.initialize.assert_not_called()

    async def test_mixed_stdio_and_remote_checks(
        self,
        write_config: Callable[..., str],
    ) -> None:
        """Both stdio and remote health checks run for mixed configs."""
        import httpx

        path = write_config(
            {
                "mcpServers": {
                    "local": {"command": "missing-cmd", "args": []},
                    "remote": {"type": "sse", "url": "http://down:9999"},
                }
            }
        )

        mock_http = AsyncMock()
        mock_http.head.side_effect = httpx.TransportError("refused")
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("deepagents_cli.mcp_tools.shutil.which", return_value=None),
            patch("httpx.AsyncClient", return_value=mock_http),
            pytest.raises(RuntimeError) as exc_info,
        ):
            await get_mcp_tools(path)

        error_msg = str(exc_info.value)
        assert "missing-cmd" in error_msg
        assert "down:9999" in error_msg


class TestToolOrdering:
    """Tools returned by get_mcp_tools are sorted deterministically."""

    @pytest.fixture(autouse=True)
    def _bypass_health_checks(self) -> Generator[None]:
        """Bypass pre-flight health checks for all tests in this class."""
        with (
            patch("deepagents_cli.mcp_tools._check_stdio_server"),
            patch(
                "deepagents_cli.mcp_tools._check_remote_server",
                new_callable=AsyncMock,
            ),
        ):
            yield

    async def test_tools_sorted_alphabetically_by_name(
        self,
        write_config: Callable[..., str],
        fake_create_session: tuple[AsyncMock, list],
    ) -> None:
        """Tools from a single server are sorted by prefixed name."""
        path = write_config(
            {"mcpServers": {"srv": {"command": "node", "args": ["server.js"]}}}
        )
        mock_session, _ = fake_create_session
        mock_session.list_tools = AsyncMock(
            return_value=_make_tool_page(
                [
                    _make_mcp_tool("zeta"),
                    _make_mcp_tool("alpha"),
                    _make_mcp_tool("mu"),
                ]
            )
        )

        tools, _, _ = await get_mcp_tools(path)

        assert [t.name for t in tools] == ["srv_alpha", "srv_mu", "srv_zeta"]

    async def test_tools_sorted_across_multiple_servers(
        self,
        write_config: Callable[..., str],
    ) -> None:
        """Tools from multiple servers are interleaved alphabetically."""
        path = write_config(
            {
                "mcpServers": {
                    "beta": {"command": "node", "args": ["b.js"]},
                    "alpha": {"command": "node", "args": ["a.js"]},
                }
            }
        )

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        # Order matters: servers are iterated in config insertion order.
        mock_session.list_tools = AsyncMock(
            side_effect=[
                _make_tool_page([_make_mcp_tool("write")]),
                _make_tool_page([_make_mcp_tool("read")]),
            ]
        )

        @asynccontextmanager
        async def _fake(connection, *, mcp_callbacks=None):  # noqa: ANN001, ARG001
            yield mock_session

        with patch("langchain_mcp_adapters.sessions.create_session", _fake):
            tools, _, _ = await get_mcp_tools(path)

        assert [t.name for t in tools] == ["alpha_read", "beta_write"]


class TestLoadToolsFromConfigHeaders:
    async def test_headers_are_resolved_before_connection(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Env-var references in headers are resolved before being passed to
        `create_session` as part of the Connection config."""
        from deepagents_cli.mcp_tools import _load_tools_from_config

        monkeypatch.setenv("DA_TOKEN", "tok-123")

        recorded: list = []
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=_make_tool_page([]))

        @asynccontextmanager
        async def _fake(connection, *, mcp_callbacks=None):  # noqa: ANN001, ARG001
            recorded.append(connection)
            yield mock_session

        with (
            patch("langchain_mcp_adapters.sessions.create_session", _fake),
            patch(
                "deepagents_cli.mcp_tools._check_remote_server",
                new=AsyncMock(return_value=None),
            ),
        ):
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


class TestLoadToolsFromConfigOAuth:
    async def test_missing_tokens_raise_login_instruction(
        self,
        fake_home: Path,
    ) -> None:
        from deepagents_cli.mcp_tools import _load_tools_from_config

        config = {
            "mcpServers": {
                "notion": {
                    "transport": "http",
                    "url": "https://mcp.notion.com/mcp",
                    "auth": "oauth",
                }
            }
        }
        with patch(
            "deepagents_cli.mcp_tools._check_remote_server",
            new=AsyncMock(return_value=None),
        ):
            with pytest.raises(RuntimeError, match="deepagents mcp login notion"):
                await _load_tools_from_config(config)

    async def test_existing_tokens_attach_oauth_provider(
        self,
        fake_home: Path,
    ) -> None:
        from mcp.client.auth import OAuthClientProvider
        from mcp.shared.auth import OAuthToken

        from deepagents_cli.mcp_auth import FileTokenStorage
        from deepagents_cli.mcp_tools import _load_tools_from_config

        storage = FileTokenStorage("notion")
        await storage.set_tokens(OAuthToken(access_token="at", token_type="Bearer"))

        recorded: list = []
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=_make_tool_page([]))

        @asynccontextmanager
        async def _fake(connection, *, mcp_callbacks=None):  # noqa: ANN001, ARG001
            recorded.append(connection)
            yield mock_session

        with (
            patch("langchain_mcp_adapters.sessions.create_session", _fake),
            patch(
                "deepagents_cli.mcp_tools._check_remote_server",
                new=AsyncMock(return_value=None),
            ),
        ):
            config = {
                "mcpServers": {
                    "notion": {
                        "transport": "http",
                        "url": "https://mcp.notion.com/mcp",
                        "auth": "oauth",
                    }
                }
            }
            await _load_tools_from_config(config)

        assert isinstance(recorded[0].get("auth"), OAuthClientProvider)


class TestCachedSessionProxy:
    """Proxy tools route through MCPSessionManager with retry semantics."""

    @pytest.fixture(autouse=True)
    def _bypass_health_checks(self) -> Generator[None]:
        """Bypass pre-flight health checks for all tests in this class."""
        with (
            patch("deepagents_cli.mcp_tools._check_stdio_server"),
            patch(
                "deepagents_cli.mcp_tools._check_remote_server",
                new_callable=AsyncMock,
            ),
        ):
            yield

    @staticmethod
    def _fake_call_tool_result(text: str = "ok") -> MagicMock:
        """Build a mock `CallToolResult` compatible with `_convert_call_tool_result`."""
        from mcp.types import CallToolResult, TextContent

        return CallToolResult(content=[TextContent(type="text", text=text)])

    async def test_first_call_opens_cached_session(
        self,
        write_config: Callable[..., str],
    ) -> None:
        """First proxy-tool call opens a session in the manager and caches it."""
        path = write_config(
            {"mcpServers": {"srv": {"command": "node", "args": ["s.js"]}}}
        )

        sessions: list[AsyncMock] = []
        session_ctx_entries = 0

        def _new_session() -> AsyncMock:
            s = AsyncMock()
            s.initialize = AsyncMock()
            s.list_tools = AsyncMock(
                return_value=_make_tool_page([_make_mcp_tool("echo")])
            )
            s.call_tool = AsyncMock(return_value=self._fake_call_tool_result("hello"))
            sessions.append(s)
            return s

        @asynccontextmanager
        async def _fake(connection, *, mcp_callbacks=None):  # noqa: ANN001, ARG001
            nonlocal session_ctx_entries
            session_ctx_entries += 1
            yield _new_session()

        with patch("langchain_mcp_adapters.sessions.create_session", _fake):
            tools, _, _ = await get_mcp_tools(path)
            assert [t.name for t in tools] == ["srv_echo"]

            # Invoke the proxy tool — should open a second session (first was
            # the throwaway discovery session).
            content = await tools[0].ainvoke({})

        assert session_ctx_entries == 2, (
            "expected one discovery session + one cached runtime session"
        )
        # The runtime session had call_tool invoked with the right tool name.
        runtime_session = sessions[1]
        runtime_session.call_tool.assert_awaited_once_with("echo", {})
        assert "hello" in str(content)

    async def test_second_call_reuses_cached_session(
        self,
        write_config: Callable[..., str],
    ) -> None:
        """Back-to-back proxy-tool calls share one runtime session."""
        path = write_config(
            {"mcpServers": {"srv": {"command": "node", "args": ["s.js"]}}}
        )

        sessions: list[AsyncMock] = []

        def _new_session() -> AsyncMock:
            s = AsyncMock()
            s.initialize = AsyncMock()
            s.list_tools = AsyncMock(
                return_value=_make_tool_page([_make_mcp_tool("echo")])
            )
            s.call_tool = AsyncMock(return_value=self._fake_call_tool_result())
            sessions.append(s)
            return s

        @asynccontextmanager
        async def _fake(connection, *, mcp_callbacks=None):  # noqa: ANN001, ARG001
            yield _new_session()

        with patch("langchain_mcp_adapters.sessions.create_session", _fake):
            tools, _, _ = await get_mcp_tools(path)
            await tools[0].ainvoke({})
            await tools[0].ainvoke({})

        # discovery + one runtime session — NOT two runtime sessions.
        assert len(sessions) == 2
        runtime_session = sessions[1]
        assert runtime_session.call_tool.await_count == 2

    async def test_transient_error_invalidates_and_retries(
        self,
        write_config: Callable[..., str],
    ) -> None:
        """A transient transport error triggers invalidate + retry-once."""
        from anyio import ClosedResourceError

        path = write_config(
            {"mcpServers": {"srv": {"command": "node", "args": ["s.js"]}}}
        )

        all_sessions: list[AsyncMock] = []

        def _new_session(*, dead: bool = False) -> AsyncMock:
            s = AsyncMock()
            s.initialize = AsyncMock()
            s.list_tools = AsyncMock(
                return_value=_make_tool_page([_make_mcp_tool("echo")])
            )
            if dead:
                s.call_tool = AsyncMock(side_effect=ClosedResourceError())
            else:
                s.call_tool = AsyncMock(return_value=self._fake_call_tool_result())
            all_sessions.append(s)
            return s

        call_counter = {"n": 0}

        @asynccontextmanager
        async def _fake(connection, *, mcp_callbacks=None):  # noqa: ANN001, ARG001
            call_counter["n"] += 1
            # discovery (#1), first runtime session is dead (#2), retry opens
            # a fresh session (#3)
            yield _new_session(dead=(call_counter["n"] == 2))

        with patch("langchain_mcp_adapters.sessions.create_session", _fake):
            tools, _, _ = await get_mcp_tools(path)
            await tools[0].ainvoke({})

        assert call_counter["n"] == 3, (
            "discovery + dead runtime + retry = 3 sessions opened"
        )
        # The retry used a live session that succeeded.
        assert all_sessions[2].call_tool.await_count == 1

    async def test_repeated_transient_error_surfaces_tool_exception(
        self,
        write_config: Callable[..., str],
    ) -> None:
        """Second failure after invalidate converts to a `ToolException`."""
        from anyio import ClosedResourceError
        from langchain_core.tools import ToolException

        path = write_config(
            {"mcpServers": {"srv": {"command": "node", "args": ["s.js"]}}}
        )

        def _new_session(*, dead: bool) -> AsyncMock:
            s = AsyncMock()
            s.initialize = AsyncMock()
            s.list_tools = AsyncMock(
                return_value=_make_tool_page([_make_mcp_tool("echo")])
            )
            s.call_tool = AsyncMock(
                side_effect=ClosedResourceError() if dead else None
            )
            return s

        call_counter = {"n": 0}

        @asynccontextmanager
        async def _fake(connection, *, mcp_callbacks=None):  # noqa: ANN001, ARG001
            call_counter["n"] += 1
            # discovery session is "alive" so list_tools succeeds; both runtime
            # sessions (post-discovery) fail with ClosedResourceError.
            yield _new_session(dead=(call_counter["n"] >= 2))

        with patch("langchain_mcp_adapters.sessions.create_session", _fake):
            tools, _, _ = await get_mcp_tools(path)
            with pytest.raises(ToolException, match="failed after one retry"):
                await tools[0].ainvoke({})

    async def test_logical_tool_exception_is_not_retried(
        self,
        write_config: Callable[..., str],
    ) -> None:
        """A `ToolException` from the MCP server propagates without retry."""
        from langchain_core.tools import ToolException
        from mcp.types import CallToolResult, TextContent

        path = write_config(
            {"mcpServers": {"srv": {"command": "node", "args": ["s.js"]}}}
        )

        call_counter = {"n": 0}
        runtime_session: AsyncMock | None = None

        def _new_session() -> AsyncMock:
            nonlocal runtime_session
            s = AsyncMock()
            s.initialize = AsyncMock()
            s.list_tools = AsyncMock(
                return_value=_make_tool_page([_make_mcp_tool("echo")])
            )
            # Return isError=True; _convert_call_tool_result turns that into
            # a ToolException.
            s.call_tool = AsyncMock(
                return_value=CallToolResult(
                    content=[TextContent(type="text", text="boom")], isError=True
                )
            )
            if call_counter["n"] >= 1:
                runtime_session = s
            return s

        @asynccontextmanager
        async def _fake(connection, *, mcp_callbacks=None):  # noqa: ANN001, ARG001
            call_counter["n"] += 1
            yield _new_session()

        with patch("langchain_mcp_adapters.sessions.create_session", _fake):
            tools, _, _ = await get_mcp_tools(path)
            with pytest.raises(ToolException, match="boom"):
                await tools[0].ainvoke({})

        # Only discovery + one runtime session — NO retry.
        assert call_counter["n"] == 2
        assert runtime_session is not None
        assert runtime_session.call_tool.await_count == 1
