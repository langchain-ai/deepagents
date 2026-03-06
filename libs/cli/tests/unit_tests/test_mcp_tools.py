"""Tests for MCP tools configuration loading and validation."""

import json
from collections.abc import Callable
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepagents_cli.mcp_tools import MCPSessionManager, get_mcp_tools, load_mcp_config

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
def mock_mcp_session():
    """Fixture for creating a mock MCP session context manager."""
    mock_session = AsyncMock()
    mock_session_cm = MagicMock()
    mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_cm.__aexit__ = AsyncMock(return_value=None)
    return mock_session, mock_session_cm


@pytest.fixture
def mock_mcp_client(
    mock_mcp_session: tuple[AsyncMock, MagicMock],
) -> tuple[MagicMock, AsyncMock]:
    """Fixture for creating a mock MultiServerMCPClient."""
    mock_session, mock_session_cm = mock_mcp_session
    mock_client = MagicMock()
    mock_client.session = MagicMock(return_value=mock_session_cm)
    return mock_client, mock_session


@pytest.fixture
def mock_tools():
    """Fixture providing mock tool objects."""
    mock_tool1 = MagicMock()
    mock_tool1.name = "read_file"
    mock_tool1.description = "Read a file"

    mock_tool2 = MagicMock()
    mock_tool2.name = "write_file"
    mock_tool2.description = "Write a file"

    return [mock_tool1, mock_tool2]


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


class TestGetMCPTools:
    """Test MCP tools loading from configuration."""

    @patch("langchain_mcp_adapters.tools.load_mcp_tools")
    @patch("langchain_mcp_adapters.client.MultiServerMCPClient")
    async def test_get_mcp_tools_success(
        self,
        mock_client_class: MagicMock,
        mock_load_tools: AsyncMock,
        write_config: Callable[..., str],
        valid_config_data: dict,
        mock_mcp_client: tuple,
        mock_tools: list,
    ) -> None:
        """Test successful loading of MCP tools."""
        path = write_config(valid_config_data)

        # Setup mocks
        mock_client, mock_session = mock_mcp_client
        mock_client_class.return_value = mock_client
        mock_load_tools.return_value = mock_tools

        tools, manager = await get_mcp_tools(path)

        # Verify client was initialized with correct connection config
        mock_client_class.assert_called_once()
        connections = mock_client_class.call_args.kwargs["connections"]
        assert connections["filesystem"]["command"] == "npx"
        assert connections["filesystem"]["transport"] == "stdio"
        assert connections["filesystem"]["args"] == [
            "-y",
            "@modelcontextprotocol/server-filesystem",
            "/tmp",
        ]

        # Verify session was created and tools were loaded
        mock_client.session.assert_called_once_with("filesystem")
        mock_load_tools.assert_called_once_with(
            mock_session, server_name="filesystem", tool_name_prefix=True
        )
        assert len(tools) == 2
        assert tools[0].name == "read_file"
        assert tools[1].name == "write_file"
        assert isinstance(manager, MCPSessionManager)

        # Clean up
        await manager.cleanup()

    @patch("langchain_mcp_adapters.client.MultiServerMCPClient")
    async def test_get_mcp_tools_server_spawn_failure(
        self, mock_client_class: MagicMock, write_config: Callable[..., str]
    ) -> None:
        """Test handling of MCP server spawn failure."""
        path = write_config(
            {
                "mcpServers": {
                    "filesystem": {
                        "command": "nonexistent-command",
                        "args": [],
                        "env": {},
                    }
                }
            }
        )

        # Setup mock client to raise an exception
        mock_client_class.side_effect = Exception("Command not found")

        with pytest.raises(
            RuntimeError, match=r"Failed to connect to MCP servers.*Command not found"
        ):
            await get_mcp_tools(path)

    @patch("langchain_mcp_adapters.tools.load_mcp_tools")
    @patch("langchain_mcp_adapters.client.MultiServerMCPClient")
    async def test_get_mcp_tools_get_tools_failure(
        self,
        mock_client_class: MagicMock,
        mock_load_tools: AsyncMock,
        write_config: Callable[..., str],
        valid_config_data: dict,
        mock_mcp_client: tuple,
    ) -> None:
        """Test handling of failure during load_mcp_tools call."""
        path = write_config(valid_config_data)

        # Setup mocks
        mock_client, _ = mock_mcp_client
        mock_client_class.return_value = mock_client
        mock_load_tools.side_effect = Exception("Server protocol error")

        with pytest.raises(
            RuntimeError,
            match=r"Failed to connect to MCP servers.*Server protocol error",
        ):
            await get_mcp_tools(path)

    @patch("langchain_mcp_adapters.tools.load_mcp_tools")
    @patch("langchain_mcp_adapters.client.MultiServerMCPClient")
    async def test_get_mcp_tools_multiple_servers(
        self,
        mock_client_class: MagicMock,
        mock_load_tools: AsyncMock,
        write_config: Callable[..., str],
    ) -> None:
        """Test loading tools from multiple MCP servers."""
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

        # Create mock tools from different servers
        mock_tools_fs = [MagicMock(name="read_file"), MagicMock(name="write_file")]
        mock_tools_search = [MagicMock(name="web_search")]

        # Setup mock client with session support
        mock_session_fs = AsyncMock()
        mock_session_search = AsyncMock()

        # Mock session context managers for both servers
        def mock_session_cm(server_name: str) -> MagicMock:
            session = (
                mock_session_fs if server_name == "filesystem" else mock_session_search
            )
            cm = MagicMock()
            cm.__aenter__ = AsyncMock(return_value=session)
            cm.__aexit__ = AsyncMock(return_value=None)
            return cm

        mock_client = MagicMock()
        mock_client.session.side_effect = mock_session_cm
        mock_client_class.return_value = mock_client

        # Mock load_mcp_tools to return different tools for each session
        def mock_load_side_effect(
            session: AsyncMock, **_kwargs: object
        ) -> list[MagicMock]:
            if session == mock_session_fs:
                return mock_tools_fs
            return mock_tools_search

        mock_load_tools.side_effect = mock_load_side_effect

        tools, manager = await get_mcp_tools(path)

        # Verify both servers were registered
        call_kwargs = mock_client_class.call_args.kwargs
        connections = call_kwargs["connections"]
        assert len(connections) == 2
        assert "filesystem" in connections
        assert "brave-search" in connections
        assert connections["brave-search"]["env"]["BRAVE_API_KEY"] == "test-key"

        # Verify sessions were created for both servers
        assert mock_client.session.call_count == 2

        # Verify tools from all servers were returned
        assert len(tools) == 3

        # Clean up
        await manager.cleanup()

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

    @patch("langchain_mcp_adapters.tools.load_mcp_tools")
    @patch("langchain_mcp_adapters.client.MultiServerMCPClient")
    async def test_get_mcp_tools_env_variables_passed(
        self,
        mock_client_class: MagicMock,
        mock_load_tools: AsyncMock,
        write_config: Callable[..., str],
        mock_mcp_client: tuple,
    ) -> None:
        """Test that environment variables are correctly passed to MCP client."""
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

        # Setup mocks
        mock_client, _ = mock_mcp_client
        mock_client_class.return_value = mock_client
        mock_load_tools.return_value = []

        _, manager = await get_mcp_tools(path)

        # Verify env variables were passed correctly
        connections = mock_client_class.call_args.kwargs["connections"]
        assert connections["github"]["env"]["GITHUB_TOKEN"] == "ghp_test123"
        assert (
            connections["github"]["env"]["GITHUB_API_URL"] == "https://api.github.com"
        )

        # Clean up
        await manager.cleanup()

    @patch("langchain_mcp_adapters.tools.load_mcp_tools")
    @patch("langchain_mcp_adapters.client.MultiServerMCPClient")
    async def test_get_mcp_tools_env_none_when_not_provided(
        self,
        mock_client_class: MagicMock,
        mock_load_tools: AsyncMock,
        write_config: Callable[..., str],
        mock_mcp_client: tuple,
    ) -> None:
        """Test that env is None (inherit parent env) when not provided in config."""
        path = write_config(
            {
                "mcpServers": {
                    "simple": {
                        "command": "simple-server",
                    }
                }
            }
        )

        # Setup mocks
        mock_client, _ = mock_mcp_client
        mock_client_class.return_value = mock_client
        mock_load_tools.return_value = []

        _, manager = await get_mcp_tools(path)

        connections = mock_client_class.call_args.kwargs["connections"]
        assert connections["simple"]["env"] is None

        await manager.cleanup()

    @patch("langchain_mcp_adapters.tools.load_mcp_tools")
    @patch("langchain_mcp_adapters.client.MultiServerMCPClient")
    async def test_get_mcp_tools_headers_passed_for_sse(
        self,
        mock_client_class: MagicMock,
        mock_load_tools: AsyncMock,
        write_config: Callable[..., str],
        mock_mcp_client: tuple,
    ) -> None:
        """Test that headers are correctly passed to SSE MCP client."""
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

        # Setup mocks
        mock_client, _ = mock_mcp_client
        mock_client_class.return_value = mock_client
        mock_load_tools.return_value = []

        _, manager = await get_mcp_tools(path)

        # Verify headers were passed correctly
        connections = mock_client_class.call_args.kwargs["connections"]
        assert connections["api"]["transport"] == "sse"
        assert connections["api"]["headers"]["Authorization"] == "Bearer token123"
        assert connections["api"]["headers"]["X-API-Key"] == "key456"

        # Clean up
        await manager.cleanup()

    @patch("langchain_mcp_adapters.tools.load_mcp_tools")
    @patch("langchain_mcp_adapters.client.MultiServerMCPClient")
    async def test_get_mcp_tools_headers_passed_for_http(
        self,
        mock_client_class: MagicMock,
        mock_load_tools: AsyncMock,
        write_config: Callable[..., str],
        mock_mcp_client: tuple,
    ) -> None:
        """Test that headers are correctly passed to HTTP MCP client."""
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

        # Setup mocks
        mock_client, _ = mock_mcp_client
        mock_client_class.return_value = mock_client
        mock_load_tools.return_value = []

        _, manager = await get_mcp_tools(path)

        # Verify headers were passed and transport is correct
        connections = mock_client_class.call_args.kwargs["connections"]
        assert connections["api"]["transport"] == "streamable_http"
        assert connections["api"]["headers"]["Authorization"] == "Bearer secret"

        # Clean up
        await manager.cleanup()

    @patch("langchain_mcp_adapters.tools.load_mcp_tools")
    @patch("langchain_mcp_adapters.client.MultiServerMCPClient")
    async def test_get_mcp_tools_no_headers_when_not_provided(
        self,
        mock_client_class: MagicMock,
        mock_load_tools: AsyncMock,
        write_config: Callable[..., str],
        mock_mcp_client: tuple,
    ) -> None:
        """Test that headers key is not added when not provided in config."""
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

        # Setup mocks
        mock_client, _ = mock_mcp_client
        mock_client_class.return_value = mock_client
        mock_load_tools.return_value = []

        _, manager = await get_mcp_tools(path)

        # Verify headers key is not present
        connections = mock_client_class.call_args.kwargs["connections"]
        assert "headers" not in connections["api"]

        # Clean up
        await manager.cleanup()
