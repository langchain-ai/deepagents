"""Tests for MCP tools configuration loading and validation."""

import json
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
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],  # noqa: S108
                "env": {},
            }
        }
    }


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

    @pytest.mark.asyncio
    async def test_load_valid_config(self, tmp_path: Path, valid_config_data: dict) -> None:
        """Test loading a valid MCP configuration file."""
        config_file = tmp_path / "mcp-config.json"
        config_file.write_text(json.dumps(valid_config_data))

        config = load_mcp_config(str(config_file))

        assert config == valid_config_data

    @pytest.mark.asyncio
    async def test_load_config_file_not_found(self, tmp_path: Path) -> None:
        """Test that FileNotFoundError is raised for missing config file."""
        nonexistent_file = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError, match="MCP config file not found"):
            load_mcp_config(str(nonexistent_file))

    @pytest.mark.asyncio
    async def test_load_config_invalid_json(self, tmp_path: Path) -> None:
        """Test that JSONDecodeError is raised for invalid JSON."""
        config_file = tmp_path / "invalid.json"
        config_file.write_text("{invalid json")

        with pytest.raises(json.JSONDecodeError, match="Invalid JSON in MCP config file"):
            load_mcp_config(str(config_file))

    @pytest.mark.asyncio
    async def test_load_config_missing_mcpservers_field(self, tmp_path: Path) -> None:
        """Test that ValueError is raised when mcpServers field is missing."""
        config_file = tmp_path / "missing-field.json"
        config_data = {"someOtherField": "value"}
        config_file.write_text(json.dumps(config_data))

        with pytest.raises(ValueError, match="must contain 'mcpServers' field"):
            load_mcp_config(str(config_file))

    @pytest.mark.asyncio
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
    async def test_load_config_invalid_mcpservers(
        self,
        tmp_path: Path,
        config_data: dict,
        expected_error: str,
        exception_type: type[Exception],
    ) -> None:
        """Test that appropriate exception is raised for invalid mcpServers field."""
        config_file = tmp_path / "invalid.json"
        config_file.write_text(json.dumps(config_data))

        with pytest.raises(exception_type, match=expected_error):
            load_mcp_config(str(config_file))

    @pytest.mark.asyncio
    async def test_load_config_server_missing_command(self, tmp_path: Path) -> None:
        """Test that ValueError is raised when server config is missing command."""
        config_file = tmp_path / "no-command.json"
        config_data = {
            "mcpServers": {
                "filesystem": {
                    "args": ["/tmp"],  # noqa: S108
                    # Missing "command" field
                }
            }
        }
        config_file.write_text(json.dumps(config_data))

        with pytest.raises(ValueError, match=r"filesystem.*missing required 'command' field"):
            load_mcp_config(str(config_file))

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("server_config", "expected_error"),
        [
            ("not a dict", "filesystem.*config must be a dictionary"),
            ({"command": "npx", "args": "not a list"}, "filesystem.*'args' must be a list"),
            (
                {"command": "npx", "args": ["/tmp"], "env": ["not", "a", "dict"]},  # noqa: S108
                "filesystem.*'env' must be a dictionary",
            ),
        ],
    )
    async def test_load_config_invalid_field_types(
        self, tmp_path: Path, server_config: dict | str, expected_error: str
    ) -> None:
        """Test that TypeError is raised for invalid server config field types."""
        config_file = tmp_path / "invalid-field.json"
        config_data = {"mcpServers": {"filesystem": server_config}}
        config_file.write_text(json.dumps(config_data))

        with pytest.raises(TypeError, match=expected_error):
            load_mcp_config(str(config_file))

    @pytest.mark.asyncio
    async def test_load_config_optional_fields(self, tmp_path: Path) -> None:
        """Test that args and env are optional fields."""
        config_file = tmp_path / "minimal.json"
        config_data = {
            "mcpServers": {
                "simple": {
                    "command": "simple-server",
                    # No args or env - should be valid
                }
            }
        }
        config_file.write_text(json.dumps(config_data))

        config = load_mcp_config(str(config_file))

        assert config == config_data
        assert "simple" in config["mcpServers"]

    @pytest.mark.asyncio
    async def test_load_config_multiple_servers(self, tmp_path: Path) -> None:
        """Test loading config with multiple MCP servers."""
        config_file = tmp_path / "multi-server.json"
        config_data = {
            "mcpServers": {
                "filesystem": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],  # noqa: S108
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
        config_file.write_text(json.dumps(config_data))

        config = load_mcp_config(str(config_file))

        assert len(config["mcpServers"]) == 3
        assert "filesystem" in config["mcpServers"]
        assert "brave-search" in config["mcpServers"]
        assert "github" in config["mcpServers"]

    @pytest.mark.asyncio
    async def test_load_config_sse_server(self, tmp_path: Path) -> None:
        """Test loading config with SSE server type."""
        config_file = tmp_path / "sse-server.json"
        config_data = {
            "mcpServers": {
                "remote-api": {
                    "type": "sse",
                    "url": "https://api.example.com/mcp",
                }
            }
        }
        config_file.write_text(json.dumps(config_data))

        config = load_mcp_config(str(config_file))

        assert config == config_data
        assert config["mcpServers"]["remote-api"]["type"] == "sse"
        assert config["mcpServers"]["remote-api"]["url"] == "https://api.example.com/mcp"

    @pytest.mark.asyncio
    async def test_load_config_http_server(self, tmp_path: Path) -> None:
        """Test loading config with HTTP server type."""
        config_file = tmp_path / "http-server.json"
        config_data = {
            "mcpServers": {
                "web-api": {
                    "type": "http",
                    "url": "https://api.example.com/mcp",
                }
            }
        }
        config_file.write_text(json.dumps(config_data))

        config = load_mcp_config(str(config_file))

        assert config == config_data
        assert config["mcpServers"]["web-api"]["type"] == "http"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("server_name", "server_type"),
        [
            ("remote-api", "sse"),
            ("web-api", "http"),
        ],
    )
    async def test_load_config_remote_server_missing_url(
        self, tmp_path: Path, server_name: str, server_type: str
    ) -> None:
        """Test that ValueError is raised when SSE/HTTP server is missing url field."""
        config_file = tmp_path / "remote-no-url.json"
        config_data = {"mcpServers": {server_name: {"type": server_type}}}
        config_file.write_text(json.dumps(config_data))

        with pytest.raises(ValueError, match=f"{server_name}.*missing required 'url' field"):
            load_mcp_config(str(config_file))

    @pytest.mark.asyncio
    async def test_load_config_mixed_server_types(self, tmp_path: Path) -> None:
        """Test loading config with mixed stdio, SSE, and HTTP servers."""
        config_file = tmp_path / "mixed-servers.json"
        config_data = {
            "mcpServers": {
                "filesystem": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],  # noqa: S108
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
        config_file.write_text(json.dumps(config_data))

        config = load_mcp_config(str(config_file))

        assert len(config["mcpServers"]) == 3
        assert "command" in config["mcpServers"]["filesystem"]
        assert config["mcpServers"]["remote-sse"]["type"] == "sse"
        assert config["mcpServers"]["remote-http"]["type"] == "http"

    @pytest.mark.asyncio
    async def test_load_config_sse_with_headers(self, tmp_path: Path) -> None:
        """Test loading SSE server config with custom headers."""
        config_file = tmp_path / "sse-headers.json"
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
        config_file.write_text(json.dumps(config_data))

        config = load_mcp_config(str(config_file))

        headers = config["mcpServers"]["authenticated-api"]["headers"]
        assert headers["Authorization"] == "Bearer token123"

    @pytest.mark.asyncio
    async def test_load_config_http_with_headers(self, tmp_path: Path) -> None:
        """Test loading HTTP server config with custom headers."""
        config_file = tmp_path / "http-headers.json"
        config_data = {
            "mcpServers": {
                "authenticated-api": {
                    "type": "http",
                    "url": "https://api.example.com/mcp",
                    "headers": {"Authorization": "Bearer secret"},
                }
            }
        }
        config_file.write_text(json.dumps(config_data))

        config = load_mcp_config(str(config_file))

        headers = config["mcpServers"]["authenticated-api"]["headers"]
        assert headers["Authorization"] == "Bearer secret"

    @pytest.mark.asyncio
    async def test_load_config_invalid_headers_type(self, tmp_path: Path) -> None:
        """Test that TypeError is raised when headers is not a dictionary."""
        config_file = tmp_path / "invalid-headers.json"
        config_data = {
            "mcpServers": {
                "api": {
                    "type": "sse",
                    "url": "https://api.example.com/mcp",
                    "headers": ["not", "a", "dict"],
                }
            }
        }
        config_file.write_text(json.dumps(config_data))

        with pytest.raises(TypeError, match=r"api.*'headers' must be a dictionary"):
            load_mcp_config(str(config_file))


class TestGetMCPTools:
    """Test MCP tools loading from configuration."""

    @patch("deepagents_cli.mcp_tools.load_mcp_tools")
    @patch("deepagents_cli.mcp_tools.MultiServerMCPClient")
    @pytest.mark.asyncio
    async def test_get_mcp_tools_success(
        self,
        mock_client_class: MagicMock,
        mock_load_tools: AsyncMock,
        tmp_path: Path,
        valid_config_data: dict,
        mock_mcp_client: tuple,
        mock_tools: list,
    ) -> None:
        """Test successful loading of MCP tools."""
        # Create a valid config file
        config_file = tmp_path / "mcp-config.json"
        config_file.write_text(json.dumps(valid_config_data))

        # Setup mocks
        mock_client, mock_session = mock_mcp_client
        mock_client_class.return_value = mock_client
        mock_load_tools.return_value = mock_tools

        tools, manager = await get_mcp_tools(str(config_file))

        # Verify client was initialized with correct connection config
        mock_client_class.assert_called_once()
        connections = mock_client_class.call_args.kwargs["connections"]
        assert connections["filesystem"]["command"] == "npx"
        assert connections["filesystem"]["transport"] == "stdio"
        assert connections["filesystem"]["args"] == [
            "-y",
            "@modelcontextprotocol/server-filesystem",
            "/tmp",  # noqa: S108
        ]

        # Verify session was created and tools were loaded
        mock_client.session.assert_called_once_with("filesystem")
        mock_load_tools.assert_called_once_with(mock_session)
        assert len(tools) == 2
        assert tools[0].name == "read_file"
        assert tools[1].name == "write_file"
        assert isinstance(manager, MCPSessionManager)

        # Clean up
        await manager.cleanup()

    @patch("deepagents_cli.mcp_tools.MultiServerMCPClient")
    @pytest.mark.asyncio
    async def test_get_mcp_tools_server_spawn_failure(
        self, mock_client_class: MagicMock, tmp_path: Path
    ) -> None:
        """Test handling of MCP server spawn failure."""
        # Create a valid config file
        config_file = tmp_path / "mcp-config.json"
        config_data = {
            "mcpServers": {
                "filesystem": {
                    "command": "nonexistent-command",
                    "args": [],
                    "env": {},
                }
            }
        }
        config_file.write_text(json.dumps(config_data))

        # Setup mock client to raise an exception
        mock_client_class.side_effect = Exception("Command not found")

        with pytest.raises(
            RuntimeError, match=r"Failed to connect to MCP servers.*Command not found"
        ):
            await get_mcp_tools(str(config_file))

    @patch("deepagents_cli.mcp_tools.load_mcp_tools")
    @patch("deepagents_cli.mcp_tools.MultiServerMCPClient")
    @pytest.mark.asyncio
    async def test_get_mcp_tools_get_tools_failure(
        self,
        mock_client_class: MagicMock,
        mock_load_tools: AsyncMock,
        tmp_path: Path,
        valid_config_data: dict,
        mock_mcp_client: tuple,
    ) -> None:
        """Test handling of failure during load_mcp_tools call."""
        # Create a valid config file
        config_file = tmp_path / "mcp-config.json"
        config_file.write_text(json.dumps(valid_config_data))

        # Setup mocks
        mock_client, _ = mock_mcp_client
        mock_client_class.return_value = mock_client
        mock_load_tools.side_effect = Exception("Server protocol error")

        with pytest.raises(
            RuntimeError, match=r"Failed to connect to MCP servers.*Server protocol error"
        ):
            await get_mcp_tools(str(config_file))

    @patch("deepagents_cli.mcp_tools.load_mcp_tools")
    @patch("deepagents_cli.mcp_tools.MultiServerMCPClient")
    @pytest.mark.asyncio
    async def test_get_mcp_tools_multiple_servers(
        self, mock_client_class: MagicMock, mock_load_tools: AsyncMock, tmp_path: Path
    ) -> None:
        """Test loading tools from multiple MCP servers."""
        # Create config with multiple servers
        config_file = tmp_path / "multi-server.json"
        config_data = {
            "mcpServers": {
                "filesystem": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],  # noqa: S108
                    "env": {},
                },
                "brave-search": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-brave-search"],
                    "env": {"BRAVE_API_KEY": "test-key"},
                },
            }
        }
        config_file.write_text(json.dumps(config_data))

        # Create mock tools from different servers
        mock_tools_fs = [MagicMock(name="read_file"), MagicMock(name="write_file")]
        mock_tools_search = [MagicMock(name="web_search")]

        # Setup mock client with session support
        mock_session_fs = AsyncMock()
        mock_session_search = AsyncMock()

        # Mock session context managers for both servers
        def mock_session_cm(server_name: str) -> MagicMock:
            session = mock_session_fs if server_name == "filesystem" else mock_session_search
            cm = MagicMock()
            cm.__aenter__ = AsyncMock(return_value=session)
            cm.__aexit__ = AsyncMock(return_value=None)
            return cm

        mock_client = MagicMock()
        mock_client.session.side_effect = mock_session_cm
        mock_client_class.return_value = mock_client

        # Mock load_mcp_tools to return different tools for each session
        async def mock_load_side_effect(session: AsyncMock) -> list[MagicMock]:
            if session == mock_session_fs:
                return mock_tools_fs
            return mock_tools_search

        mock_load_tools.side_effect = mock_load_side_effect

        tools, manager = await get_mcp_tools(str(config_file))

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

    @pytest.mark.asyncio
    async def test_get_mcp_tools_invalid_config(self, tmp_path: Path) -> None:
        """Test that config validation errors are propagated."""
        # Create invalid config (missing command)
        config_file = tmp_path / "invalid.json"
        config_data = {
            "mcpServers": {
                "filesystem": {
                    "args": ["/tmp"],  # noqa: S108
                    # Missing command field
                }
            }
        }
        config_file.write_text(json.dumps(config_data))

        with pytest.raises(ValueError, match="missing required 'command' field"):
            await get_mcp_tools(str(config_file))

    @patch("deepagents_cli.mcp_tools.load_mcp_tools")
    @patch("deepagents_cli.mcp_tools.MultiServerMCPClient")
    @pytest.mark.asyncio
    async def test_get_mcp_tools_env_variables_passed(
        self,
        mock_client_class: MagicMock,
        mock_load_tools: AsyncMock,
        tmp_path: Path,
        mock_mcp_client: tuple,
    ) -> None:
        """Test that environment variables are correctly passed to MCP client."""
        config_file = tmp_path / "with-env.json"
        config_data = {
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
        config_file.write_text(json.dumps(config_data))

        # Setup mocks
        mock_client, _ = mock_mcp_client
        mock_client_class.return_value = mock_client
        mock_load_tools.return_value = []

        _, manager = await get_mcp_tools(str(config_file))

        # Verify env variables were passed correctly
        connections = mock_client_class.call_args.kwargs["connections"]
        assert connections["github"]["env"]["GITHUB_TOKEN"] == "ghp_test123"  # noqa: S105
        assert connections["github"]["env"]["GITHUB_API_URL"] == "https://api.github.com"

        # Clean up
        await manager.cleanup()

    @patch("deepagents_cli.mcp_tools.load_mcp_tools")
    @patch("deepagents_cli.mcp_tools.MultiServerMCPClient")
    @pytest.mark.asyncio
    async def test_get_mcp_tools_headers_passed_for_sse(
        self,
        mock_client_class: MagicMock,
        mock_load_tools: AsyncMock,
        tmp_path: Path,
        mock_mcp_client: tuple,
    ) -> None:
        """Test that headers are correctly passed to SSE MCP client."""
        config_file = tmp_path / "sse-with-headers.json"
        config_data = {
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
        config_file.write_text(json.dumps(config_data))

        # Setup mocks
        mock_client, _ = mock_mcp_client
        mock_client_class.return_value = mock_client
        mock_load_tools.return_value = []

        _, manager = await get_mcp_tools(str(config_file))

        # Verify headers were passed correctly
        connections = mock_client_class.call_args.kwargs["connections"]
        assert connections["api"]["transport"] == "sse"
        assert connections["api"]["headers"]["Authorization"] == "Bearer token123"
        assert connections["api"]["headers"]["X-API-Key"] == "key456"

        # Clean up
        await manager.cleanup()

    @patch("deepagents_cli.mcp_tools.load_mcp_tools")
    @patch("deepagents_cli.mcp_tools.MultiServerMCPClient")
    @pytest.mark.asyncio
    async def test_get_mcp_tools_headers_passed_for_http(
        self,
        mock_client_class: MagicMock,
        mock_load_tools: AsyncMock,
        tmp_path: Path,
        mock_mcp_client: tuple,
    ) -> None:
        """Test that headers are correctly passed to HTTP MCP client."""
        config_file = tmp_path / "http-with-headers.json"
        config_data = {
            "mcpServers": {
                "api": {
                    "type": "http",
                    "url": "https://api.example.com/mcp",
                    "headers": {"Authorization": "Bearer secret"},
                }
            }
        }
        config_file.write_text(json.dumps(config_data))

        # Setup mocks
        mock_client, _ = mock_mcp_client
        mock_client_class.return_value = mock_client
        mock_load_tools.return_value = []

        _, manager = await get_mcp_tools(str(config_file))

        # Verify headers were passed and transport is correct
        connections = mock_client_class.call_args.kwargs["connections"]
        assert connections["api"]["transport"] == "streamable_http"
        assert connections["api"]["headers"]["Authorization"] == "Bearer secret"

        # Clean up
        await manager.cleanup()

    @patch("deepagents_cli.mcp_tools.load_mcp_tools")
    @patch("deepagents_cli.mcp_tools.MultiServerMCPClient")
    @pytest.mark.asyncio
    async def test_get_mcp_tools_no_headers_when_not_provided(
        self,
        mock_client_class: MagicMock,
        mock_load_tools: AsyncMock,
        tmp_path: Path,
        mock_mcp_client: tuple,
    ) -> None:
        """Test that headers key is not added when not provided in config."""
        config_file = tmp_path / "sse-no-headers.json"
        config_data = {
            "mcpServers": {
                "api": {
                    "type": "sse",
                    "url": "https://api.example.com/mcp",
                }
            }
        }
        config_file.write_text(json.dumps(config_data))

        # Setup mocks
        mock_client, _ = mock_mcp_client
        mock_client_class.return_value = mock_client
        mock_load_tools.return_value = []

        _, manager = await get_mcp_tools(str(config_file))

        # Verify headers key is not present
        connections = mock_client_class.call_args.kwargs["connections"]
        assert "headers" not in connections["api"]

        # Clean up
        await manager.cleanup()
