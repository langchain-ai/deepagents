"""Tests for MCP tools configuration loading and validation."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepagents_cli.mcp_tools import get_mcp_tools, load_mcp_config


class TestLoadMCPConfig:
    """Test MCP configuration file loading and validation."""

    def test_load_valid_config(self, tmp_path: Path) -> None:
        """Test loading a valid MCP configuration file."""
        config_file = tmp_path / "mcp-config.json"
        config_data = {
            "mcpServers": {
                "filesystem": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                    "env": {},
                }
            }
        }
        config_file.write_text(json.dumps(config_data))

        # Run async function in event loop
        import asyncio

        config = asyncio.run(load_mcp_config(str(config_file)))

        assert config == config_data
        assert "mcpServers" in config
        assert "filesystem" in config["mcpServers"]
        assert config["mcpServers"]["filesystem"]["command"] == "npx"

    def test_load_config_file_not_found(self, tmp_path: Path) -> None:
        """Test that FileNotFoundError is raised for missing config file."""
        nonexistent_file = tmp_path / "nonexistent.json"

        import asyncio

        with pytest.raises(FileNotFoundError) as exc_info:
            asyncio.run(load_mcp_config(str(nonexistent_file)))

        assert "MCP config file not found" in str(exc_info.value)

    def test_load_config_invalid_json(self, tmp_path: Path) -> None:
        """Test that JSONDecodeError is raised for invalid JSON."""
        config_file = tmp_path / "invalid.json"
        config_file.write_text("{invalid json")

        import asyncio

        with pytest.raises(json.JSONDecodeError) as exc_info:
            asyncio.run(load_mcp_config(str(config_file)))

        assert "Invalid JSON in MCP config file" in str(exc_info.value)

    def test_load_config_missing_mcpservers_field(self, tmp_path: Path) -> None:
        """Test that ValueError is raised when mcpServers field is missing."""
        config_file = tmp_path / "missing-field.json"
        config_data = {"someOtherField": "value"}
        config_file.write_text(json.dumps(config_data))

        import asyncio

        with pytest.raises(ValueError) as exc_info:
            asyncio.run(load_mcp_config(str(config_file)))

        assert "must contain 'mcpServers' field" in str(exc_info.value)

    def test_load_config_mcpservers_not_dict(self, tmp_path: Path) -> None:
        """Test that ValueError is raised when mcpServers is not a dictionary."""
        config_file = tmp_path / "bad-type.json"
        config_data = {"mcpServers": ["not", "a", "dict"]}
        config_file.write_text(json.dumps(config_data))

        import asyncio

        with pytest.raises(ValueError) as exc_info:
            asyncio.run(load_mcp_config(str(config_file)))

        assert "'mcpServers' field must be a dictionary" in str(exc_info.value)

    def test_load_config_empty_mcpservers(self, tmp_path: Path) -> None:
        """Test that ValueError is raised when mcpServers is empty."""
        config_file = tmp_path / "empty.json"
        config_data = {"mcpServers": {}}
        config_file.write_text(json.dumps(config_data))

        import asyncio

        with pytest.raises(ValueError) as exc_info:
            asyncio.run(load_mcp_config(str(config_file)))

        assert "'mcpServers' field is empty" in str(exc_info.value)

    def test_load_config_server_missing_command(self, tmp_path: Path) -> None:
        """Test that ValueError is raised when server config is missing command."""
        config_file = tmp_path / "no-command.json"
        config_data = {
            "mcpServers": {
                "filesystem": {
                    "args": ["/tmp"],
                    # Missing "command" field
                }
            }
        }
        config_file.write_text(json.dumps(config_data))

        import asyncio

        with pytest.raises(ValueError) as exc_info:
            asyncio.run(load_mcp_config(str(config_file)))

        assert "missing required 'command' field" in str(exc_info.value)
        assert "filesystem" in str(exc_info.value)

    def test_load_config_server_config_not_dict(self, tmp_path: Path) -> None:
        """Test that ValueError is raised when server config is not a dict."""
        config_file = tmp_path / "bad-server-config.json"
        config_data = {"mcpServers": {"filesystem": "not a dict"}}
        config_file.write_text(json.dumps(config_data))

        import asyncio

        with pytest.raises(ValueError) as exc_info:
            asyncio.run(load_mcp_config(str(config_file)))

        assert "config must be a dictionary" in str(exc_info.value)
        assert "filesystem" in str(exc_info.value)

    def test_load_config_args_not_list(self, tmp_path: Path) -> None:
        """Test that ValueError is raised when args is not a list."""
        config_file = tmp_path / "bad-args.json"
        config_data = {
            "mcpServers": {
                "filesystem": {
                    "command": "npx",
                    "args": "not a list",  # Should be a list
                }
            }
        }
        config_file.write_text(json.dumps(config_data))

        import asyncio

        with pytest.raises(ValueError) as exc_info:
            asyncio.run(load_mcp_config(str(config_file)))

        assert "'args' must be a list" in str(exc_info.value)
        assert "filesystem" in str(exc_info.value)

    def test_load_config_env_not_dict(self, tmp_path: Path) -> None:
        """Test that ValueError is raised when env is not a dict."""
        config_file = tmp_path / "bad-env.json"
        config_data = {
            "mcpServers": {
                "filesystem": {
                    "command": "npx",
                    "args": ["/tmp"],
                    "env": ["not", "a", "dict"],  # Should be a dict
                }
            }
        }
        config_file.write_text(json.dumps(config_data))

        import asyncio

        with pytest.raises(ValueError) as exc_info:
            asyncio.run(load_mcp_config(str(config_file)))

        assert "'env' must be a dictionary" in str(exc_info.value)
        assert "filesystem" in str(exc_info.value)

    def test_load_config_optional_fields(self, tmp_path: Path) -> None:
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

        import asyncio

        config = asyncio.run(load_mcp_config(str(config_file)))

        assert config == config_data
        assert "simple" in config["mcpServers"]

    def test_load_config_multiple_servers(self, tmp_path: Path) -> None:
        """Test loading config with multiple MCP servers."""
        config_file = tmp_path / "multi-server.json"
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
        config_file.write_text(json.dumps(config_data))

        import asyncio

        config = asyncio.run(load_mcp_config(str(config_file)))

        assert len(config["mcpServers"]) == 3
        assert "filesystem" in config["mcpServers"]
        assert "brave-search" in config["mcpServers"]
        assert "github" in config["mcpServers"]


class TestGetMCPTools:
    """Test MCP tools loading from configuration."""

    @patch("deepagents_cli.mcp_tools.MultiServerMCPClient")
    def test_get_mcp_tools_success(self, mock_client_class: MagicMock, tmp_path: Path) -> None:
        """Test successful loading of MCP tools."""
        # Create a valid config file
        config_file = tmp_path / "mcp-config.json"
        config_data = {
            "mcpServers": {
                "filesystem": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                    "env": {},
                }
            }
        }
        config_file.write_text(json.dumps(config_data))

        # Create mock tools
        mock_tool1 = MagicMock()
        mock_tool1.name = "read_file"
        mock_tool1.description = "Read a file"

        mock_tool2 = MagicMock()
        mock_tool2.name = "write_file"
        mock_tool2.description = "Write a file"

        # Setup mock client
        mock_client = AsyncMock()
        mock_client.get_tools = AsyncMock(return_value=[mock_tool1, mock_tool2])
        mock_client_class.return_value = mock_client

        import asyncio

        tools, client = asyncio.run(get_mcp_tools(str(config_file)))

        # Verify client was initialized with correct connection config
        mock_client_class.assert_called_once()
        call_kwargs = mock_client_class.call_args.kwargs
        assert "connections" in call_kwargs
        connections = call_kwargs["connections"]
        assert "filesystem" in connections
        assert connections["filesystem"]["command"] == "npx"
        assert connections["filesystem"]["transport"] == "stdio"
        assert connections["filesystem"]["args"] == [
            "-y",
            "@modelcontextprotocol/server-filesystem",
            "/tmp",
        ]

        # Verify tools were retrieved
        mock_client.get_tools.assert_called_once()
        assert len(tools) == 2
        assert tools[0].name == "read_file"
        assert tools[1].name == "write_file"

    @patch("deepagents_cli.mcp_tools.MultiServerMCPClient")
    def test_get_mcp_tools_server_spawn_failure(
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

        import asyncio

        with pytest.raises(RuntimeError) as exc_info:
            asyncio.run(get_mcp_tools(str(config_file)))

        assert "Failed to connect to MCP servers" in str(exc_info.value)
        assert "Command not found" in str(exc_info.value)

    @patch("deepagents_cli.mcp_tools.MultiServerMCPClient")
    def test_get_mcp_tools_get_tools_failure(
        self, mock_client_class: MagicMock, tmp_path: Path
    ) -> None:
        """Test handling of failure during get_tools call."""
        # Create a valid config file
        config_file = tmp_path / "mcp-config.json"
        config_data = {
            "mcpServers": {
                "filesystem": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                    "env": {},
                }
            }
        }
        config_file.write_text(json.dumps(config_data))

        # Setup mock client to fail on get_tools
        mock_client = AsyncMock()
        mock_client.get_tools = AsyncMock(side_effect=Exception("Server protocol error"))
        mock_client_class.return_value = mock_client

        import asyncio

        with pytest.raises(RuntimeError) as exc_info:
            asyncio.run(get_mcp_tools(str(config_file)))

        assert "Failed to connect to MCP servers" in str(exc_info.value)
        assert "Server protocol error" in str(exc_info.value)

    @patch("deepagents_cli.mcp_tools.MultiServerMCPClient")
    def test_get_mcp_tools_multiple_servers(
        self, mock_client_class: MagicMock, tmp_path: Path
    ) -> None:
        """Test loading tools from multiple MCP servers."""
        # Create config with multiple servers
        config_file = tmp_path / "multi-server.json"
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
            }
        }
        config_file.write_text(json.dumps(config_data))

        # Create mock tools from different servers
        mock_tools = [
            MagicMock(name="read_file"),
            MagicMock(name="write_file"),
            MagicMock(name="web_search"),
        ]

        # Setup mock client
        mock_client = AsyncMock()
        mock_client.get_tools = AsyncMock(return_value=mock_tools)
        mock_client_class.return_value = mock_client

        import asyncio

        tools, client = asyncio.run(get_mcp_tools(str(config_file)))

        # Verify both servers were registered
        call_kwargs = mock_client_class.call_args.kwargs
        connections = call_kwargs["connections"]
        assert len(connections) == 2
        assert "filesystem" in connections
        assert "brave-search" in connections
        assert connections["brave-search"]["env"]["BRAVE_API_KEY"] == "test-key"

        # Verify tools from all servers were returned
        assert len(tools) == 3

    def test_get_mcp_tools_invalid_config(self, tmp_path: Path) -> None:
        """Test that config validation errors are propagated."""
        # Create invalid config (missing command)
        config_file = tmp_path / "invalid.json"
        config_data = {
            "mcpServers": {
                "filesystem": {
                    "args": ["/tmp"],
                    # Missing command field
                }
            }
        }
        config_file.write_text(json.dumps(config_data))

        import asyncio

        with pytest.raises(ValueError) as exc_info:
            asyncio.run(get_mcp_tools(str(config_file)))

        assert "missing required 'command' field" in str(exc_info.value)

    @patch("deepagents_cli.mcp_tools.MultiServerMCPClient")
    def test_get_mcp_tools_env_variables_passed(
        self, mock_client_class: MagicMock, tmp_path: Path
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

        # Setup mock client
        mock_client = AsyncMock()
        mock_client.get_tools = AsyncMock(return_value=[])
        mock_client_class.return_value = mock_client

        import asyncio

        asyncio.run(get_mcp_tools(str(config_file)))

        # Verify env variables were passed correctly
        call_kwargs = mock_client_class.call_args.kwargs
        connections = call_kwargs["connections"]
        assert connections["github"]["env"]["GITHUB_TOKEN"] == "ghp_test123"
        assert connections["github"]["env"]["GITHUB_API_URL"] == "https://api.github.com"
