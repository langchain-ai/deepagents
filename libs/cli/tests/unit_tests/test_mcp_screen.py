"""Tests for MCPScreen."""

import json
from pathlib import Path
from typing import ClassVar

import pytest
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Tree

from deepagents_cli.widgets.mcp_screen import MCPScreen


class MCPScreenTestApp(App):
    """Test app for MCPScreen."""

    def __init__(self, config_path: Path | None = None) -> None:
        super().__init__()
        self.dismissed = False
        self._config_path = config_path

    def compose(self) -> ComposeResult:
        yield Container(id="main")

    def show_mcp(self) -> None:
        """Show the MCP screen."""

        def handle_result(_result: None) -> None:
            self.dismissed = True

        self.push_screen(MCPScreen(config_path=self._config_path), handle_result)


class AppWithEscapeBinding(App):
    """Test app with conflicting escape binding."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "interrupt", "Interrupt", show=False, priority=True),
    ]

    def __init__(self, config_path: Path | None = None) -> None:
        super().__init__()
        self.dismissed = False
        self.interrupt_called = False
        self._config_path = config_path

    def compose(self) -> ComposeResult:
        yield Container(id="main")

    def action_interrupt(self) -> None:
        """Handle escape - dismiss modal if present."""
        if isinstance(self.screen, ModalScreen):
            self.screen.dismiss(None)
            return
        self.interrupt_called = True

    def show_mcp(self) -> None:
        """Show the MCP screen."""

        def handle_result(_result: None) -> None:
            self.dismissed = True

        self.push_screen(MCPScreen(config_path=self._config_path), handle_result)


class TestMCPScreenEscapeKey:
    """Tests for ESC key dismissing the modal."""

    @pytest.mark.asyncio
    async def test_escape_dismisses_modal(self, tmp_path: Path) -> None:
        """Pressing ESC should dismiss the modal."""
        config = tmp_path / ".mcp.json"
        config.write_text(json.dumps({"mcpServers": {}}))
        app = MCPScreenTestApp(config_path=config)
        async with app.run_test() as pilot:
            app.show_mcp()
            await pilot.pause()

            await pilot.press("escape")
            await pilot.pause()

            assert app.dismissed is True

    @pytest.mark.asyncio
    async def test_escape_with_conflicting_app_binding(self, tmp_path: Path) -> None:
        """ESC should dismiss modal even when app has its own escape binding."""
        config = tmp_path / ".mcp.json"
        config.write_text(json.dumps({"mcpServers": {}}))
        app = AppWithEscapeBinding(config_path=config)
        async with app.run_test() as pilot:
            app.show_mcp()
            await pilot.pause()

            await pilot.press("escape")
            await pilot.pause()

            assert app.dismissed is True
            assert app.interrupt_called is False


class TestMCPScreenConfigLoading:
    """Tests for loading .mcp.json config."""

    @pytest.mark.asyncio
    async def test_missing_config_shows_empty_message(self, tmp_path: Path) -> None:
        """Missing .mcp.json should show 'no servers' message."""
        config = tmp_path / ".mcp.json"  # Does not exist
        app = MCPScreenTestApp(config_path=config)
        async with app.run_test() as pilot:
            app.show_mcp()
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, MCPScreen)
            tree = screen.query_one("#mcp-tree", Tree)
            children = list(tree.root.children)
            assert len(children) == 1
            assert "No MCP servers configured" in str(children[0].label)

    @pytest.mark.asyncio
    async def test_valid_config_shows_servers(self, tmp_path: Path) -> None:
        """Valid .mcp.json should list all servers."""
        config = tmp_path / ".mcp.json"
        config.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "google-workspace": {
                            "type": "http",
                            "url": "https://example.com/mcp",
                        },
                        "open-brain": {
                            "type": "sse",
                            "url": "https://brain.example.com",
                        },
                    }
                }
            )
        )
        app = MCPScreenTestApp(config_path=config)
        async with app.run_test() as pilot:
            app.show_mcp()
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, MCPScreen)
            tree = screen.query_one("#mcp-tree", Tree)
            children = list(tree.root.children)
            assert len(children) == 2
            labels = [str(c.label) for c in children]
            assert any("google-workspace" in label for label in labels)
            assert any("open-brain" in label for label in labels)

    @pytest.mark.asyncio
    async def test_malformed_json_shows_empty(self, tmp_path: Path) -> None:
        """Malformed JSON should gracefully show empty tree."""
        config = tmp_path / ".mcp.json"
        config.write_text("{ invalid json !!!")
        app = MCPScreenTestApp(config_path=config)
        async with app.run_test() as pilot:
            app.show_mcp()
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, MCPScreen)
            tree = screen.query_one("#mcp-tree", Tree)
            children = list(tree.root.children)
            assert len(children) == 1
            assert "No MCP servers configured" in str(children[0].label)

    @pytest.mark.asyncio
    async def test_empty_servers_dict_shows_empty(self, tmp_path: Path) -> None:
        """Empty mcpServers dict should show empty message."""
        config = tmp_path / ".mcp.json"
        config.write_text(json.dumps({"mcpServers": {}}))
        app = MCPScreenTestApp(config_path=config)
        async with app.run_test() as pilot:
            app.show_mcp()
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, MCPScreen)
            tree = screen.query_one("#mcp-tree", Tree)
            children = list(tree.root.children)
            assert len(children) == 1
            assert "No MCP servers configured" in str(children[0].label)


class TestMCPScreenServerType:
    """Tests for server type derivation."""

    @pytest.mark.asyncio
    async def test_explicit_type_field(self, tmp_path: Path) -> None:
        """Server with explicit 'type' field should use it."""
        config = tmp_path / ".mcp.json"
        config.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "my-server": {"type": "sse", "url": "https://x.com"},
                    }
                }
            )
        )
        app = MCPScreenTestApp(config_path=config)
        async with app.run_test() as pilot:
            app.show_mcp()
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, MCPScreen)
            tree = screen.query_one("#mcp-tree", Tree)
            label = str(next(iter(tree.root.children)).label)
            assert "sse" in label

    @pytest.mark.asyncio
    async def test_stdio_derived_from_command(self, tmp_path: Path) -> None:
        """Server with 'command' key should derive type as 'stdio'."""
        config = tmp_path / ".mcp.json"
        config.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "local-server": {
                            "command": "npx",
                            "args": ["-y", "mcp-server"],
                        },
                    }
                }
            )
        )
        app = MCPScreenTestApp(config_path=config)
        async with app.run_test() as pilot:
            app.show_mcp()
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, MCPScreen)
            tree = screen.query_one("#mcp-tree", Tree)
            label = str(next(iter(tree.root.children)).label)
            assert "stdio" in label

    @pytest.mark.asyncio
    async def test_http_derived_from_url(self, tmp_path: Path) -> None:
        """Server with only 'url' key should derive type as 'http'."""
        config = tmp_path / ".mcp.json"
        config.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "remote": {"url": "https://example.com/mcp"},
                    }
                }
            )
        )
        app = MCPScreenTestApp(config_path=config)
        async with app.run_test() as pilot:
            app.show_mcp()
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, MCPScreen)
            tree = screen.query_one("#mcp-tree", Tree)
            label = str(next(iter(tree.root.children)).label)
            assert "http" in label
