"""Capture SVG screenshots of the step-into feature for PR demo."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Container, Vertical
from textual.widgets import Static

# Add the CLI package to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepagents_cli.widgets.approval import ApprovalMenu
from deepagents_cli.widgets.subagent_banner import SubagentBanner

SCREENSHOTS_DIR = Path(__file__).resolve().parent.parent / "screenshots"


class ApprovalDemoApp(App):
    """Minimal app to screenshot the approval menu with step-into option."""

    DEFAULT_CSS = """
    Screen {
        background: #1e1e2e;
    }

    #demo-header {
        color: #94a3b8;
        padding: 1 2;
    }

    #demo-container {
        padding: 0 2;
    }

    /* Approval menu styles (from app.tcss) */
    .approval-menu {
        border: heavy yellow;
        padding: 1;
        margin: 0 0;
        height: auto;
        max-height: 24;
    }

    .approval-title {
        text-style: bold;
        color: yellow;
        padding: 0 0 1 0;
    }

    .approval-option {
        padding: 0 1;
        height: 1;
    }

    .approval-option-selected {
        background: #3b82f6;
        color: white;
        text-style: bold;
    }

    .approval-help {
        color: #64748b;
        padding: 1 0 0 0;
    }

    .approval-separator {
        color: #475569;
    }

    .approval-command {
        padding: 0 0 1 0;
    }

    .tool-info-scroll {
        height: auto;
        max-height: 10;
    }

    .tool-info-container {
        height: auto;
    }

    .approval-description {
        padding: 0 0 1 0;
    }

    .approval-security-warning {
        padding: 0 0 1 0;
    }

    .approval-options-container {
        height: auto;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static(
            "[bold #94a3b8]deepagents>[/] /task Research the auth implementation",
            id="demo-header",
        )
        with Container(id="demo-container"):
            yield ApprovalMenu(
                {
                    "name": "task",
                    "description": "Launch a subagent to research auth patterns",
                    "args": {
                        "subagent_type": "general-purpose",
                        "description": "Research the authentication implementation",
                        "prompt": "Investigate how auth is handled in the codebase",
                    },
                },
            )


class BannerDemoApp(App):
    """Minimal app to screenshot the subagent banner."""

    DEFAULT_CSS = """
    Screen {
        background: #1e1e2e;
    }

    #chat-area {
        height: 1fr;
        padding: 1 2;
    }

    .msg {
        color: #94a3b8;
        padding: 0 0 1 0;
    }

    .msg-agent {
        color: #e2e8f0;
        padding: 0 0 1 0;
    }

    #prompt-line {
        color: #60a5fa;
        text-style: bold;
        padding: 1 2 0 2;
        dock: bottom;
    }
    """

    def compose(self) -> ComposeResult:
        yield SubagentBanner(id="subagent-banner")
        with Vertical(id="chat-area"):
            yield Static(
                "[dim]Stepped into subagent (approve + interactive)[/]",
                classes="msg",
            )
            yield Static(
                "[bold #60a5fa]general-purpose[/] [dim]agent running...[/]",
                classes="msg",
            )
            yield Static(
                "Found 3 authentication modules:\n"
                "  [#a78bfa]auth/oauth.py[/] - OAuth2 provider integration\n"
                "  [#a78bfa]auth/jwt.py[/] - JWT token handling\n"
                "  [#a78bfa]auth/middleware.py[/] - Request auth middleware",
                classes="msg-agent",
            )
        yield Static(
            "[bold #60a5fa][general-purpose:1] >[/] _",
            id="prompt-line",
        )

    async def on_mount(self) -> None:
        banner = self.query_one("#subagent-banner", SubagentBanner)
        banner.show(subagent_type="general-purpose", depth=1)


class ErrorStateDemoApp(App):
    """Demo: what the UI looks like when stuck in a dead subagent context."""

    DEFAULT_CSS = """
    Screen { background: #1e1e2e; }
    #chat-area { height: 1fr; padding: 1 2; }
    .msg { color: #94a3b8; padding: 0 0 1 0; }
    .msg-error { color: #ef4444; padding: 0 0 1 0; }
    #prompt-line { color: #60a5fa; text-style: bold; padding: 1 2 0 2; dock: bottom; }
    """

    def compose(self) -> ComposeResult:
        yield SubagentBanner(id="subagent-banner")
        with Vertical(id="chat-area"):
            yield Static(
                "[dim]Stepped into subagent (approve + interactive)[/]",
                classes="msg",
            )
            yield Static(
                "[bold red]Agent error: Connection timeout[/]",
                classes="msg-error",
            )
            yield Static(
                "[dim]BUG: Context stuck at depth=1, banner still shows.[/]\n"
                "[dim]User must type /return or /clear to recover.[/]",
                classes="msg",
            )
        yield Static(
            "[bold #60a5fa][researcher:1] >[/] _",
            id="prompt-line",
        )

    async def on_mount(self) -> None:
        banner = self.query_one("#subagent-banner", SubagentBanner)
        banner.show(subagent_type="researcher", depth=1)


async def capture_screenshots() -> None:
    """Run demo apps and save SVG screenshots."""
    SCREENSHOTS_DIR.mkdir(exist_ok=True)

    # Screenshot 1: Approval menu with step-into
    app1 = ApprovalDemoApp()
    async with app1.run_test(size=(80, 28)) as pilot:
        await pilot.pause()
        await pilot.pause()
        await pilot.press("down")
        await pilot.press("down")
        await pilot.pause()
        path1 = app1.save_screenshot(
            filename="step_into_approval.svg",
            path=str(SCREENSHOTS_DIR),
        )
        print(f"Saved: {path1}")

    # Screenshot 2: Banner + subagent session
    app2 = BannerDemoApp()
    async with app2.run_test(size=(80, 18)) as pilot:
        await pilot.pause()
        await pilot.pause()
        path2 = app2.save_screenshot(
            filename="subagent_banner.svg",
            path=str(SCREENSHOTS_DIR),
        )
        print(f"Saved: {path2}")

    # Screenshot 3: Error state - stuck in dead subagent
    app3 = ErrorStateDemoApp()
    async with app3.run_test(size=(80, 16)) as pilot:
        await pilot.pause()
        await pilot.pause()
        path3 = app3.save_screenshot(
            filename="error_state_stuck.svg",
            path=str(SCREENSHOTS_DIR),
        )
        print(f"Saved: {path3}")


if __name__ == "__main__":
    asyncio.run(capture_screenshots())
