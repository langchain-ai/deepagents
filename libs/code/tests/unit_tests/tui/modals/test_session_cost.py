"""Tests for the persistent session cost warning modal."""

from textual.app import App
from textual.widgets import Static

from deepagents_code.tui.modals.session_cost import SessionCostWarningScreen


async def test_warning_copy_and_click_persistence() -> None:
    """Show the cost and recovery suggestions until keyboard acknowledgment."""
    app: App[None] = App()
    screen = SessionCostWarningScreen(cost_usd=5.25, threshold=5.0)
    async with app.run_test() as pilot:
        app.push_screen(screen)
        await pilot.pause()
        body = str(screen.query_one(".session-cost-warning-body", Static).render())
        assert "$5.25" in body
        assert "$5.00" in body
        assert "/offload" in body
        assert "/clear" in body

        await pilot.click(".session-cost-warning-body")
        await pilot.pause()
        assert app.screen is screen
