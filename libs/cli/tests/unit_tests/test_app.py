"""Unit tests for DeepAgentsApp."""

from deepagents_cli.app import DeepAgentsApp


class TestAppBindings:
    """Test app keybindings."""

    def test_toggle_tool_output_has_ctrl_e_binding(self) -> None:
        """Ctrl+E should be bound to toggle_tool_output."""
        bindings_by_key = {b.key: b for b in DeepAgentsApp.BINDINGS}
        ctrl_e = bindings_by_key.get("ctrl+e")

        assert ctrl_e is not None
        assert ctrl_e.action == "toggle_tool_output"
