"""Unit tests for DeepAgentsApp."""

from textual.binding import Binding

from deepagents_cli.app import DeepAgentsApp


class TestAppBindings:
    """Test app keybindings."""

    def test_toggle_tool_output_has_ctrl_e_binding(self) -> None:
        """Ctrl+E should be bound to toggle_tool_output with priority."""
        bindings = [b for b in DeepAgentsApp.BINDINGS if isinstance(b, Binding)]
        bindings_by_key = {b.key: b for b in bindings}
        ctrl_e = bindings_by_key.get("ctrl+e")

        assert ctrl_e is not None
        assert ctrl_e.action == "toggle_tool_output"
        assert ctrl_e.priority is True

    def test_ctrl_o_not_bound_to_toggle_tool_output(self) -> None:
        """Ctrl+O should not exist (replaced by Ctrl+E)."""
        bindings = [b for b in DeepAgentsApp.BINDINGS if isinstance(b, Binding)]
        bindings_by_key = {b.key: b for b in bindings}
        assert "ctrl+o" not in bindings_by_key
