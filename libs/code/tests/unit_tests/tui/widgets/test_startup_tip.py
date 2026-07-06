"""Unit tests for the startup tip widget."""

from unittest.mock import patch

import pytest
from textual.content import Content

from deepagents_code._env_vars import HIDE_SPLASH_TIPS
from deepagents_code.tui.widgets.startup_tip import (
    _TIPS,
    StartupTip,
    show_startup_tip,
)

_PICK_TIP = "deepagents_code.tui.widgets.startup_tip._pick_tip"


class TestStartupTip:
    """Tests for the bottom startup tip widget."""

    def test_returns_content(self) -> None:
        """The widget renders Textual `Content`."""
        assert isinstance(StartupTip("Use /help").render(), Content)

    def test_renders_tip_text(self) -> None:
        """The widget labels and renders the selected tip."""
        assert StartupTip("Use /copy").render().plain == "Tip: Use /copy"

    def test_selects_weighted_tip_when_omitted(self) -> None:
        """A tip is selected when no explicit text is provided."""
        with patch(_PICK_TIP, return_value="Use /copy") as pick_tip:
            widget = StartupTip()

        assert widget.tip == "Use /copy"
        assert widget.render().plain == "Tip: Use /copy"
        pick_tip.assert_called_once()

    def test_show_startup_tip_defaults_to_true(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Startup tips are visible by default."""
        monkeypatch.delenv(HIDE_SPLASH_TIPS, raising=False)

        assert show_startup_tip() is True

    def test_hide_splash_tips_env_var_hides_tip(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`HIDE_SPLASH_TIPS` disables the bottom tip widget."""
        monkeypatch.setenv(HIDE_SPLASH_TIPS, "1")

        assert show_startup_tip() is False

    def test_startup_cmd_tip_registered(self) -> None:
        """The `--startup-cmd` flag keeps a discoverability tip."""
        assert any("--startup-cmd" in tip for tip in _TIPS)

    def test_incognito_shell_tip_registered(self) -> None:
        """The `!!` shell mode keeps a discoverability tip."""
        assert any("!!" in tip and "incognito" in tip.lower() for tip in _TIPS)

    def test_copy_command_tip_registered(self) -> None:
        """The `/copy` command keeps a discoverability tip."""
        assert "Use /copy to copy the latest assistant message" in _TIPS

    def test_workflow_subagent_tip_registered(self) -> None:
        """The workflow trigger phrase keeps its higher relative weight."""
        tip = "Ask for a workflow to fan work out to subagents in parallel"
        assert _TIPS[tip] == 3
