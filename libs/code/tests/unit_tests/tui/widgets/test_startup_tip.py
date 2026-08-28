"""Unit tests for the startup tip widget."""

from unittest.mock import patch

import pytest
from textual.content import Content

from deepagents_code._env_vars import HIDE_SPLASH_TIPS
from deepagents_code.tui.widgets.startup_tip import (
    _TIP_EXTERNAL_EDITOR,
    _TIP_SHIFT_TAB_WITH_YOLO,
    _TIP_SHIFT_TAB_WITHOUT_YOLO,
    _TIPS,
    StartupTip,
    _active_tips,
    _pick_tip,
    show_startup_tip,
)

_PICK_TIP = "deepagents_code.tui.widgets.startup_tip._pick_tip"
_CHOICES = "deepagents_code.tui.widgets.startup_tip.random.choices"
_IS_YOLO_SWITCHER = "deepagents_code.config.is_yolo_switcher_enabled"


@pytest.fixture(autouse=True)
def _clear_editor_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.delenv("EDITOR", raising=False)


class TestStartupTip:
    """Tests for the bottom startup tip widget."""

    def test_returns_content(self) -> None:
        """The widget renders Textual `Content`."""
        assert isinstance(StartupTip("Use /help").render(), Content)

    def test_renders_tip_text(self) -> None:
        """The widget labels and renders the selected tip."""
        rendered = StartupTip("Use /copy").render()
        assert isinstance(rendered, Content)
        assert rendered.plain == "Tip: Use /copy"

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

    def test_incognito_shell_tip_registered(self) -> None:
        """The `!!` shell mode keeps a discoverability tip."""
        assert any("!!" in tip and "incognito" in tip.lower() for tip in _TIPS)

    def test_copy_command_tip_registered(self) -> None:
        """The `/copy` command keeps a discoverability tip."""
        assert any("/copy" in tip for tip in _TIPS)

    def test_show_reasoning_tip_registered(self) -> None:
        """The reasoning display flag keeps a discoverability tip."""
        assert any("--show-reasoning" in tip for tip in _TIPS)
