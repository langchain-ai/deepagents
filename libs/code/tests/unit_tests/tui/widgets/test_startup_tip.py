"""Unit tests for the startup tip widget."""

from typing import TYPE_CHECKING

from deepagents_code._env_vars import HIDE_SPLASH_TIPS
from deepagents_code.tui.widgets.startup_tip import show_startup_tip

if TYPE_CHECKING:
    import pytest


class TestStartupTip:
    """Tests for the bottom startup tip widget."""

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
