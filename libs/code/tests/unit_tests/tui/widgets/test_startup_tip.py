"""Unit tests for the startup tip widget."""

from unittest.mock import patch

import pytest
from textual.content import Content

from deepagents_code._env_vars import HIDE_SPLASH_TIPS
from deepagents_code.tui.widgets.startup_tip import (
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

    def test_selects_weighted_tip_when_omitted(self) -> None:
        """A tip is selected when no explicit text is provided."""
        with patch(_PICK_TIP, return_value="Use /copy") as pick_tip:
            widget = StartupTip()

        assert widget.tip == "Use /copy"
        rendered = widget.render()
        assert isinstance(rendered, Content)
        assert rendered.plain == "Tip: Use /copy"
        pick_tip.assert_called_once()

    def test_pick_tip_returns_registered_tip(self) -> None:
        """`_pick_tip` only ever returns a tip drawn from the active registry."""
        with patch(_IS_YOLO_SWITCHER, return_value=True):
            for _ in range(100):
                assert _pick_tip() in _TIPS

    def test_pick_tip_weights_by_registry_values(self) -> None:
        """`_pick_tip` passes the active registry's relative weights to the draw."""
        with (
            patch(_IS_YOLO_SWITCHER, return_value=True),
            patch(_CHOICES, return_value=["Use /copy"]) as choices,
        ):
            assert _pick_tip() == "Use /copy"

        choices.assert_called_once()
        args, kwargs = choices.call_args
        assert args[0] == list(_TIPS.keys())
        assert kwargs["weights"] == list(_TIPS.values())

    def test_active_tips_includes_yolo_when_switcher_enabled(self) -> None:
        """Enabled YOLO switcher keeps the three-mode Shift+Tab tip."""
        tips = _active_tips(yolo_switcher_enabled=True)

        assert _TIP_SHIFT_TAB_WITH_YOLO in tips
        assert _TIP_SHIFT_TAB_WITHOUT_YOLO not in tips
        assert tips[_TIP_SHIFT_TAB_WITH_YOLO] == _TIPS[_TIP_SHIFT_TAB_WITH_YOLO]

    def test_active_tips_hides_yolo_when_switcher_disabled(self) -> None:
        """Disabled YOLO switcher advertises Manual/Auto only."""
        tips = _active_tips(yolo_switcher_enabled=False)

        assert _TIP_SHIFT_TAB_WITH_YOLO not in tips
        assert tips[_TIP_SHIFT_TAB_WITHOUT_YOLO] == _TIPS[_TIP_SHIFT_TAB_WITH_YOLO]

    def test_pick_tip_uses_resolved_yolo_switcher_setting(self) -> None:
        """`_pick_tip` resolves the live switcher setting when choosing tips."""
        with (
            patch(_IS_YOLO_SWITCHER, return_value=False) as resolved,
            patch(
                _CHOICES,
                return_value=[_TIP_SHIFT_TAB_WITHOUT_YOLO],
            ) as choices,
        ):
            assert _pick_tip() == _TIP_SHIFT_TAB_WITHOUT_YOLO

        resolved.assert_called_once_with()
        args, kwargs = choices.call_args
        assert _TIP_SHIFT_TAB_WITHOUT_YOLO in args[0]
        assert _TIP_SHIFT_TAB_WITH_YOLO not in args[0]
        assert kwargs["weights"][args[0].index(_TIP_SHIFT_TAB_WITHOUT_YOLO)] == 2

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

    def test_workflow_subagent_tip_registered(self) -> None:
        """The workflow trigger phrase keeps an above-baseline weight."""
        tip = "Ask for a workflow to fan work out to subagents in parallel"
        assert _TIPS[tip] > 1
