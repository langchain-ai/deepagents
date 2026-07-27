"""Startup tip widget shown above the chat input."""

from __future__ import annotations

import random
from typing import Any

from textual.content import Content
from textual.widgets import Static

from deepagents_code._env_vars import HIDE_SPLASH_TIPS, is_env_truthy

_TIP_SHIFT_TAB_WITH_YOLO = "Press Shift+Tab to cycle Manual, Auto, and YOLO modes"
"""Tip used when `startup.yolo_switcher` keeps YOLO in the approval cycle."""

_TIP_SHIFT_TAB_WITHOUT_YOLO = "Press Shift+Tab to toggle Manual and Auto modes"
"""Tip used when orgs/users disable YOLO entry via the approval switcher."""

_TIPS: dict[str, int] = {
    "Use @ to reference files and / for commands": 3,
    "Try /threads to resume a previous conversation": 2,
    "Use /offload when your conversation gets long": 2,
    "Use /copy to copy the latest message": 3,
    "Use /cost to see estimated model spend": 1,
    "Use /tools to list the tools available to the agent": 1,
    "Use /mcp login <server> to authenticate MCP servers": 1,
    "Use /remember to save learnings from this conversation": 1,
    "Use /model to switch models mid-conversation": 2,
    "Use /effort to change the current model's reasoning effort": 1,
    "Press ctrl+x to compose prompts in your external editor": 1,
    "Use /skill:<name> to invoke a skill directly": 1,
    "Use /theme to customize the TUI's colors": 1,
    "Use /skill-creator to build reusable agent skills": 1,
    "Ask for a workflow to fan work out to subagents in parallel": 3,
    "Use /timestamps to show or hide message timestamp footers": 1,
    "Use /agents to browse and switch between your available agents": 2,
    _TIP_SHIFT_TAB_WITH_YOLO: 2,
    "Use !! for incognito shell commands that stay out of model context": 1,
    "Deep Agents can explain its own features and look up its docs. Ask it how to use.": 3,  # noqa: E501
}
"""Tips shown above the chat input. One is chosen at random per launch,
weighted by these relative selection weights.

The Shift+Tab tip is varied at pick time via `_active_tips` so disabled-YOLO
installs do not advertise a switcher path that policy has removed.
"""

# Fail fast at import if the registry is ever emptied or given a non-positive
# weight: `random.choices` would otherwise raise a cryptic error at widget
# construction. `_TIPS` is a hardcoded constant, so this never fires in
# practice — it just guards future edits.
if not _TIPS:
    msg = "_TIPS must not be empty"
    raise ValueError(msg)
if any(weight <= 0 for weight in _TIPS.values()):
    msg = "_TIPS weights must be positive"
    raise ValueError(msg)


def _active_tips(*, yolo_switcher_enabled: bool | None = None) -> dict[str, int]:
    """Return the weighted tip registry for the current switcher policy.

    Args:
        yolo_switcher_enabled: Override for whether YOLO appears in the
            Shift+Tab cycle. When omitted, resolves `startup.yolo_switcher`.

    Returns:
        Weighted tip map appropriate for the active YOLO switcher setting.
    """
    if yolo_switcher_enabled is None:
        from deepagents_code.config import is_yolo_switcher_enabled

        yolo_switcher_enabled = is_yolo_switcher_enabled()

    tips = dict(_TIPS)
    if yolo_switcher_enabled:
        return tips

    # Replace the YOLO cycle tip with the Manual/Auto-only wording so the
    # splash never claims Shift+Tab can enter unrestricted mode when policy
    # has removed that entry from the switcher.
    weight = tips.pop(_TIP_SHIFT_TAB_WITH_YOLO, None)
    if weight is not None:
        tips[_TIP_SHIFT_TAB_WITHOUT_YOLO] = weight
    return tips


def _pick_tip(*, yolo_switcher_enabled: bool | None = None) -> str:
    """Pick one startup tip using the configured relative weights.

    Args:
        yolo_switcher_enabled: Optional override forwarded to `_active_tips`.

    Returns:
        Tip text selected from the active tip registry.
    """
    tips = _active_tips(yolo_switcher_enabled=yolo_switcher_enabled)
    return random.choices(list(tips.keys()), weights=list(tips.values()), k=1)[0]  # noqa: S311


def show_startup_tip() -> bool:
    """Return whether startup tips should be shown.

    Returns:
        `True` when startup tips are enabled for the current process.
    """
    return not is_env_truthy(HIDE_SPLASH_TIPS)


class StartupTip(Static):
    """One startup tip displayed above the chat input."""

    DEFAULT_CSS = """
    StartupTip {
        height: auto;
        color: $text-muted;
        text-style: dim italic;
        padding: 0 1;
    }
    """

    def __init__(self, tip: str | None = None, **kwargs: Any) -> None:
        """Initialize the startup tip widget.

        Args:
            tip: Tip text to display. When omitted, one weighted tip is selected.
            **kwargs: Additional arguments passed to `Static`.
        """
        self.tip: str = tip if tip is not None else _pick_tip()
        # Styling (dim italic, muted color) is owned by DEFAULT_CSS, which
        # applies to the whole widget — no need to restyle the spans here.
        super().__init__(Content.assemble("Tip: ", self.tip), **kwargs)
