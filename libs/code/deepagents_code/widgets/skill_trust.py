"""Trust prompt for skills that resolve outside trusted skill directories.

When a `/skill:<name>` invocation reads a `SKILL.md` whose resolved path (via
symlink) falls outside every trusted skill root, `load_skill_content` refuses
the read. Rather than forcing the user to quit, edit env/config, and relaunch,
this non-blocking modal asks for an in-the-moment decision. Allowing persists
the resolved target directory to the skill trust store.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import Static

if TYPE_CHECKING:
    from textual.app import ComposeResult


class SkillTrustScreen(ModalScreen[bool]):
    """Approval overlay for a skill resolving outside trusted directories.

    Dismisses with `True` when the user allows the resolved target and `False`
    when the user declines. Esc is treated as deny so the user is never forced
    into reading from an untrusted location they did not explicitly choose.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "confirm", "Allow", show=False, priority=True),
        Binding("escape", "cancel", "Deny", show=False, priority=True),
    ]

    CSS = """
    SkillTrustScreen {
        align: center middle;
    }

    SkillTrustScreen > Vertical {
        width: 72;
        max-width: 90%;
        height: auto;
        background: $surface;
        border: solid $warning;
        padding: 1 2;
    }

    SkillTrustScreen .skill-trust-title {
        text-style: bold;
        color: $warning;
        text-align: center;
        margin-bottom: 1;
    }

    SkillTrustScreen .skill-trust-body {
        height: auto;
        color: $text;
        margin-bottom: 1;
    }

    SkillTrustScreen .skill-trust-help {
        height: 1;
        color: $text-muted;
        text-style: italic;
        text-align: center;
    }
    """

    def __init__(self, skill_name: str, target_dir: str) -> None:
        """Initialize the prompt.

        Args:
            skill_name: Name of the skill being invoked.
            target_dir: Resolved directory the skill's `SKILL.md` lives in.
        """
        super().__init__()
        self._skill_name = skill_name
        self._target_dir = target_dir

    def compose(self) -> ComposeResult:
        """Compose the skill trust dialog.

        Yields:
            Title, body, and help-row widgets parented inside a `Vertical`.
        """
        with Vertical():
            yield Static(
                "Allow skill from outside trusted directories?",
                classes="skill-trust-title",
                markup=False,
            )
            yield Static(
                Content.from_markup(
                    "Skill [bold]$name[/bold] resolves via symlink to "
                    "[bold]$dir[/bold], outside your trusted skill "
                    "directories. Allowing reads instructions from there and "
                    "remembers this location for future sessions.",
                    name=self._skill_name,
                    dir=self._target_dir,
                ),
                classes="skill-trust-body",
                markup=False,
            )
            yield Static(
                "Enter to allow, Esc to deny",
                classes="skill-trust-help",
                markup=False,
            )

    def action_confirm(self) -> None:
        """Dismiss with `True`."""
        self.dismiss(True)

    def action_cancel(self) -> None:
        """Dismiss with `False`.

        The method name must stay `cancel`: the app owns a priority `escape`
        binding that, for an active `ModalScreen`, dispatches to
        `action_cancel` if present and otherwise falls through to
        `dismiss(None)`. Renaming this would silently regress Esc to a
        `None` dismiss instead of an explicit deny.
        """
        self.dismiss(False)
