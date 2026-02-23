import pytest
from textual.pilot import Pilot

from deepagents_cli.widgets.feedback_choice import FeedbackChoiceScreen


async def test_feedback_choice_screen_mounts():
    """Test that FeedbackChoiceScreen can be instantiated and mounted."""
    app = FeedbackChoiceScreen()
    assert app is not None
