"""Contract guard binding goal-tool eval gates to middleware reality."""

from deepagents_code.goal_tools import (
    GOAL_TOOL_NAMES,
    REMOVED_GOAL_TOOL_NAMES,
    GoalToolsMiddleware,
)


def test_gated_goal_tool_names_match_middleware() -> None:
    actual = frozenset(tool.name for tool in GoalToolsMiddleware().tools)
    assert actual == GOAL_TOOL_NAMES


def test_removed_read_tools_are_not_registered() -> None:
    """The removed read tools must not come back under the same names.

    Re-registering one would make the injected goal-state notice's "do not call
    any goal or rubric read tool" instruction wrong, and would silently undo the
    tool-schema reduction the notice exists to pay for.
    """
    actual = frozenset(tool.name for tool in GoalToolsMiddleware().tools)

    assert not actual & REMOVED_GOAL_TOOL_NAMES
    # The two sets describe disjoint eras and must never overlap.
    assert not GOAL_TOOL_NAMES & REMOVED_GOAL_TOOL_NAMES
