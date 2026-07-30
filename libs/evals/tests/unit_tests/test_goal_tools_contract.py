"""Contract guard binding goal-tool eval gates to middleware reality."""

from deepagents_code.goal_tools import GOAL_TOOL_NAMES, GoalToolsMiddleware


def test_gated_goal_tool_names_match_middleware() -> None:
    actual = frozenset(tool.name for tool in GoalToolsMiddleware().tools)
    assert actual == GOAL_TOOL_NAMES
