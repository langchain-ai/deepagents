"""Contract guard binding the goal-tool eval gates to middleware reality.

`tests/evals/test_goal_tools.py` gates behavior with `tool_not_called("get_rubric")`
/ `tool_not_called("get_goal")` / `tool_not_called("update_goal")`. `ToolNotCalled`
passes whenever nothing matches, so a `name` no tool actually emits passes
*vacuously* — if a goal tool is renamed (or a literal is mistyped), those gates
would keep passing while silently asserting nothing.

This deterministic unit test is the loud backstop for that: it pins the exact
names the eval gates reference to the tools `GoalToolsMiddleware` exposes, so a
rename breaks the fast suite immediately instead of rotting the eval gate. Keep
this set in sync with the `tool_not_called(...)` literals in
`test_goal_tools.py`.
"""

from __future__ import annotations

from deepagents_code.goal_tools import GoalToolsMiddleware

# The literals the goal-tool eval gates depend on. Must stay in lockstep with
# the `tool_not_called(...)` calls in `tests/evals/test_goal_tools.py`.
GATED_GOAL_TOOL_NAMES = frozenset({"get_rubric", "get_goal", "update_goal"})


def test_gated_goal_tool_names_exist_on_middleware() -> None:
    """Every eval-gated tool name must be a real `GoalToolsMiddleware` tool."""
    actual = {tool.name for tool in GoalToolsMiddleware().tools}
    missing = GATED_GOAL_TOOL_NAMES - actual
    assert not missing, (
        f"Eval gates in test_goal_tools.py reference tool names that "
        f"GoalToolsMiddleware no longer exposes: {sorted(missing)}. A "
        f"`tool_not_called(...)` gate on a nonexistent name passes vacuously — "
        f"rename the gate literals to match {sorted(actual)}."
    )
