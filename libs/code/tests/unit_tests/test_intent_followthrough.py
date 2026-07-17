"""Tests for the intent-follow-through middleware."""

from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from deepagents_code.intent_followthrough import (
    FOLLOW_THROUGH_REMINDER,
    IntentFollowThroughMiddleware,
    _decide,
)


def _runtime() -> SimpleNamespace:
    """Return a stand-in runtime; the middleware ignores it."""
    return SimpleNamespace(context=None)


class TestDecide:
    def test_plan_only_turn_with_url_is_re_driven(self) -> None:
        """A plan-only reply to a URL request is re-driven to a tool call."""
        state = {
            "messages": [
                HumanMessage(
                    content="why'd this fail? https://github.com/o/r/actions/runs/1"
                ),
                AIMessage(
                    content="I'm checking the PR context and the failed job's logs..."
                ),
            ]
        }
        result = _decide(state)
        assert result is not None
        assert result["jump_to"] == "model"
        assert result["messages"][0].content == FOLLOW_THROUGH_REMINDER

    def test_pr_reference_triggers_re_drive(self) -> None:
        """An issue/PR number reference counts as an external resource."""
        state = {
            "messages": [
                HumanMessage(content="reword issue #42 to be clearer"),
                AIMessage(content="Let me review the issue wording first."),
            ]
        }
        assert _decide(state) is not None

    def test_git_action_triggers_re_drive(self) -> None:
        """A git action verb counts as needing tool work."""
        state = {
            "messages": [
                HumanMessage(content="commit and push these changes"),
                AIMessage(content="I'll run the commit now."),
            ]
        }
        assert _decide(state) is not None

    def test_no_external_resource_is_left_alone(self) -> None:
        """A plan-only reply to a self-contained request is not re-driven."""
        state = {
            "messages": [
                HumanMessage(content="write me a haiku about autumn"),
                AIMessage(content="Let me review the request..."),
            ]
        }
        assert _decide(state) is None

    def test_tool_call_present_is_left_alone(self) -> None:
        """A turn that already called a tool is never re-driven."""
        ai = AIMessage(
            content="I'm checking the logs",
            tool_calls=[{"name": "shell", "args": {}, "id": "1"}],
        )
        state = {"messages": [HumanMessage(content="why did pr #3 fail?"), ai]}
        assert _decide(state) is None

    def test_completed_answer_is_left_alone(self) -> None:
        """A completed answer without an intent statement is not re-driven."""
        state = {
            "messages": [
                HumanMessage(content="commit this"),
                AIMessage(content="Done. Committed as abc123."),
            ]
        }
        assert _decide(state) is None

    def test_does_not_loop_after_one_nudge(self) -> None:
        """A second plan-only reply after a reminder is not re-driven again."""
        state = {
            "messages": [
                HumanMessage(content="fix pr #12"),
                AIMessage(content="I will review the diff."),
                HumanMessage(content=FOLLOW_THROUGH_REMINDER),
                AIMessage(content="I'm checking the diff again..."),
            ]
        }
        assert _decide(state) is None


class TestMiddlewareHooks:
    def test_after_model_re_drives(self) -> None:
        """The sync hook re-drives a plan-only external-resource turn."""
        mw = IntentFollowThroughMiddleware()
        state = {
            "messages": [
                HumanMessage(content="why'd this fail? https://ci.example/run/9"),
                AIMessage(content="I'm reading the failed job's logs..."),
            ]
        }
        result = mw.after_model(state, _runtime())
        assert result is not None
        assert result["jump_to"] == "model"

    async def test_aafter_model_re_drives(self) -> None:
        """The async hook mirrors the sync hook."""
        mw = IntentFollowThroughMiddleware()
        state = {
            "messages": [
                HumanMessage(content="why'd this fail? https://ci.example/run/9"),
                AIMessage(content="I'm reading the failed job's logs..."),
            ]
        }
        result = await mw.aafter_model(state, _runtime())
        assert result is not None
        assert result["jump_to"] == "model"
