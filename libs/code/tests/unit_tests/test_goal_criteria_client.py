"""Tests for the TUI boundary of server-side goal criteria generation."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

from deepagents_code.app import DeepAgentsApp
from deepagents_code.tui.widgets.messages import ErrorMessage

if TYPE_CHECKING:
    from deepagents_code.goal_rubric import GoalCriteriaRequest


def _app(*, supports_goal_criteria: bool = True) -> DeepAgentsApp:
    agent = MagicMock()
    agent.channels = (
        {"goal_criteria_request": object()} if supports_goal_criteria else {}
    )
    app = DeepAgentsApp(agent=agent, thread_id="thread-1")
    app._ui_adapter = MagicMock()
    app._session_state = MagicMock()
    return app


def test_cancelling_goal_does_not_reject_unrelated_approval() -> None:
    app = _app()
    worker = MagicMock()
    approval = MagicMock()
    app._goal_proposal_worker = worker
    app._pending_approval_widget = approval

    app._cancel_goal_proposal_worker()

    approval.action_select_reject.assert_not_called()
    worker.cancel.assert_called_once_with()


async def test_tui_submits_typed_request_through_normal_agent_runner() -> None:
    app = _app()
    run = AsyncMock()
    request: GoalCriteriaRequest = {
        "request_id": "request-1",
        "kind": "create",
        "objective": "ship it",
    }

    with patch.object(app, "_run_agent_task", run):
        await app._run_goal_criteria_request(request)

    run.assert_awaited_once_with(
        "",
        graph_input={
            "messages": [],
            "goal_criteria_request": request,
        },
    )
    assert app._agent_running is True


async def test_tui_rejects_local_agent_without_criteria_middleware() -> None:
    app = _app(supports_goal_criteria=False)
    run = AsyncMock()
    mount = AsyncMock()
    request: GoalCriteriaRequest = {
        "request_id": "request-unsupported",
        "kind": "create",
        "objective": "ship it",
    }

    with (
        patch.object(app, "_run_agent_task", run),
        patch.object(app, "_mount_message", mount),
    ):
        await app._run_goal_criteria_request(request)

    run.assert_not_awaited()
    mount.assert_awaited_once()
    assert mount.await_args is not None
    message = mount.await_args.args[0]
    assert isinstance(message, ErrorMessage)
    assert "does not support goal criteria generation" in str(message._content)


async def test_criteria_run_forwards_profile_override_context() -> None:
    app = _app()
    app._model_override = "test:switched"
    app._model_params_override = {"temperature": 0}
    app._profile_override = {"max_input_tokens": 180_000}
    execute = AsyncMock()

    with (
        patch(
            "deepagents_code.tui.textual_adapter.execute_task_textual",
            execute,
        ),
        patch.object(app, "_cleanup_agent_task", new_callable=AsyncMock),
    ):
        await app._run_agent_task(
            "",
            graph_input={
                "messages": [],
                "goal_criteria_request": {
                    "request_id": "request-profile",
                    "kind": "create",
                    "objective": "ship it",
                },
            },
        )

    assert execute.await_args is not None
    context = execute.await_args.kwargs["context"]
    assert context["model"] == "test:switched"
    assert context["model_params"] == {"temperature": 0}
    assert context["profile_overrides"] == {"max_input_tokens": 180_000}


async def test_create_request_contains_data_not_a_model_prompt() -> None:
    app = _app()
    submit = AsyncMock()

    with (
        patch.object(app, "_run_goal_criteria_request", submit),
        patch("deepagents_code.app.uuid.uuid4") as uuid4,
    ):
        uuid4.return_value.hex = "request-2"
        await app._propose_goal_rubric(
            "ship it",
            feedback="make it concrete",
            previous_criteria="- old",
        )

    submit.assert_awaited_once_with(
        {
            "request_id": "request-2",
            "kind": "create",
            "objective": "ship it",
            "feedback": "make it concrete",
            "previous_criteria": "- old",
        }
    )


async def test_amendment_request_contains_current_state_and_feedback() -> None:
    app = _app()
    app._active_goal = "ship login"
    app._active_rubric = "- passwords work"
    submit = AsyncMock()

    with (
        patch.object(app, "_run_goal_criteria_request", submit),
        patch("deepagents_code.app.uuid.uuid4") as uuid4,
    ):
        uuid4.return_value.hex = "request-3"
        await app._propose_goal_amendment("add passkeys")

    submit.assert_awaited_once_with(
        {
            "request_id": "request-3",
            "kind": "amend",
            "objective": "ship login",
            "criteria": "- passwords work",
            "feedback": "add passkeys",
        }
    )


async def test_criteria_request_requires_a_running_server() -> None:
    app = _app()
    app._session_state = None  # server prerequisites unmet
    mount = AsyncMock()
    run = AsyncMock()
    request: GoalCriteriaRequest = {
        "request_id": "request-no-server",
        "kind": "create",
        "objective": "ship it",
    }

    with (
        patch.object(app, "_mount_message", mount),
        patch.object(app, "_run_agent_task", run),
    ):
        await app._run_goal_criteria_request(request)

    run.assert_not_awaited()
    mount.assert_awaited_once()
    await_args = mount.await_args
    assert await_args is not None
    message = await_args.args[0]
    assert isinstance(message, ErrorMessage)
    assert "requires the Deep Agents Code server" in str(message._content)


async def test_failed_criteria_turn_shows_actionable_message() -> None:
    app = _app()
    mount = AsyncMock()
    execute = AsyncMock(side_effect=RuntimeError("An internal error occurred"))

    with (
        patch(
            "deepagents_code.tui.textual_adapter.execute_task_textual",
            execute,
        ),
        patch.object(app, "_cleanup_agent_task", new_callable=AsyncMock),
        patch.object(app, "_mount_message", mount),
        patch(
            "deepagents_code.app._langsmith_gateway_key_mismatch",
            return_value=None,
        ),
    ):
        await app._run_agent_task(
            "",
            graph_input={
                "messages": [],
                "goal_criteria_request": {
                    "request_id": "request-fail",
                    "kind": "create",
                    "objective": "ship it",
                },
            },
        )

    # The redactable server exception text is replaced by a self-contained,
    # actionable message so remote deployments still guide the user.
    assert mount.await_args is not None
    body = str(mount.await_args.args[0]._content)
    assert "Could not generate acceptance criteria" in body
    assert "internal error occurred" not in body


async def test_goal_submission_never_constructs_a_model_client_side() -> None:
    app = _app()
    execute = AsyncMock()

    with (
        patch(
            "deepagents_code.tui.textual_adapter.execute_task_textual",
            execute,
        ),
        patch.object(app, "_cleanup_agent_task", new_callable=AsyncMock),
        patch("deepagents_code.config.create_model") as create_model,
        patch("deepagents_code.goal_rubric.create_goal_criteria_agent") as make_agent,
        patch("deepagents_code.app.uuid.uuid4") as uuid4,
    ):
        uuid4.return_value.hex = "request-behavioral"
        await app._propose_goal_rubric("add refresh tokens")

    # The client submits a typed request through the normal graph stream...
    assert execute.await_args is not None
    graph_input = execute.await_args.kwargs["graph_input"]
    assert graph_input["goal_criteria_request"]["objective"] == "add refresh tokens"
    # ...and never constructs or wires a model client-side.
    create_model.assert_not_called()
    make_agent.assert_not_called()
