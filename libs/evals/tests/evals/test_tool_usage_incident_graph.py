"""Eval tests for incident-management graph tool usage.

A synthetic operational incident-management domain with many entities and tools.
The agent receives only graph lookup/search tools and must compose them to answer
questions efficiently.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from deepagents import create_deep_agent
from langchain_quickjs import CodeInterpreterMiddleware

from deepagents_evals.mock_tools.incident_graph import (
    INCIDENT_GRAPH_TOOLS,
    incident_graph_tool_error_middleware as _incident_graph_tool_error_middleware,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

from tests.evals.utils import (
    TrajectoryScorer,
    final_text_contains,
    run_agent_async,
    tool_call,
)

pytestmark = [
    pytest.mark.eval_category("tool_use"),
    pytest.mark.eval_tier("baseline"),
    pytest.mark.repl("quickjs"),
]


def _create_agent(model: BaseChatModel, repl_name: str | None):
    """Create an agent implementation."""
    middleware = [_incident_graph_tool_error_middleware]
    tools = None
    if repl_name == "quickjs":
        middleware.append(CodeInterpreterMiddleware(ptc=INCIDENT_GRAPH_TOOLS))
    elif repl_name is None:
        tools = INCIDENT_GRAPH_TOOLS
    else:
        msg = f'Unknown repl_name "{repl_name}"'
        raise ValueError(msg)
    return create_deep_agent(model=model, tools=tools, middleware=middleware)


@pytest.fixture
def agent(model: BaseChatModel, repl_name: str | None):
    """Get an agent implementation."""
    return _create_agent(model, repl_name)


@pytest.mark.langsmith
async def test_single_tool_list_incident_ids(agent, model: BaseChatModel) -> None:
    await run_agent_async(
        agent,
        model=model,
        query="What are all the incident IDs in the system?",
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("41017"),
            final_text_contains("41029"),
            final_text_contains("41043"),
            final_text_contains("41058"),
        )
        .expect(
            agent_steps=2,
            tool_call_requests=1,
            tool_calls=[tool_call(name="list_incident_ids", step=1)],
        ),
    )


@pytest.mark.langsmith
async def test_two_tools_current_incident_service_name(agent, model: BaseChatModel) -> None:
    await run_agent_async(
        agent,
        model=model,
        query="What service is affected by the current incident?",
        scorer=TrajectoryScorer()
        .success(final_text_contains("payments-api"))
        .expect(
            agent_steps=4,
            tool_call_requests=3,
            tool_calls=[
                tool_call(name="get_current_incident_id", step=1),
                tool_call(
                    name="get_incident_service", step=2, args_contains={"incident_id": 41017}
                ),
                tool_call(name="get_service_name", step=3, args_contains={"service_id": 8401}),
            ],
        ),
    )


@pytest.mark.langsmith
async def test_three_tools_find_service_owner_team(agent, model: BaseChatModel) -> None:
    await run_agent_async(
        agent,
        model=model,
        query="Which team owns checkout-web?",
        scorer=TrajectoryScorer()
        .success(final_text_contains("Checkout Experience"))
        .expect(
            agent_steps=4,
            tool_call_requests=3,
            tool_calls=[
                tool_call(
                    name="find_services_by_name", step=1, args_contains={"name": "checkout-web"}
                ),
                tool_call(name="get_service_team", step=2, args_contains={"service_id": 8514}),
                tool_call(name="get_team_name", step=3, args_contains={"team_id": 562}),
            ],
        ),
    )


@pytest.mark.langsmith
async def test_multi_question_current_incident_service_and_incident_oncall(
    agent,
    model: BaseChatModel,
) -> None:
    await run_agent_async(
        agent,
        model=model,
        query=(
            "Answer both questions: 1) What service is affected by the current incident? "
            "2) Who is the on-call engineer for incident 41029?"
        ),
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("payments-api"),
            final_text_contains("Cara Singh"),
        )
        .expect(
            agent_steps=5,
            tool_call_requests=7,
            tool_calls=[
                tool_call(name="get_current_incident_id"),
                tool_call(name="get_incident_service", args_contains={"incident_id": 41017}),
                tool_call(name="get_incident_service", args_contains={"incident_id": 41029}),
                tool_call(name="get_service_team", args_contains={"service_id": 8514}),
                tool_call(name="get_team_oncall_engineer", args_contains={"team_id": 562}),
                tool_call(name="get_service_name", args_contains={"service_id": 8401}),
                tool_call(name="get_engineer_name", args_contains={"engineer_id": 7381}),
            ],
        ),
    )


@pytest.mark.langsmith
async def test_multi_question_incident_oncall_and_incident_environment(
    agent,
    model: BaseChatModel,
) -> None:
    await run_agent_async(
        agent,
        model=model,
        query=(
            "Answer both questions: 1) Who is the on-call engineer for incident 41029? "
            "2) What environment and region is incident 41058 running in?"
        ),
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("Cara Singh"),
            final_text_contains("staging", case_insensitive=True),
            final_text_contains("us-west-2"),
        )
        .expect(
            agent_steps=5,
            tool_call_requests=8,
            tool_calls=[
                tool_call(name="get_incident_service", args_contains={"incident_id": 41029}),
                tool_call(name="get_service_team", args_contains={"service_id": 8514}),
                tool_call(name="get_team_oncall_engineer", args_contains={"team_id": 562}),
                tool_call(name="get_engineer_name", args_contains={"engineer_id": 7381}),
                tool_call(name="get_incident_service", args_contains={"incident_id": 41058}),
                tool_call(name="get_service_environment", args_contains={"service_id": 8799}),
                tool_call(name="get_environment_name", args_contains={"environment_id": 442}),
                tool_call(name="get_environment_region", args_contains={"environment_id": 442}),
            ],
        ),
    )


@pytest.mark.langsmith
async def test_multi_question_incident_oncall_and_service_with_most_firing_alerts(
    agent,
    model: BaseChatModel,
) -> None:
    await run_agent_async(
        agent,
        model=model,
        query=(
            "Answer both questions: 1) Who is on call for incident 41029? "
            "2) Which service currently has the most firing alerts?"
        ),
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("Cara Singh"),
            final_text_contains("payments-api"),
            final_text_contains("2", case_insensitive=False),
        )
        .expect(
            agent_steps=6,
            tool_call_requests=13,
            tool_calls=[
                tool_call(name="get_incident_service", args_contains={"incident_id": 41029}),
                tool_call(name="get_service_team", args_contains={"service_id": 8514}),
                tool_call(name="get_team_oncall_engineer", args_contains={"team_id": 562}),
                tool_call(name="get_engineer_name", args_contains={"engineer_id": 7381}),
                tool_call(name="get_service_name", args_contains={"service_id": 8401}),
            ],
        ),
    )


@pytest.mark.langsmith
async def test_multi_question_three_independent_simple_lookups(agent, model: BaseChatModel) -> None:
    await run_agent_async(
        agent,
        model=model,
        query=(
            "Answer all three questions: 1) What is the severity of incident 41017? "
            "2) What is the default branch for repo 9217? "
            "3) What is the region for environment 442?"
        ),
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("sev1", case_insensitive=True),
            final_text_contains("main"),
            final_text_contains("us-west-2"),
        )
        .expect(
            agent_steps=2,
            tool_call_requests=3,
            tool_calls=[
                # All three lookups are independent, so the optimal trajectory issues them
                # together in one tool-calling step and then answers in the final step.
                tool_call(
                    name="get_incident_severity", step=1, args_contains={"incident_id": 41017}
                ),
                tool_call(name="get_repo_default_branch", step=1, args_contains={"repo_id": 9217}),
                tool_call(
                    name="get_environment_region", step=1, args_contains={"environment_id": 442}
                ),
            ],
        ),
    )


@pytest.mark.langsmith
async def test_four_tools_incident_to_oncall_name(agent, model: BaseChatModel) -> None:
    await run_agent_async(
        agent,
        model=model,
        query="Who is the on-call engineer for incident 41029?",
        scorer=TrajectoryScorer()
        .success(final_text_contains("Cara Singh"))
        .expect(
            agent_steps=5,
            tool_call_requests=4,
            tool_calls=[
                tool_call(
                    name="get_incident_service", step=1, args_contains={"incident_id": 41029}
                ),
                tool_call(name="get_service_team", step=2, args_contains={"service_id": 8514}),
                tool_call(name="get_team_oncall_engineer", step=3, args_contains={"team_id": 562}),
                tool_call(name="get_engineer_name", step=4, args_contains={"engineer_id": 7381}),
            ],
        ),
    )


@pytest.mark.langsmith
async def test_four_tools_service_runbook_url(agent, model: BaseChatModel) -> None:
    await run_agent_async(
        agent,
        model=model,
        query="What is the runbook URL for payments-api?",
        scorer=TrajectoryScorer()
        .success(final_text_contains("https://runbooks.example.com/payments-api-5xx"))
        .expect(
            agent_steps=4,
            tool_call_requests=3,
            tool_calls=[
                tool_call(
                    name="find_services_by_name", step=1, args_contains={"name": "payments-api"}
                ),
                tool_call(name="get_service_runbook", step=2, args_contains={"service_id": 8401}),
                tool_call(name="get_runbook_url", step=3, args_contains={"runbook_id": 12041}),
            ],
        ),
    )


@pytest.mark.langsmith
async def test_five_tools_incident_latest_deploy_and_repo(agent, model: BaseChatModel) -> None:
    await run_agent_async(
        agent,
        model=model,
        query="For incident 41017, what repo was most recently deployed and what version was it?",
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("payments-service"),
            final_text_contains("payments-api@2024.08.12.1"),
        )
        .expect(
            agent_steps=4,
            tool_call_requests=5,
            tool_calls=[
                tool_call(
                    name="get_incident_service", step=1, args_contains={"incident_id": 41017}
                ),
                tool_call(name="get_service_repo", step=2, args_contains={"service_id": 8401}),
                tool_call(
                    name="get_latest_deploy_for_service", step=2, args_contains={"service_id": 8401}
                ),
                tool_call(name="get_repo_name", step=3, args_contains={"repo_id": 9104}),
                tool_call(name="get_deploy_version", step=3, args_contains={"deploy_id": 66011}),
            ],
        ),
    )


@pytest.mark.langsmith
async def test_five_tools_incident_environment_name_and_region(agent, model: BaseChatModel) -> None:
    await run_agent_async(
        agent,
        model=model,
        query="What environment and region is incident 41058 running in?",
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("staging"),
            final_text_contains("us-west-2"),
        )
        .expect(
            agent_steps=4,
            tool_call_requests=4,
            tool_calls=[
                tool_call(
                    name="get_incident_service", step=1, args_contains={"incident_id": 41058}
                ),
                tool_call(
                    name="get_service_environment", step=2, args_contains={"service_id": 8799}
                ),
                tool_call(
                    name="get_environment_name", step=3, args_contains={"environment_id": 442}
                ),
                tool_call(
                    name="get_environment_region", step=3, args_contains={"environment_id": 442}
                ),
            ],
        ),
    )


@pytest.mark.langsmith
async def test_five_tools_service_dependency_names_parallel(agent, model: BaseChatModel) -> None:
    await run_agent_async(
        agent,
        model=model,
        query="What services does checkout-web depend on? Give me the dependency names.",
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("payments-api"),
            final_text_contains("identity-api"),
        )
        .expect(
            agent_steps=4,
            tool_call_requests=4,
            tool_calls=[
                tool_call(
                    name="find_services_by_name", step=1, args_contains={"name": "checkout-web"}
                ),
                tool_call(
                    name="list_service_dependencies", step=2, args_contains={"service_id": 8514}
                ),
                tool_call(name="get_service_name", step=3, args_contains={"service_id": 8401}),
                tool_call(name="get_service_name", step=3, args_contains={"service_id": 8627}),
            ],
        ),
    )


@pytest.mark.langsmith
async def test_five_tools_service_alert_names_parallel(agent, model: BaseChatModel) -> None:
    await run_agent_async(
        agent,
        model=model,
        query="List the alert names for payments-api.",
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("payments-api 5xx rate"),
            final_text_contains("payments-api latency p95"),
        )
        .expect(
            agent_steps=4,
            tool_call_requests=4,
            tool_calls=[
                tool_call(
                    name="find_services_by_name", step=1, args_contains={"name": "payments-api"}
                ),
                tool_call(
                    name="list_service_alert_ids", step=2, args_contains={"service_id": 8401}
                ),
                tool_call(name="get_alert_name", step=3, args_contains={"alert_id": 55101}),
                tool_call(name="get_alert_name", step=3, args_contains={"alert_id": 55114}),
            ],
        ),
    )


@pytest.mark.langsmith
async def test_six_tools_current_incident_oncall_name_and_email(
    agent, model: BaseChatModel
) -> None:
    await run_agent_async(
        agent,
        model=model,
        query="For the current incident, who is on call and what is their email address?",
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("Ben Ortiz"),
            final_text_contains("ben@ops.example.com"),
        )
        .expect(
            agent_steps=6,
            tool_call_requests=6,
            tool_calls=[
                tool_call(name="get_current_incident_id", step=1),
                tool_call(
                    name="get_incident_service", step=2, args_contains={"incident_id": 41017}
                ),
                tool_call(name="get_service_team", step=3, args_contains={"service_id": 8401}),
                tool_call(name="get_team_oncall_engineer", step=4, args_contains={"team_id": 481}),
                tool_call(name="get_engineer_name", step=5, args_contains={"engineer_id": 7243}),
                tool_call(name="get_engineer_email", step=5, args_contains={"engineer_id": 7243}),
            ],
        ),
    )


@pytest.mark.langsmith
async def test_six_tools_service_repo_and_branch(agent, model: BaseChatModel) -> None:
    await run_agent_async(
        agent,
        model=model,
        query="What repository backs identity-api and what is its default branch?",
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("identity-service"),
            final_text_contains("main"),
        )
        .expect(
            agent_steps=4,
            tool_call_requests=4,
            tool_calls=[
                tool_call(
                    name="find_services_by_name", step=1, args_contains={"name": "identity-api"}
                ),
                tool_call(name="get_service_repo", step=2, args_contains={"service_id": 8627}),
                tool_call(name="get_repo_name", step=3, args_contains={"repo_id": 9346}),
                tool_call(name="get_repo_default_branch", step=3, args_contains={"repo_id": 9346}),
            ],
        ),
    )


@pytest.mark.langsmith
async def test_six_tools_incident_title_severity_and_status(agent, model: BaseChatModel) -> None:
    await run_agent_async(
        agent,
        model=model,
        query="For incident 41043, tell me its title, severity, and status.",
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("Identity login error burst"),
            final_text_contains("sev2", case_insensitive=True),
            final_text_contains("resolved", case_insensitive=True),
        )
        .expect(
            agent_steps=2,
            tool_call_requests=3,
            tool_calls=[
                tool_call(name="get_incident_title", args_contains={"incident_id": 41043}),
                tool_call(name="get_incident_severity", args_contains={"incident_id": 41043}),
                tool_call(name="get_incident_status", args_contains={"incident_id": 41043}),
            ],
        ),
    )


@pytest.mark.langsmith
async def test_six_tools_current_incident_metrics_parallel(agent, model: BaseChatModel) -> None:
    await run_agent_async(
        agent,
        model=model,
        query="For the current incident's service, what are the current error_rate and latency_p95 metrics?",
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("12.4%"),
            final_text_contains("1.8s"),
        )
        .expect(
            agent_steps=4,
            tool_call_requests=4,
            tool_calls=[
                tool_call(name="get_current_incident_id", step=1),
                tool_call(
                    name="get_incident_service", step=2, args_contains={"incident_id": 41017}
                ),
                tool_call(
                    name="get_metric_value",
                    step=3,
                    args_contains={"service_id": 8401, "metric_name": "error_rate"},
                ),
                tool_call(
                    name="get_metric_value",
                    step=3,
                    args_contains={"service_id": 8401, "metric_name": "latency_p95"},
                ),
            ],
        ),
    )


@pytest.mark.langsmith
async def test_aggregation_active_incident_count_by_team(agent, model: BaseChatModel) -> None:
    await run_agent_async(
        agent,
        model=model,
        query=(
            "How many active incidents belong to each team, and which team has the most active incidents? "
            "Please include the team names and counts."
        ),
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("Payments Platform"),
            final_text_contains("Checkout Experience"),
            final_text_contains("1"),
        )
        .expect(
            agent_steps=6,
            tool_call_requests=13,
            tool_calls=[
                tool_call(name="list_incident_ids", step=1),
                tool_call(name="get_incident_status", step=2, args_contains={"incident_id": 41017}),
                tool_call(name="get_incident_status", step=2, args_contains={"incident_id": 41029}),
                tool_call(name="get_incident_status", step=2, args_contains={"incident_id": 41043}),
                tool_call(name="get_incident_status", step=2, args_contains={"incident_id": 41058}),
                tool_call(
                    name="get_incident_service", step=3, args_contains={"incident_id": 41017}
                ),
                tool_call(
                    name="get_incident_service", step=3, args_contains={"incident_id": 41029}
                ),
                tool_call(
                    name="get_incident_service", step=3, args_contains={"incident_id": 41058}
                ),
                tool_call(name="get_service_team", step=4, args_contains={"service_id": 8401}),
                tool_call(name="get_service_team", step=4, args_contains={"service_id": 8514}),
                tool_call(name="get_service_team", step=4, args_contains={"service_id": 8799}),
                tool_call(name="get_team_name", step=5, args_contains={"team_id": 481}),
                tool_call(name="get_team_name", step=5, args_contains={"team_id": 562}),
            ],
        ),
    )


@pytest.mark.langsmith
async def test_comparison_active_incident_most_dependencies(agent, model: BaseChatModel) -> None:
    await run_agent_async(
        agent,
        model=model,
        query=(
            "Among the active incidents, which incident affects the service with the most dependencies? "
            "Return the incident ID, incident title, service name, and dependency count."
        ),
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("41029"),
            final_text_contains("Checkout page latency spike"),
            final_text_contains("checkout-web"),
            final_text_contains("2"),
        )
        .expect(
            agent_steps=5,
            tool_call_requests=14,
            tool_calls=[
                tool_call(name="list_incident_ids", step=1),
                tool_call(name="get_incident_status", step=2, args_contains={"incident_id": 41017}),
                tool_call(name="get_incident_status", step=2, args_contains={"incident_id": 41029}),
                tool_call(name="get_incident_status", step=2, args_contains={"incident_id": 41043}),
                tool_call(name="get_incident_status", step=2, args_contains={"incident_id": 41058}),
                tool_call(
                    name="get_incident_service", step=3, args_contains={"incident_id": 41017}
                ),
                tool_call(
                    name="get_incident_service", step=3, args_contains={"incident_id": 41029}
                ),
                tool_call(
                    name="get_incident_service", step=3, args_contains={"incident_id": 41058}
                ),
                tool_call(
                    name="list_service_dependencies", step=4, args_contains={"service_id": 8401}
                ),
                tool_call(
                    name="list_service_dependencies", step=4, args_contains={"service_id": 8514}
                ),
                tool_call(
                    name="list_service_dependencies", step=4, args_contains={"service_id": 8799}
                ),
                tool_call(name="get_incident_title", step=5, args_contains={"incident_id": 41029}),
                tool_call(name="get_service_name", step=5, args_contains={"service_id": 8514}),
            ],
        ),
    )


@pytest.mark.langsmith
async def test_latest_selection_active_incident_most_recent_deploy(
    agent, model: BaseChatModel
) -> None:
    await run_agent_async(
        agent,
        model=model,
        query=(
            "Across the services involved in active incidents, which service had the most recent deploy? "
            "Return the service name, repo name, deploy version, and deploy timestamp."
        ),
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("checkout-web"),
            final_text_contains("checkout-frontend"),
            final_text_contains("checkout-web@2024.08.12.3"),
            final_text_contains("2024-08-12T09:05:00Z"),
        )
        .expect(
            agent_steps=6,
            tool_call_requests=15,
            tool_calls=[
                tool_call(name="list_incident_ids", step=1),
                tool_call(name="get_incident_status", step=2, args_contains={"incident_id": 41017}),
                tool_call(name="get_incident_status", step=2, args_contains={"incident_id": 41029}),
                tool_call(name="get_incident_status", step=2, args_contains={"incident_id": 41043}),
                tool_call(name="get_incident_status", step=2, args_contains={"incident_id": 41058}),
                tool_call(
                    name="get_incident_service", step=3, args_contains={"incident_id": 41017}
                ),
                tool_call(
                    name="get_incident_service", step=3, args_contains={"incident_id": 41029}
                ),
                tool_call(
                    name="get_incident_service", step=3, args_contains={"incident_id": 41058}
                ),
                tool_call(
                    name="get_latest_deploy_for_service", step=4, args_contains={"service_id": 8401}
                ),
                tool_call(
                    name="get_latest_deploy_for_service", step=4, args_contains={"service_id": 8514}
                ),
                tool_call(
                    name="get_latest_deploy_for_service", step=4, args_contains={"service_id": 8799}
                ),
                tool_call(name="get_deploy_timestamp", step=5, args_contains={"deploy_id": 66011}),
                tool_call(name="get_deploy_timestamp", step=5, args_contains={"deploy_id": 66037}),
                tool_call(name="get_deploy_timestamp", step=5, args_contains={"deploy_id": 66059}),
                tool_call(name="get_service_name", step=5, args_contains={"service_id": 8514}),
                tool_call(name="get_service_repo", step=5, args_contains={"service_id": 8514}),
                tool_call(name="get_deploy_version", step=5, args_contains={"deploy_id": 66037}),
            ],
        ),
    )


@pytest.mark.langsmith
async def test_metric_ranking_active_incident_highest_latency(agent, model: BaseChatModel) -> None:
    await run_agent_async(
        agent,
        model=model,
        query=(
            "Among the active incidents affecting customer-facing services with a latency_p95 metric, "
            "which incident is tied to the service with the highest latency_p95, and which team owns that service?"
        ),
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("41029"),
            final_text_contains("checkout-web"),
            final_text_contains("2.4s"),
            final_text_contains("Checkout Experience"),
        )
        .expect(
            agent_steps=7,
            tool_call_requests=12,
            tool_calls=[
                tool_call(name="list_incident_ids", step=1),
                tool_call(name="get_incident_status", step=2, args_contains={"incident_id": 41017}),
                tool_call(name="get_incident_status", step=2, args_contains={"incident_id": 41029}),
                tool_call(name="get_incident_status", step=2, args_contains={"incident_id": 41043}),
                tool_call(name="get_incident_status", step=2, args_contains={"incident_id": 41058}),
                tool_call(
                    name="get_incident_service", step=3, args_contains={"incident_id": 41017}
                ),
                tool_call(
                    name="get_incident_service", step=3, args_contains={"incident_id": 41029}
                ),
                tool_call(
                    name="get_incident_service", step=3, args_contains={"incident_id": 41058}
                ),
                tool_call(
                    name="get_metric_value",
                    step=4,
                    args_contains={"service_id": 8401, "metric_name": "latency_p95"},
                ),
                tool_call(
                    name="get_metric_value",
                    step=4,
                    args_contains={"service_id": 8514, "metric_name": "latency_p95"},
                ),
                tool_call(name="get_service_name", step=5, args_contains={"service_id": 8514}),
                tool_call(name="get_service_team", step=5, args_contains={"service_id": 8514}),
                tool_call(name="get_team_name", step=6, args_contains={"team_id": 562}),
            ],
        ),
    )


@pytest.mark.langsmith
async def test_alert_aggregation_service_with_most_firing_alerts(
    agent, model: BaseChatModel
) -> None:
    await run_agent_async(
        agent,
        model=model,
        query="Which service has the most firing alerts right now, and what are the names of those alerts?",
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("payments-api"),
            final_text_contains("payments-api 5xx rate"),
            final_text_contains("payments-api latency p95"),
            final_text_contains("2"),
        )
        .expect(
            agent_steps=6,
            tool_call_requests=16,
            tool_calls=[
                tool_call(
                    name="find_services_by_name", step=1, args_contains={"name": "payments-api"}
                ),
                tool_call(
                    name="find_services_by_name", step=1, args_contains={"name": "checkout-web"}
                ),
                tool_call(
                    name="find_services_by_name", step=1, args_contains={"name": "identity-api"}
                ),
                tool_call(
                    name="find_services_by_name", step=1, args_contains={"name": "analytics-worker"}
                ),
                tool_call(
                    name="list_service_alert_ids", step=2, args_contains={"service_id": 8401}
                ),
                tool_call(
                    name="list_service_alert_ids", step=2, args_contains={"service_id": 8514}
                ),
                tool_call(
                    name="list_service_alert_ids", step=2, args_contains={"service_id": 8627}
                ),
                tool_call(
                    name="list_service_alert_ids", step=2, args_contains={"service_id": 8799}
                ),
                tool_call(name="get_alert_status", step=3, args_contains={"alert_id": 55101}),
                tool_call(name="get_alert_status", step=3, args_contains={"alert_id": 55114}),
                tool_call(name="get_alert_status", step=3, args_contains={"alert_id": 55128}),
                tool_call(name="get_alert_status", step=3, args_contains={"alert_id": 55139}),
                tool_call(name="get_alert_status", step=3, args_contains={"alert_id": 55152}),
                tool_call(name="get_service_name", step=4, args_contains={"service_id": 8401}),
                tool_call(name="get_alert_name", step=5, args_contains={"alert_id": 55101}),
                tool_call(name="get_alert_name", step=5, args_contains={"alert_id": 55114}),
            ],
        ),
    )


@pytest.mark.langsmith
async def test_dependency_reasoning_active_incident_depending_on_identity_api(
    agent,
    model: BaseChatModel,
) -> None:
    await run_agent_async(
        agent,
        model=model,
        query=(
            "Which active incident affects a service that depends on identity-api, and who is the on-call engineer "
            "for the owning team? Include the engineer email too."
        ),
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("41029"),
            final_text_contains("Checkout page latency spike"),
            final_text_contains("Cara Singh"),
            final_text_contains("cara@ops.example.com"),
        )
        .expect(
            agent_steps=7,
            tool_call_requests=16,
            tool_calls=[
                tool_call(name="list_incident_ids", step=1),
                tool_call(name="get_incident_status", step=2, args_contains={"incident_id": 41017}),
                tool_call(name="get_incident_status", step=2, args_contains={"incident_id": 41029}),
                tool_call(name="get_incident_status", step=2, args_contains={"incident_id": 41043}),
                tool_call(name="get_incident_status", step=2, args_contains={"incident_id": 41058}),
                tool_call(
                    name="get_incident_service", step=3, args_contains={"incident_id": 41017}
                ),
                tool_call(
                    name="get_incident_service", step=3, args_contains={"incident_id": 41029}
                ),
                tool_call(
                    name="get_incident_service", step=3, args_contains={"incident_id": 41058}
                ),
                tool_call(
                    name="list_service_dependencies", step=4, args_contains={"service_id": 8401}
                ),
                tool_call(
                    name="list_service_dependencies", step=4, args_contains={"service_id": 8514}
                ),
                tool_call(
                    name="list_service_dependencies", step=4, args_contains={"service_id": 8799}
                ),
                tool_call(name="get_incident_title", step=5, args_contains={"incident_id": 41029}),
                tool_call(name="get_service_team", step=5, args_contains={"service_id": 8514}),
                tool_call(name="get_team_oncall_engineer", step=6, args_contains={"team_id": 562}),
                tool_call(name="get_engineer_name", step=7, args_contains={"engineer_id": 7381}),
                tool_call(name="get_engineer_email", step=7, args_contains={"engineer_id": 7381}),
            ],
        ),
    )
