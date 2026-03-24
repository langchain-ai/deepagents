"""Eval tests for incident-management graph tool usage.

A synthetic operational incident-management domain with many entities and tools.
The agent receives only graph lookup/search tools and must compose them to answer
questions efficiently.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypedDict, TypeVar, overload

import pytest
from deepagents import create_deep_agent
from langchain_core.tools import ToolException, tool

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

from tests.evals.utils import (
    TrajectoryScorer,
    final_text_contains,
    run_agent_async,
    tool_call,
)


class Engineer(TypedDict):
    id: int
    name: str
    email: str
    team_id: int


class Team(TypedDict):
    id: int
    name: str
    oncall_engineer_id: int


class Repo(TypedDict):
    id: int
    name: str
    default_branch: str


class Runbook(TypedDict):
    id: int
    title: str
    url: str


class Environment(TypedDict):
    id: int
    name: str
    region: str


class Service(TypedDict):
    id: int
    name: str
    team_id: int
    repo_id: int
    runbook_id: int
    environment_id: int
    dependency_ids: list[int]


class Incident(TypedDict):
    id: int
    title: str
    service_id: int
    severity: Literal["sev1", "sev2", "sev3"]
    status: Literal["active", "resolved"]
    started_at: str


class Alert(TypedDict):
    id: int
    service_id: int
    name: str
    status: Literal["firing", "resolved"]


class Deploy(TypedDict):
    id: int
    service_id: int
    repo_id: int
    version: str
    deployed_at: str


class MetricSnapshot(TypedDict):
    service_id: int
    metric_name: Literal["error_rate", "latency_p95", "auth_failure_rate", "queue_depth"]
    value: str


class IncidentSearchResult(TypedDict):
    id: int
    title: str


class ServiceSearchResult(TypedDict):
    id: int
    name: str


class EngineerSearchResult(TypedDict):
    id: int
    name: str


class TeamSearchResult(TypedDict):
    id: int
    name: str


DataItemT = TypeVar(
    "DataItemT", Incident, Service, Engineer, Team, Repo, Runbook, Environment, Alert, Deploy
)


ENGINEER_DATA: list[Engineer] = [
    {"id": 7118, "name": "Alice Kim", "email": "alice@ops.example.com", "team_id": 481},
    {"id": 7243, "name": "Ben Ortiz", "email": "ben@ops.example.com", "team_id": 481},
    {"id": 7381, "name": "Cara Singh", "email": "cara@ops.example.com", "team_id": 562},
    {"id": 7459, "name": "Diego Park", "email": "diego@ops.example.com", "team_id": 562},
    {"id": 7526, "name": "Evan Brooks", "email": "evan@ops.example.com", "team_id": 693},
    {"id": 7684, "name": "Farah Chen", "email": "farah@ops.example.com", "team_id": 693},
]

TEAM_DATA: list[Team] = [
    {"id": 481, "name": "Payments Platform", "oncall_engineer_id": 7243},
    {"id": 562, "name": "Checkout Experience", "oncall_engineer_id": 7381},
    {"id": 693, "name": "Identity", "oncall_engineer_id": 7684},
]

REPO_DATA: list[Repo] = [
    {"id": 9104, "name": "payments-service", "default_branch": "main"},
    {"id": 9217, "name": "checkout-frontend", "default_branch": "main"},
    {"id": 9346, "name": "identity-service", "default_branch": "main"},
    {"id": 9482, "name": "shared-observability", "default_branch": "main"},
]

RUNBOOK_DATA: list[Runbook] = [
    {
        "id": 12041,
        "title": "Payments API 5xx Response Runbook",
        "url": "https://runbooks.example.com/payments-api-5xx",
    },
    {
        "id": 12058,
        "title": "Checkout Latency Runbook",
        "url": "https://runbooks.example.com/checkout-latency",
    },
    {
        "id": 12073,
        "title": "Authentication Failure Runbook",
        "url": "https://runbooks.example.com/auth-failures",
    },
]

ENVIRONMENT_DATA: list[Environment] = [
    {"id": 301, "name": "production", "region": "us-east-1"},
    {"id": 442, "name": "staging", "region": "us-west-2"},
]

SERVICE_DATA: list[Service] = [
    {
        "id": 8401,
        "name": "payments-api",
        "team_id": 481,
        "repo_id": 9104,
        "runbook_id": 12041,
        "environment_id": 301,
        "dependency_ids": [8627],
    },
    {
        "id": 8514,
        "name": "checkout-web",
        "team_id": 562,
        "repo_id": 9217,
        "runbook_id": 12058,
        "environment_id": 301,
        "dependency_ids": [8401, 8627],
    },
    {
        "id": 8627,
        "name": "identity-api",
        "team_id": 693,
        "repo_id": 9346,
        "runbook_id": 12073,
        "environment_id": 301,
        "dependency_ids": [],
    },
    {
        "id": 8799,
        "name": "analytics-worker",
        "team_id": 481,
        "repo_id": 9482,
        "runbook_id": 0,
        "environment_id": 442,
        "dependency_ids": [8401],
    },
]

INCIDENT_DATA: list[Incident] = [
    {
        "id": 41017,
        "title": "Payments API elevated 5xx",
        "service_id": 8401,
        "severity": "sev1",
        "status": "active",
        "started_at": "2024-08-12T09:14:00Z",
    },
    {
        "id": 41029,
        "title": "Checkout page latency spike",
        "service_id": 8514,
        "severity": "sev2",
        "status": "active",
        "started_at": "2024-08-12T09:20:00Z",
    },
    {
        "id": 41043,
        "title": "Identity login error burst",
        "service_id": 8627,
        "severity": "sev2",
        "status": "resolved",
        "started_at": "2024-08-11T17:02:00Z",
    },
    {
        "id": 41058,
        "title": "Analytics backlog growth",
        "service_id": 8799,
        "severity": "sev3",
        "status": "active",
        "started_at": "2024-08-12T08:05:00Z",
    },
]

ALERT_DATA: list[Alert] = [
    {"id": 55101, "service_id": 8401, "name": "payments-api 5xx rate", "status": "firing"},
    {"id": 55114, "service_id": 8401, "name": "payments-api latency p95", "status": "firing"},
    {"id": 55128, "service_id": 8514, "name": "checkout-web latency p95", "status": "firing"},
    {"id": 55139, "service_id": 8627, "name": "identity-api auth failures", "status": "resolved"},
    {"id": 55152, "service_id": 8799, "name": "analytics-worker queue depth", "status": "firing"},
]

DEPLOY_DATA: list[Deploy] = [
    {
        "id": 66011,
        "service_id": 8401,
        "repo_id": 9104,
        "version": "payments-api@2024.08.12.1",
        "deployed_at": "2024-08-12T08:58:00Z",
    },
    {
        "id": 66024,
        "service_id": 8401,
        "repo_id": 9104,
        "version": "payments-api@2024.08.11.4",
        "deployed_at": "2024-08-11T21:10:00Z",
    },
    {
        "id": 66037,
        "service_id": 8514,
        "repo_id": 9217,
        "version": "checkout-web@2024.08.12.3",
        "deployed_at": "2024-08-12T09:05:00Z",
    },
    {
        "id": 66048,
        "service_id": 8627,
        "repo_id": 9346,
        "version": "identity-api@2024.08.11.7",
        "deployed_at": "2024-08-11T16:40:00Z",
    },
    {
        "id": 66059,
        "service_id": 8799,
        "repo_id": 9482,
        "version": "observability@2024.08.10.2",
        "deployed_at": "2024-08-10T11:30:00Z",
    },
]

METRIC_SNAPSHOT_DATA: list[MetricSnapshot] = [
    {"service_id": 8401, "metric_name": "error_rate", "value": "12.4%"},
    {"service_id": 8401, "metric_name": "latency_p95", "value": "1.8s"},
    {"service_id": 8514, "metric_name": "latency_p95", "value": "2.4s"},
    {"service_id": 8627, "metric_name": "auth_failure_rate", "value": "0.2%"},
    {"service_id": 8799, "metric_name": "queue_depth", "value": "18420"},
]

CURRENT_INCIDENT_ID = 41017


@overload
def _similarity_search(
    data: list[Incident], query: str, key: Literal["title"]
) -> list[IncidentSearchResult]: ...


@overload
def _similarity_search(
    data: list[Service], query: str, key: Literal["name"]
) -> list[ServiceSearchResult]: ...


@overload
def _similarity_search(
    data: list[Engineer], query: str, key: Literal["name"]
) -> list[EngineerSearchResult]: ...


@overload
def _similarity_search(
    data: list[Team], query: str, key: Literal["name"]
) -> list[TeamSearchResult]: ...


def _similarity_search(
    data: list[dict[str, object]], query: str, key: str
) -> list[dict[str, object]]:
    def _score(x: str) -> float:
        return len(set(x.lower()) & set(query.lower())) / len(set(x.lower()) | set(query.lower()))

    ranked = sorted(data, key=lambda item: _score(item[key]), reverse=True)
    return [{"id": item["id"], key: item[key]} for item in ranked]


def _get_by_id(data: list[DataItemT], item_id: int, label: str) -> DataItemT:
    for item in data:
        if item["id"] == item_id:
            return item
    msg = f"{label} ID {item_id} cannot be resolved"
    raise ToolException(msg)


def _get_metric_snapshot(service_id: int, metric_name: str) -> MetricSnapshot:
    for metric in METRIC_SNAPSHOT_DATA:
        if metric["service_id"] == service_id and metric["metric_name"] == metric_name:
            return metric
    msg = f"Metric {metric_name} for service {service_id} cannot be resolved"
    raise ToolException(msg)


@tool
def get_current_incident_id() -> int:
    """Get the current incident ID."""
    return CURRENT_INCIDENT_ID


@tool
def list_incident_ids() -> list[int]:
    """List all incident IDs."""
    return [incident["id"] for incident in INCIDENT_DATA]


@tool
def find_incidents_by_title(title: str) -> list[dict]:
    """Find incidents with a similar title.

    Args:
        title: The incident title to search for.
    """
    return _similarity_search(INCIDENT_DATA, title, "title")


@tool
def find_services_by_name(name: str) -> list[ServiceSearchResult]:
    """Find services with a similar name.

    Args:
        name: The service name to search for.
    """
    return _similarity_search(SERVICE_DATA, name, "name")


@tool
def find_engineers_by_name(name: str) -> list[EngineerSearchResult]:
    """Find engineers with a similar name.

    Args:
        name: The engineer name to search for.
    """
    return _similarity_search(ENGINEER_DATA, name, "name")


@tool
def find_teams_by_name(name: str) -> list[dict]:
    """Find teams with a similar name.

    Args:
        name: The team name to search for.
    """
    return _similarity_search(TEAM_DATA, name, "name")


@tool
def get_incident_title(incident_id: int) -> str:
    """Get the title for an incident.

    Args:
        incident_id: The incident ID.
    """
    return _get_by_id(INCIDENT_DATA, incident_id, "Incident")["title"]


@tool
def get_incident_service(incident_id: int) -> int:
    """Get the affected service ID for an incident.

    Args:
        incident_id: The incident ID.
    """
    return _get_by_id(INCIDENT_DATA, incident_id, "Incident")["service_id"]


@tool
def get_incident_severity(incident_id: int) -> str:
    """Get the severity for an incident.

    Args:
        incident_id: The incident ID.
    """
    return _get_by_id(INCIDENT_DATA, incident_id, "Incident")["severity"]


@tool
def get_incident_status(incident_id: int) -> str:
    """Get the status for an incident.

    Args:
        incident_id: The incident ID.
    """
    return _get_by_id(INCIDENT_DATA, incident_id, "Incident")["status"]


@tool
def get_incident_started_at(incident_id: int) -> str:
    """Get the start timestamp for an incident.

    Args:
        incident_id: The incident ID.
    """
    return _get_by_id(INCIDENT_DATA, incident_id, "Incident")["started_at"]


@tool
def get_service_name(service_id: int) -> str:
    """Get the name of a service.

    Args:
        service_id: The service ID.
    """
    return _get_by_id(SERVICE_DATA, service_id, "Service")["name"]


@tool
def get_service_team(service_id: int) -> int:
    """Get the owner team ID for a service.

    Args:
        service_id: The service ID.
    """
    return _get_by_id(SERVICE_DATA, service_id, "Service")["team_id"]


@tool
def get_service_repo(service_id: int) -> int:
    """Get the repo ID for a service.

    Args:
        service_id: The service ID.
    """
    return _get_by_id(SERVICE_DATA, service_id, "Service")["repo_id"]


@tool
def get_service_runbook(service_id: int) -> int:
    """Get the runbook ID for a service.

    Args:
        service_id: The service ID.
    """
    return _get_by_id(SERVICE_DATA, service_id, "Service")["runbook_id"]


@tool
def get_service_environment(service_id: int) -> int:
    """Get the environment ID for a service.

    Args:
        service_id: The service ID.
    """
    return _get_by_id(SERVICE_DATA, service_id, "Service")["environment_id"]


@tool
def list_service_dependencies(service_id: int) -> list[int]:
    """List dependency service IDs for a service.

    Args:
        service_id: The service ID.
    """
    return _get_by_id(SERVICE_DATA, service_id, "Service")["dependency_ids"]


@tool
def list_service_alert_ids(service_id: int) -> list[int]:
    """List alert IDs for a service.

    Args:
        service_id: The service ID.
    """
    return [alert["id"] for alert in ALERT_DATA if alert["service_id"] == service_id]


@tool
def get_latest_deploy_for_service(service_id: int) -> int:
    """Get the most recent deploy ID for a service.

    Args:
        service_id: The service ID.
    """
    deploys = [deploy for deploy in DEPLOY_DATA if deploy["service_id"] == service_id]
    if not deploys:
        msg = f"No deploys found for service {service_id}"
        raise ToolException(msg)
    latest = max(deploys, key=lambda deploy: deploy["deployed_at"])
    return latest["id"]


@tool
def get_metric_value(service_id: int, metric_name: str) -> str:
    """Get the current value of a named metric for a service.

    Args:
        service_id: The service ID.
        metric_name: The metric name.
    """
    return _get_metric_snapshot(service_id, metric_name)["value"]


@tool
def get_team_name(team_id: int) -> str:
    """Get the team name for a team ID.

    Args:
        team_id: The team ID.
    """
    return _get_by_id(TEAM_DATA, team_id, "Team")["name"]


@tool
def get_team_oncall_engineer(team_id: int) -> int:
    """Get the on-call engineer ID for a team.

    Args:
        team_id: The team ID.
    """
    return _get_by_id(TEAM_DATA, team_id, "Team")["oncall_engineer_id"]


@tool
def get_engineer_name(engineer_id: int) -> str:
    """Get the name of an engineer.

    Args:
        engineer_id: The engineer ID.
    """
    return _get_by_id(ENGINEER_DATA, engineer_id, "Engineer")["name"]


@tool
def get_engineer_email(engineer_id: int) -> str:
    """Get the email of an engineer.

    Args:
        engineer_id: The engineer ID.
    """
    return _get_by_id(ENGINEER_DATA, engineer_id, "Engineer")["email"]


@tool
def get_engineer_team(engineer_id: int) -> int:
    """Get the team ID for an engineer.

    Args:
        engineer_id: The engineer ID.
    """
    return _get_by_id(ENGINEER_DATA, engineer_id, "Engineer")["team_id"]


@tool
def get_repo_name(repo_id: int) -> str:
    """Get the repository name for a repo ID.

    Args:
        repo_id: The repo ID.
    """
    return _get_by_id(REPO_DATA, repo_id, "Repo")["name"]


@tool
def get_repo_default_branch(repo_id: int) -> str:
    """Get the default branch for a repo.

    Args:
        repo_id: The repo ID.
    """
    return _get_by_id(REPO_DATA, repo_id, "Repo")["default_branch"]


@tool
def get_runbook_title(runbook_id: int) -> str:
    """Get the title of a runbook.

    Args:
        runbook_id: The runbook ID.
    """
    return _get_by_id(RUNBOOK_DATA, runbook_id, "Runbook")["title"]


@tool
def get_runbook_url(runbook_id: int) -> str:
    """Get the URL of a runbook.

    Args:
        runbook_id: The runbook ID.
    """
    return _get_by_id(RUNBOOK_DATA, runbook_id, "Runbook")["url"]


@tool
def get_environment_name(environment_id: int) -> str:
    """Get the environment name.

    Args:
        environment_id: The environment ID.
    """
    return _get_by_id(ENVIRONMENT_DATA, environment_id, "Environment")["name"]


@tool
def get_environment_region(environment_id: int) -> str:
    """Get the environment region.

    Args:
        environment_id: The environment ID.
    """
    return _get_by_id(ENVIRONMENT_DATA, environment_id, "Environment")["region"]


@tool
def get_alert_name(alert_id: int) -> str:
    """Get the alert name for an alert ID.

    Args:
        alert_id: The alert ID.
    """
    return _get_by_id(ALERT_DATA, alert_id, "Alert")["name"]


@tool
def get_alert_status(alert_id: int) -> str:
    """Get the alert status for an alert ID.

    Args:
        alert_id: The alert ID.
    """
    return _get_by_id(ALERT_DATA, alert_id, "Alert")["status"]


@tool
def get_deploy_version(deploy_id: int) -> str:
    """Get the version string for a deploy.

    Args:
        deploy_id: The deploy ID.
    """
    return _get_by_id(DEPLOY_DATA, deploy_id, "Deploy")["version"]


@tool
def get_deploy_timestamp(deploy_id: int) -> str:
    """Get the deployment timestamp for a deploy.

    Args:
        deploy_id: The deploy ID.
    """
    return _get_by_id(DEPLOY_DATA, deploy_id, "Deploy")["deployed_at"]


INCIDENT_GRAPH_TOOLS = [
    get_current_incident_id,
    list_incident_ids,
    find_incidents_by_title,
    find_services_by_name,
    find_engineers_by_name,
    find_teams_by_name,
    get_incident_title,
    get_incident_service,
    get_incident_severity,
    get_incident_status,
    get_incident_started_at,
    get_service_name,
    get_service_team,
    get_service_repo,
    get_service_runbook,
    get_service_environment,
    list_service_dependencies,
    list_service_alert_ids,
    get_latest_deploy_for_service,
    get_metric_value,
    get_team_name,
    get_team_oncall_engineer,
    get_engineer_name,
    get_engineer_email,
    get_engineer_team,
    get_repo_name,
    get_repo_default_branch,
    get_runbook_title,
    get_runbook_url,
    get_environment_name,
    get_environment_region,
    get_alert_name,
    get_alert_status,
    get_deploy_version,
    get_deploy_timestamp,
]


@pytest.mark.langsmith
async def test_single_tool_list_incident_ids(model: BaseChatModel) -> None:
    agent = create_deep_agent(model=model, tools=INCIDENT_GRAPH_TOOLS)
    await run_agent_async(
        agent,
        model=model,
        query="What are all the incident IDs in the system?",
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("101"),
            final_text_contains("102"),
            final_text_contains("103"),
            final_text_contains("104"),
        )
        .expect(
            agent_steps=2,
            tool_call_requests=1,
            tool_calls=[tool_call(name="list_incident_ids", step=1)],
        ),
    )


@pytest.mark.langsmith
async def test_two_tools_current_incident_service_name(model: BaseChatModel) -> None:
    agent = create_deep_agent(model=model, tools=INCIDENT_GRAPH_TOOLS)
    await run_agent_async(
        agent,
        model=model,
        query="What service is affected by the current incident?",
        scorer=TrajectoryScorer()
        .success(final_text_contains("payments-api"))
        .expect(
            agent_steps=3,
            tool_call_requests=2,
            tool_calls=[
                tool_call(name="get_current_incident_id", step=1),
                tool_call(
                    name="get_incident_service", step=2, args_contains={"incident_id": 41017}
                ),
            ],
        ),
    )


@pytest.mark.langsmith
async def test_three_tools_find_service_owner_team(model: BaseChatModel) -> None:
    agent = create_deep_agent(model=model, tools=INCIDENT_GRAPH_TOOLS)
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
async def test_four_tools_incident_to_oncall_name(model: BaseChatModel) -> None:
    agent = create_deep_agent(model=model, tools=INCIDENT_GRAPH_TOOLS)
    await run_agent_async(
        agent,
        model=model,
        query="Who is the on-call engineer for incident 102?",
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
async def test_four_tools_service_runbook_url(model: BaseChatModel) -> None:
    agent = create_deep_agent(model=model, tools=INCIDENT_GRAPH_TOOLS)
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
async def test_five_tools_incident_latest_deploy_and_repo(model: BaseChatModel) -> None:
    agent = create_deep_agent(model=model, tools=INCIDENT_GRAPH_TOOLS)
    await run_agent_async(
        agent,
        model=model,
        query="For incident 101, what repo was most recently deployed and what version was it?",
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("payments-service"),
            final_text_contains("payments-api@2024.08.12.1"),
        )
        .expect(
            agent_steps=5,
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
async def test_five_tools_incident_environment_name_and_region(model: BaseChatModel) -> None:
    agent = create_deep_agent(model=model, tools=INCIDENT_GRAPH_TOOLS)
    await run_agent_async(
        agent,
        model=model,
        query="What environment and region is incident 104 running in?",
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
async def test_five_tools_service_dependency_names_parallel(model: BaseChatModel) -> None:
    agent = create_deep_agent(model=model, tools=INCIDENT_GRAPH_TOOLS)
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
async def test_five_tools_service_alert_names_parallel(model: BaseChatModel) -> None:
    agent = create_deep_agent(model=model, tools=INCIDENT_GRAPH_TOOLS)
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
async def test_six_tools_current_incident_oncall_name_and_email(model: BaseChatModel) -> None:
    agent = create_deep_agent(model=model, tools=INCIDENT_GRAPH_TOOLS)
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
            agent_steps=5,
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
async def test_six_tools_service_repo_and_branch(model: BaseChatModel) -> None:
    agent = create_deep_agent(model=model, tools=INCIDENT_GRAPH_TOOLS)
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
async def test_six_tools_incident_title_severity_and_status(model: BaseChatModel) -> None:
    agent = create_deep_agent(model=model, tools=INCIDENT_GRAPH_TOOLS)
    await run_agent_async(
        agent,
        model=model,
        query="For incident 103, tell me its title, severity, and status.",
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("Identity login error burst"),
            final_text_contains("sev2"),
            final_text_contains("resolved"),
        )
        .expect(
            agent_steps=3,
            tool_call_requests=3,
            tool_calls=[
                tool_call(name="get_incident_title", step=1, args_contains={"incident_id": 41043}),
                tool_call(
                    name="get_incident_severity", step=1, args_contains={"incident_id": 41043}
                ),
                tool_call(name="get_incident_status", step=1, args_contains={"incident_id": 41043}),
            ],
        ),
    )


@pytest.mark.langsmith
async def test_six_tools_current_incident_metrics_parallel(model: BaseChatModel) -> None:
    agent = create_deep_agent(model=model, tools=INCIDENT_GRAPH_TOOLS)
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
