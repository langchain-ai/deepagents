"""Mock incident-management graph tools for the incident-graph eval suite.

A synthetic operational incident-management domain with many entities and tools.
The agent receives only graph lookup/search tools and must compose them to
answer questions efficiently.

Extracted from `tests/evals/test_tool_usage_incident_graph.py` so both the
pytest suite and the Harbor sandbox dispatcher share the same tool definitions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypedDict

from langchain.agents.middleware.types import ToolCallRequest, wrap_tool_call
from langchain_core.messages import ToolMessage
from langchain_core.tools import ToolException, tool

if TYPE_CHECKING:
    from collections.abc import Callable

    from langgraph.types import Command


# ---------------------------------------------------------------------------
# TypedDicts for entity records
# ---------------------------------------------------------------------------


class Engineer(TypedDict):
    """Engineer record."""

    id: int
    name: str
    email: str
    team_id: int


class Team(TypedDict):
    """Team record."""

    id: int
    name: str
    oncall_engineer_id: int


class Repo(TypedDict):
    """Repository record."""

    id: int
    name: str
    default_branch: str


class Runbook(TypedDict):
    """Runbook record."""

    id: int
    title: str
    url: str


class Environment(TypedDict):
    """Deployment environment record."""

    id: int
    name: str
    region: str


class Service(TypedDict):
    """Service record."""

    id: int
    name: str
    team_id: int
    repo_id: int
    runbook_id: int
    environment_id: int
    dependency_ids: list[int]


class Incident(TypedDict):
    """Incident record."""

    id: int
    title: str
    service_id: int
    severity: Literal["sev1", "sev2", "sev3"]
    status: Literal["active", "resolved"]
    started_at: str


class Alert(TypedDict):
    """Alert record."""

    id: int
    service_id: int
    name: str
    status: Literal["firing", "resolved"]


class Deploy(TypedDict):
    """Deployment record."""

    id: int
    service_id: int
    repo_id: int
    version: str
    deployed_at: str


class MetricSnapshot(TypedDict):
    """Metric snapshot record."""

    service_id: int
    metric_name: Literal["error_rate", "latency_p95", "auth_failure_rate", "queue_depth"]
    value: str


class IncidentSearchResult(TypedDict):
    """Search result for incidents."""

    id: int
    title: str


class ServiceSearchResult(TypedDict):
    """Search result for services."""

    id: int
    name: str


class EngineerSearchResult(TypedDict):
    """Search result for engineers."""

    id: int
    name: str


class TeamSearchResult(TypedDict):
    """Search result for teams."""

    id: int
    name: str


# ---------------------------------------------------------------------------
# Static data
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Internal helpers (not exposed as tools)
# ---------------------------------------------------------------------------


def _rank_by_similarity[ItemT](
    data: list[ItemT], query: str, value: Callable[[ItemT], str]
) -> list[ItemT]:
    def _score(x: str) -> float:
        return len(set(x.lower()) & set(query.lower())) / len(set(x.lower()) | set(query.lower()))

    return sorted(data, key=lambda item: _score(value(item)), reverse=True)


def _search_incidents_by_title(title: str) -> list[dict]:
    return [
        {"id": incident["id"], "title": incident["title"]}
        for incident in _rank_by_similarity(
            INCIDENT_DATA, title, lambda incident: incident["title"]
        )
    ]


def _search_services_by_name(name: str) -> list[ServiceSearchResult]:
    return [
        {"id": service["id"], "name": service["name"]}
        for service in _rank_by_similarity(SERVICE_DATA, name, lambda service: service["name"])
    ]


def _search_engineers_by_name(name: str) -> list[EngineerSearchResult]:
    return [
        {"id": engineer["id"], "name": engineer["name"]}
        for engineer in _rank_by_similarity(ENGINEER_DATA, name, lambda engineer: engineer["name"])
    ]


def _search_teams_by_name(name: str) -> list[dict]:
    return [
        {"id": team["id"], "name": team["name"]}
        for team in _rank_by_similarity(TEAM_DATA, name, lambda team: team["name"])
    ]


def _get_by_id[
    DataItemT: (Incident, Service, Engineer, Team, Repo, Runbook, Environment, Alert, Deploy)
](data: list[DataItemT], item_id: int, label: str) -> DataItemT:
    for item in data:
        if item["id"] == item_id:
            return item
    msg = f"{label} ID {item_id} cannot be resolved"
    raise ToolException(msg)


def _get_metric_snapshot(service_id: int, metric_name: str) -> MetricSnapshot | None:
    for metric in METRIC_SNAPSHOT_DATA:
        if metric["service_id"] == service_id and metric["metric_name"] == metric_name:
            return metric
    return None


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


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
    return _search_incidents_by_title(title)


@tool
def find_services_by_name(name: str) -> list[ServiceSearchResult]:
    """Find services with a similar name.

    Args:
        name: The service name to search for.
    """
    return _search_services_by_name(name)


@tool
def find_engineers_by_name(name: str) -> list[EngineerSearchResult]:
    """Find engineers with a similar name.

    Args:
        name: The engineer name to search for.
    """
    return _search_engineers_by_name(name)


@tool
def find_teams_by_name(name: str) -> list[dict]:
    """Find teams with a similar name.

    Args:
        name: The team name to search for.
    """
    return _search_teams_by_name(name)


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
    metric = _get_metric_snapshot(service_id, metric_name)
    if metric is None:
        msg = f"Metric {metric_name!r} is not available for service {service_id}"
        raise ToolException(msg)
    return metric["value"]


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


@wrap_tool_call
async def incident_graph_tool_error_middleware(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], Any],
) -> ToolMessage | Command[Any]:
    """Wrap tool errors into `ToolMessage` with `status="error"`."""
    try:
        return await handler(request)
    except ToolException as e:
        tool_call = request.tool_call
        return ToolMessage(
            content=str(e),
            name=tool_call["name"],
            tool_call_id=tool_call["id"],
            status="error",
        )
