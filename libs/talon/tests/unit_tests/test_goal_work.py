from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest
from langchain_core.tools import tool
from langgraph.prebuilt.tool_node import ToolCallRequest

from deepagents_talon.cron import CronJobStore, CronOrigin
from deepagents_talon.cron.scheduler import SILENT_SENTINEL
from deepagents_talon.goals import GoalController, GoalLedger, GoalRequest, GoalScope
from deepagents_talon.goals.controller import MAX_OPEN_ITEMS
from deepagents_talon.goals.scope import GOAL_READ_SCOPE, GoalScopeMiddleware, ReadScope
from deepagents_talon.host import TalonHost
from deepagents_talon.interfaces import AgentRequest, AgentResult
from deepagents_talon.runtime import _goal_scope
from tests.conftest import RecordingChannel
from tests.test_host import BlockingAgent, _config

if TYPE_CHECKING:
    from pathlib import Path

    from deepagents_talon.cron import CronJob

NOW = datetime(2026, 11, 20, 12, 0, tzinfo=UTC)
TTL = timedelta(minutes=31)
OWNER = "U-owner"
ORIGIN = CronOrigin(conversation_id="C1", channel="slack", sender_id=OWNER)
SCOPE = GoalScope(channel="slack", conversation_id="C1", owner_id=OWNER)
SOURCES = "tool:linear_list_issues; path:/srv/reports; the SEC board"


def _discovery(key: str = "SEC-1", version: str = "v1") -> str:
    candidate = {"source_key": key, "source_version": version, "evidence": "new P1"}
    return "Found one.\n" + json.dumps({"candidate": candidate, "summary": "one new P1"})


class _Clock:
    def __init__(self) -> None:
        self.now = NOW

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def clock() -> _Clock:
    return _Clock()


@pytest.fixture
def store(tmp_path: Path) -> CronJobStore:
    return CronJobStore(assistant_id="test", cron_dir=tmp_path / "cron")


@pytest.fixture
def ledger(tmp_path: Path) -> GoalLedger:
    return GoalLedger(goal_dir=tmp_path / "goals")


@pytest.fixture
def controller(ledger: GoalLedger, store: CronJobStore, clock: _Clock) -> GoalController:
    return GoalController(ledger=ledger, store=store, now=clock)


def _activate(controller: GoalController) -> CronJob:
    for argument in (
        "drive security outcomes",
        f"edit sources: {SOURCES}\nprogress: closed SEC issues per week",
        "confirm 2",
    ):
        controller.handle(GoalRequest(scope=SCOPE, origin=ORIGIN, argument=argument))
    [job] = [job for job in controller.store.list_jobs() if job.goal_id]
    return job


def _run(controller: GoalController, job: CronJob, reply: str | None, failure=None) -> str:
    turn = controller.prepare_fire(job, ttl=TTL)
    assert turn is not None
    controller.finish_turn(turn, text=reply, failure=failure)
    return turn.item_id or "discovery"


def test_discovery_adds_one_planned_item_and_deduplicates(
    controller: GoalController, ledger: GoalLedger
) -> None:
    job = _activate(controller)
    goal_id = job.goal_id

    _run(controller, job, _discovery())
    [item] = ledger.list_items(goal_id)
    item_turn = controller.prepare_fire(job, ttl=TTL)

    assert item.status == "planned"
    assert item_turn is not None
    assert item_turn.item_id == item.id
    assert item_turn.scope == ReadScope(
        tools=frozenset({"linear_list_issues"}), paths=("/srv/reports",)
    )
    controller.finish_turn(item_turn, text='{"status": "waiting_for_source"}')
    _run(controller, job, _discovery())
    assert ledger.get_item(item.id).status == "waiting_for_source"
    _run(controller, job, _discovery().replace("new P1", "source is available"))
    [resumed] = ledger.list_items(goal_id)
    assert (resumed.id, resumed.status) == (item.id, "planned")


def test_item_turn_records_verified_completion(
    controller: GoalController, ledger: GoalLedger
) -> None:
    job = _activate(controller)
    _run(controller, job, _discovery())

    item_id = _run(
        controller,
        job,
        '{"status": "done", "outcome": "verified", "artifact": "https://x/SEC-1"}',
    )

    item = ledger.get_item(item_id)
    assert (item.status, item.outcome, item.artifact) == ("done", "verified", "https://x/SEC-1")


@pytest.mark.parametrize(
    "reply",
    [
        "I looked around and everything is fine.",
        '{"status": "done"}',
        '{"status": "awaiting_decision"}',
    ],
)
def test_invalid_item_result_is_an_error_with_backoff(
    controller: GoalController, ledger: GoalLedger, reply: str
) -> None:
    job = _activate(controller)
    _run(controller, job, _discovery())

    item_id = _run(controller, job, reply)

    item = ledger.get_item(item_id)
    assert item.status == "planned"
    assert item.retry_at is not None
    assert ledger.list_runs(job.goal_id)[-1].result == "error"


def test_proposed_write_becomes_pending_owner_decision(
    controller: GoalController, ledger: GoalLedger
) -> None:
    job = _activate(controller)
    _run(controller, job, _discovery())
    proposal = "create a Linear issue in SEC titled 'Rotate leaked key'"

    item_id = _run(
        controller, job, json.dumps({"status": "awaiting_decision", "proposal": proposal})
    )

    assert ledger.get_item(item_id).status == "awaiting_decision"
    [decision] = ledger.list_decisions(job.goal_id, states={"pending"})
    assert decision.item_id == item_id
    assert "Waiting on you: 1 decision(s)." in controller.handle(
        GoalRequest(scope=SCOPE, origin=ORIGIN, argument="status")
    )


def test_timeout_flags_item_instead_of_retrying(
    controller: GoalController, ledger: GoalLedger
) -> None:
    job = _activate(controller)
    _run(controller, job, _discovery())

    item_id = _run(controller, job, None, failure="timeout")

    assert ledger.get_item(item_id).status == "needs_attention"
    assert controller.prepare_fire(job, ttl=TTL).item_id is None


def test_parked_work_leaves_room_for_discovery(
    controller: GoalController, ledger: GoalLedger
) -> None:
    job = _activate(controller)
    for index in range(MAX_OPEN_ITEMS):
        _run(controller, job, _discovery(key=f"SEC-{index}"))
        _run(controller, job, None, failure="timeout")

    assert _run(controller, job, _discovery(key="SEC-0")) == "discovery"
    assert all(item.status == "needs_attention" for item in ledger.list_items(job.goal_id))


def test_paused_goal_or_full_queue_makes_no_turn(
    controller: GoalController, store: CronJobStore
) -> None:
    job = _activate(controller)
    store.edit_job(job.id, origin=job.origin, enabled=False)

    assert controller.prepare_fire(job, ttl=TTL) is None

    store.edit_job(job.id, origin=job.origin, enabled=True)
    controller.reconcile()
    for index in range(MAX_OPEN_ITEMS):
        controller.ledger.add_candidate(job.goal_id, source_key=f"SEC-{index}")
    assert controller.prepare_fire(job, ttl=TTL) is None


def test_reconcile_marks_interrupted_turns_unknown(
    controller: GoalController, ledger: GoalLedger, clock: _Clock
) -> None:
    job = _activate(controller)
    _run(controller, job, _discovery())
    turn = controller.prepare_fire(job, ttl=TTL)
    assert turn is not None

    clock.now = NOW + TTL
    controller.reconcile()

    assert ledger.list_runs(job.goal_id)[-1].result == "unknown"
    assert ledger.get_item(turn.item_id).status == "needs_attention"


def test_read_scope_allows_only_named_read_only_tools_and_paths() -> None:
    @tool
    def linear_list_issues(team: str) -> str:
        """List issues."""
        return team

    @tool
    def linear_create_issue(title: str) -> str:
        """Create an issue."""
        return title

    @tool
    def read_file(file_path: str) -> str:
        """Read a file."""
        return file_path

    linear_list_issues.metadata = {"mcp": {"tool": {"annotations": {"read_only_hint": True}}}}
    linear_create_issue.metadata = {"mcp": {"tool": {"annotations": {"read_only_hint": True}}}}
    scope = ReadScope.from_sources(["tool:linear_list_issues", "path:/srv/reports", "SEC board"])
    unannotated = ReadScope.from_sources(["tool:linear_create_issue"])

    assert scope.allows_call(linear_list_issues, {"team": "SEC"})
    assert not scope.allows_tool(linear_create_issue)
    assert unannotated.allows_tool(linear_create_issue)
    linear_create_issue.metadata = {"mcp": {"tool": None}}
    assert not unannotated.allows_tool(linear_create_issue)
    assert scope.allows_call(read_file, {"file_path": "/srv/reports/q4.md"})
    assert not scope.allows_call(read_file, {"file_path": "/srv/reports/../secrets.env"})
    assert not scope.allows_call(read_file, {"file_path": "/srv/reports-old/x"})
    assert not ReadScope().allows_tool(read_file)


async def test_middleware_refuses_out_of_scope_calls_only_in_goal_turns() -> None:
    @tool
    def write_file(file_path: str, content: str) -> str:
        """Write a file."""
        return file_path + content

    middleware = GoalScopeMiddleware()
    call = {
        "name": "write_file",
        "args": {"file_path": "/srv/reports/x", "content": "y"},
        "id": "1",
    }
    request = ToolCallRequest(tool_call=call, tool=write_file, state={}, runtime=None)

    async def handler(_request: ToolCallRequest) -> str:
        return "ran"

    assert await middleware.awrap_tool_call(request, handler) == "ran"
    token = GOAL_READ_SCOPE.set(ReadScope(paths=("/srv/reports",)))
    try:
        refused = await middleware.awrap_tool_call(request, handler)
    finally:
        GOAL_READ_SCOPE.reset(token)
    assert refused.status == "error"
    assert "only read" in refused.content


def test_runtime_scopes_only_scheduled_turns() -> None:
    def scope_for(metadata: dict[str, object]) -> ReadScope | None:
        request = AgentRequest(conversation_id="c", text="t", metadata=metadata)
        with _goal_scope(request):
            return GOAL_READ_SCOPE.get()

    injected = {"goal_read_scope": {"tools": ["x"], "paths": []}}

    assert scope_for(injected) is None
    assert scope_for({**injected, "trigger": "cron"}) == ReadScope(tools=frozenset({"x"}))
    assert scope_for({"trigger": "cron", "goal_read_scope": "bogus"}) == ReadScope()


class _ScriptedAgent(BlockingAgent):
    def __init__(self, replies: list[str]) -> None:
        super().__init__()
        self.replies = replies

    async def invoke(self, request: AgentRequest) -> AgentResult:
        self.requests.append(request)
        return AgentResult(text=self.replies.pop(0))


async def test_host_runs_scoped_goal_turn_and_stays_silent(tmp_path: Path) -> None:
    channel = RecordingChannel(provider="slack")
    agent = _ScriptedAgent([_discovery()])
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path / "cron")
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    ledger = GoalLedger(goal_dir=host.config.goal_dir)
    job = _activate(GoalController(ledger=ledger, store=store))

    class _Scheduler:
        def __init__(self) -> None:
            self.store = store

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

    host.scheduler = _Scheduler()
    await host.start()
    try:
        result = await host.run_scheduled_job(job)
    finally:
        await host.stop()

    [request] = agent.requests
    assert result == SILENT_SENTINEL
    assert request.metadata["trigger"] == "cron"
    assert request.metadata["goal_read_scope"] == {
        "tools": ["linear_list_issues"],
        "paths": ["/srv/reports"],
    }
    assert "untrusted data" in request.text
    assert [item.status for item in ledger.list_items(job.goal_id)] == ["planned"]
    assert channel.sent == []
