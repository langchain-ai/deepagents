from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from deepagents_talon.cron import CronJobStore, CronOrigin
from deepagents_talon.goals import GoalController, GoalLedger, GoalRequest, GoalScope
from deepagents_talon.host import TalonHost
from deepagents_talon.interfaces import AgentRequest, AgentResult, ChannelMessage, SendResult
from tests.conftest import RecordingChannel
from tests.test_host import BlockingAgent, _config

if TYPE_CHECKING:
    from pathlib import Path

    from deepagents_talon.cron import CronJob

NOW = datetime(2026, 11, 20, 12, 0, tzinfo=UTC)
TTL = timedelta(minutes=31)
OWNER = "U-owner"
ORIGIN = CronOrigin(conversation_id="C1", channel="slack", sender_id=OWNER)
PROPOSAL = "create a Linear issue in SEC titled 'Rotate leaked key'"
EVIDENCE = "secret key ABC123 appears in build log"


class _Clock:
    def __init__(self) -> None:
        self.now = NOW

    def __call__(self) -> datetime:
        return self.now


def _discovery() -> str:
    candidate = {"source_key": "SEC-1", "source_version": "v1", "evidence": EVIDENCE}
    return json.dumps({"candidate": candidate})


def _scope(*, direct: bool = False, owner: str = OWNER) -> GoalScope:
    return GoalScope(channel="slack", conversation_id="C1", owner_id=owner, direct=direct)


def _send(controller: GoalController, argument: str, *, owner: str = OWNER, direct: bool = False):
    request = GoalRequest(
        scope=_scope(direct=direct, owner=owner), origin=ORIGIN, argument=argument
    )
    return controller.handle_command(request)


def _activate(controller: GoalController, *, direct: bool = False) -> CronJob:
    for argument in (
        "drive security outcomes",
        "edit sources: tool:linear_list_issues\nprogress: closed SEC issues per week",
        "confirm 2",
    ):
        _send(controller, argument, direct=direct)
    [job] = [job for job in controller.store.list_jobs() if job.goal_id]
    return job


def _turn(controller: GoalController, job: CronJob, reply: str) -> str | None:
    turn = controller.prepare_fire(job, ttl=TTL)
    assert turn is not None
    controller.finish_turn(turn, text=reply)
    return turn.item_id


def _propose(controller: GoalController, job: CronJob) -> str:
    _turn(controller, job, _discovery())
    item_id = _turn(
        controller, job, json.dumps({"status": "awaiting_decision", "proposal": PROPOSAL})
    )
    assert item_id is not None
    return item_id


@pytest.fixture
def clock() -> _Clock:
    return _Clock()


@pytest.fixture
def controller(tmp_path: Path, clock: _Clock) -> GoalController:
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path / "cron")
    return GoalController(ledger=GoalLedger(goal_dir=tmp_path / "goals"), store=store, now=clock)


@pytest.mark.parametrize("chat", ["dm", "shared"])
def test_decision_request_is_queued_and_redacted_outside_dms(
    controller: GoalController, chat: str
) -> None:
    direct = chat == "dm"
    job = _activate(controller, direct=direct)
    _propose(controller, job)

    [event] = controller.ledger.list_outbox(states={"pending"})
    [decision] = controller.ledger.list_decisions(job.goal_id, states={"pending"})

    assert event.event_id == f"decision:{decision.id}"
    assert PROPOSAL in event.payload
    assert f"/goal approve {decision.id}" in event.payload
    assert (EVIDENCE in event.payload) is direct


def test_approval_hands_exact_action_to_interactive_turn_then_waits_for_owner(
    controller: GoalController,
) -> None:
    job = _activate(controller)
    item_id = _propose(controller, job)
    [decision] = controller.ledger.list_decisions(job.goal_id, states={"pending"})

    refused = _send(controller, f"approve {decision.id}", owner="U-other")
    reply = _send(controller, f"approve {decision.id}")

    assert "Only the person who started" in refused.text
    assert refused.action is None
    assert reply.action is not None
    assert PROPOSAL in reply.action.prompt
    assert controller.ledger.get_item(item_id).status == "running_interactive"
    assert controller.prepare_fire(job, ttl=TTL).item_id is None

    controller.interactive_finished(reply.action, cancelled=False)
    assert controller.ledger.get_item(item_id).status == "needs_attention"
    assert f"item {item_id} needs you" in _send(controller, "status").text

    done = _send(controller, f"done {item_id} https://linear.app/SEC-9")
    item = controller.ledger.get_item(item_id)
    assert done.text == f"Marked item {item_id} done (verified)."
    assert (item.status, item.artifact) == ("done", "https://linear.app/SEC-9")


def test_reject_and_expired_approval_run_nothing(controller: GoalController, clock: _Clock) -> None:
    job = _activate(controller)
    first = _propose(controller, job)
    [decision] = controller.ledger.list_decisions(job.goal_id, states={"pending"})

    rejected = _send(controller, f"reject {decision.id}")

    assert rejected.action is None
    assert controller.ledger.get_item(first).status == "abandoned"

    second = {"candidate": {"source_key": "SEC-2", "source_version": "v1"}}
    _turn(controller, job, json.dumps(second))
    item_id = _turn(
        controller, job, json.dumps({"status": "awaiting_decision", "proposal": PROPOSAL})
    )
    [late] = controller.ledger.list_decisions(job.goal_id, states={"pending"})
    clock.now = NOW + timedelta(days=8)

    expired = _send(controller, f"approve {late.id}")

    assert expired.action is None
    assert "expired" in expired.text
    assert controller.ledger.get_item(item_id).status == "abandoned"


def test_weekly_digest_is_queued_once_per_week(controller: GoalController, clock: _Clock) -> None:
    job = _activate(controller)
    _turn(controller, job, json.dumps({"candidate": None}))

    clock.now = NOW + timedelta(days=7)
    controller.prepare_fire(job, ttl=TTL)
    controller.ledger.reconcile_leases(now=clock.now + TTL)
    controller.prepare_fire(job, ttl=TTL)

    digests = [e for e in controller.ledger.list_outbox() if e.event_id.startswith("digest:")]
    assert len(digests) == 1
    assert digests[0].payload.startswith("Weekly goal update.")


class _ScriptedAgent(BlockingAgent):
    def __init__(self, replies: list[str]) -> None:
        super().__init__()
        self.replies = replies

    async def invoke(self, request: AgentRequest) -> AgentResult:
        self.requests.append(request)
        return AgentResult(text=self.replies.pop(0))


class _FlakyChannel(RecordingChannel):
    def __init__(self) -> None:
        super().__init__(provider="slack")
        self.fail = False

    async def send_message(self, conversation_id: str, text: str) -> SendResult:
        if self.fail:
            return SendResult(success=False, error="down", retryable=False)
        return await super().send_message(conversation_id, text)


class _Scheduler:
    def __init__(self, store: CronJobStore) -> None:
        self.store = store

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None


async def test_host_delivers_decision_and_runs_approved_action(tmp_path: Path) -> None:
    replies = [
        _discovery(),
        json.dumps({"status": "awaiting_decision", "proposal": PROPOSAL}),
        "Created SEC-9.",
    ]
    agent = _ScriptedAgent(replies)
    channel = _FlakyChannel()
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path / "cron")
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    host.scheduler = _Scheduler(store)
    ledger = GoalLedger(goal_dir=host.config.goal_dir)
    job = _activate(GoalController(ledger=ledger, store=store))
    await host.start()
    try:
        await host.run_scheduled_job(job)
        await host.run_scheduled_job(job)
        [decision] = ledger.list_decisions(job.goal_id, states={"pending"})
        approve = ChannelMessage(
            conversation_id="C1", text=f"/goal approve {decision.id}", sender_id=OWNER
        )
        await host.receive_message(channel, approve)
        for _ in range(50):
            if ledger.get_item(decision.item_id).status == "needs_attention":
                break
            await asyncio.sleep(0.01)
    finally:
        await host.stop()

    sent = [text for _, text in channel.sent]
    assert any(f"/goal approve {decision.id}" in text for text in sent)
    assert any(text.startswith(f"Approved decision {decision.id}") for text in sent)
    assert PROPOSAL in agent.requests[-1].text
    assert "goal_read_scope" not in agent.requests[-1].metadata
    assert ledger.list_outbox()[0].state == "sent"
    assert ledger.get_item(decision.item_id).status == "needs_attention"


async def test_failed_delivery_is_left_unknown_and_not_resent(tmp_path: Path) -> None:
    channel = _FlakyChannel()
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path / "cron")
    host = TalonHost(config=_config(tmp_path), agent=BlockingAgent(), channels=[channel])
    host.scheduler = _Scheduler(store)
    ledger = GoalLedger(goal_dir=host.config.goal_dir)
    job = _activate(GoalController(ledger=ledger, store=store))
    ledger.enqueue(job.goal_id, event_id="note-1", payload="update")
    channel.fail = True
    await host.start()
    try:
        await host.receive_message(
            channel, ChannelMessage(conversation_id="C1", text="/goal status", sender_id=OWNER)
        )
        channel.fail = False
        await host.receive_message(
            channel, ChannelMessage(conversation_id="C1", text="/goal status", sender_id=OWNER)
        )
    finally:
        await host.stop()

    [event] = ledger.list_outbox()
    assert event.state == "unknown"
    assert "Updates not confirmed delivered: 1." in channel.sent[-1][1]
    assert all(text != "update" for _, text in channel.sent)


async def test_overlapping_deliveries_send_once(tmp_path: Path, controller: GoalController) -> None:
    job = _activate(controller)
    controller.ledger.enqueue(job.goal_id, event_id="note-1", payload="update")
    [event] = controller.ledger.list_outbox()
    channel = _FlakyChannel()
    host = TalonHost(config=_config(tmp_path), agent=BlockingAgent(), channels=[channel])

    await asyncio.gather(
        host._deliver_goal_event(controller, event), host._deliver_goal_event(controller, event)
    )

    assert channel.sent == [("C1", "update")]
    assert controller.ledger.list_outbox()[0].state == "sent"
