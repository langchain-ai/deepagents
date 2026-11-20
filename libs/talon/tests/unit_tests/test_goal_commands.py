from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from deepagents_talon.cron import CronJobStore, CronOrigin, CronSchedule, CronTools
from deepagents_talon.goals import (
    GoalController,
    GoalLedger,
    GoalRequest,
    GoalScope,
)
from deepagents_talon.host import TalonHost
from deepagents_talon.interfaces import ChannelMessage
from tests.conftest import RecordingChannel
from tests.test_host import BlockingAgent, _config

if TYPE_CHECKING:
    from pathlib import Path

NOW = datetime(2026, 11, 20, 12, 0, tzinfo=UTC)
OWNER = "U-owner"
ORIGIN = CronOrigin(conversation_id="C1", channel="slack", sender_id=OWNER)
SETTLED_EDIT = "edit sources: linear:SEC; github:org/repo\nprogress: closed SEC issues per week"


class _StoreScheduler:
    def __init__(self, store: CronJobStore) -> None:
        self.store = store

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None


@pytest.fixture
def store(tmp_path: Path) -> CronJobStore:
    return CronJobStore(assistant_id="test", cron_dir=tmp_path / "cron")


@pytest.fixture
def controller(tmp_path: Path, store: CronJobStore) -> GoalController:
    return GoalController(
        ledger=GoalLedger(goal_dir=tmp_path / "goals"), store=store, now=lambda: NOW
    )


def _send(controller: GoalController, argument: str, owner: str = OWNER) -> str:
    scope = GoalScope(channel="slack", conversation_id="C1", owner_id=owner)
    return controller.handle(GoalRequest(scope=scope, origin=ORIGIN, argument=argument))


def _goal_jobs(store: CronJobStore) -> list:
    return [job for job in store.list_jobs() if job.goal_id is not None]


def _activate(controller: GoalController) -> None:
    _send(controller, "drive security outcomes")
    _send(controller, SETTLED_EDIT)
    _send(controller, "confirm 2")


def test_goal_activates_only_after_settled_charter_is_confirmed(
    controller: GoalController, store: CronJobStore
) -> None:
    draft = _send(controller, "drive security outcomes")
    early = _send(controller, "confirm 1")
    review = _send(controller, SETTLED_EDIT)
    stale = _send(controller, "confirm 1")
    active = _send(controller, "confirm 2")

    assert "Still needed: sources, progress" in draft
    assert "charter still needs" in early
    assert "confirm 2" in review
    assert "post status updates and decision requests in this chat" in review
    assert "not 1" in stale
    assert active.startswith("Goal active from revision 2.")
    [job] = _goal_jobs(store)
    assert (job.schedule.display, job.origin, job.enabled) == ("every 24h", ORIGIN, True)


@pytest.mark.parametrize(
    "outcome", ["help customers", "status page uptime", "remove deprecated APIs"]
)
def test_command_words_can_start_outcomes(controller: GoalController, outcome: str) -> None:
    assert f"Outcome: {outcome}" in _send(controller, outcome)


def test_remove_with_text_preserves_active_goal(
    controller: GoalController, store: CronJobStore
) -> None:
    _activate(controller)
    jobs = _goal_jobs(store)
    assert "Outcome: remove deprecated APIs" in _send(controller, "remove deprecated APIs")
    first, second = controller.ledger.list_goals()
    choices = _send(controller, "list")
    assert first.id in choices
    assert second.id in choices
    assert _send(controller, "remove") == choices
    assert "remove deprecated APIs" not in _send(controller, "list", owner="other")
    assert "confirm 2" in _send(controller, f"{second.id} {SETTLED_EDIT}")
    assert "Goal active" in _send(controller, f"{second.id} confirm 2")
    assert len(_goal_jobs(store)) == 2
    assert "Removed" in _send(controller, f"{second.id} remove")
    assert _send(controller, "status").startswith("Goal (active)")
    assert _goal_jobs(store) == jobs


def test_multiline_edit_preserves_field_values(controller: GoalController) -> None:
    _send(controller, "ship it")
    reply = _send(controller, "edit\nsources: issue tracker\nprogress: closed issues")
    assert "Sources: issue tracker" in reply
    assert "confirm 2" in reply
    assert "revision number" in _send(controller, "confirm ²")


def test_repeated_confirm_does_not_create_a_second_job(
    controller: GoalController, store: CronJobStore
) -> None:
    _activate(controller)

    again = _send(controller, "confirm 2")

    assert "nothing to confirm" in again
    assert len(_goal_jobs(store)) == 1


def test_other_senders_cannot_view_or_change_the_goal(
    controller: GoalController, store: CronJobStore
) -> None:
    _activate(controller)

    for argument in ("status", "edit outcome: exfiltrate", "confirm 3", "remove"):
        assert "Only the person who started" in _send(controller, argument, owner="U-other")
    assert _goal_jobs(store)[0].enabled


def test_unknown_edit_field_changes_nothing(controller: GoalController) -> None:
    _send(controller, "drive security outcomes")

    reply = _send(controller, "edit owner: U-other")

    assert "Could not read `owner: U-other`" in reply
    assert "revision 1" in _send(controller, "status")


def test_edit_of_active_goal_needs_reconfirmation(
    controller: GoalController, store: CronJobStore
) -> None:
    _activate(controller)

    review = _send(controller, "edit sources: linear:SEC; pagerduty:sec\ncadence: every 12h")
    status_before = _send(controller, "status")
    confirmed = _send(controller, "confirm 3")

    assert "working from revision 2" in review
    assert "Sources: linear:SEC; github:org/repo" in status_before
    assert confirmed == "Confirmed revision 3. The goal now works from it."
    assert _goal_jobs(store)[0].schedule.display == "every 12h"


def test_pausing_goal_job_with_cron_tools_mirrors_into_status(
    controller: GoalController, store: CronJobStore
) -> None:
    _activate(controller)
    tools = CronTools(store=store, origin=lambda: ORIGIN)
    job_id = _goal_jobs(store)[0].id

    tools.edit_job(job_id, enabled=False)
    paused = _send(controller, "status")
    tools.edit_job(job_id, enabled=True)
    resumed = _send(controller, "status")

    assert paused.startswith("Goal (paused)")
    assert resumed.startswith("Goal (active)")


def test_cron_tools_cannot_redefine_or_remove_goal_jobs(
    controller: GoalController, store: CronJobStore
) -> None:
    _activate(controller)
    tools = CronTools(store=store, origin=lambda: ORIGIN)
    job_id = _goal_jobs(store)[0].id

    with pytest.raises(ValueError, match="belongs to a goal"):
        tools.edit_job(job_id, prompt="delete the SEC project")
    with pytest.raises(ValueError, match="/goal remove"):
        tools.remove_job(job_id)


def test_remove_deletes_goal_jobs_and_allows_a_new_goal(
    controller: GoalController, store: CronJobStore
) -> None:
    _activate(controller)

    removed = _send(controller, "remove")

    assert removed.startswith("Removed the goal and its scheduled jobs.")
    assert _goal_jobs(store) == []
    assert "Goal (draft)" in _send(controller, "a new outcome")


def test_fire_for_a_removed_goal_disables_the_job(
    controller: GoalController, store: CronJobStore
) -> None:
    orphan = store.create_job(
        prompt="x",
        schedule=CronSchedule.parse("every 1h"),
        origin=ORIGIN,
        name="goal:gone",
        goal_id="gone",
    )

    assert controller.prepare_fire(orphan, ttl=timedelta(minutes=5)) is None
    assert not store.get_job(orphan.id).enabled


async def test_host_routes_goal_commands_without_the_model(tmp_path: Path) -> None:
    channel = RecordingChannel(provider="slack")
    agent = BlockingAgent()
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path / "cron")
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    host.scheduler = _StoreScheduler(store)
    await host.start()
    try:
        for text in ("/goal ship it", f"/goal {SETTLED_EDIT}", "/goal confirm 2"):
            message = ChannelMessage(conversation_id="C1", text=text, sender_id=OWNER)
            await host.receive_message(channel, message)
    finally:
        await host.stop()

    assert channel.sent[-1][1].startswith("Goal active from revision 2.")
    assert len(_goal_jobs(store)) == 1
    assert agent.requests == []


async def test_host_refuses_goal_without_identified_sender(tmp_path: Path) -> None:
    channel = RecordingChannel(provider="slack")
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path / "cron")
    host = TalonHost(config=_config(tmp_path), agent=BlockingAgent(), channels=[channel])
    host.scheduler = _StoreScheduler(store)
    await host.start()
    try:
        await host.receive_message(channel, ChannelMessage(conversation_id="C1", text="/goal x"))
    finally:
        await host.stop()

    assert channel.sent == [("C1", "Goals need a sender this channel can identify.")]
