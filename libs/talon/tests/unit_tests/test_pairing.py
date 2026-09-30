"""Sender pairing: unknown DM senders get a code only an operator can approve."""

from __future__ import annotations

import asyncio
import stat
import sys
from typing import TYPE_CHECKING, cast

import pytest

from deepagents_talon import __main__ as talon_main
from deepagents_talon.channels.base import ChannelExposure, ExposureMode
from deepagents_talon.channels.discord import (
    DiscordChannel,
    DiscordChannelConfig,
    _DiscordInboundMessage,
    _DiscordInboundReaction,
)
from deepagents_talon.channels.telegram import (
    TelegramChannel,
    TelegramChannelConfig,
    _TelegramTransport,
)
from deepagents_talon.config import TalonConfig
from deepagents_talon.cron import CronJobStore, CronOrigin, CronSchedule
from deepagents_talon.host import ScheduledRunRevokedError, TalonHost
from deepagents_talon.interfaces import (
    AgentRequest,
    AgentResult,
    ChannelMessage,
    ChannelReaction,
)
from deepagents_talon.pairing import (
    APPROVED_NOTICE,
    CODE_ALPHABET,
    CODE_LENGTH,
    CODE_TTL_SECONDS,
    MAX_PENDING_PER_CHANNEL,
    PAIRING_FILENAME,
    PairingStore,
    SenderPairing,
    format_code,
)
from tests.channels.test_discord import RecordingGateway
from tests.channels.test_telegram import (
    RecordingTransport,
    _make_reaction_update,
    _make_update,
)
from tests.test_host import StubBackground

if TYPE_CHECKING:
    from pathlib import Path

OPERATOR = "op-1"
OPERATOR_DM = "dm-op"
STRANGER = "stranger-1"
STRANGER_DM = "dm-stranger"


class Clock:
    def __init__(self) -> None:
        self.now = 1_700_000_000.0

    def __call__(self) -> float:
        return self.now


class BlockingAgent:
    def __init__(self) -> None:
        self.requests: list[AgentRequest] = []
        self.released = asyncio.Event()

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def recover_interrupted(self, conversation_id: str) -> None:  # noqa: ARG002  # test fake
        return None

    async def invoke(self, request: AgentRequest) -> AgentResult:
        self.requests.append(request)
        if request.text == "block":
            await self.released.wait()
        return AgentResult(text=f"reply:{request.text}")


class StoreScheduler:
    def __init__(self, store: CronJobStore) -> None:
        self.store = store

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None


def _pairing(tmp_path: Path, clock: Clock, *, reply: bool = True) -> SenderPairing:
    return SenderPairing(
        store=PairingStore(tmp_path / PAIRING_FILENAME),
        provider="discord",
        env_sender_ids=frozenset({OPERATOR}),
        reply=reply,
        clock=clock,
    )


def _discord(
    tmp_path: Path,
    pairing: SenderPairing | None,
    *,
    mode: ExposureMode = ExposureMode.SELF,
) -> tuple[DiscordChannel, RecordingGateway]:
    gateway = RecordingGateway()
    config = DiscordChannelConfig(
        bot_token="test-token",  # noqa: S106  # inert test token
        inbound_media_dir=tmp_path / "inbound",
        outbound_media_dir=tmp_path,
        exposure=ChannelExposure(mode=mode, operator_ids=frozenset({OPERATOR})),
        pairing=pairing,
    )
    return DiscordChannel(config, gateway=gateway), gateway


def _dm(sender: str, text: str, *, channel_id: str | None = None, is_dm: bool = True):
    return _DiscordInboundMessage(
        channel_id=channel_id or (OPERATOR_DM if sender == OPERATOR else STRANGER_DM),
        message_id="m1",
        sender_id=sender,
        text=text,
        is_dm=is_dm,
        from_self=False,
    )


async def _collecting(channel: DiscordChannel) -> list[ChannelMessage]:
    received: list[ChannelMessage] = []

    async def handler(message: ChannelMessage) -> None:
        received.append(message)

    channel.set_message_handler(handler)
    await channel.start()
    return received


def _issued_code(gateway: RecordingGateway) -> str:
    (conversation, text), *_ = gateway.sent_text
    assert conversation == STRANGER_DM
    code = text.split("code: ")[1].split(".")[0]
    return code.replace("-", "")


# --- Store ---------------------------------------------------------------


def test_codes_use_the_unambiguous_alphabet(tmp_path: Path) -> None:
    store = PairingStore(tmp_path / PAIRING_FILENAME)
    code = store.request("discord", STRANGER, STRANGER_DM, now=0)

    assert code is not None
    assert len(code) == CODE_LENGTH
    assert set(code) <= set(CODE_ALPHABET)


def test_store_file_is_private(tmp_path: Path) -> None:
    store = PairingStore(tmp_path / PAIRING_FILENAME)
    store.request("discord", STRANGER, STRANGER_DM, now=0)

    assert stat.S_IMODE((tmp_path / PAIRING_FILENAME).stat().st_mode) == 0o600


def test_valid_code_is_single_use(tmp_path: Path) -> None:
    store = PairingStore(tmp_path / PAIRING_FILENAME)
    code = cast("str", store.request("discord", STRANGER, STRANGER_DM, now=0))

    paired = store.approve("discord", format_code(code).lower(), now=1)

    assert paired is not None
    assert paired.sender_id == STRANGER
    assert store.approve("discord", code, now=2) is None


def test_expired_code_fails(tmp_path: Path) -> None:
    store = PairingStore(tmp_path / PAIRING_FILENAME)
    code = cast("str", store.request("discord", STRANGER, STRANGER_DM, now=0))

    assert store.approve("discord", code, now=CODE_TTL_SECONDS) is None
    assert not store.is_paired("discord", STRANGER)


def test_code_does_not_cross_channels(tmp_path: Path) -> None:
    store = PairingStore(tmp_path / PAIRING_FILENAME)
    code = cast("str", store.request("discord", STRANGER, STRANGER_DM, now=0))

    assert store.approve("telegram", code, now=1) is None
    assert store.approve("discord", code, now=1) is not None


def test_code_admits_only_the_sender_it_was_issued_to(tmp_path: Path) -> None:
    store = PairingStore(tmp_path / PAIRING_FILENAME)
    code = cast("str", store.request("discord", STRANGER, STRANGER_DM, now=0))
    store.request("discord", "other", "dm-other", now=0)

    store.approve("discord", code, now=1)

    assert store.is_paired("discord", STRANGER)
    assert not store.is_paired("discord", "other")


def test_repeat_requests_reuse_the_pending_code(tmp_path: Path) -> None:
    store = PairingStore(tmp_path / PAIRING_FILENAME)

    first = store.request("discord", STRANGER, STRANGER_DM, now=0)
    second = store.request("discord", STRANGER, STRANGER_DM, now=60)
    renewed = store.request("discord", STRANGER, STRANGER_DM, now=CODE_TTL_SECONDS)

    assert first is not None
    assert second is None
    assert renewed is not None


def test_pending_requests_are_capped_per_channel(tmp_path: Path) -> None:
    store = PairingStore(tmp_path / PAIRING_FILENAME)
    codes = [
        store.request("discord", f"s{index}", f"dm{index}", now=0)
        for index in range(MAX_PENDING_PER_CHANNEL + 1)
    ]

    assert all(codes[:MAX_PENDING_PER_CHANNEL])
    assert codes[-1] is None
    assert store.request("telegram", "s0", "dm0", now=0) is not None


def test_corrupt_store_fails_closed(tmp_path: Path) -> None:
    store = PairingStore(tmp_path / PAIRING_FILENAME)
    code = cast("str", store.request("discord", STRANGER, STRANGER_DM, now=0))
    store.approve("discord", code, now=1)
    (tmp_path / PAIRING_FILENAME).write_text('{"version": 1, "channels": {"discord": 5}}')

    assert not store.is_paired("discord", STRANGER)


def test_symlinked_store_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "elsewhere.json"
    target.write_text('{"version": 1, "channels": {}}')
    (tmp_path / PAIRING_FILENAME).symlink_to(target)
    store = PairingStore(tmp_path / PAIRING_FILENAME)

    with pytest.raises(OSError):  # noqa: PT011  # O_NOFOLLOW raises ELOOP
        store.request("discord", STRANGER, STRANGER_DM, now=0)
    assert target.read_text() == '{"version": 1, "channels": {}}'


# --- Discord adapter -------------------------------------------------------


async def test_unknown_dm_sender_is_rejected_and_sent_a_code(tmp_path: Path) -> None:
    channel, gateway = _discord(tmp_path, _pairing(tmp_path, Clock()))
    received = await _collecting(channel)

    await gateway.deliver_message(_dm(STRANGER, "hello"))
    await gateway.deliver_message(_dm(STRANGER, "hello again"))

    assert received == []
    assert len(gateway.sent_text) == 1
    assert len(_issued_code(gateway)) == CODE_LENGTH


async def test_silent_mode_records_the_request_without_replying(tmp_path: Path) -> None:
    pairing = _pairing(tmp_path, Clock(), reply=False)
    channel, gateway = _discord(tmp_path, pairing)
    await _collecting(channel)

    await gateway.deliver_message(_dm(STRANGER, "hello"))

    assert gateway.sent_text == []
    state = pairing.store.state("discord", now=pairing.now())
    assert list(state.pending) == [STRANGER]


async def test_guild_messages_never_issue_codes(tmp_path: Path) -> None:
    pairing = _pairing(tmp_path, Clock())
    channel, gateway = _discord(tmp_path, pairing, mode=ExposureMode.ALLOWLIST)
    await _collecting(channel)

    await gateway.deliver_message(_dm(STRANGER, "hello", channel_id="guild-chan", is_dm=False))

    assert gateway.sent_text == []
    assert pairing.store.state("discord", now=pairing.now()).pending == {}


async def test_paired_sender_is_admitted_in_any_chat(tmp_path: Path) -> None:
    pairing = _pairing(tmp_path, Clock())
    channel, gateway = _discord(tmp_path, pairing, mode=ExposureMode.ALLOWLIST)
    received = await _collecting(channel)
    await gateway.deliver_message(_dm(STRANGER, "hello"))
    pairing.store.approve("discord", _issued_code(gateway), now=pairing.now())

    await gateway.deliver_message(_dm(STRANGER, "in dm"))
    await gateway.deliver_message(_dm(STRANGER, "in guild", channel_id="guild", is_dm=False))

    assert [message.text for message in received] == ["in dm", "in guild"]


async def test_env_allowlisted_users_stay_dm_only_with_pairing(tmp_path: Path) -> None:
    pairing = SenderPairing(
        store=PairingStore(tmp_path / PAIRING_FILENAME),
        provider="discord",
        env_sender_ids=frozenset({OPERATOR, "allowed-1"}),
    )
    channel, gateway = _discord(tmp_path, pairing)
    received = await _collecting(channel)

    await gateway.deliver_message(_dm("allowed-1", "in dm", channel_id="dm-allowed"))
    await gateway.deliver_message(_dm("allowed-1", "in guild", channel_id="guild", is_dm=False))

    assert [message.text for message in received] == ["in dm"]
    assert gateway.sent_text == []


async def test_paired_sender_reactions_count_in_any_chat(tmp_path: Path) -> None:
    pairing = _pairing(tmp_path, Clock())
    code = cast("str", pairing.store.request("discord", STRANGER, STRANGER_DM, now=0))
    pairing.store.approve("discord", code, now=0)
    channel, gateway = _discord(tmp_path, pairing)
    reactions: list[ChannelReaction] = []

    async def handler(reaction: ChannelReaction) -> None:
        reactions.append(reaction)

    channel.set_reaction_handler(handler)
    await channel.start()

    for channel_id in ("guild-chan", STRANGER_DM):
        await gateway.deliver_reaction(
            _DiscordInboundReaction(
                channel_id=channel_id, message_id="m1", sender_id=STRANGER, emoji="👍"
            )
        )

    assert [reaction.conversation_id for reaction in reactions] == ["guild-chan", STRANGER_DM]


def test_pairing_is_opt_in_and_refused_with_open_exposure(tmp_path: Path) -> None:
    base = {"AGENT_ASSISTANT_ID": "a", "DEEPAGENTS_TALON_DISCORD_BOT_TOKEN": "t"}
    self_env = {**base, "DEEPAGENTS_TALON_DISCORD_OPERATOR_ID": OPERATOR}

    def build(env: dict[str, str]) -> DiscordChannelConfig:
        return DiscordChannelConfig.from_talon_config(TalonConfig.from_env(env, base_home=tmp_path))

    assert build(self_env).pairing is None
    pairing = build({**self_env, "DEEPAGENTS_TALON_DISCORD_PAIRING": "enabled"}).pairing
    assert pairing is not None
    assert OPERATOR in pairing.env_sender_ids
    with pytest.raises(ValueError, match="open exposure"):
        build(
            {
                **base,
                "DEEPAGENTS_TALON_DISCORD_EXPOSURE": "open",
                "DEEPAGENTS_TALON_DISCORD_OPEN_ACK": "allow-arbitrary-senders",
                "DEEPAGENTS_TALON_DISCORD_PAIRING": "enabled",
            }
        )


# --- Telegram adapter ------------------------------------------------------


async def test_telegram_private_chat_gets_a_code_then_access(tmp_path: Path) -> None:
    transport = RecordingTransport()
    pairing = SenderPairing(store=PairingStore(tmp_path / PAIRING_FILENAME), provider="telegram")
    channel = TelegramChannel(
        TelegramChannelConfig(
            bot_token="test-token",  # noqa: S106  # inert test token
            session_dir=tmp_path / "telegram",
            exposure=ChannelExposure(operator_ids=frozenset({"999"})),
            pairing=pairing,
        ),
        transport=cast("_TelegramTransport", transport),
    )
    received: list[ChannelMessage] = []

    async def handler(message: ChannelMessage) -> None:
        received.append(message)

    channel.set_message_handler(handler)

    await channel._process_update(_make_update(sender_id=222, chat_id=222, text="hi"))
    code = cast("str", pairing.store.state("telegram", now=pairing.now()).pending["222"].code)
    pairing.store.approve("telegram", code, now=pairing.now())
    await channel._process_update(_make_update(sender_id=222, chat_id=222, text="again"))

    replies = [params for method, params in transport.calls if method == "sendMessage"]
    assert len(replies) == 1
    assert format_code(code) in str(replies[0]["text"])
    assert [message.text for message in received] == ["again"]

    reactions: list[ChannelReaction] = []

    async def on_reaction(reaction: ChannelReaction) -> None:
        reactions.append(reaction)

    channel.set_reaction_handler(on_reaction)
    for chat_id, chat_type in ((-100500, "supergroup"), (222, "private")):
        await channel._process_update(
            _make_reaction_update(sender_id=222, chat_id=chat_id, chat_type=chat_type)
        )

    assert [reaction.conversation_id for reaction in reactions] == ["-100500", "222"]


# --- Host: operator approval and revocation --------------------------------


async def _host(tmp_path: Path, clock: Clock):
    config = TalonConfig.from_env({"AGENT_ASSISTANT_ID": "test"}, base_home=tmp_path)
    pairing = SenderPairing(
        store=PairingStore(config.home / PAIRING_FILENAME),
        provider="discord",
        env_sender_ids=frozenset({OPERATOR}),
        clock=clock,
    )
    channel, gateway = _discord(tmp_path, pairing)
    cron = CronJobStore(assistant_id="test", cron_dir=config.cron_dir)
    agent = BlockingAgent()
    host = TalonHost(config=config, agent=agent, channels=[channel], scheduler=StoreScheduler(cron))
    await host.start()
    return host, gateway, agent, cron


async def _wait_for_request(agent: BlockingAgent) -> None:
    for _ in range(100):
        if agent.requests:
            return
        await asyncio.sleep(0)
    msg = "agent received no request"
    raise AssertionError(msg)


async def test_operator_approves_code_from_their_dm(tmp_path: Path) -> None:
    host, gateway, agent, _ = await _host(tmp_path, Clock())
    await gateway.deliver_message(_dm(STRANGER, "let me in"))
    code = _issued_code(gateway)

    await gateway.deliver_message(_dm(OPERATOR, f"/pair approve {format_code(code)}"))
    await gateway.deliver_message(_dm(STRANGER, "hello agent"))
    await _wait_for_request(agent)
    await host.stop()

    assert (STRANGER_DM, APPROVED_NOTICE) in gateway.sent_text
    assert (OPERATOR_DM, f"Paired sender {STRANGER}.") in gateway.sent_text
    assert [request.text for request in agent.requests] == ["hello agent"]


async def test_non_operator_cannot_approve_and_never_reaches_the_agent(tmp_path: Path) -> None:
    host, gateway, agent, _ = await _host(tmp_path, Clock())
    await gateway.deliver_message(_dm(STRANGER, "let me in"))
    code = _issued_code(gateway)
    await gateway.deliver_message(_dm(OPERATOR, "/pair approve WRONGCODE"))
    await gateway.deliver_message(_dm(STRANGER, f"/pair approve {code}"))
    helper = "helper-1"
    store = PairingStore(tmp_path / "test" / PAIRING_FILENAME)
    store.approve(
        "discord", cast("str", store.request("discord", helper, "dm-helper", now=0)), now=0
    )

    await gateway.deliver_message(_dm(helper, f"/pair approve {code}", channel_id="dm-helper"))
    await gateway.deliver_message(_dm(OPERATOR, f"/pair approve {code}"))
    await host.stop()

    assert ("dm-helper", "Only an operator can manage sender pairing.") in gateway.sent_text
    assert (OPERATOR_DM, "No live pairing request matches that code.") in gateway.sent_text
    assert (OPERATOR_DM, f"Paired sender {STRANGER}.") in gateway.sent_text
    assert agent.requests == []


async def test_pair_is_refused_outside_a_dm(tmp_path: Path) -> None:
    host, gateway, agent, _ = await _host(tmp_path, Clock())

    await gateway.deliver_message(_dm(OPERATOR, "/pair list", channel_id="guild", is_dm=False))
    await host.stop()

    assert gateway.sent_text == [("guild", "Run /pair in a direct message with this assistant.")]
    assert agent.requests == []


def _job(cron: CronJobStore, conversation_id: str, sender_id: str | None):
    return cron.create_job(
        prompt="report",
        schedule=CronSchedule.parse("every 1h"),
        origin=CronOrigin(conversation_id=conversation_id, channel="discord", sender_id=sender_id),
    )


async def test_revoke_stops_the_run_pauses_jobs_and_blocks_the_sender(tmp_path: Path) -> None:
    host, gateway, agent, cron = await _host(tmp_path, Clock())
    await gateway.deliver_message(_dm(STRANGER, "let me in"))
    await gateway.deliver_message(_dm(OPERATOR, f"/pair approve {_issued_code(gateway)}"))
    dm_job = _job(cron, STRANGER_DM, STRANGER)
    shared_job = _job(cron, "guild", STRANGER)
    operator_job = _job(cron, "guild", OPERATOR)
    await gateway.deliver_message(_dm(STRANGER, "block", channel_id="guild", is_dm=False))
    await _wait_for_request(agent)

    await gateway.deliver_message(_dm(OPERATOR, f"/pair revoke {STRANGER}"))
    await gateway.deliver_message(_dm(STRANGER, "still here?", channel_id="guild", is_dm=False))
    await host.stop()

    assert (
        OPERATOR_DM,
        f"Revoked sender {STRANGER}. Stopped their current run. "
        "Paused 2 scheduled job(s) they created.",
    ) in gateway.sent_text
    assert [request.text for request in agent.requests] == ["block"]
    assert [(saved.id, saved.enabled) for saved in cron.list_jobs()] == [
        (dm_job.id, False),
        (shared_job.id, False),
        (operator_job.id, True),
    ]


async def _wait_for_reply(gateway: RecordingGateway, reply: tuple[str, str]) -> None:
    # Waiting for the reply, not just the request, keeps the next message from
    # landing while the turn is still settling.
    for _ in range(200):
        if reply in gateway.sent_text:
            return
        await asyncio.sleep(0)
    msg = f"no reply {reply!r}"
    raise AssertionError(msg)


class RunningBackground(StubBackground):
    """Background workers that are still running, so they have no results yet."""

    def results(self, owner: str) -> dict[str, str]:  # noqa: ARG002  # test fake
        return {}


class BackgroundWorkAgent(BlockingAgent):
    """Leaves a background worker running for any turn that asks for one."""

    def __init__(self) -> None:
        super().__init__()
        self.background = RunningBackground()

    async def invoke(self, request: AgentRequest) -> AgentResult:
        if request.text == "spawn":
            self.background.pending.add(request.conversation_id)
        return await super().invoke(request)


async def test_revoke_stops_background_work_after_someone_else_speaks(tmp_path: Path) -> None:
    config = TalonConfig.from_env({"AGENT_ASSISTANT_ID": "test"}, base_home=tmp_path)
    pairing = SenderPairing(
        store=PairingStore(config.home / PAIRING_FILENAME),
        provider="discord",
        env_sender_ids=frozenset({OPERATOR}),
    )
    channel, gateway = _discord(tmp_path, pairing)
    agent = BackgroundWorkAgent()
    host = TalonHost(config=config, agent=agent, channels=[channel])
    await host.start()
    await gateway.deliver_message(_dm(STRANGER, "let me in"))
    await gateway.deliver_message(_dm(OPERATOR, f"/pair approve {_issued_code(gateway)}"))
    await gateway.deliver_message(_dm(STRANGER, "spawn", channel_id="guild", is_dm=False))
    await _wait_for_reply(gateway, ("guild", "reply:spawn"))
    await gateway.deliver_message(_dm(OPERATOR, "my turn", channel_id="guild", is_dm=False))
    await _wait_for_reply(gateway, ("guild", "reply:my turn"))

    await gateway.deliver_message(_dm(OPERATOR, f"/pair revoke {STRANGER}"))
    await host.stop()

    assert agent.background.pending == set()
    assert (OPERATOR_DM, f"Revoked sender {STRANGER}. Stopped their current run.") in (
        gateway.sent_text
    )


async def _approved_stranger_with_job(tmp_path: Path):
    host, gateway, agent, cron = await _host(tmp_path, Clock())
    await gateway.deliver_message(_dm(STRANGER, "let me in"))
    await gateway.deliver_message(_dm(OPERATOR, f"/pair approve {_issued_code(gateway)}"))
    job = cron.create_job(
        prompt="block",
        schedule=CronSchedule.parse("every 1h"),
        origin=CronOrigin(conversation_id=STRANGER_DM, channel="discord", sender_id=STRANGER),
    )
    return host, gateway, agent, job


async def test_revoke_stops_a_scheduled_run_in_progress(tmp_path: Path) -> None:
    host, gateway, agent, job = await _approved_stranger_with_job(tmp_path)
    run = asyncio.create_task(host.run_scheduled_job(job))
    await _wait_for_request(agent)

    await gateway.deliver_message(_dm(OPERATOR, f"/pair revoke {STRANGER}"))

    with pytest.raises(ScheduledRunRevokedError):
        await run
    await host.stop()
    assert (
        OPERATOR_DM,
        f"Revoked sender {STRANGER}. Paused 1 scheduled job(s) they created. "
        "Stopped 1 scheduled run(s) in progress.",
    ) in gateway.sent_text


async def test_scheduled_run_passes_its_creator_to_jobs_it_creates(tmp_path: Path) -> None:
    host, _, agent, cron = await _host(tmp_path, Clock())
    job = cron.create_job(
        prompt="report",
        schedule=CronSchedule.parse("every 1h"),
        origin=CronOrigin(conversation_id="guild", channel="discord", sender_id=STRANGER),
    )

    await host.run_scheduled_job(job)
    await host.stop()

    assert agent.requests[0].metadata["cron_origin_sender_id"] == STRANGER


async def test_shutdown_still_cancels_a_scheduled_run(tmp_path: Path) -> None:
    host, _, agent, job = await _approved_stranger_with_job(tmp_path)
    run = asyncio.create_task(host.run_scheduled_job(job))
    await _wait_for_request(agent)

    run.cancel()

    with pytest.raises(asyncio.CancelledError):
        await run
    await host.stop()


async def test_env_senders_cannot_be_revoked_through_pairing(tmp_path: Path) -> None:
    host, gateway, _, _ = await _host(tmp_path, Clock())

    await gateway.deliver_message(_dm(OPERATOR, f"/pair revoke {OPERATOR}"))
    await host.stop()

    assert gateway.sent_text == [
        (
            OPERATOR_DM,
            f"Sender {OPERATOR} is configured in env; edit the env and restart to remove it.",
        )
    ]


# --- CLI -------------------------------------------------------------------


def _cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *args: str) -> int:
    monkeypatch.setenv("DEEPAGENTS_TALON_HOME", str(tmp_path))
    monkeypatch.setenv("DEEPAGENTS_TALON_ASSISTANT_ID", "test")
    monkeypatch.setattr(sys, "argv", ["deepagents-talon", "pairing", *args])
    with pytest.raises(SystemExit) as exited:
        talon_main.main()
    return cast("int", exited.value.code)


def test_cli_approves_lists_and_revokes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = PairingStore(tmp_path / "test" / PAIRING_FILENAME)
    code = cast("str", store.request("telegram", STRANGER, STRANGER_DM, now=int(1e10)))

    assert _cli(monkeypatch, tmp_path, "list", "telegram") == 0
    assert format_code(code) in capsys.readouterr().out
    assert _cli(monkeypatch, tmp_path, "approve", "discord", code) == 1
    assert _cli(monkeypatch, tmp_path, "approve", "telegram", code) == 0
    assert store.is_paired("telegram", STRANGER)
    cron = CronJobStore(assistant_id="test", cron_dir=tmp_path / "test" / "cron")
    job = cron.create_job(
        prompt="report",
        schedule=CronSchedule.parse("every 1h"),
        origin=CronOrigin(conversation_id=STRANGER_DM, channel="telegram", sender_id=STRANGER),
    )
    capsys.readouterr()

    assert _cli(monkeypatch, tmp_path, "revoke", "telegram", STRANGER) == 0
    assert not store.is_paired("telegram", STRANGER)
    follow_up = f"deepagents-talon pairing pause-jobs telegram {STRANGER}"
    assert follow_up in capsys.readouterr().out
    assert _cli(monkeypatch, tmp_path, *follow_up.split()[2:]) == 0
    assert [(saved.id, saved.enabled) for saved in cron.list_jobs()] == [(job.id, False)]
