"""Sender pairing on Slack, where operators approve through `/talon pair`."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from deepagents_talon.channels.base import ChannelExposure, ExposureMode
from deepagents_talon.channels.slack import (
    SlackChannel,
    SlackChannelConfig,
    _convert_command,
    _SlackInboundCommand,
    _SlackInboundMessage,
    _SlackInboundReaction,
)
from deepagents_talon.config import TalonConfig
from deepagents_talon.host import TalonHost
from deepagents_talon.pairing import (
    APPROVED_NOTICE,
    PAIRING_FILENAME,
    PairingStore,
    SenderPairing,
    format_code,
)
from tests.channels.test_slack import CapturingResponder, RecordingGateway, _command_payload
from tests.unit_tests.test_pairing import BlockingAgent, _cli, _wait_for_request

if TYPE_CHECKING:
    from pathlib import Path

    from deepagents_talon.interfaces import ChannelMessage, ChannelReaction

OPERATOR = "UOPERATOR"
OPERATOR_DM = "DOPERATOR"
COWORKER = "UCOWORKER"
COWORKER_DM = "DCOWORKER"


def _pairing(home: Path) -> SenderPairing:
    return SenderPairing(
        store=PairingStore(home / PAIRING_FILENAME),
        provider="slack",
        env_sender_ids=frozenset({OPERATOR}),
    )


def _slack(pairing: SenderPairing | None, tmp_path: Path) -> tuple[SlackChannel, RecordingGateway]:
    gateway = RecordingGateway()
    config = SlackChannelConfig(
        bot_token="xoxb-test",  # noqa: S106  # inert test token
        app_token="xapp-test",  # noqa: S106  # inert test token
        inbound_media_dir=tmp_path / "inbound",
        outbound_media_dir=tmp_path,
        exposure=ChannelExposure(mode=ExposureMode.SELF, operator_ids=frozenset({OPERATOR})),
        pairing=pairing,
    )
    return SlackChannel(config, gateway=gateway), gateway


def _message(sender: str, text: str, *, channel_id: str | None = None, is_dm: bool = True):
    return _SlackInboundMessage(
        channel_id=channel_id or (OPERATOR_DM if sender == OPERATOR else COWORKER_DM),
        ts="1700000000.000100",
        thread_ts=None,
        sender_id=sender,
        text=text,
        is_dm=is_dm,
    )


def _talon(sender: str, text: str) -> tuple[_SlackInboundCommand, CapturingResponder]:
    responder = CapturingResponder()
    name, _, argument = text.partition(" ")
    command = _SlackInboundCommand(
        command=name,
        channel_id=OPERATOR_DM if sender == OPERATOR else COWORKER_DM,
        sender_id=sender,
        trigger_id="trig-1",
        responder=responder,
        argument=argument or None,
    )
    return command, responder


def _issued_code(gateway: RecordingGateway) -> str:
    (channel_id, text, _), *_ = gateway.posts
    assert channel_id == COWORKER_DM
    return text.split("code: ")[1].split(".")[0].replace("-", "")


async def _collecting(channel: SlackChannel) -> list[ChannelMessage]:
    received: list[ChannelMessage] = []

    async def handler(message: ChannelMessage) -> None:
        received.append(message)

    channel.set_message_handler(handler)
    await channel.start()
    return received


def test_command_argument_is_passed_through() -> None:
    bare = _convert_command(_command_payload(text="new"))
    pair = _convert_command(_command_payload(text="pair approve  K7QM-3XRD "))

    assert bare is not None
    assert bare.argument is None
    assert pair is not None
    assert (pair.command, pair.argument) == ("pair", "approve  K7QM-3XRD")


async def test_unknown_dm_gets_one_code_and_mentions_get_none(tmp_path: Path) -> None:
    channel, gateway = _slack(_pairing(tmp_path), tmp_path)
    received = await _collecting(channel)

    await gateway.handle_message(_message(COWORKER, "hi"))
    await gateway.handle_message(_message(COWORKER, "hi again"))
    await gateway.handle_message(_message("USTRANGER", "hey", channel_id="C1", is_dm=False))

    assert received == []
    assert len(gateway.posts) == 1
    assert len(_issued_code(gateway)) == len("K7QM3XRD")


async def test_paired_coworker_is_admitted_in_their_dm_only(tmp_path: Path) -> None:
    pairing = _pairing(tmp_path)
    channel, gateway = _slack(pairing, tmp_path)
    received = await _collecting(channel)
    await gateway.handle_message(_message(COWORKER, "hi"))
    pairing.store.approve("slack", _issued_code(gateway), now=pairing.now())
    reactions: list[ChannelReaction] = []

    async def on_reaction(reaction: ChannelReaction) -> None:
        reactions.append(reaction)

    channel.set_reaction_handler(on_reaction)

    await gateway.handle_message(_message(COWORKER, "in dm"))
    await gateway.handle_message(_message(COWORKER, "in channel", channel_id="C1", is_dm=False))
    for channel_id in ("C1", COWORKER_DM):
        await gateway.handle_reaction(
            _SlackInboundReaction(
                channel_id=channel_id, message_ts="1.1", sender_id=COWORKER, reaction="+1"
            )
        )

    assert [message.text for message in received] == ["in dm"]
    assert [reaction.conversation_id for reaction in reactions] == [COWORKER_DM]


async def test_operator_pairs_a_coworker_with_talon_pair(tmp_path: Path) -> None:
    config = TalonConfig.from_env({"AGENT_ASSISTANT_ID": "test"}, base_home=tmp_path)
    channel, gateway = _slack(_pairing(config.home), tmp_path)
    agent = BlockingAgent()
    host = TalonHost(config=config, agent=agent, channels=[channel])
    await host.start()
    await gateway.handle_message(_message(COWORKER, "let me in"))
    code = format_code(_issued_code(gateway))

    refused, refused_responder = _talon(COWORKER, f"pair approve {code}")
    await gateway.handle_command(refused)
    approve, responder = _talon(OPERATOR, f"pair approve {code}")
    await gateway.handle_command(approve)
    await gateway.handle_message(_message(COWORKER, "hello agent"))
    await _wait_for_request(agent)
    await host.stop()

    assert refused_responder.rejects == ["This assistant does not accept commands from you."]
    assert responder.sends == [f"Paired sender {COWORKER}."]
    assert (COWORKER_DM, APPROVED_NOTICE, None) in gateway.posts
    assert [request.text for request in agent.requests] == ["hello agent"]


def test_slack_pairing_is_opt_in_and_refused_with_open_exposure(tmp_path: Path) -> None:
    base = {
        "AGENT_ASSISTANT_ID": "a",
        "DEEPAGENTS_TALON_SLACK_BOT_TOKEN": "xoxb-t",
        "DEEPAGENTS_TALON_SLACK_APP_TOKEN": "xapp-t",
    }
    self_env = {**base, "DEEPAGENTS_TALON_SLACK_OPERATOR_ID": OPERATOR}

    def build(env: dict[str, str]) -> SlackChannelConfig:
        return SlackChannelConfig.from_talon_config(TalonConfig.from_env(env, base_home=tmp_path))

    assert build(self_env).pairing is None
    pairing = cast(
        "SenderPairing", build({**self_env, "DEEPAGENTS_TALON_SLACK_PAIRING": "enabled"}).pairing
    )
    assert (pairing.provider, OPERATOR in pairing.env_sender_ids) == ("slack", True)
    with pytest.raises(ValueError, match="open exposure"):
        build(
            {
                **base,
                "DEEPAGENTS_TALON_SLACK_EXPOSURE": "open",
                "DEEPAGENTS_TALON_SLACK_OPEN_ACK": "allow-arbitrary-senders",
                "DEEPAGENTS_TALON_SLACK_PAIRING": "enabled",
            }
        )


def test_cli_approves_slack_codes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = PairingStore(tmp_path / "test" / PAIRING_FILENAME)
    code = cast("str", store.request("slack", COWORKER, COWORKER_DM, now=int(1e10)))

    assert _cli(monkeypatch, tmp_path, "approve", "slack", code) == 0
    assert store.is_paired("slack", COWORKER)
