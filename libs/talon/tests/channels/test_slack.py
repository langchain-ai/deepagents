from __future__ import annotations

import io
import urllib.request
from typing import TYPE_CHECKING

import pytest

from deepagents_talon.channels import slack as slack_module
from deepagents_talon.channels.base import ChannelExposure, ChannelMediaError, ExposureMode
from deepagents_talon.channels.slack import (
    MAX_TEXT_CHARS,
    SlackChannel,
    SlackChannelConfig,
    _convert_command,
    _convert_event,
    _convert_reaction,
    _RefuseRedirects,
    _SlackFile,
    _SlackInboundCommand,
    _SlackInboundMessage,
    _SlackInboundReaction,
    download_slack_file,
    format_markdown_for_slack,
)
from deepagents_talon.config import TalonConfig
from deepagents_talon.interfaces import ChannelMedia

if TYPE_CHECKING:
    from pathlib import Path

BOT = "UBOT"
OPERATOR = "UOPERATOR"


class RecordingGateway:
    """Fake `_SlackGateway` that records sends and lets tests deliver events."""

    def __init__(self) -> None:
        self.bot_id = BOT
        self.posts: list[tuple[str, str, str | None]] = []
        self.uploads: list[tuple[str, Path, str | None, str | None]] = []
        self.updates: list[tuple[str, str, str]] = []
        self._next_ts = 0
        self.handle_message = None
        self.handle_reaction = None
        self.handle_connection = None
        self.handle_command = None

    async def start(self, *, handle_message, handle_reaction, handle_connection, handle_command):
        self.handle_message = handle_message
        self.handle_reaction = handle_reaction
        self.handle_connection = handle_connection
        self.handle_command = handle_command

    async def stop(self):
        pass

    async def post_message(self, channel_id, text, *, thread_ts):
        self.posts.append((channel_id, text, thread_ts))
        self._next_ts += 1
        return f"1700000000.00000{self._next_ts}"

    async def upload_file(self, channel_id, file_path, *, thread_ts, comment):
        self.uploads.append((channel_id, file_path, thread_ts, comment))

    async def update_message(self, channel_id, ts, text):
        self.updates.append((channel_id, ts, text))


class CapturingResponder:
    def __init__(self) -> None:
        self.rejects: list[str] = []
        self.sends: list[str] = []

    async def reject(self, text):
        self.rejects.append(text)

    async def send(self, text):
        self.sends.append(text)


def _channel(
    tmp_path: Path,
    *,
    exposure: ChannelExposure | None = None,
    allowed_user_ids: frozenset[str] = frozenset(),
) -> tuple[SlackChannel, RecordingGateway, list, list]:
    gateway = RecordingGateway()
    channel = SlackChannel(
        SlackChannelConfig(
            bot_token="xoxb-test",  # noqa: S106  # inert test token
            app_token="xapp-test",  # noqa: S106  # inert test token
            inbound_media_dir=tmp_path / "inbound",
            outbound_media_dir=tmp_path,
            exposure=exposure
            or ChannelExposure(mode=ExposureMode.SELF, operator_ids=frozenset({OPERATOR})),
            allowed_user_ids=allowed_user_ids,
            max_media_bytes=1000,
        ),
        gateway=gateway,
    )
    messages: list = []
    reactions: list = []

    async def on_message(message):
        messages.append(message)

    async def on_reaction(reaction):
        reactions.append(reaction)

    channel.set_message_handler(on_message)
    channel.set_reaction_handler(on_reaction)
    return channel, gateway, messages, reactions


def _dm(text: str = "hi", *, sender: str = OPERATOR) -> _SlackInboundMessage:
    return _SlackInboundMessage(
        channel_id="D1",
        ts="1700000000.000100",
        thread_ts=None,
        sender_id=sender,
        text=text,
        is_dm=True,
    )


# Configuration


def _talon_config(tmp_path: Path, env: dict[str, str]) -> TalonConfig:
    return TalonConfig.from_env({"AGENT_ASSISTANT_ID": "assistant", **env}, base_home=tmp_path)


def test_config_requires_both_tokens(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="BOT_TOKEN"):
        SlackChannelConfig.from_talon_config(_talon_config(tmp_path, {}))
    with pytest.raises(ValueError, match="APP_TOKEN"):
        SlackChannelConfig.from_talon_config(
            _talon_config(tmp_path, {"DEEPAGENTS_TALON_SLACK_BOT_TOKEN": "xoxb-1"}),
        )


def test_config_parses_exposure_and_hides_tokens(tmp_path: Path) -> None:
    config = SlackChannelConfig.from_talon_config(
        _talon_config(
            tmp_path,
            {
                "DEEPAGENTS_TALON_SLACK_BOT_TOKEN": "xoxb-secret",
                "DEEPAGENTS_TALON_SLACK_APP_TOKEN": "xapp-secret",
                "DEEPAGENTS_TALON_SLACK_EXPOSURE": "allowlist",
                "DEEPAGENTS_TALON_SLACK_ALLOWLIST_CHATS": "C1, C2",
                "DEEPAGENTS_TALON_SLACK_ALLOWLIST_USERS": "U1",
            },
        ),
    )
    assert config.exposure.mode is ExposureMode.ALLOWLIST
    assert config.exposure.conversations == frozenset({"C1", "C2"})
    assert config.allowed_user_ids == frozenset({"U1"})
    assert "secret" not in repr(config)


def test_config_self_exposure_requires_operator(tmp_path: Path) -> None:
    env = {
        "DEEPAGENTS_TALON_SLACK_BOT_TOKEN": "xoxb-1",
        "DEEPAGENTS_TALON_SLACK_APP_TOKEN": "xapp-1",
    }
    with pytest.raises(ValueError, match="OPERATOR_ID"):
        SlackChannelConfig.from_talon_config(_talon_config(tmp_path, env))


# Event conversion and the reply-loop guard


def _event(**overrides: object) -> dict:
    event: dict = {
        "type": "message",
        "channel_type": "im",
        "channel": "D1",
        "user": OPERATOR,
        "text": "hello",
        "ts": "1700000000.000100",
    }
    event.update(overrides)
    return event


@pytest.mark.parametrize(
    "overrides",
    [
        {"user": BOT},
        {"bot_id": "B123"},
        {"subtype": "message_changed"},
        {"subtype": "bot_message"},
        {"user": None},
        # A mention in a channel also arrives as a plain `message`; only
        # `app_mention` may produce a turn, or each mention would run twice.
        {"channel_type": "channel", "channel": "C1"},
    ],
)
def test_convert_event_drops_events_that_must_not_reach_the_agent(overrides: dict) -> None:
    assert _convert_event(_event(**overrides), bot_id=BOT) is None


def test_convert_event_keeps_dm_file_share() -> None:
    event = _event(
        subtype="file_share",
        files=[
            {
                "id": "F1",
                "url_private_download": "https://files.slack.com/f",
                "name": "a.png",
                "mimetype": "image/png",
                "size": 10,
            },
        ],
    )
    message = _convert_event(event, bot_id=BOT)
    assert message is not None
    assert message.files[0].file_id == "F1"


def test_channel_mention_opens_a_thread_conversation() -> None:
    event = _event(type="app_mention", channel_type=None, channel="C1", text=f"<@{BOT}> do it")
    message = _convert_event(event, bot_id=BOT)
    assert message is not None
    assert message.text == "do it"
    assert message.conversation_id == "C1:1700000000.000100"


def test_thread_reply_joins_the_thread_conversation() -> None:
    event = _event(
        type="app_mention",
        channel_type=None,
        channel="C1",
        ts="1700000009.000900",
        thread_ts="1700000000.000100",
    )
    message = _convert_event(event, bot_id=BOT)
    assert message is not None
    assert message.conversation_id == "C1:1700000000.000100"


def test_dm_conversation_is_the_dm_channel() -> None:
    message = _convert_event(_event(thread_ts="1700000000.000001"), bot_id=BOT)
    assert message is not None
    assert message.conversation_id == "D1"


# Exposure


async def test_self_exposure_admits_only_the_operator(tmp_path: Path) -> None:
    channel, gateway, messages, _ = _channel(tmp_path)
    await channel.start()
    await gateway.handle_message(_dm(sender=OPERATOR))
    await gateway.handle_message(_dm(sender="USTRANGER"))
    assert [message.sender_id for message in messages] == [OPERATOR]
    assert messages[0].metadata["provider"] == "slack"


async def test_allowlisted_channel_admits_every_thread(tmp_path: Path) -> None:
    exposure = ChannelExposure(mode=ExposureMode.ALLOWLIST, conversations=frozenset({"C1"}))
    channel, gateway, messages, _ = _channel(tmp_path, exposure=exposure)
    await channel.start()
    for channel_id in ("C1", "C2"):
        await gateway.handle_message(
            _SlackInboundMessage(
                channel_id=channel_id,
                ts="1700000000.000100",
                thread_ts=None,
                sender_id="UANY",
                text="hi",
                is_dm=False,
            ),
        )
    assert [message.conversation_id for message in messages] == ["C1:1700000000.000100"]


async def test_allowlisted_user_may_dm(tmp_path: Path) -> None:
    exposure = ChannelExposure(mode=ExposureMode.ALLOWLIST)
    channel, gateway, messages, _ = _channel(
        tmp_path, exposure=exposure, allowed_user_ids=frozenset({"UFRIEND"})
    )
    await channel.start()
    await gateway.handle_message(_dm(sender="UFRIEND"))
    await gateway.handle_message(_dm(sender="USTRANGER"))
    assert [message.sender_id for message in messages] == ["UFRIEND"]


# Outbound


async def test_thread_reply_is_posted_into_the_thread(tmp_path: Path) -> None:
    channel, gateway, _, _ = _channel(tmp_path)
    result = await channel.send_message("C1:1700000000.000100", "hello")
    assert gateway.posts == [("C1", "hello", "1700000000.000100")]
    assert result.message_id is not None


async def test_dm_reply_is_posted_top_level(tmp_path: Path) -> None:
    channel, gateway, _, _ = _channel(tmp_path)
    await channel.send_message("D1", "hello")
    assert gateway.posts == [("D1", "hello", None)]


@pytest.mark.parametrize("conversation_id", ["", "D1:", "C1:abc", "general", "C1:1.2:3"])
async def test_malformed_conversation_id_is_rejected(tmp_path: Path, conversation_id: str) -> None:
    channel, gateway, _, _ = _channel(tmp_path)
    with pytest.raises(ValueError, match="not a Slack conversation id"):
        await channel.send_message(conversation_id, "hello")
    assert gateway.posts == []


async def test_long_text_is_split_into_slack_sized_posts(tmp_path: Path) -> None:
    channel, gateway, _, _ = _channel(tmp_path)
    await channel.send_message("D1", "word " * 2000)
    assert len(gateway.posts) > 1
    assert all(len(text) <= MAX_TEXT_CHARS for _, text, _ in gateway.posts)


async def test_media_is_uploaded_into_the_thread(tmp_path: Path) -> None:
    image = tmp_path / "chart.png"
    image.write_bytes(b"\x89PNG")
    channel, gateway, _, _ = _channel(tmp_path)
    await channel.send_media(
        "C1:1700000000.000100",
        ChannelMedia(path=image, media_type="image", caption="**Chart**"),
    )
    assert gateway.uploads == [("C1", image.resolve(), "1700000000.000100", "*Chart*")]


def test_markdown_becomes_mrkdwn() -> None:
    assert format_markdown_for_slack("**bold** and *it*") == "*bold* and _it_"
    assert format_markdown_for_slack("[docs](https://x.dev/a?b=1&c=2)") == (
        "<https://x.dev/a?b=1&amp;c=2|docs>"
    )


def test_markdown_cannot_form_slack_control_sequences() -> None:
    # Raw `<!channel>` would notify everyone in the channel; a non-web link
    # target must not become a Slack link either.
    assert format_markdown_for_slack("<!channel> hi") == "&lt;!channel&gt; hi"
    assert "<" not in format_markdown_for_slack("[x](!channel)")
    assert format_markdown_for_slack("a < b & c") == "a &lt; b &amp; c"


def test_code_is_escaped_but_not_reformatted() -> None:
    assert format_markdown_for_slack("`**x** <y>`") == "`**x** &lt;y&gt;`"
    assert format_markdown_for_slack("```\n__init__\n```") == "```\n__init__\n```"


# Reactions


def test_convert_reaction_drops_own_reactions() -> None:
    event = {
        "type": "reaction_added",
        "user": BOT,
        "reaction": "+1",
        "item": {"type": "message", "channel": "D1", "ts": "1.1"},
    }
    assert _convert_reaction(event, bot_id=BOT) is None


async def test_thumbs_up_on_a_thread_message_approves_in_that_thread(tmp_path: Path) -> None:
    channel, gateway, _, reactions = _channel(tmp_path)
    await channel.start()
    sent = await channel.send_message("C1:1700000000.000100", "Approve?")
    assert sent.message_id is not None
    await gateway.handle_reaction(
        _SlackInboundReaction(
            channel_id="C1",
            message_ts=sent.message_id,
            sender_id=OPERATOR,
            reaction="+1::skin-tone-3",
        ),
    )
    assert len(reactions) == 1
    assert reactions[0].emoji == "\U0001f44d"
    assert reactions[0].conversation_id == "C1:1700000000.000100"
    assert reactions[0].message_id == sent.message_id


async def test_reaction_from_non_operator_is_ignored(tmp_path: Path) -> None:
    channel, gateway, _, reactions = _channel(tmp_path)
    await channel.start()
    await gateway.handle_reaction(
        _SlackInboundReaction(
            channel_id="D1", message_ts="1.1", sender_id="USTRANGER", reaction="thumbsup"
        ),
    )
    assert reactions == []


# Slash command


def _command_payload(**overrides: object) -> dict:
    payload: dict = {
        "command": "/talon",
        "text": "new",
        "user_id": OPERATOR,
        "channel_id": "D1",
        "response_url": "https://hooks.slack.com/commands/T/1/abc",
        "trigger_id": "trig-1",
    }
    payload.update(overrides)
    return payload


def test_convert_command_defaults_to_help_and_rejects_foreign_response_urls() -> None:
    command = _convert_command(_command_payload(text=""))
    assert command is not None
    assert command.command == "help"
    assert _convert_command(_command_payload(response_url="https://evil.test/x")) is None
    assert _convert_command(_command_payload(response_url="http://hooks.slack.com/x")) is None
    assert _convert_command(_command_payload(command="/other")) is None


def _command(name: str, *, sender: str = OPERATOR, channel_id: str = "D1"):
    responder = CapturingResponder()
    return (
        _SlackInboundCommand(
            command=name,
            channel_id=channel_id,
            sender_id=sender,
            trigger_id="trig-1",
            responder=responder,
        ),
        responder,
    )


async def test_command_dispatches_its_typed_equivalent(tmp_path: Path) -> None:
    channel, gateway, messages, _ = _channel(tmp_path)
    await channel.start()
    command, responder = _command("new")
    await gateway.handle_command(command)
    assert [message.text for message in messages] == ["/new"]
    # The recording handler posts nothing, so the user still gets an answer.
    assert responder.sends == ["Done."]


async def test_command_reply_goes_to_the_response_url(tmp_path: Path) -> None:
    channel, gateway, _, _ = _channel(tmp_path)

    async def reply(message):
        await channel.send_message(message.conversation_id, "Started fresh.")

    channel.set_message_handler(reply)
    await channel.start()
    command, responder = _command("new")
    await gateway.handle_command(command)
    assert responder.sends == ["Started fresh."]
    assert gateway.posts == []


@pytest.mark.parametrize(
    ("name", "sender", "channel_id", "expected"),
    [
        ("new", "USTRANGER", "D1", "does not accept commands"),
        ("nonsense", OPERATOR, "D1", "not available"),
        ("new", OPERATOR, "C1", "direct message"),
    ],
)
async def test_command_rejections_are_private(
    tmp_path: Path, name: str, sender: str, channel_id: str, expected: str
) -> None:
    channel, gateway, messages, _ = _channel(tmp_path)
    await channel.start()
    command, responder = _command(name, sender=sender, channel_id=channel_id)
    await gateway.handle_command(command)
    assert messages == []
    assert len(responder.rejects) == 1
    assert expected in responder.rejects[0]


# Inbound media download


class _FakeResponse(io.BytesIO):
    def __init__(self, body: bytes, headers: dict[str, str] | None = None) -> None:
        super().__init__(body)
        self.headers = headers or {}


class _FakeOpener:
    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        self.requests: list[urllib.request.Request] = []

    def open(self, request, timeout):
        del timeout
        self.requests.append(request)
        return self.response


def _use_opener(monkeypatch: pytest.MonkeyPatch, response: _FakeResponse) -> _FakeOpener:
    opener = _FakeOpener(response)
    monkeypatch.setattr(slack_module, "_build_opener", lambda: opener)
    return opener


def test_download_sends_the_token_only_to_slack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opener = _use_opener(monkeypatch, _FakeResponse(b"data"))
    destination = tmp_path / "out.bin"
    download_slack_file(
        "https://files.slack.com/files-pri/T-F/a.png",
        destination,
        token="xoxb-t",  # noqa: S106  # inert test token
        timeout=1,
        max_bytes=100,
    )
    assert destination.read_bytes() == b"data"
    assert destination.stat().st_mode & 0o777 == 0o600
    assert opener.requests[0].get_header("Authorization") == "Bearer xoxb-t"


@pytest.mark.parametrize(
    "url",
    [
        "http://files.slack.com/a",
        "https://files.slack.com.evil.test/a",
        "https://evil.test/a",
        "https://slack.com/a",
    ],
)
def test_download_refuses_other_hosts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    opener = _use_opener(monkeypatch, _FakeResponse(b"data"))
    with pytest.raises(ChannelMediaError, match="unexpected host"):
        download_slack_file(
            url,
            tmp_path / "out.bin",
            token="xoxb-t",  # noqa: S106  # inert test token
            timeout=1,
            max_bytes=100,
        )
    assert opener.requests == []


def test_download_refuses_redirects() -> None:
    request = urllib.request.Request("https://files.slack.com/a")
    with pytest.raises(ChannelMediaError, match="redirected"):
        _RefuseRedirects().redirect_request(request, None, 302, "Found", {}, "https://evil.test/")


@pytest.mark.parametrize(
    "response",
    [
        _FakeResponse(b"x" * 10, {"content-length": "500"}),
        _FakeResponse(b"x" * 500),
    ],
)
def test_download_enforces_the_size_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, response: _FakeResponse
) -> None:
    _use_opener(monkeypatch, response)
    destination = tmp_path / "out.bin"
    with pytest.raises(ChannelMediaError, match="too large"):
        download_slack_file(
            "https://files.slack.com/a",
            destination,
            token="xoxb-t",  # noqa: S106  # inert test token
            timeout=1,
            max_bytes=100,
        )
    assert not destination.exists()


def test_download_does_not_follow_a_planted_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_opener(monkeypatch, _FakeResponse(b"data"))
    target = tmp_path / "target"
    target.write_bytes(b"original")
    destination = tmp_path / "out.bin"
    destination.symlink_to(target)
    with pytest.raises(OSError, match=r"symbolic links|Too many levels"):
        download_slack_file(
            "https://files.slack.com/a",
            destination,
            token="xoxb-t",  # noqa: S106  # inert test token
            timeout=1,
            max_bytes=100,
        )
    assert target.read_bytes() == b"original"


async def test_inbound_file_is_downloaded_and_attached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_opener(monkeypatch, _FakeResponse(b"\x89PNG"))
    channel, gateway, messages, _ = _channel(tmp_path)
    await channel.start()
    inbound = _SlackInboundMessage(
        channel_id="D1",
        ts="1700000000.000100",
        thread_ts=None,
        sender_id=OPERATOR,
        text="look",
        is_dm=True,
        files=(_SlackFile("F1", "https://files.slack.com/a", "a.png", "image/png", 4),),
    )
    await gateway.handle_message(inbound)
    metadata = messages[0].metadata
    assert metadata["has_media"] is True
    assert metadata["media_type"] == "image"
    assert str(metadata["media_path"]).endswith(".png")


async def test_oversized_inbound_file_is_reported_not_downloaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opener = _use_opener(monkeypatch, _FakeResponse(b""))
    channel, gateway, messages, _ = _channel(tmp_path)
    await channel.start()
    inbound = _SlackInboundMessage(
        channel_id="D1",
        ts="1700000000.000100",
        thread_ts=None,
        sender_id=OPERATOR,
        text="look",
        is_dm=True,
        files=(_SlackFile("F1", "https://files.slack.com/a", "a.png", "image/png", 5000),),
    )
    await gateway.handle_message(inbound)
    assert messages[0].metadata["has_media"] is False
    assert "too large" in str(messages[0].metadata["media_error"])
    assert opener.requests == []


class _BrokenResponse(_FakeResponse):
    def read(self, size=-1):
        if self.tell() > 0:
            msg = "connection reset"
            raise ConnectionResetError(msg)
        return super().read(size)


def test_download_failure_mid_body_leaves_no_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_opener(monkeypatch, _BrokenResponse(b"x" * 70_000))
    destination = tmp_path / "out.bin"
    with pytest.raises(ConnectionResetError):
        download_slack_file(
            "https://files.slack.com/a",
            destination,
            token="xoxb-t",  # noqa: S106  # inert test token
            timeout=1,
            max_bytes=1_000_000,
        )
    assert not destination.exists()


def test_truncated_download_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_opener(monkeypatch, _FakeResponse(b"abcd", {"content-length": "10"}))
    destination = tmp_path / "out.bin"
    with pytest.raises(ChannelMediaError, match="incomplete"):
        download_slack_file(
            "https://files.slack.com/a",
            destination,
            token="xoxb-t",  # noqa: S106  # inert test token
            timeout=1,
            max_bytes=100,
        )
    assert not destination.exists()


class _FakeWebClient:
    def __init__(self, upload_url: str) -> None:
        self.upload_url = upload_url
        self.completed: list[dict] = []

    async def files_getUploadURLExternal(self, *, filename, length):  # noqa: N802  # Slack API name
        del filename, length
        return {"upload_url": self.upload_url, "file_id": "F1"}

    async def files_completeUploadExternal(self, **kwargs: object):  # noqa: N802  # Slack API name
        self.completed.append(kwargs)


async def test_upload_streams_to_slack_and_completes_in_the_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "chart.png"
    image.write_bytes(b"\x89PNG")
    gateway = slack_module._SlackSdkGateway(bot_token="b", app_token="a", timeout_seconds=1)  # noqa: S106  # inert test token
    web = _FakeWebClient("https://files.slack.com/upload/v1/abc")
    gateway._web = web  # type: ignore[assignment]
    streamed: list[tuple[str, Path]] = []

    async def fake_stream(upload_url, file_path):
        streamed.append((upload_url, file_path))

    monkeypatch.setattr(gateway, "_stream_upload", fake_stream)
    await gateway.upload_file("C1", image, thread_ts="1.2", comment="hi")
    assert streamed == [("https://files.slack.com/upload/v1/abc", image)]
    assert web.completed == [
        {
            "files": [{"id": "F1", "title": "chart.png"}],
            "channel_id": "C1",
            "initial_comment": "hi",
            "thread_ts": "1.2",
        },
    ]


async def test_upload_refuses_a_foreign_upload_url(tmp_path: Path) -> None:
    image = tmp_path / "chart.png"
    image.write_bytes(b"\x89PNG")
    gateway = slack_module._SlackSdkGateway(bot_token="b", app_token="a", timeout_seconds=1)  # noqa: S106  # inert test token
    web = _FakeWebClient("https://evil.test/upload")
    gateway._web = web  # type: ignore[assignment]
    with pytest.raises(ChannelMediaError, match="unexpected host"):
        await gateway.upload_file("C1", image, thread_ts=None, comment=None)
    assert web.completed == []
