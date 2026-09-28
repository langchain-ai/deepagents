"""Native steering keeps one renderer and waits for server registration."""

import asyncio
from collections.abc import AsyncGenerator

import pytest

from deepagents_code.client.steering import (
    SteeredError,
    SteeringControl,
    SteeringInput,
    steerable_stream,
)


async def test_steer_waits_for_registration_and_closes_old_stream() -> None:
    control = SteeringControl()
    starting = asyncio.Event()
    registered = asyncio.Event()
    closed = asyncio.Event()

    async def stream() -> AsyncGenerator[str, None]:
        starting.set()
        try:
            await registered.wait()
            control.registered.set()
            await asyncio.Event().wait()
            yield "unreachable"
        finally:
            closed.set()

    output = steerable_stream(stream(), control)
    task = asyncio.create_task(anext(output))
    await starting.wait()
    assert control.submit("do this instead")
    await asyncio.sleep(0)
    assert not task.done()
    assert not closed.is_set()
    registered.set()
    with pytest.raises(SteeredError, match="do this instead"):
        await asyncio.wait_for(task, 1)
    assert closed.is_set()
    assert control.detached
    assert control.unsent == SteeringInput("do this instead")


async def test_pending_steer_is_preserved_when_stream_fails() -> None:
    control = SteeringControl()
    assert control.submit("keep this")

    async def stream() -> AsyncGenerator[str, None]:
        await asyncio.sleep(0)
        msg = "connection failed"
        raise RuntimeError(msg)
        yield "unreachable"

    with pytest.raises(RuntimeError, match="connection failed"):
        async for _ in steerable_stream(stream(), control):
            pass
    assert control.unsent == SteeringInput("keep this")
    assert not control.accepting


@pytest.mark.parametrize("register", [False, True])
async def test_additional_steer_survives_replacement_registration(
    *, register: bool
) -> None:
    from deepagents_code.input import MediaTracker
    from deepagents_code.media_utils import ImageData

    first_media = MediaTracker()
    first_media.add_image(ImageData("first", "png", ""))
    first = SteeringInput("first [image 1]", first_media)
    control = SteeringControl(detached=True, unsent=first)
    second_media = MediaTracker()
    second_media.add_image(ImageData("second", "png", ""))
    assert control.submit("second [image 1]", second_media)
    second_media.clear()

    async def replacement() -> AsyncGenerator[str, None]:
        # Let the steering consumer run while the first handoff is still pending.
        await asyncio.sleep(0)
        if register:
            control.unsent = None
            control.detached = False
            control.registered.set()
            await asyncio.Event().wait()
        msg = "replacement failed"
        raise RuntimeError(msg)
        yield "unreachable"

    output = steerable_stream(replacement(), control)
    if register:
        with pytest.raises(SteeredError, match="second"):
            await asyncio.wait_for(anext(output), 1)
        second = control.unsent
        assert control.pending.empty()
    else:
        with pytest.raises(RuntimeError, match="replacement failed"):
            await anext(output)
        assert control.unsent is first
        assert first.media is not None
        assert first.media.get_images() == [ImageData("first", "png", "[image 1]")]
        assert not control.pending.empty()
        second = control.pending.get_nowait()
    assert second is not None
    assert second.text == "second [image 1]"
    assert second.media is not None
    assert second.media.get_images() == [ImageData("second", "png", "[image 1]")]


async def test_normal_stream_is_unchanged() -> None:
    control = SteeringControl()

    async def stream() -> AsyncGenerator[int, None]:
        for item in range(3):
            await asyncio.sleep(0)
            yield item

    assert [item async for item in steerable_stream(stream(), control)] == [0, 1, 2]
    assert not control.detached
    assert not control.submit("too late")


async def test_unregistered_end_preserves_prompt_without_handoff() -> None:
    control = SteeringControl()
    assert control.submit("keep this")

    async def stream() -> AsyncGenerator[str, None]:
        await asyncio.sleep(0)
        yield "last event"

    assert [item async for item in steerable_stream(stream(), control)] == [
        "last event"
    ]
    assert not control.detached
    assert control.unsent == SteeringInput("keep this")


async def test_cancel_closes_stream_and_keeps_accepted_prompt() -> None:
    control = SteeringControl()
    started = asyncio.Event()
    closed = asyncio.Event()

    async def stream() -> AsyncGenerator[str, None]:
        started.set()
        try:
            await asyncio.Event().wait()
            yield "unreachable"
        finally:
            closed.set()

    output = steerable_stream(stream(), control)
    task = asyncio.create_task(anext(output))
    await started.wait()
    assert control.submit("keep this")
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()
    assert control.unsent == SteeringInput("keep this")


def test_second_pending_steer_is_not_silently_accepted() -> None:
    control = SteeringControl()
    assert control.submit("first")
    assert not control.submit("second")
    assert control.pending.get_nowait() == SteeringInput("first")


async def test_delayed_registration_keeps_submitted_media_isolated() -> None:
    from deepagents_code.input import MediaTracker
    from deepagents_code.media_utils import ImageData

    composer = MediaTracker()
    original = ImageData("original", "png", "")
    placeholder = composer.add_image(original)
    control = SteeringControl()
    started = asyncio.Event()
    registered = asyncio.Event()

    async def stream() -> AsyncGenerator[str, None]:
        started.set()
        await registered.wait()
        control.registered.set()
        await asyncio.Event().wait()
        yield "unreachable"

    task = asyncio.create_task(anext(steerable_stream(stream(), control)))
    await started.wait()
    assert control.submit(f"instead {placeholder}", composer)
    original.base64_data = "changed"
    composer.sync_to_text("next draft")
    composer.add_image(ImageData("draft", "jpeg", ""))
    registered.set()
    with pytest.raises(SteeredError) as raised:
        await asyncio.wait_for(task, 1)
    request = raised.value.request
    assert request is control.unsent
    assert request.media is not None
    assert request.media.get_images() == [ImageData("original", "png", placeholder)]
    outbound = request.media.snapshot()
    outbound.clear()
    assert request.media.get_images() == [ImageData("original", "png", placeholder)]
    assert composer.get_images() == [ImageData("draft", "jpeg", "[image 1]")]
