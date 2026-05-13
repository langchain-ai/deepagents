"""FSM tests using a recording fake compositor (no OBS, no HTTP)."""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, TypeVar

import pytest

from control_server.state_config import StateConfig
from control_server.state_machine import (
    Event,
    InvalidTransitionError,
    Phase,
    StateMachine,
)

T = TypeVar("T")


def _run(coro: Awaitable[T]) -> T:
    """Sync wrapper for async dispatch. Avoids pytest-asyncio dependency."""
    return asyncio.run(coro)  # type: ignore[arg-type]


class FakeCompositor:
    """Records every `set_scene` / `set_text` call in order."""

    def __init__(self) -> None:
        self.scenes: list[str] = []
        self.texts: list[tuple[str, str]] = []

    async def set_scene(self, name: str) -> None:
        self.scenes.append(name)

    async def set_text(self, source: str, value: str) -> None:
        self.texts.append((source, value))

    def last_text(self, source: str) -> str | None:
        """Return the most recent value written to `source`, or `None`."""
        for s, v in reversed(self.texts):
            if s == source:
                return v
        return None


@pytest.fixture
def config() -> StateConfig:
    return StateConfig(
        scenes={
            Phase.IDLE: "Idle",
            Phase.CODING: "Coding",
            Phase.SCOREBOARD: "Scoreboard",
        },
        text_prompt="PromptText",
        contestant_name_template="Contestant{n}Name",
        contestant_score_template="Contestant{n}Score",
    )


def test_full_round_drives_expected_scenes(config: StateConfig) -> None:
    comp = FakeCompositor()
    machine = StateMachine(comp, config)

    async def run() -> None:
        await machine.prime()
        await machine.dispatch(
            Event.START,
            {"prompt": "build a cat shrine", "contestants": ["Alice", "Bob"]},
        )
        assert machine.snapshot.phase is Phase.CODING
        assert machine.snapshot.prompt == "build a cat shrine"
        await machine.dispatch(Event.END, {"scores": {"Alice": 8.2, "Bob": 7.5}})
        assert machine.snapshot.phase is Phase.SCOREBOARD
        await machine.dispatch(Event.RESET, {})
        assert machine.snapshot.phase is Phase.IDLE

    _run(run())

    assert comp.last_text("PromptText") == ""
    assert comp.scenes == ["Idle", "Coding", "Scoreboard", "Idle"]
    assert comp.last_text("Contestant1Name") == ""
    assert comp.last_text("Contestant1Score") == ""


def test_illegal_transition_rejected(config: StateConfig) -> None:
    machine = StateMachine(FakeCompositor(), config)
    with pytest.raises(InvalidTransitionError):
        _run(machine.dispatch(Event.END, {}))


def test_ready_players_render_in_idle_scene(config: StateConfig) -> None:
    comp = FakeCompositor()
    machine = StateMachine(comp, config)

    async def run() -> None:
        await machine.prime()
        await machine.dispatch(Event.READY, {"contestants": ["Alice", "Bob"]})

    _run(run())

    assert machine.snapshot.phase is Phase.IDLE
    assert machine.snapshot.contestants == ["Alice", "Bob"]
    assert comp.scenes == ["Idle", "Idle"]
    assert comp.last_text("Contestant1Name") == "Alice"
    assert comp.last_text("Contestant2Name") == "Bob"
    assert comp.last_text("Contestant1Score") == ""
    assert comp.last_text("Contestant2Score") == ""


def test_ready_rejects_empty_contestants(config: StateConfig) -> None:
    machine = StateMachine(FakeCompositor(), config)

    with pytest.raises(InvalidTransitionError, match="contestants"):
        _run(machine.dispatch(Event.READY, {"contestants": []}))


def test_ready_rejected_during_coding(config: StateConfig) -> None:
    machine = StateMachine(FakeCompositor(), config)

    async def run() -> Any:
        await machine.dispatch(Event.START, {"prompt": "r1", "contestants": ["A"]})
        return await machine.dispatch(Event.READY, {"contestants": ["B"]})

    with pytest.raises(InvalidTransitionError):
        _run(run())


def test_start_from_scoreboard_rolls_next_round(config: StateConfig) -> None:
    comp = FakeCompositor()
    machine = StateMachine(comp, config)

    async def run() -> None:
        await machine.prime()
        await machine.dispatch(Event.START, {"prompt": "r1", "contestants": ["A", "B"]})
        await machine.dispatch(Event.END, {"scores": {"A": 1.0, "B": 2.0}})
        await machine.dispatch(Event.START, {"prompt": "r2", "contestants": ["C", "D"]})

    _run(run())

    assert machine.snapshot.phase is Phase.CODING
    assert machine.snapshot.prompt == "r2"
    assert machine.snapshot.contestants == ["C", "D"]
    assert machine.snapshot.scores == {}
    assert comp.last_text("Contestant1Name") == "C"
    assert comp.last_text("Contestant2Name") == "D"


def test_start_from_coding_still_rejected(config: StateConfig) -> None:
    machine = StateMachine(FakeCompositor(), config)
    _run(machine.dispatch(Event.START, {"prompt": "r1", "contestants": ["A"]}))
    with pytest.raises(InvalidTransitionError):
        _run(machine.dispatch(Event.START, {"prompt": "r2", "contestants": ["B"]}))


def test_reset_is_always_legal(config: StateConfig) -> None:
    machine = StateMachine(FakeCompositor(), config)

    async def run() -> None:
        await machine.dispatch(Event.RESET, {})
        assert machine.snapshot.phase is Phase.IDLE
        await machine.dispatch(Event.START, {"prompt": "x", "contestants": ["A"]})
        await machine.dispatch(Event.RESET, {})
        assert machine.snapshot.phase is Phase.IDLE
        await machine.dispatch(Event.START, {"prompt": "x", "contestants": ["A"]})
        await machine.dispatch(Event.END, {"scores": {"A": 1.0}})
        await machine.dispatch(Event.RESET, {})
        assert machine.snapshot.phase is Phase.IDLE

    _run(run())


def test_score_mapped_to_contestant_slot(config: StateConfig) -> None:
    comp = FakeCompositor()
    machine = StateMachine(comp, config)

    async def run() -> None:
        await machine.prime()
        await machine.dispatch(
            Event.START,
            {"prompt": "x", "contestants": ["Bob", "Alice"]},
        )
        await machine.dispatch(Event.END, {"scores": {"Alice": 9.9, "Bob": 3.0}})

    _run(run())

    assert comp.last_text("Contestant1Score") == "3.00"   # Bob
    assert comp.last_text("Contestant2Score") == "9.90"   # Alice


def test_missing_score_leaves_slot_blank(config: StateConfig) -> None:
    comp = FakeCompositor()
    machine = StateMachine(comp, config)

    async def run() -> None:
        await machine.prime()
        await machine.dispatch(
            Event.START,
            {"prompt": "x", "contestants": ["Alice", "Bob"]},
        )
        await machine.dispatch(Event.END, {"scores": {"Alice": 8.0}})

    _run(run())

    assert comp.last_text("Contestant1Score") == "8.00"
    assert comp.last_text("Contestant2Score") == ""


def test_score_sources_are_optional() -> None:
    config = StateConfig(
        scenes={
            Phase.IDLE: "coding",
            Phase.CODING: "coding",
            Phase.SCOREBOARD: "coding",
        },
        contestant_name_template="Contestant{n}Name",
        contestant_score_template=None,
    )
    comp = FakeCompositor()
    machine = StateMachine(comp, config)

    async def run() -> None:
        await machine.dispatch(Event.START, {"prompt": "r1", "contestants": ["Alice", "Bob"]})
        await machine.dispatch(Event.END, {"scores": {"Alice": 8.2, "Bob": 7.5}})

    _run(run())

    assert comp.last_text("Contestant1Name") == "Alice"
    assert comp.last_text("Contestant2Name") == "Bob"
    assert comp.last_text("Contestant1Score") is None
    assert comp.last_text("Contestant2Score") is None


def test_text_sources_are_optional() -> None:
    config = StateConfig(
        scenes={
            Phase.IDLE: "coding",
            Phase.CODING: "coding",
            Phase.SCOREBOARD: "coding",
        },
        text_prompt=None,
        contestant_name_template=None,
        contestant_score_template=None,
    )
    comp = FakeCompositor()
    machine = StateMachine(comp, config)

    async def run() -> None:
        await machine.prime()
        await machine.dispatch(Event.START, {"prompt": "r1", "contestants": ["Alice", "Bob"]})
        await machine.dispatch(Event.END, {"scores": {"Alice": 8.2, "Bob": 7.5}})

    _run(run())

    assert comp.scenes == ["coding", "coding", "coding"]
    assert comp.texts == []


def test_end_rejects_missing_scores(config: StateConfig) -> None:
    machine = StateMachine(FakeCompositor(), config)
    _run(machine.dispatch(Event.START, {"prompt": "x", "contestants": ["Alice"]}))

    with pytest.raises(InvalidTransitionError, match="scores"):
        _run(machine.dispatch(Event.END, {}))


def test_end_rejects_non_numeric_scores(config: StateConfig) -> None:
    machine = StateMachine(FakeCompositor(), config)
    _run(machine.dispatch(Event.START, {"prompt": "x", "contestants": ["Alice"]}))

    with pytest.raises(InvalidTransitionError, match="numeric"):
        _run(machine.dispatch(Event.END, {"scores": {"Alice": "nope"}}))


def test_contestants_beyond_slot_cap_are_dropped(config: StateConfig) -> None:
    comp = FakeCompositor()
    machine = StateMachine(comp, config)

    async def run() -> None:
        await machine.prime()
        await machine.dispatch(
            Event.START,
            {"prompt": "x", "contestants": ["A", "B", "C"]},
        )

    _run(run())

    assert comp.last_text("Contestant1Name") == "A"
    assert comp.last_text("Contestant2Name") == "B"
    assert comp.last_text("Contestant3Name") is None
