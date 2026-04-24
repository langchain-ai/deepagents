"""FSM tests using a recording fake compositor (no OBS required)."""

from __future__ import annotations

import pytest

from obs_runner.config import Config
from obs_runner.state_machine import (
    Event,
    InvalidTransitionError,
    Phase,
    StateMachine,
)


class FakeCompositor:
    """Records every `set_scene` / `set_text` call in order."""

    def __init__(self) -> None:
        self.scenes: list[str] = []
        self.texts: list[tuple[str, str]] = []

    def set_scene(self, name: str) -> None:
        self.scenes.append(name)

    def set_text(self, source: str, value: str) -> None:
        self.texts.append((source, value))

    def last_text(self, source: str) -> str | None:
        """Return the most recent value written to `source`, or `None`."""
        for s, v in reversed(self.texts):
            if s == source:
                return v
        return None


@pytest.fixture
def config() -> Config:
    return Config(
        scenes={
            Phase.IDLE: "Idle",
            Phase.CODING: "Coding",
            Phase.SCOREBOARD: "Scoreboard",
        },
    )


def test_full_round_drives_expected_scenes(config: Config) -> None:
    comp = FakeCompositor()
    machine = StateMachine(comp, config)
    machine.prime()

    machine.dispatch(
        Event.START,
        {
            "prompt": "build a cat shrine",
            "contestants": ["Alice", "Bob"],
        },
    )
    assert machine.snapshot.phase is Phase.CODING
    assert machine.snapshot.prompt == "build a cat shrine"
    assert comp.last_text("PromptText") == "build a cat shrine"
    assert comp.last_text("RoundNum") is None
    assert comp.last_text("Contestant1Name") == "Alice"
    assert comp.last_text("Contestant2Name") == "Bob"
    assert comp.last_text("Contestant1Score") == ""
    assert comp.last_text("Contestant2Score") == ""
    assert comp.last_text("Contestant3Name") is None

    machine.dispatch(Event.END, {"scores": {"Alice": 8.2, "Bob": 7.5}})
    assert machine.snapshot.phase is Phase.SCOREBOARD
    assert comp.last_text("Contestant1Score") == "8.20"
    assert comp.last_text("Contestant2Score") == "7.50"
    assert comp.last_text("Contestant1Name") == "Alice"
    assert comp.last_text("Contestant2Name") == "Bob"

    machine.dispatch(Event.RESET, {})
    assert machine.snapshot.phase is Phase.IDLE
    assert comp.scenes == ["Idle", "Coding", "Scoreboard", "Idle"]
    assert comp.last_text("Contestant1Name") == ""
    assert comp.last_text("Contestant1Score") == ""


def test_illegal_transition_rejected(config: Config) -> None:
    machine = StateMachine(FakeCompositor(), config)
    with pytest.raises(InvalidTransitionError):
        machine.dispatch(Event.END, {})


def test_start_from_scoreboard_rolls_next_round(config: Config) -> None:
    """A producer should not need an explicit reset between rounds."""
    comp = FakeCompositor()
    machine = StateMachine(comp, config)
    machine.prime()
    machine.dispatch(Event.START, {"prompt": "r1", "contestants": ["A", "B"]})
    machine.dispatch(Event.END, {"scores": {"A": 1.0, "B": 2.0}})
    # Start straight from SCOREBOARD with a new prompt/contestants.
    machine.dispatch(Event.START, {"prompt": "r2", "contestants": ["C", "D"]})
    assert machine.snapshot.phase is Phase.CODING
    assert machine.snapshot.prompt == "r2"
    assert machine.snapshot.contestants == ["C", "D"]
    # Previous scores are replaced by the empty default on re-entry.
    assert machine.snapshot.scores == {}
    assert comp.last_text("Contestant1Name") == "C"
    assert comp.last_text("Contestant2Name") == "D"


def test_start_from_coding_still_rejected(config: Config) -> None:
    """`start` from CODING must 409 — otherwise it hides a missing END."""
    machine = StateMachine(FakeCompositor(), config)
    machine.dispatch(Event.START, {"prompt": "r1", "contestants": ["A"]})
    with pytest.raises(InvalidTransitionError):
        machine.dispatch(Event.START, {"prompt": "r2", "contestants": ["B"]})


def test_reset_is_always_legal(config: Config) -> None:
    """Reset must be an idempotent panic-button from every phase.

    Producers double-click the Reset control between rounds; a 409 on
    the no-op path is an annoyance, not a signal.
    """
    machine = StateMachine(FakeCompositor(), config)
    # From IDLE
    machine.dispatch(Event.RESET, {})
    assert machine.snapshot.phase is Phase.IDLE
    # From CODING
    machine.dispatch(Event.START, {"prompt": "x", "contestants": ["A"]})
    machine.dispatch(Event.RESET, {})
    assert machine.snapshot.phase is Phase.IDLE
    # From SCOREBOARD
    machine.dispatch(Event.START, {"prompt": "x", "contestants": ["A"]})
    machine.dispatch(Event.END, {"scores": {"A": 1.0}})
    machine.dispatch(Event.RESET, {})
    assert machine.snapshot.phase is Phase.IDLE


def test_score_mapped_to_contestant_slot(config: Config) -> None:
    """Scores must land in the same slot as their contestant in `CODING`.

    Slot order is stable across phases so viewers can follow a
    contestant's position from the coding scene into the scoreboard.
    """
    comp = FakeCompositor()
    machine = StateMachine(comp, config)
    machine.prime()
    machine.dispatch(
        Event.START,
        {
            "prompt": "x",
            "contestants": ["Bob", "Alice"],
        },
    )
    machine.dispatch(
        Event.END,
        {"scores": {"Alice": 9.9, "Bob": 3.0}},
    )

    assert comp.last_text("Contestant1Score") == "3.00"   # Bob
    assert comp.last_text("Contestant2Score") == "9.90"   # Alice


def test_missing_score_leaves_slot_blank(config: Config) -> None:
    comp = FakeCompositor()
    machine = StateMachine(comp, config)
    machine.prime()
    machine.dispatch(
        Event.START,
        {"prompt": "x", "contestants": ["Alice", "Bob"]},
    )
    machine.dispatch(Event.END, {"scores": {"Alice": 8.0}})

    assert comp.last_text("Contestant1Score") == "8.00"
    assert comp.last_text("Contestant2Score") == ""


def test_contestants_beyond_slot_cap_are_dropped(config: Config) -> None:
    comp = FakeCompositor()
    machine = StateMachine(comp, config)
    machine.prime()
    machine.dispatch(
        Event.START,
        {"prompt": "x", "contestants": ["A", "B", "C"]},
    )
    assert comp.last_text("Contestant1Name") == "A"
    assert comp.last_text("Contestant2Name") == "B"
    # No slot 3 was written since CONTESTANT_SLOTS == 2.
    assert comp.last_text("Contestant3Name") is None
