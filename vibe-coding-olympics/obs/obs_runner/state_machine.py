"""Round state machine for the Vibe Coding Olympics MVP.

Three phases, three events, deterministic transitions:

```
IDLE  --start--> CODING  --end--> SCOREBOARD  --reset--> IDLE
```

Each successful transition fires an on-entry hook that writes the
compositor — e.g. entering `CODING` switches OBS to the coding scene
and writes the prompt/round-number/contestants to their text inputs.

Intentionally hand-rolled: a `Phase` enum, an `Event` enum, and a
`dict[(Phase, Event), Phase]` transition table. No library, no decorators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from obs_runner.compositor import CompositorProtocol
    from obs_runner.config import Config


CONTESTANT_SLOTS = 2


class Phase(StrEnum):
    """Game phases. String values are the wire format for `GET /state`."""

    IDLE = "idle"
    CODING = "coding"
    SCOREBOARD = "scoreboard"


class Event(StrEnum):
    """Transition triggers accepted by `StateMachine.dispatch`."""

    START = "start"
    END = "end"
    RESET = "reset"


_TRANSITIONS: dict[tuple[Phase, Event], Phase] = {
    (Phase.IDLE, Event.START): Phase.CODING,
    # `start` from SCOREBOARD rolls straight into the next round — a
    # producer rarely wants to park in IDLE between rounds. `start`
    # from CODING is still 409 because that would mask a forgotten END.
    (Phase.SCOREBOARD, Event.START): Phase.CODING,
    (Phase.CODING, Event.END): Phase.SCOREBOARD,
    # `reset` is a panic-button: always legal, always lands in IDLE.
    # Idempotent from IDLE so the UI can double-click without a 409.
    (Phase.IDLE, Event.RESET): Phase.IDLE,
    (Phase.CODING, Event.RESET): Phase.IDLE,
    (Phase.SCOREBOARD, Event.RESET): Phase.IDLE,
}


class InvalidTransitionError(Exception):
    """Raised when an event is not valid for the current phase."""


@dataclass
class Snapshot:
    """Serializable view of the current machine state.

    Carried forward across phases so `GET /state` can show what the
    current scene is displaying.
    """

    phase: Phase = Phase.IDLE
    prompt: str | None = None
    contestants: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)


class StateMachine:
    """Owns the current phase and fires compositor writes on entry.

    Not thread-safe — FastAPI serializes dispatch through the event loop.
    """

    def __init__(self, compositor: CompositorProtocol, config: Config) -> None:
        """Initialize the machine in `IDLE` without touching OBS.

        Args:
            compositor: Object implementing `set_scene` and `set_text`.
            config: Resolved scene/text-source names.
        """
        self._compositor = compositor
        self._config = config
        self._snapshot = Snapshot()

    @property
    def snapshot(self) -> Snapshot:
        """Return the latest snapshot (by reference — do not mutate)."""
        return self._snapshot

    def prime(self) -> None:
        """Push the IDLE scene so OBS reflects the starting phase.

        Called once on server startup. A `dispatch(start)` from IDLE is
        what normally triggers scene writes; without priming, OBS may
        linger on whatever scene the operator last selected manually.
        """
        self._enter_idle()

    def dispatch(self, event: Event, payload: dict[str, Any]) -> Snapshot:
        """Advance the machine by `event`, applying `payload` on entry.

        Args:
            event: Transition trigger.
            payload: Event-specific data. Shape is validated by the API
                layer before it reaches the machine.

        Returns:
            The post-transition snapshot.

        Raises:
            InvalidTransitionError: If `event` is not legal for the
                current phase.
        """
        current = self._snapshot.phase
        key = (current, event)
        if key not in _TRANSITIONS:
            msg = f"event '{event.value}' not valid from phase '{current.value}'"
            raise InvalidTransitionError(msg)

        target = _TRANSITIONS[key]
        if target is Phase.CODING:
            self._enter_coding(payload)
        elif target is Phase.SCOREBOARD:
            self._enter_scoreboard(payload)
        elif target is Phase.IDLE:
            self._enter_idle()
        return self._snapshot

    def _write_contestant_slots(
        self, contestants: list[str], scores: dict[str, float] | None
    ) -> None:
        """Write names + scores into per-slot sources; clear the rest.

        Slots are 1-indexed. Contestants beyond `CONTESTANT_SLOTS` are
        silently dropped — the Olympics layout can only show N at a time.

        Args:
            contestants: Ordered contestant names. Slot `i` receives
                `contestants[i-1]`.
            scores: Optional per-name score map. When provided, each
                slot's score source gets `"X.XX"`; otherwise cleared.
        """
        cfg = self._config
        for slot in range(1, CONTESTANT_SLOTS + 1):
            index = slot - 1
            name = contestants[index] if index < len(contestants) else ""
            self._compositor.set_text(cfg.name_source(slot), name)

            if scores is None or not name:
                score_text = ""
            else:
                value = scores.get(name)
                score_text = f"{value:.2f}" if value is not None else ""
            self._compositor.set_text(cfg.score_source(slot), score_text)

    def _enter_idle(self) -> None:
        """Clear round state and switch OBS to the idle scene."""
        self._snapshot = Snapshot(phase=Phase.IDLE)
        cfg = self._config
        self._compositor.set_scene(cfg.scenes[Phase.IDLE])
        self._compositor.set_text(cfg.text_prompt, "")
        self._write_contestant_slots([], None)

    def _enter_coding(self, payload: dict[str, Any]) -> None:
        """Switch to the coding scene and write round metadata."""
        prompt = str(payload.get("prompt", ""))
        contestants = list(payload.get("contestants") or [])

        self._snapshot = Snapshot(
            phase=Phase.CODING,
            prompt=prompt,
            contestants=contestants,
        )
        cfg = self._config
        self._compositor.set_scene(cfg.scenes[Phase.CODING])
        self._compositor.set_text(cfg.text_prompt, prompt)
        self._write_contestant_slots(contestants, None)

    def _enter_scoreboard(self, payload: dict[str, Any]) -> None:
        """Switch to the scoreboard scene and render per-slot scores.

        Scores are mapped to the same slot their contestant occupied in
        `CODING`, preserving visual position across the round.
        """
        raw = payload.get("scores") or {}
        scores = {str(k): float(v) for k, v in raw.items()}
        self._snapshot = Snapshot(
            phase=Phase.SCOREBOARD,
            prompt=self._snapshot.prompt,
            contestants=self._snapshot.contestants,
            scores=scores,
        )
        cfg = self._config
        self._compositor.set_scene(cfg.scenes[Phase.SCOREBOARD])
        self._write_contestant_slots(self._snapshot.contestants, scores)
