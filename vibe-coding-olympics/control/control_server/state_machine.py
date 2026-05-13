"""Round state machine for the Vibe Coding Olympics control plane.

Three phases, four events, deterministic transitions:

```
IDLE  --start--> CODING  --end--> SCOREBOARD  --reset--> IDLE
```

Each successful transition fires an on-entry hook that drives the
remote OBS compositor — e.g. entering `CODING` switches OBS to the
coding scene and writes the prompt/contestants to their text inputs.

The FSM lives in the control plane because every operator action that
changes phase already originates here (round start, end, reset, player
ready). OBS is a downstream renderer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from control_server.compositor import CompositorProtocol
    from control_server.state_config import StateConfig


CONTESTANT_SLOTS = 2


class Phase(StrEnum):
    """Game phases. String values are the wire format for state payloads."""

    IDLE = "idle"
    CODING = "coding"
    SCOREBOARD = "scoreboard"


class Event(StrEnum):
    """Transition triggers accepted by `StateMachine.dispatch`."""

    START = "start"
    END = "end"
    READY = "ready"
    RESET = "reset"


_TRANSITIONS: dict[tuple[Phase, Event], Phase] = {
    (Phase.IDLE, Event.READY): Phase.IDLE,
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
    """Serializable view of the current machine state."""

    phase: Phase = Phase.IDLE
    prompt: str | None = None
    contestants: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)


class StateMachine:
    """Owns the current phase and fires compositor writes on entry.

    Not thread-safe — FastAPI serializes dispatch through the event loop.
    """

    def __init__(self, compositor: CompositorProtocol, config: StateConfig) -> None:
        """Initialize the machine in `IDLE` without touching OBS.

        Args:
            compositor: Object implementing async `set_scene` and `set_text`.
            config: Scene names and text-source templates.
        """
        self._compositor = compositor
        self._config = config
        self._snapshot = Snapshot()

    @property
    def snapshot(self) -> Snapshot:
        """Return the latest snapshot (by reference — do not mutate)."""
        return self._snapshot

    async def prime(self) -> None:
        """Push the IDLE scene so OBS reflects the starting phase.

        Called once on server startup. Without priming, OBS may linger
        on whatever scene the operator last selected manually.
        """
        await self._enter_idle()

    async def dispatch(self, event: Event, payload: dict[str, Any]) -> Snapshot:
        """Advance the machine by `event`, applying `payload` on entry.

        Args:
            event: Transition trigger.
            payload: Event-specific data.

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
        if event is Event.READY:
            await self._enter_ready(payload)
        elif target is Phase.CODING:
            await self._enter_coding(payload)
        elif target is Phase.SCOREBOARD:
            await self._enter_scoreboard(payload)
        elif target is Phase.IDLE:
            await self._enter_idle()
        return self._snapshot

    def _contestants_from_payload(self, payload: dict[str, Any]) -> list[str]:
        """Return validated contestant names from an event payload."""
        raw = payload.get("contestants")
        if not isinstance(raw, list):
            msg = "payload must include a `contestants` list"
            raise InvalidTransitionError(msg)
        contestants = [name.strip() for name in raw if isinstance(name, str)]
        if not contestants:
            msg = "payload `contestants` must include at least one name"
            raise InvalidTransitionError(msg)
        return contestants

    async def _write_contestant_slots(
        self, contestants: list[str], scores: dict[str, float] | None
    ) -> None:
        """Write names + scores into configured per-slot sources."""
        cfg = self._config
        for slot in range(1, CONTESTANT_SLOTS + 1):
            index = slot - 1
            name = contestants[index] if index < len(contestants) else ""
            name_source = cfg.name_source(slot)
            if name_source is not None:
                await self._compositor.set_text(name_source, name)

            if scores is None or not name:
                score_text = ""
            else:
                value = scores.get(name)
                score_text = f"{value:.2f}" if value is not None else ""
            score_source = cfg.score_source(slot)
            if score_source is not None:
                await self._compositor.set_text(score_source, score_text)

    async def _enter_idle(self) -> None:
        """Clear round state and switch OBS to the configured idle scene."""
        self._snapshot = Snapshot(phase=Phase.IDLE)
        cfg = self._config
        await self._compositor.set_scene(cfg.scenes[Phase.IDLE])
        if cfg.text_prompt is not None:
            await self._compositor.set_text(cfg.text_prompt, "")
        await self._write_contestant_slots([], None)

    async def _enter_ready(self, payload: dict[str, Any]) -> None:
        """Render ready player names while waiting in the idle phase."""
        contestants = self._contestants_from_payload(payload)
        self._snapshot = Snapshot(phase=Phase.IDLE, contestants=contestants)
        cfg = self._config
        await self._compositor.set_scene(cfg.scenes[Phase.IDLE])
        await self._write_contestant_slots(contestants, None)

    async def _enter_coding(self, payload: dict[str, Any]) -> None:
        """Switch to the coding scene and write round metadata."""
        prompt = str(payload.get("prompt", ""))
        contestants = self._contestants_from_payload(payload)

        self._snapshot = Snapshot(
            phase=Phase.CODING,
            prompt=prompt,
            contestants=contestants,
        )
        cfg = self._config
        await self._compositor.set_scene(cfg.scenes[Phase.CODING])
        if cfg.text_prompt is not None:
            await self._compositor.set_text(cfg.text_prompt, prompt)
        await self._write_contestant_slots(contestants, None)

    async def _enter_scoreboard(self, payload: dict[str, Any]) -> None:
        """Switch to the scoreboard scene and render per-slot scores."""
        raw = payload.get("scores")
        if not isinstance(raw, Mapping):
            msg = "payload must include a `scores` object"
            raise InvalidTransitionError(msg)
        try:
            scores = {str(k): float(v) for k, v in raw.items()}
        except (TypeError, ValueError) as exc:
            msg = "payload `scores` values must be numeric"
            raise InvalidTransitionError(msg) from exc
        self._snapshot = Snapshot(
            phase=Phase.SCOREBOARD,
            prompt=self._snapshot.prompt,
            contestants=self._snapshot.contestants,
            scores=scores,
        )
        cfg = self._config
        await self._compositor.set_scene(cfg.scenes[Phase.SCOREBOARD])
        await self._write_contestant_slots(self._snapshot.contestants, scores)
