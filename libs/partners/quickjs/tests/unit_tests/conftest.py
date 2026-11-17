from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from langchain_quickjs import middleware
from langchain_quickjs._repl import _Registry, _SlotConfig, _TurnSlot

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


@pytest.fixture(autouse=True)
def registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Registry]:
    """Give each test its own slot registry and close its slots afterwards."""
    reg = _Registry()
    monkeypatch.setattr(middleware, "_REGISTRY", reg)
    yield reg
    for slot in reg._slots.values():
        slot.close()


@pytest.fixture
def make_repl() -> Iterator[Callable[..., _TurnSlot]]:
    """Build turn slots outside the registry; closed when the test ends."""
    slots: list[_TurnSlot] = []

    def _make(**config: Any) -> _TurnSlot:
        slots.append(_TurnSlot("quickjs-test", _SlotConfig(**config)))
        return slots[-1]

    yield _make
    for slot in slots:
        slot.close()


@pytest.fixture
def repl(make_repl: Callable[..., _TurnSlot]) -> _TurnSlot:
    """A turn slot with default settings: one Context per turn."""
    return make_repl()
