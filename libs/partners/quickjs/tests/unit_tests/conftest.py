from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from langchain_quickjs import middleware
from langchain_quickjs._repl import _SLOT_CLASSES, _Registry, _Slot, _SlotConfig

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
def make_repl() -> Iterator[Callable[..., _Slot]]:
    """Build slots outside the registry; closed when the test ends."""
    slots: list[_Slot] = []

    def _make(**config: Any) -> _Slot:
        slot_config = _SlotConfig(**config)
        slots.append(_SLOT_CLASSES[slot_config.mode]("quickjs-test", slot_config))
        return slots[-1]

    yield _make
    for slot in slots:
        slot.close()


@pytest.fixture
def repl(make_repl: Callable[..., _Slot]) -> _Slot:
    """A slot with default settings (`thread` mode)."""
    return make_repl()
