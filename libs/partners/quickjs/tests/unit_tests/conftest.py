from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import pytest

from langchain_quickjs import CodeInterpreterMiddleware, middleware
from langchain_quickjs._repl import _Registry, _Slot, _SlotConfig

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


@pytest.fixture(autouse=True, params=["worker", "inline"])
def execution(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """Run every unit test under both execution modes.

    Middleware and slots built without an explicit `execution` get this run's
    mode. Tests marked `worker_only` check worker threads and skip inline.
    """
    mode = request.param
    if mode == "inline" and request.node.get_closest_marker("worker_only"):
        pytest.skip("checks worker threads")
    init = CodeInterpreterMiddleware.__init__

    @functools.wraps(init)
    def init_with_mode(self: CodeInterpreterMiddleware, *args: Any, **kw: Any) -> None:
        kw.setdefault("execution", mode)
        init(self, *args, **kw)

    monkeypatch.setattr(CodeInterpreterMiddleware, "__init__", init_with_mode)
    return mode


@pytest.fixture(autouse=True)
def registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Registry]:
    """Give each test its own slot registry and close its slots afterwards."""
    reg = _Registry()
    monkeypatch.setattr(middleware, "_REGISTRY", reg)
    yield reg
    for slot in reg._slots.values():
        slot.close()


@pytest.fixture
def make_repl(execution: str) -> Iterator[Callable[..., _Slot]]:
    """Build slots outside the registry; closed when the test ends."""
    slots: list[_Slot] = []

    def _make(**config: Any) -> _Slot:
        config.setdefault("execution", execution)
        slots.append(_Slot("quickjs-test", _SlotConfig(**config)))
        return slots[-1]

    yield _make
    for slot in slots:
        slot.close()


@pytest.fixture
def repl(make_repl: Callable[..., _Slot]) -> _Slot:
    """A slot with default settings (`thread` mode)."""
    return make_repl()
