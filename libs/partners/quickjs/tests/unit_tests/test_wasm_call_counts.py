"""Upper bounds on WASM export calls for values crossing from JS to Python.

`quickjs-rs` marshals a JS value into Python one leaf at a time, and every
step is a WASM export call costing about 5 us. Counting the calls gives a
deterministic proxy for that cost: a codec fix shows up as a drop, and a
regression fails the bound. Lower a bound when a fix lands so it ratchets.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

import pytest
from quickjs_rs import Context, Runtime, _engine

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

# Counts measured with quickjs-rs 0.2.5. Each includes the fixed cost of one eval.
NUMBERS_1K_MAX = 11_019
ROWS_1K_MAX = 96_019
# Extra calls for one host call over an eval that makes none.
FS_ARGS_HOST_CALL_MAX = 84


@pytest.fixture
def export_calls(monkeypatch: pytest.MonkeyPatch) -> Counter[str]:
    """Count every WASM export call made through `_engine._Instance.c`."""
    calls: Counter[str] = Counter()
    original = _engine._Instance.c

    def counting(self: _engine._Instance, name: str, *args: Any) -> Any:
        calls[name] += 1
        return original(self, name, *args)

    monkeypatch.setattr(_engine._Instance, "c", counting)
    return calls


@pytest.fixture
def ctx(export_calls: Counter[str]) -> Iterator[Context]:
    # Depends on `export_calls` so the patch is in place before the runtime exists.
    with Runtime() as runtime:
        context = runtime.new_context(timeout=60)
        context.register("fsargs", lambda _op, _args: "x", is_async=False)
        context.eval("globalThis.nums = Array.from({length: 1000}, (_, i) => i * 1.5)")
        context.eval(
            "globalThis.rows = Array.from({length: 1000}, "
            "(_, i) => ({id: i, name: 'row-' + i, score: i / 7}))"
        )
        yield context


def _count(calls: Counter[str], run: Callable[[], object]) -> int:
    calls.clear()
    run()
    return sum(calls.values())


def test_returning_1k_numbers(ctx: Context, export_calls: Counter[str]) -> None:
    count = _count(export_calls, lambda: ctx.eval("nums"))
    # A count of zero means the patch no longer sees the calls.
    assert 0 < count <= NUMBERS_1K_MAX


def test_returning_1k_row_objects(ctx: Context, export_calls: Counter[str]) -> None:
    count = _count(export_calls, lambda: ctx.eval("rows"))
    assert 0 < count <= ROWS_1K_MAX


def test_host_call_with_fs_shaped_args(
    ctx: Context, export_calls: Counter[str]
) -> None:
    baseline = _count(export_calls, lambda: ctx.eval("1"))
    call = "fsargs('readFile', {path: '/a.txt', encoding: 'utf8'})"
    count = _count(export_calls, lambda: ctx.eval(call)) - baseline
    assert baseline > 0
    assert 0 < count <= FS_ARGS_HOST_CALL_MAX
